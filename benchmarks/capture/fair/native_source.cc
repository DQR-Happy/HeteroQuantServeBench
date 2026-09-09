// Common native, open-loop source. Only the selected real adapter is dlopened.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
using Clock = std::chrono::steady_clock;
static uint64_t now() { return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(); }
static void pin(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set)) throw std::runtime_error("affinity");
}
static void wait_until(uint64_t target) {
    auto t = now();
    if (target > t + 80000) {
        auto sleep_target = target - 30000;
        timespec ts{time_t(sleep_target / 1000000000), long(sleep_target % 1000000000)};
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
    }
    while (now() < target) std::this_thread::yield();
}
static bool sample(uint64_t i, uint64_t seed, uint64_t every) {
    if (!every) return false;
    uint64_t z = i + seed + 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return ((z ^ (z >> 31)) % every) == 0;
}
#pragma pack(push, 1)
struct Receipt { uint64_t i, begin, scheduled, duration; uint8_t code; };
#pragma pack(pop)
static_assert(sizeof(Receipt) == 33);
static double cpu_time(const rusage& u) { return u.ru_utime.tv_sec + u.ru_stime.tv_sec + (u.ru_utime.tv_usec + u.ru_stime.tv_usec) / 1e6; }

int main(int argc, char** argv) {
    try {
        if (argc != 13) throw std::runtime_error("lib encoded out run offsets bodies producers H_ns sample_every seed skew warmups");
        const std::string lib = argv[1], encoded = argv[2], out = argv[3], run = argv[4];
        const unsigned producers = std::stoul(argv[7]);
        const uint64_t horizon = std::stoull(argv[8]), every = std::stoull(argv[9]), seed = std::stoull(argv[10]);
        const bool skew = std::stoi(argv[11]);
        const unsigned warmups = std::stoul(argv[12]);
        if (!producers || producers > 16) throw std::runtime_error("producer bound");
        cpu_set_t mask; sched_getaffinity(0, sizeof(mask), &mask);
        std::vector<int> cpus;
        for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &mask)) cpus.push_back(i);
        if (cpus.size() != 4) throw std::runtime_error("requires exact four-CPU profile");
        std::vector<int> pc{cpus[0], cpus[2], cpus[3]};
        pin(cpus[1]); // All background threads inherit collector affinity.
        std::ifstream schedule(argv[5], std::ios::binary | std::ios::ate);
        auto bytes = schedule.tellg();
        if (bytes < 0 || bytes % 8 || bytes > 8 * 4000000) throw std::runtime_error("schedule bound");
        std::vector<uint64_t> offsets(size_t(bytes) / 8);
        schedule.seekg(0); schedule.read(reinterpret_cast<char*>(offsets.data()), bytes);
        std::ifstream bodies_file(argv[6], std::ios::binary);
        uint32_t count = 0; bodies_file.read(reinterpret_cast<char*>(&count), 4);
        if (!count || count > 100000) throw std::runtime_error("body template bound");
        std::vector<std::string> bodies;
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t length = 0; bodies_file.read(reinterpret_cast<char*>(&length), 4);
            if (length > 32 * 1024 * 1024) throw std::runtime_error("body size bound");
            std::string value(length, '\0'); bodies_file.read(value.data(), length);
            if (!bodies_file) throw std::runtime_error("truncated body templates");
            bodies.push_back(std::move(value));
        }
        std::vector<std::vector<uint64_t>> indices(producers);
        for (uint64_t i = 0; i < offsets.size(); ++i) {
            unsigned p = skew && producers > 1 ? (i % 100 < 90 ? 0 : 1 + (i % 100 - 90) % (producers - 1)) : i % producers;
            indices[p].push_back(i);
        }
        using Open = int(*)(const char*); using Offer = int(*)(const char*, uint32_t, uint64_t, uint64_t, const char*); using Close = int(*)();
        Open open_fn = nullptr; Offer offer = nullptr; Close close_fn = nullptr;
        const bool explicit_ack = std::getenv("HQSB_FAIR_ADAPTER") && std::string(std::getenv("HQSB_FAIR_ADAPTER")) == "hqsb";
        uint64_t startup = now();
        if (lib != "null") {
            void* handle = dlopen(lib.c_str(), RTLD_NOW | (lib.find("python_bridge") != std::string::npos ? RTLD_GLOBAL : RTLD_LOCAL));
            if (!handle) throw std::runtime_error(dlerror());
            bool hindsight = lib.find("hindsight") != std::string::npos;
            open_fn = reinterpret_cast<Open>(dlsym(handle, hindsight ? "hh_open" : "hf_open"));
            offer = reinterpret_cast<Offer>(dlsym(handle, hindsight ? "hh_offer" : "hf_offer"));
            close_fn = reinterpret_cast<Close>(dlsym(handle, hindsight ? "hh_close" : "hf_close"));
            if (!open_fn || !offer || !close_fn || open_fn(encoded.c_str())) throw std::runtime_error("adapter open");
        }
        uint64_t opened = now(), t0 = 0;
        std::atomic<unsigned> ready{0}; std::atomic<bool> go{false};
        std::vector<std::vector<Receipt>> receipts(producers);
        std::vector<uint64_t> logical(producers), errors(producers);
        std::vector<std::thread> threads;
        for (unsigned p = 0; p < producers; ++p) threads.emplace_back([&, p] {
            pin(pc[p % pc.size()]);
            // Pre-touch the same planned receipt budget, including unattempted debt.
            std::vector<Receipt> local(indices[p].size());
            for (unsigned w = 0; w < warmups; ++w) if (offer) offer("warmup", p, w, 0, bodies[w % bodies.size()].c_str());
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            size_t n = 0; uint64_t total = 0, bad = 0;
            for (auto i : indices[p]) {
                const uint64_t scheduled = t0 + offsets[i];
                wait_until(scheduled);
                if (now() > t0 + horizon + 250000000) break;
                const auto& body = bodies[i % bodies.size()];
                bool timed = sample(i, seed, every);
                uint64_t begin = timed ? now() : 0;
                int result = offer ? offer(run.c_str(), p, i, scheduled, body.c_str()) : 0;
                uint64_t end = timed ? now() : 0;
                const int code = !offer ? 7 : explicit_ack ? result : result == 1 ? 1 : result ? 6 : 1;
                local[n++] = {i, begin, scheduled, timed ? end - begin : 0, uint8_t(code)};
                total += 64 + body.size(); bad += code == 6;
            }
            local.resize(n); receipts[p] = std::move(local); logical[p] = total; errors[p] = bad;
        });
        while (ready.load(std::memory_order_acquire) != producers) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        rusage a{}, b{}; getrusage(RUSAGE_SELF, &a);
        t0 = now() + 50000000; go.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        auto attempts_end = now(), close_start = now();
        if (close_fn && close_fn()) throw std::runtime_error("adapter close/fsync");
        auto ended = now(); getrusage(RUSAGE_SELF, &b);
        uint64_t total = 0, bad = 0, attempted = 0;
        for (unsigned p = 0; p < producers; ++p) {
            std::ofstream file(out + "/receipts-" + std::to_string(p) + ".bin", std::ios::binary);
            file.write(reinterpret_cast<const char*>(receipts[p].data()), receipts[p].size() * sizeof(Receipt));
            if (!file) throw std::runtime_error("receipt I/O");
            total += logical[p]; bad += errors[p]; attempted += receipts[p].size();
        }
        std::ofstream report(out + "/native.json");
        report << std::setprecision(15) << "{\"scheduled\":" << offsets.size() << ",\"attempted\":" << attempted
               << ",\"t0_ns\":" << t0 << ",\"H_ns\":" << horizon << ",\"attempts_end_ns\":" << attempts_end
               << ",\"endpoint_end_ns\":" << ended << ",\"startup_ns\":" << opened-startup
               << ",\"close_start_ns\":" << close_start << ",\"logical_attempted_bytes\":" << total
               << ",\"adapter_errors\":" << bad << ",\"cpu_seconds\":" << cpu_time(b)-cpu_time(a)
               << ",\"maxrss_KiB\":" << b.ru_maxrss << ",\"minor_faults\":" << b.ru_minflt-a.ru_minflt
               << ",\"major_faults\":" << b.ru_majflt-a.ru_majflt << ",\"voluntary_switches\":" << b.ru_nvcsw-a.ru_nvcsw
               << ",\"involuntary_switches\":" << b.ru_nivcsw-a.ru_nivcsw << "}";
        return report ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

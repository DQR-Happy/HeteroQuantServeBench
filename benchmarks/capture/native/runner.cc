// Native CORE diagnostics. Same wire body and receipt format as Python track.
// No GPU/runtime dependency. Compile/link to the real NanoLog bridge.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <utility>
#include <unistd.h>
#include <vector>

extern "C" int hf_open(const char*) noexcept;
extern "C" int hf_offer(const char*, uint32_t, uint64_t, uint64_t, const char*) noexcept;
extern "C" int hf_close() noexcept;

using Clock = std::chrono::steady_clock;
static uint64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}
static void pin(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset)) throw std::runtime_error("affinity failed");
}
static void wait_until(uint64_t target) {
    auto now = now_ns();
    if (target > now + 100000) std::this_thread::sleep_for(std::chrono::nanoseconds(target - now - 50000));
    while (now_ns() < target) std::this_thread::yield();
}
#pragma pack(push, 1)
struct Receipt { uint64_t index, call, scheduled, duration; uint8_t code; };
#pragma pack(pop)
static_assert(sizeof(Receipt) == 33);

int main(int argc, char** argv) {
    try {
        if (argc != 11) throw std::runtime_error("adapter dir run rate seconds body producers seed arrival events");
        const std::string adapter = argv[1], dir = argv[2], run = argv[3], arrival = argv[9];
        const uint64_t rate = std::stoull(argv[4]), body_size = std::stoull(argv[6]), seed = std::stoull(argv[8]);
        const double seconds = std::stod(argv[5]);
        const unsigned producers = std::stoul(argv[7]);
        const uint64_t n = std::stoull(argv[10]);
        if ((adapter != "nanolog" && adapter != "null") || !rate || !producers || producers > 16 || n > 2000000 || body_size > 16384)
            throw std::runtime_error("unsupported native case");
        cpu_set_t allowed;
        if (sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("get affinity failed");
        std::vector<int> cpus;
        for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &allowed)) cpus.push_back(i);
        if (cpus.size() < 2) throw std::runtime_error("requires >=2 CPUs");
        const int collector = cpus[1];
        std::vector<int> producer_cpus{cpus[0]};
        for (size_t i = 2; i < cpus.size(); ++i) producer_cpus.push_back(cpus[i]);
        const auto startup = now_ns();
        if (adapter == "nanolog" && hf_open((dir + "/nanolog.bin").c_str())) throw std::runtime_error("open failed");
        const auto opened = now_ns();
        // Pin NanoLog's pre-existing background thread, not the benchmark's producers.
        if (DIR* tasks = opendir("/proc/self/task")) {
            while (auto* task = readdir(tasks)) {
                if (task->d_name[0] == '.') continue;
                int tid = std::stoi(task->d_name);
                if (tid == getpid()) continue;
                cpu_set_t one; CPU_ZERO(&one); CPU_SET(collector, &one);
                if (sched_setaffinity(tid, sizeof(one), &one)) throw std::runtime_error("collector affinity failed");
            }
            closedir(tasks);
        }
        std::vector<std::string> bodies;
        for (unsigned kind = 0; kind < 64; ++kind) {
            std::string name = "module_" + std::to_string(kind) + "_";
            name.resize(16, 'm');
            bodies.push_back("{\"name\":\"" + name + "\",\"data\":\"" + std::string(body_size, char('a' + kind % 26)) + "\"}");
        }
        std::atomic<bool> go{false};
        std::atomic<unsigned> ready{0};
        uint64_t t0 = 0, stop = 0;
        std::vector<std::vector<Receipt>> receipts(producers);
        std::vector<uint64_t> bytes(producers), errors(producers);
        std::vector<std::thread> threads;
        for (unsigned p = 0; p < producers; ++p) threads.emplace_back([&, p] {
            pin(producer_cpus[p % producer_cpus.size()]);
            std::vector<Receipt> local_receipts;
            local_receipts.reserve((n + producers - 1) / producers);
            uint64_t local_bytes = 0, local_errors = 0;
            if (adapter == "nanolog") for (unsigned i = 0; i < 100; ++i)
                hf_offer("warmup", p, i, 0, bodies[i % 64].c_str());
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            wait_until(t0);
            for (uint64_t i = p; i < n; i += producers) {
                const uint64_t scheduled = arrival == "closed" ? now_ns() : t0 + (i * 1000000000ull / rate);
                if (arrival != "closed") wait_until(scheduled);
                if (now_ns() > stop) break;
                uint64_t z = i + seed + 0x9e3779b97f4a7c15ull;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                bool sampled = ((z ^ (z >> 31)) & 63) == 0;
                const auto begin = sampled ? now_ns() : 0;
                int result = adapter == "nanolog" ? hf_offer(run.c_str(), p, i, scheduled, bodies[i % 64].c_str()) : 0;
                const auto end = sampled ? now_ns() : 0;
                local_receipts.push_back({i, sampled ? begin : 0, scheduled, sampled ? end - begin : 0,
                                      uint8_t(adapter == "null" ? 7 : result == 0 ? 1 : 6)});
                local_bytes += 64 + bodies[i % 64].size();
                local_errors += result != 0;
            }
            receipts[p] = std::move(local_receipts);
            bytes[p] = local_bytes;
            errors[p] = local_errors;
        });
        while (ready.load(std::memory_order_acquire) < producers) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        rusage before{}, after{};
        getrusage(RUSAGE_SELF, &before);
        t0 = now_ns() + 50000000;
        stop = t0 + uint64_t((seconds + 1) * 1e9);
        go.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        const auto attempts_end = now_ns(), close_start = now_ns();
        if (adapter == "nanolog" && hf_close()) throw std::runtime_error("close/fsync failed");
        const auto closed = now_ns();
        getrusage(RUSAGE_SELF, &after);
        double cpu = after.ru_utime.tv_sec - before.ru_utime.tv_sec + after.ru_stime.tv_sec - before.ru_stime.tv_sec
                     + (after.ru_utime.tv_usec - before.ru_utime.tv_usec + after.ru_stime.tv_usec - before.ru_stime.tv_usec) / 1e6;
        for (unsigned p = 0; p < producers; ++p) {
            std::ofstream stream(dir + "/receipts-" + std::to_string(p) + ".bin", std::ios::binary);
            stream.write(reinterpret_cast<const char*>(receipts[p].data()), receipts[p].size() * sizeof(Receipt));
            if (!stream) throw std::runtime_error("receipt write failed");
        }
        std::ostringstream config;
        config << "{\"id\":\"native-" << arrival << "-" << rate << "-b" << body_size << "-p" << producers
               << "\",\"workload\":\"W01/W02/W10\",\"run_id\":\"" << run << "\",\"seed\":" << seed
               << ",\"rate\":" << rate << ",\"duration_s\":" << seconds << ",\"arrival\":\"" << arrival
               << "\",\"body_bytes\":" << body_size << ",\"producers\":" << producers << "}";
        std::ofstream config_file(dir + "/config.json"); config_file << config.str();
        uint64_t total_bytes = 0; for (auto b : bytes) total_bytes += b;
        uint64_t total_errors = 0; for (auto value : errors) total_errors += value;
        std::ofstream out(dir + "/worker.json");
        out << std::setprecision(15) << "{\"case\":" << config.str() << ",\"adapter\":\"" << adapter
            << "\",\"track\":\"NATIVE_CORE\",\"scheduled\":" << n << ",\"errors\":"
            << (total_errors ? "[\"adapter_error\"]" : "[]") << ",\"acknowledgement\":\"unavailable_void_api\","
            << "\"H_ns\":" << (arrival == "closed" ? attempts_end - t0 : uint64_t(seconds * 1e9))
            << ",\"t0_ns\":" << t0 << ",\"attempts_end_ns\":" << attempts_end << ",\"endpoint_end_ns\":" << closed
            << ",\"startup_ns\":" << opened - startup << ",\"logical_attempted_bytes\":" << total_bytes
            << ",\"terminal\":{\"endpoint\":\"" << (adapter == "null" ? "discard_control" : "D2-close")
            << "\",\"close_start_ns\":" << close_start << ",\"close_end_ns\":" << closed << "},"
            << "\"cpu_seconds\":" << cpu << ",\"maxrss_KiB\":" << after.ru_maxrss
            << ",\"minor_faults\":" << after.ru_minflt - before.ru_minflt << ",\"major_faults\":" << after.ru_majflt - before.ru_majflt
            << ",\"voluntary_switches\":" << after.ru_nvcsw - before.ru_nvcsw << ",\"involuntary_switches\":" << after.ru_nivcsw - before.ru_nivcsw << "}";
        if (!out) throw std::runtime_error("report write failed");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

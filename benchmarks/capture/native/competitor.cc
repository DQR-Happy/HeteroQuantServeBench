// Controlled CPU/memory-bandwidth competitor, only inside a benchmark job.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <csignal>
#include <string>
#include <sys/prctl.h>
#include <unistd.h>
#include <thread>
#include <vector>

static volatile std::sig_atomic_t running = 1;
static void stop(int) { running = 0; }

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    auto parent = getppid();
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (parent == 1 || getppid() != parent) return 0;
    std::signal(SIGTERM, stop);
    size_t mib = std::stoul(argv[1]);
    double duty = std::stod(argv[2]);
    if (mib > 256 || mib < 1 || duty <= 0 || duty > 1) return 1;
    std::vector<uint64_t> memory(mib * 1024 * 1024 / sizeof(uint64_t), 1);
    volatile uint64_t* data = memory.data();
    while (running) {
        auto start = std::chrono::steady_clock::now();
        auto busy_end = start + std::chrono::microseconds(int(100000 * duty));
        while (running && std::chrono::steady_clock::now() < busy_end)
            for (size_t i = 0; i < memory.size(); i += 8) data[i] = data[i] + 1;
        if (duty < 1) std::this_thread::sleep_until(start + std::chrono::milliseconds(100));
    }
}

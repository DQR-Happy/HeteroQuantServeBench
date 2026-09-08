// Upstream Hindsight C client + C agent API, with a benchmark full-stream sink.
// This is NOT the paper's distributed trigger/Go collector deployment.
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
extern "C" {
#include "hindsight.h"
#include "agentapi.h"
}

static std::atomic<bool> stopping{false}, failed{false};
static std::thread consumer;
static HindsightAgentAPI* agent;
static std::string service, path;

#pragma pack(push, 1)
struct Header { char run[64]; uint32_t producer; uint64_t sequence, scheduled; uint32_t bytes; char reserved[8]; };
#pragma pack(pop)
static_assert(sizeof(Header) == 96);
static_assert(sizeof(TraceHeader) == 32, "decoder requires this recorded upstream ABI");

static void collect() {
    std::ofstream out(path, std::ios::binary);
    while (true) {
        CompleteBuffers batch{};
        hindsight_agentapi_get_complete_nonblocking(agent, &batch);
        if (!batch.count) {
            if (stopping.load(std::memory_order_acquire)) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            continue;
        }
        AvailableBuffers returned{};
        returned.count = batch.count;
        for (size_t i = 0; i < batch.count; ++i) {
            const int id = batch.bufs[i].buffer_id;
            if (id < 0 || size_t(id) >= agent->mgr.meta->capacity) { failed = true; return; }
            const char* src = agent->mgr.pool + size_t(id) * agent->mgr.meta->buffer_size;
            TraceHeader header{};
            std::memcpy(&header, src, sizeof(header));
            if (header.size < sizeof(header)
                    || header.size > agent->mgr.meta->buffer_size) { failed = true; return; }
            uint32_t size = header.size;
            out.write(reinterpret_cast<const char*>(&size), sizeof(size));
            out.write(src, size);
            returned.bufs[i].buffer_id = id;
        }
        if (!out) { failed = true; return; }
        hindsight_agentapi_put_available_blocking(agent, &returned);
    }
    out.flush();
    if (!out) failed = true;
}

extern "C" int hh_open(const char* filename) noexcept {
    try {
        path = filename;
        service = "hqsbbench_" + std::to_string(getpid());
        HindsightConfig config = hindsight_default_config();
        config.pool_capacity = 8192; config.buffer_size = 4096; // 32MiB pool.
        config.breadcrumbs_capacity = 8192; config.triggers_capacity = 8192;
        static char address[] = "127.0.0.1:0";
        config.address = address;
        hindsight_init_with_config(service.c_str(), config);
        agent = hindsight_agentapi_init(service.c_str());
        consumer = std::thread(collect);
        return 0;
    } catch (...) { return -1; }
}

extern "C" int hh_offer(const char* run, uint32_t producer, uint64_t seq,
                         uint64_t scheduled, const char* body) noexcept {
    try {
        Header header{};
        if (std::strlen(run) >= sizeof(header.run) || std::strlen(body) > 32 * 1024 * 1024) return -1;
        std::memcpy(header.run, run, std::strlen(run));
        header.producer = producer; header.sequence = seq; header.scheduled = scheduled;
        header.bytes = std::strlen(body);
        uint64_t trace = run[0] == 'w' ? (uint64_t(1) << 63) | (uint64_t(producer) << 32) | seq : seq + 1;
        hindsight_begin(trace);
        hindsight_tracepoint(reinterpret_cast<char*>(&header), sizeof(header));
        hindsight_tracepoint(const_cast<char*>(body), header.bytes);
        hindsight_end();
        return failed ? -1 : 0; // API is void: no per-record accepted acknowledgement.
    } catch (...) { return -1; }
}

extern "C" int hh_close() noexcept {
    try {
        stopping.store(true, std::memory_order_release);
        consumer.join();
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return -1;
        int status = ::fsync(fd);
        if (::close(fd)) status = -1;
        for (const char* suffix : {"pool", "available_queue", "complete_queue", "breadcrumbs_queue", "triggers_queue"})
            if (::unlink(("/dev/shm/" + service + "__" + suffix).c_str()) && errno != ENOENT) status = -1;
        return failed ? -1 : status;
    } catch (...) { return -1; }
}

// Real in-process Perfetto recording service and SDK; official decoder separate.
#include "perfetto.h"
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <unistd.h>
PERFETTO_DEFINE_CATEGORIES(perfetto::Category("hqsb"));
PERFETTO_TRACK_EVENT_STATIC_STORAGE();
static std::unique_ptr<perfetto::TracingSession> session;
static int output_fd = -1;
extern "C" int hf_open(const char* path) noexcept {
    try {
        int profile = std::getenv("HQSB_FAIR_PROFILE") ? std::atoi(std::getenv("HQSB_FAIR_PROFILE")) : 1;
        const uint32_t buffers[] = {4096, 16384, 65536};
        const uint32_t periods[] = {10, 50, 100};
        if (profile < 0 || profile > 2) return -1;
        perfetto::TracingInitArgs args;
        args.backends = perfetto::kInProcessBackend;
        args.shmem_size_hint_kb = 4096;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();
        perfetto::TraceConfig config;
        auto* buffer = config.add_buffers();
        buffer->set_size_kb(buffers[profile]);
        // Official streaming mode uses RING_BUFFER; overrun is counted as loss
        // by the independent per-identity oracle, never treated as sampling.
        buffer->set_fill_policy(perfetto::protos::gen::TraceConfig::BufferConfig::RING_BUFFER);
        config.set_write_into_file(true);
        config.set_file_write_period_ms(periods[profile]);
        config.add_data_sources()->mutable_config()->set_name("track_event");
        output_fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (output_fd < 0) return -1;
        session = perfetto::Tracing::NewTrace(perfetto::kInProcessBackend);
        session->Setup(config, output_fd);
        session->StartBlocking();
        return 0;
    } catch (...) { return -1; }
}
extern "C" int hf_offer(const char* run, uint32_t p, uint64_t i, uint64_t ts, const char* body) noexcept {
    try {
        TRACE_EVENT_INSTANT("hqsb", "capture", "run_id", perfetto::DynamicString(run),
                           "producer_id", p, "source_seq", i, "source_timestamp", ts,
                           "body", perfetto::DynamicString(body));
        return 0;
    } catch (...) { return -1; }
}
extern "C" int hf_close() noexcept {
    try {
        perfetto::TrackEvent::Flush();
        bool ok = session->FlushBlocking(30000);
        session->StopBlocking();
        int status = fsync(output_fd);
        if (close(output_fd)) status = -1;
        return ok ? status : -1;
    } catch (...) { return -1; }
}

#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#include "lttng_provider.h"
#include <chrono>
#include <thread>
extern "C" int hf_open(const char*) noexcept {
    for (int i = 0; i < 500; ++i) {
        if (lttng_ust_tracepoint_enabled(hqsb_capture, event)) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return -1; // Never benchmark an accidentally disabled tracepoint as success.
}
extern "C" int hf_offer(const char* r, uint32_t p, uint64_t i, uint64_t ts, const char* b) noexcept {
    lttng_ust_tracepoint(hqsb_capture, event, r, p, i, ts, b);
    return 0;
}
extern "C" int hf_close() noexcept { return 0; } // Controller stop/destroy/fsync is D2.

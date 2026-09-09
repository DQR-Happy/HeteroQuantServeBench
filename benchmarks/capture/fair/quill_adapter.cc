// Real Quill frontend/backend. Formatting and required argument copies included.
#include <cstdint>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include "quill/Backend.h"
#include "quill/Frontend.h"
#include "quill/LogMacros.h"
#include "quill/sinks/FileSink.h"
#ifndef HQSB_QUILL_QUEUE_BYTES
#define HQSB_QUILL_QUEUE_BYTES 131072
#endif
struct Options : quill::FrontendOptions {
    static constexpr size_t initial_queue_capacity = HQSB_QUILL_QUEUE_BYTES;
    static constexpr size_t unbounded_queue_max_capacity = 64ull * 1024 * 1024;
};
using Frontend = quill::FrontendImpl<Options>;
static quill::LoggerImpl<Options>* logger;
static std::string filename;
extern "C" int hf_open(const char* path) noexcept {
    try {
        filename = path;
        quill::Backend::start();
        quill::FileSinkConfig cfg;
        cfg.set_open_mode('w');
        auto sink = Frontend::create_or_get_sink<quill::FileSink>(filename, cfg);
        quill::PatternFormatterOptions format;
        format.format_pattern = "%(message)";
        logger = Frontend::create_or_get_logger("hqsb", sink, format);
        return 0;
    } catch (...) { return -1; }
}
extern "C" int hf_offer(const char* run, uint32_t p, uint64_t i, uint64_t ts, const char* body) noexcept {
    try {
        LOG_INFO(logger, "HQSB|{}|{}|{}|{}|{}", run, p, i, ts, body);
        return 0; // No per-record acceptance acknowledgement in this contract.
    } catch (...) { return -1; }
}
extern "C" int hf_close() noexcept {
    try {
        logger->flush_log();
        quill::Backend::stop();
        int fd = open(filename.c_str(), O_RDONLY);
        if (fd < 0) return -1;
        int status = fsync(fd);
        if (close(fd)) status = -1;
        return status;
    } catch (...) { return -1; }
}

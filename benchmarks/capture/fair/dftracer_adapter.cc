// Official DFTracer public C custom-region ABI. Wheel omits transitive
// cpp-logger headers; these declarations match its installed dftracer.h.
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <string>
#include <unistd.h>
extern "C" {
struct DFTracerData;
void initialize_no_bind(const char*, const char*, int*);
void mark_used(const char*);
void finalize();
DFTracerData* initialize_region(const char*, const char*, int);
void update_metadata_int(DFTracerData*, const char*, int);
void update_metadata_string(DFTracerData*, const char*, const char*);
void finalize_region(DFTracerData*);
}
static std::string prefix;
static std::string json_string_contents(const char* input) {
    // Stock metadata writer interpolates strings into JSON without escaping.
    // Canonical input bodies are JSON text; escape them inside the timed call.
    std::string output;
    for (; *input; ++input) {
        if (*input == '\\' || *input == '"') output += '\\';
        output += *input;
    }
    return output;
}
extern "C" int hf_open(const char* path) noexcept {
    try {
        prefix = path;
        setenv("DFTRACER_ENABLE", "1", 1);
        setenv("DFTRACER_INIT", "FUNCTION", 1);
        setenv("DFTRACER_DISABLE_IO", "1", 1);
        setenv("DFTRACER_INC_METADATA", "1", 1);
        setenv("DFTRACER_ENABLE_AGGREGATION", "0", 1);
        int profile = std::getenv("HQSB_FAIR_PROFILE") ? std::atoi(std::getenv("HQSB_FAIR_PROFILE")) : 1;
        setenv("DFTRACER_TRACE_COMPRESSION", profile == 0 ? "0" : "1", 1);
        // Profile 2 varies the documented collection interval, not record content.
        setenv("DFTRACER_TRACE_INTERVAL_MS", profile == 2 ? "50" : "10", 1);
        initialize_no_bind(path, nullptr, nullptr);
        mark_used("CPP_APP");
        return 0;
    } catch (...) { return -1; }
}
extern "C" int hf_offer(const char* run, uint32_t p, uint64_t i, uint64_t ts, const char* body) noexcept {
    try {
        const auto sequence = std::to_string(i), timestamp = std::to_string(ts);
        const auto escaped_body = json_string_contents(body);
        auto* event = initialize_region("capture", "HQSB", 0);
        update_metadata_string(event, "run_id", run);
        update_metadata_int(event, "producer_id", int(p));
        // Stock API has int32 and strings; decimal strings preserve uint64 fields.
        update_metadata_string(event, "source_seq", sequence.c_str());
        update_metadata_string(event, "source_timestamp", timestamp.c_str());
        update_metadata_string(event, "body", escaped_body.c_str());
        finalize_region(event);
        return 0;
    } catch (...) { return -1; }
}
extern "C" int hf_close() noexcept {
    try {
        finalize();
        auto base = std::filesystem::path(prefix);
        for (const auto& item : std::filesystem::directory_iterator(base.parent_path())) {
            if (!item.is_regular_file() || item.path().filename().string().find(base.filename().string()) != 0) continue;
            int fd = open(item.path().c_str(), O_RDONLY);
            if (fd < 0) return -1;
            int result = fsync(fd); close(fd);
            if (result) return -1;
        }
        return 0;
    } catch (...) { return -1; }
}

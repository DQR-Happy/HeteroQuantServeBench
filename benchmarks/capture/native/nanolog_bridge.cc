// Real NanoLog C++17 adapter. No C++ exception may cross the C ABI.
#include <array>
#include "NanoLogCpp17.h"
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <string>

static std::string output_path;

extern "C" int hf_open(const char* path) noexcept {
    try {
        output_path = path;
        NanoLog::setLogFile(path);
        NanoLog::setLogLevel(NanoLog::LogLevels::NOTICE);
        return 0;
    } catch (...) { return -1; }
}

extern "C" int hf_offer(const char* run, uint32_t producer, uint64_t seq,
                         uint64_t scheduled, const char* body) noexcept {
    try {
        // ASCII body JSON is NUL-free; all fields survive offline decompression.
        NANO_LOG(NOTICE, "HQSB|%s|%u|%llu|%llu|%s\n",
                 run, producer, static_cast<unsigned long long>(seq),
                 static_cast<unsigned long long>(scheduled), body);
        return 0; // submitted, not an acknowledgement promised by NanoLog.
    } catch (...) { return -1; }
}

extern "C" int hf_close() noexcept {
    try {
        NanoLog::sync();
        int fd = ::open(output_path.c_str(), O_RDONLY);
        if (fd < 0) return -1;
        int result = ::fsync(fd);
        if (::close(fd) != 0) return -1;
        return result;
    } catch (...) { return -1; }
}

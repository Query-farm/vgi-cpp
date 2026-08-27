// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "portable.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <process.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vgi::portable {

int64_t current_process_id() {
#if defined(_WIN32)
    return static_cast<int64_t>(::_getpid());
#else
    return static_cast<int64_t>(::getpid());
#endif
}

std::string current_user_tag() {
#if defined(_WIN32)
    const char* user = std::getenv("USERNAME");
    return user ? std::string(user) : std::string("unknown");
#else
    return std::to_string(static_cast<uint64_t>(::getuid()));
#endif
}

bool TryClaimFile(const std::string& path, const std::string& context) {
#if defined(_WIN32)
    HANDLE h = ::CreateFileA(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD err = ::GetLastError();
        if (err == ERROR_FILE_EXISTS) return false;
        throw std::runtime_error(context + ": cannot claim " + path + ": Windows error " +
                                  std::to_string(err));
    }
    ::CloseHandle(h);
    return true;
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        if (errno == EEXIST) return false;
        throw std::runtime_error(context + ": cannot claim " + path + ": " + std::strerror(errno));
    }
    ::close(fd);
    return true;
#endif
}

}  // namespace vgi::portable

// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Small OS-portability helpers for the handful of places dispatcher code
// needs something POSIX-only (`<unistd.h>`'s `getpid()`, an atomic
// exclusive-create used as a cross-process claim primitive) - kept here,
// not spread across catalog.cpp/function_dispatch.cpp/storage.cpp as
// per-file `#ifdef _WIN32` blocks, so there is exactly one place that knows
// what the Windows equivalent of each is.

#pragma once

#include <cstdint>
#include <string>

namespace vgi::portable {

// Cross-platform process id (`::getpid()` on POSIX, `_getpid()` on
// Windows) - used only to seed unique-enough ids (transaction/execution
// ids, temp-file suffixes) alongside a counter/timestamp, never as a real
// identity or security boundary.
int64_t current_process_id();

// A short, stable-per-user tag for namespacing a shared temp directory -
// POSIX's `getuid()` as a decimal string. `std::filesystem::temp_directory_
// path()` is a SHARED, system-wide location on POSIX (`/tmp`), so this tag
// is what keeps two different local users' worker state from colliding
// there. Windows has no equivalent need (its temp directory is already
// per-user, e.g. `%LOCALAPPDATA%\Temp`) - this returns the `USERNAME`
// environment variable there instead, purely for a human-readable
// directory name, not because collision-avoidance requires it.
std::string current_user_tag();

// Atomically creates `path` as a new, empty file, returning false (not
// throwing) if it already exists - the same "exclusive create" claim
// primitive on both platforms (POSIX `open()` with `O_CREAT|O_EXCL`;
// Windows `CreateFileA` with `CREATE_NEW`), used as a lightweight
// cross-process claim/lock. Throws `std::runtime_error` (message prefixed
// with `context`) on any OTHER failure.
bool TryClaimFile(const std::string& path, const std::string& context);

}  // namespace vgi::portable

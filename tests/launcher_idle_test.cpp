// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
//
// A launched worker exits once it has been idle for --idle-timeout.
//
// The DuckDB extension's launcher (vgi_rpc.launcher in the reference) spawns
// `<worker> --unix PATH --idle-timeout SEC`, keeps no handle on the process,
// and relies on the worker to exit SEC seconds after its last client leaves.
// This worker used to accept --idle-timeout and ignore it, so every launched
// C++ worker lived until it was killed. These tests launch the built example
// worker the way the launcher does and wait for it to leave on its own.

#include <catch2/catch_test_macros.hpp>

#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <vgi_rpc/client.h>

#include "vgi/generated/vgi_protocol_names.hpp"
#include "vgi/generated/vgi_protocol_version.hpp"

#ifndef VGI_EXAMPLE_WORKER_PATH
#error "VGI_EXAMPLE_WORKER_PATH must name the built vgi-example-worker"
#endif

namespace {

using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

const std::string kWorker = VGI_EXAMPLE_WORKER_PATH;

// The worker, started with fork/exec as the launcher starts it, with stdout
// and stderr captured. Killed on destruction if it is still running.
class LaunchedWorker {
public:
    explicit LaunchedWorker(const std::vector<std::string>& args) {
        int out[2];
        int err[2];
        REQUIRE(::pipe(out) == 0);
        REQUIRE(::pipe(err) == 0);
        pid_ = ::fork();
        REQUIRE(pid_ >= 0);
        if (pid_ == 0) {
            ::dup2(out[1], STDOUT_FILENO);
            ::dup2(err[1], STDERR_FILENO);
            ::close(out[0]);
            ::close(out[1]);
            ::close(err[0]);
            ::close(err[1]);
            std::vector<std::string> owned{kWorker};
            owned.insert(owned.end(), args.begin(), args.end());
            std::vector<char*> argv;
            for (auto& arg : owned) argv.push_back(arg.data());
            argv.push_back(nullptr);
            ::execv(kWorker.c_str(), argv.data());
            ::_exit(127);
        }
        ::close(out[1]);
        ::close(err[1]);
        stdout_ = out[0];
        stderr_ = err[0];
    }

    ~LaunchedWorker() {
        if (pid_ > 0 && !status_) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
        if (stdout_ >= 0) ::close(stdout_);
        if (stderr_ >= 0) ::close(stderr_);
    }

    LaunchedWorker(const LaunchedWorker&) = delete;
    LaunchedWorker& operator=(const LaunchedWorker&) = delete;

    // The path from the `UNIX:<path>` line, printed once the socket listens.
    std::string read_unix_line() {
        const auto deadline = Clock::now() + 30s;
        std::string line;
        while (Clock::now() < deadline) {
            pollfd pfd{stdout_, POLLIN, 0};
            if (::poll(&pfd, 1, 100) <= 0) continue;
            char c = 0;
            if (::read(stdout_, &c, 1) != 1) break;  // exited before binding
            if (c != '\n') {
                line.push_back(c);
                continue;
            }
            if (line.rfind("UNIX:", 0) == 0) return line.substr(5);
            line.clear();
        }
        FAIL("worker never printed UNIX:<path>");
        return {};
    }

    // The wait status once the worker exits, or nullopt if it is still running
    // at `limit`.
    std::optional<int> wait_for_exit(std::chrono::milliseconds limit) {
        const auto deadline = Clock::now() + limit;
        while (!status_) {
            int status = 0;
            const pid_t done = ::waitpid(pid_, &status, WNOHANG);
            REQUIRE(done >= 0);
            if (done == pid_) {
                status_ = status;
                break;
            }
            if (Clock::now() >= deadline) break;
            ::usleep(10'000);
        }
        return status_;
    }

    std::string drain_stderr() {
        std::string text;
        char buffer[512];
        while (true) {
            pollfd pfd{stderr_, POLLIN, 0};
            if (::poll(&pfd, 1, 0) <= 0) break;
            const auto got = ::read(stderr_, buffer, sizeof(buffer));
            if (got <= 0) break;
            text.append(buffer, static_cast<size_t>(got));
        }
        return text;
    }

private:
    pid_t pid_ = -1;
    int stdout_ = -1;
    int stderr_ = -1;
    std::optional<int> status_;
};

// A private directory for the socket, removed afterwards. Short, because a
// socket path is capped near 104 bytes and a CI workspace path is not.
class TempDir {
public:
    TempDir() {
        auto pattern = (std::filesystem::temp_directory_path() / "vgi-idle-XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        path_ = pattern;
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

vgi_rpc::RpcClientOptions vgi_options() {
    vgi_rpc::RpcClientOptions options;
    options.protocol = std::string(::vgi::generated::VGI_PROTOCOL_NAME);
    options.protocol_version = std::string(::vgi::generated::VGI_PROTOCOL_VERSION);
    return options;
}

}  // namespace

TEST_CASE("launcher: a worker exits after --idle-timeout once its client leaves",
          "[launcher][idle]") {
    TempDir dir;
    const auto socket = (dir.path() / "w.sock").string();
    // The flags exactly as the launcher appends them.
    LaunchedWorker worker({"--unix", socket, "--idle-timeout", "0.5"});
    REQUIRE(worker.read_unix_line() == socket);

    {
        auto client = vgi_rpc::RpcClient::connect_unix(socket, vgi_options());
        CHECK_FALSE(client.list_protocols().empty());
        // Connected for longer than the timeout: a client holds the worker.
        CHECK_FALSE(worker.wait_for_exit(1500ms).has_value());
        client.close();
    }

    const auto status = worker.wait_for_exit(10s);
    REQUIRE(status.has_value());
    CHECK(WIFEXITED(*status));
    CHECK(WEXITSTATUS(*status) == 0);
    CHECK_FALSE(std::filesystem::exists(socket));
}

TEST_CASE("launcher: a malformed --idle-timeout is refused at startup", "[launcher][idle]") {
    TempDir dir;
    const auto socket = (dir.path() / "w.sock").string();
    for (const std::string value : {"soon", "-1", "5s", "nan", "inf", ""}) {
        LaunchedWorker worker({"--unix", socket, "--idle-timeout", value});
        const auto status = worker.wait_for_exit(10s);
        REQUIRE(status.has_value());
        CHECK(WIFEXITED(*status));
        CHECK(WEXITSTATUS(*status) == 2);
        CHECK(worker.drain_stderr().find("--idle-timeout") != std::string::npos);
        CHECK_FALSE(std::filesystem::exists(socket));
    }
}

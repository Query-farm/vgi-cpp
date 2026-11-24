// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "storage_seal.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <system_error>

#include <openssl/evp.h>
#include <vgi_rpc/crypto.h>

#include "portable.h"
#include "vgi/storage.h"

#if defined(_WIN32)
#include <io.h>
#else
#include <fcntl.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vgi::storage_seal {
namespace {

namespace fs = std::filesystem;

void append_lp(std::string& out, const std::string& field) {
    const auto n = static_cast<uint32_t>(field.size());
    out.push_back(static_cast<char>((n >> 24) & 0xFF));
    out.push_back(static_cast<char>((n >> 16) & 0xFF));
    out.push_back(static_cast<char>((n >> 8) & 0xFF));
    out.push_back(static_cast<char>(n & 0xFF));
    out += field;
}

struct CipherCtx {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    ~CipherCtx() { EVP_CIPHER_CTX_free(ctx); }
};

std::string describe(const Slot& slot) {
    return "framework storage payload " + slot.ns + " for execution " +
           vgi_rpc::crypto::hex_encode(reinterpret_cast<const uint8_t*>(slot.scope_id.data()),
                                       slot.scope_id.size());
}

std::string key_file_error(const std::string& path, const std::string& problem) {
    return "framework storage payloads need the host-local key file " + path + ", but " + problem +
           ". Fix or remove it, or configure VGI_SIGNING_KEY or " + "VGI_ATTACH_SECRET_KEY_FILE.";
}

std::string env_or_empty(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string{};
}

#if !defined(_WIN32)
void check_private(const std::string& path, const struct stat& st, const char* what) {
    if (st.st_uid != ::getuid()) {
        throw StoragePayloadError(
            key_file_error(path, std::string("the ") + what + " is not owned by the current user"));
    }
    if ((st.st_mode & 077) != 0) {
        throw StoragePayloadError(key_file_error(
            path, std::string("the ") + what + " is accessible to group or other users"));
    }
}
#endif

// A private temp file hard-linked into place, never overwriting: a racing
// process that wins keeps its key, and everyone reads that one.
void create_key_file(const fs::path& path) {
    const auto random = vgi_rpc::crypto::random_bytes(kKeyBytes);
    const auto suffix = vgi_rpc::crypto::random_bytes(6);
    const fs::path tmp =
        path.parent_path() / ("." + path.filename().string() + "." +
                              std::to_string(vgi::portable::current_process_id()) + "." +
                              vgi_rpc::crypto::hex_encode(suffix.data(), suffix.size()) + ".tmp");
#if defined(_WIN32)
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(random.data()),
                  static_cast<std::streamsize>(random.size()));
        if (!out) throw StoragePayloadError(key_file_error(path.string(), "it cannot be created"));
    }
    std::error_code ec;
    if (!fs::exists(path, ec)) fs::rename(tmp, path, ec);
    fs::remove(tmp, ec);
#else
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        throw StoragePayloadError(key_file_error(
            path.string(), std::string("it cannot be created (") + std::strerror(errno) + ")"));
    }
    const bool wrote =
        ::write(fd, random.data(), random.size()) == static_cast<ssize_t>(random.size()) &&
        ::fsync(fd) == 0;
    ::close(fd);
    if (!wrote) {
        ::unlink(tmp.c_str());
        throw StoragePayloadError(key_file_error(path.string(), "it cannot be written"));
    }
    if (::link(tmp.c_str(), path.c_str()) != 0 && errno != EEXIST) {
        // No hard links on this filesystem: exclusive create of the target.
        const int fd2 = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        if (fd2 >= 0) {
            const bool ok =
                ::write(fd2, random.data(), random.size()) == static_cast<ssize_t>(random.size());
            ::close(fd2);
            if (!ok) {
                ::unlink(tmp.c_str());
                throw StoragePayloadError(key_file_error(path.string(), "it cannot be written"));
            }
        }
    }
    ::unlink(tmp.c_str());
#endif
}

std::string load_host_local_key() {
    static std::mutex mutex;
    static std::map<std::string, std::string> cache;
    const std::string path_text = host_local_key_path();
    std::lock_guard<std::mutex> lock(mutex);
    if (auto it = cache.find(path_text); it != cache.end()) return it->second;

    const fs::path path(path_text);
    std::error_code ec;
    bool created_directory = false;
    if (path.has_parent_path()) {
        created_directory = fs::create_directories(path.parent_path(), ec);
        if (ec) {
            throw StoragePayloadError(key_file_error(
                path_text, "its directory cannot be created (" + ec.message() + ")"));
        }
    }
#if !defined(_WIN32)
    if (env_or_empty("VGI_ATTACH_SECRET_KEY_FILE").empty()) {
        // Created 0700 (as vgi-python's mkdir(mode=0o700)); a pre-existing
        // directory that someone else owns or can read is refused, never used.
        if (created_directory) ::chmod(path.parent_path().c_str(), 0700);
        struct stat dir_st{};
        if (::stat(path.parent_path().c_str(), &dir_st) != 0) {
            throw StoragePayloadError(key_file_error(path_text, "its directory cannot be read"));
        }
        check_private(path_text, dir_st, "directory");
    }
#endif
    if (!fs::exists(path, ec)) create_key_file(path);
#if !defined(_WIN32)
    struct stat st{};
    if (::lstat(path.c_str(), &st) != 0) {
        throw StoragePayloadError(key_file_error(
            path_text, std::string("it cannot be read (") + std::strerror(errno) + ")"));
    }
    if (!S_ISREG(st.st_mode)) {
        throw StoragePayloadError(key_file_error(path_text, "it is not a regular file"));
    }
    check_private(path_text, st, "file");
#endif
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    std::string data = buffer.str();
    if (!in.good() && !in.eof()) {
        throw StoragePayloadError(key_file_error(path_text, "it cannot be read"));
    }
    if (data.size() != kKeyBytes) {
        throw StoragePayloadError(key_file_error(path_text, "it does not hold exactly 32 bytes"));
    }
    cache.emplace(path_text, data);
    return data;
}

}  // namespace

std::string pack_int_key(int64_t value) {
    std::string out(8, '\0');
    auto bits = static_cast<uint64_t>(value);
    for (size_t i = 0; i < 8; ++i) {
        out[i] = static_cast<char>(bits & 0xFF);
        bits >>= 8;
    }
    return out;
}

std::array<uint8_t, kKeyBytes> derive_key(const std::string& ikm) {
    // RFC 5869 with an empty salt (= HashLen zero bytes); one output block.
    const std::array<uint8_t, 32> salt{};
    const auto prk = vgi_rpc::crypto::hmac_sha256(
        salt.data(), salt.size(), reinterpret_cast<const uint8_t*>(ikm.data()), ikm.size());
    std::string info(kInfo);
    info.push_back('\x01');
    return vgi_rpc::crypto::hmac_sha256(prk.data(), prk.size(),
                                        reinterpret_cast<const uint8_t*>(info.data()), info.size());
}

std::string aad(const Slot& slot) {
    std::string out(kInfo);
    append_lp(out, slot.shard_key);
    append_lp(out, slot.scope_id);
    append_lp(out, slot.ns);
    append_lp(out, slot.key);
    return out;
}

std::string seal_with(const std::string& payload, const std::string& ikm, const Slot& slot,
                      const std::optional<std::array<uint8_t, kNonceBytes>>& nonce) {
    std::array<uint8_t, kNonceBytes> iv{};
    if (nonce) {
        iv = *nonce;
    } else {
        const auto random = vgi_rpc::crypto::random_bytes(kNonceBytes);
        std::copy(random.begin(), random.end(), iv.begin());
    }
    const auto key = derive_key(ikm);
    const auto associated = aad(slot);

    CipherCtx c;
    std::string out(1 + kNonceBytes + payload.size() + kTagBytes, '\0');
    out[0] = static_cast<char>(kVersion);
    std::memcpy(out.data() + 1, iv.data(), kNonceBytes);
    auto* body = reinterpret_cast<unsigned char*>(out.data() + 1 + kNonceBytes);
    int len = 0;
    int total = 0;
    const bool ok =
        c.ctx && EVP_EncryptInit_ex(c.ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1 &&
        EVP_CIPHER_CTX_ctrl(c.ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(kNonceBytes),
                            nullptr) == 1 &&
        EVP_EncryptInit_ex(c.ctx, nullptr, nullptr, key.data(), iv.data()) == 1 &&
        EVP_EncryptUpdate(c.ctx, nullptr, &len,
                          reinterpret_cast<const unsigned char*>(associated.data()),
                          static_cast<int>(associated.size())) == 1 &&
        EVP_EncryptUpdate(c.ctx, body, &len, reinterpret_cast<const unsigned char*>(payload.data()),
                          static_cast<int>(payload.size())) == 1 &&
        (total = len, EVP_EncryptFinal_ex(c.ctx, body + total, &len) == 1) &&
        EVP_CIPHER_CTX_ctrl(c.ctx, EVP_CTRL_GCM_GET_TAG, static_cast<int>(kTagBytes),
                            body + payload.size()) == 1;
    if (!ok) throw StoragePayloadError("cannot seal " + describe(slot) + " (AES-256-GCM failed)");
    return out;
}

std::string open_with(const std::string& blob, const std::string& ikm, const Slot& slot) {
    const auto what = describe(slot);
    if (blob.size() < 1 + kNonceBytes + kTagBytes) {
        throw StoragePayloadError(what + " is too short to be a sealed payload");
    }
    if (static_cast<uint8_t>(blob[0]) != kVersion) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "%#04x",
                      static_cast<unsigned>(static_cast<uint8_t>(blob[0])));
        throw StoragePayloadError(what + " has unknown seal version " + hex);
    }
    const auto key = derive_key(ikm);
    const auto associated = aad(slot);
    const auto* iv = reinterpret_cast<const unsigned char*>(blob.data() + 1);
    const auto* body = iv + kNonceBytes;
    const size_t body_len = blob.size() - 1 - kNonceBytes - kTagBytes;
    std::array<unsigned char, kTagBytes> tag{};
    std::memcpy(tag.data(), body + body_len, kTagBytes);

    CipherCtx c;
    std::string out(body_len, '\0');
    int len = 0;
    const bool ok =
        c.ctx && EVP_DecryptInit_ex(c.ctx, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1 &&
        EVP_CIPHER_CTX_ctrl(c.ctx, EVP_CTRL_GCM_SET_IVLEN, static_cast<int>(kNonceBytes),
                            nullptr) == 1 &&
        EVP_DecryptInit_ex(c.ctx, nullptr, nullptr, key.data(), iv) == 1 &&
        EVP_DecryptUpdate(c.ctx, nullptr, &len,
                          reinterpret_cast<const unsigned char*>(associated.data()),
                          static_cast<int>(associated.size())) == 1 &&
        EVP_DecryptUpdate(c.ctx, reinterpret_cast<unsigned char*>(out.data()), &len, body,
                          static_cast<int>(body_len)) == 1 &&
        EVP_CIPHER_CTX_ctrl(c.ctx, EVP_CTRL_GCM_SET_TAG, static_cast<int>(kTagBytes), tag.data()) ==
            1 &&
        EVP_DecryptFinal_ex(c.ctx, reinterpret_cast<unsigned char*>(out.data()) + len, &len) == 1;
    if (!ok) {
        throw StoragePayloadError(what +
                                  " failed authentication (tampered, moved, or sealed under "
                                  "another key)");
    }
    return out;
}

std::string host_local_key_path() {
    if (auto explicit_path = env_or_empty("VGI_ATTACH_SECRET_KEY_FILE"); !explicit_path.empty()) {
        return explicit_path;
    }
    std::error_code ec;
    if (auto xdg = env_or_empty("XDG_RUNTIME_DIR"); !xdg.empty() && fs::is_directory(xdg, ec)) {
        return (fs::path(xdg) / "vgi" / "attach-secret-options.key").string();
    }
#if defined(_WIN32)
    const std::string user = env_or_empty("USERNAME");
#else
    std::string user = env_or_empty("LOGNAME");
    if (user.empty()) user = env_or_empty("USER");
    if (user.empty()) user = env_or_empty("LNAME");
    if (user.empty()) user = env_or_empty("USERNAME");
    if (user.empty()) {
        if (const struct passwd* entry = ::getpwuid(::getuid()); entry && entry->pw_name) {
            user = entry->pw_name;
        }
    }
#endif
    return (fs::temp_directory_path(ec) / ("vgi-" + user) / "attach-secret-options.key").string();
}

std::string resolve_ikm() {
    if (auto signing_key = env_or_empty("VGI_SIGNING_KEY"); !signing_key.empty()) {
        return signing_key;
    }
    return load_host_local_key();
}

std::string seal(const std::string& payload, const Slot& slot) {
    return seal_with(payload, resolve_ikm(), slot);
}

std::string open(const std::string& blob, const Slot& slot) {
    std::string ikm;
    try {
        ikm = resolve_ikm();
    } catch (const StoragePayloadError& error) {
        throw StoragePayloadError(std::string("no key to open framework storage payloads: ") +
                                  error.what());
    }
    return open_with(blob, ikm, slot);
}

void sealed_put(FunctionStorage& storage, const std::string& scope, const std::string& kv_key,
                const Slot& slot, const std::string& payload) {
    storage.kv_put(scope, kv_key, seal(payload, slot));
}

std::optional<std::string> sealed_get(FunctionStorage& storage, const std::string& scope,
                                      const std::string& kv_key, const Slot& slot) {
    auto stored = storage.kv_get(scope, kv_key);
    if (!stored) return std::nullopt;
    return open(*stored, slot);
}

}  // namespace vgi::storage_seal

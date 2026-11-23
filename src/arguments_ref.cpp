// © Copyright 2025, 2026 Query Farm LLC - https://query.farm

#include "arguments_ref.h"

#include <cctype>
#include <cstdlib>
#include <stdexcept>

#include <arrow/array.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/errors.h>
#include <vgi_rpc/external.h>
#include <vgi_rpc/state_ref.h>

#include "wire.h"

namespace vgi::arguments_ref {
namespace {

constexpr int kMaxHeaderDigits = 19;
constexpr char kLocationField[] = "arguments_location";
constexpr char kArgumentsField[] = "arguments";

[[noreturn]] void invalid_ref(const std::string& message) {
    // The message names the kind too: a client that shows only
    // "<type>: <message>" (the DuckDB extension) would otherwise never show
    // the reason a caller can act on.
    throw vgi_rpc::KindedError(kInvalidArgumentsRefKind, "ValueError",
                               std::string(kInvalidArgumentsRefKind) + ": " + message,
                               vgi_rpc::Code::INVALID_ARGUMENT);
}

[[noreturn]] void invalid_bind_request() {
    throw vgi_rpc::KindedError(kInvalidBindRequestKind, "ValueError", kInvalidBindRequestMessage,
                               vgi_rpc::Code::INVALID_ARGUMENT);
}

bool truthy(const char* raw) {
    if (!raw) return false;
    std::string value(raw);
    for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
        value.pop_back();
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
        value.erase(value.begin());
    return value == "1" || value == "true" || value == "yes";
}

bool is_lower_hex_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

bool has_column(const std::shared_ptr<arrow::RecordBatch>& batch, const char* name) {
    return batch && batch->schema()->GetFieldIndex(name) >= 0;
}

bool arguments_set(const std::shared_ptr<arrow::RecordBatch>& bind_call) {
    if (!has_column(bind_call, kArgumentsField)) return false;
    const auto column = wire::column(bind_call, kArgumentsField);
    return column->length() > 0 && !column->IsNull(0);
}

std::string u64le(uint64_t value) {
    std::string out(8, '\0');
    for (int i = 0; i < 8; ++i) out[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
    return out;
}

}  // namespace

int64_t bytes_from_env() {
    const char* raw_env = std::getenv(kBytesEnv);
    std::string raw = raw_env ? raw_env : "";
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) raw.pop_back();
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front())))
        raw.erase(raw.begin());
    if (raw.empty()) return kDefaultBytes;
    bool digits = true;
    for (char c : raw)
        if (!std::isdigit(static_cast<unsigned char>(c))) digits = false;
    if (!digits || raw.front() == '0' || raw.size() > kMaxHeaderDigits) {
        throw std::invalid_argument(
            std::string(kBytesEnv) +
            " must be a positive integer number of bytes (no sign, no leading zero, at most 19 "
            "digits), got '" +
            raw + "'");
    }
    // 19 digits can exceed int64; the header carries the text, the SDK keeps
    // the value only to report it.
    try {
        return std::stoll(raw);
    } catch (const std::out_of_range&) {
        throw std::invalid_argument(std::string(kBytesEnv) + " is out of range: '" + raw + "'");
    }
}

std::optional<int64_t> install_header(vgi_rpc::HttpConfig& config) {
    // The mandatory capability signal (spec §5): sent whenever the server
    // vends upload URLs and can fetch what its grant minted back by id, and
    // never otherwise. Every in-tree vgi-rpc backend named by URL implements
    // the object API; an injected one must.
    bool resolvable = !config.external_storage_url.empty();
    if (config.external_storage) {
        resolvable = dynamic_cast<vgi_rpc::ObjectStore*>(config.external_storage.get()) != nullptr;
    }
    if (!resolvable) return std::nullopt;
    if (truthy(std::getenv(kDisableEnv))) return std::nullopt;
    const auto value = bytes_from_env();
    config.extra_capability_headers[kHeader] = std::to_string(value);
    return value;
}

std::optional<Location> check(const std::shared_ptr<arrow::RecordBatch>& bind_call) {
    const bool has_arguments = arguments_set(bind_call);
    std::shared_ptr<arrow::StructArray> location;
    if (has_column(bind_call, kLocationField)) {
        location =
            std::dynamic_pointer_cast<arrow::StructArray>(wire::column(bind_call, kLocationField));
        if (!location) invalid_ref("arguments_location is not a struct");
        if (location->length() == 0 || location->IsNull(0)) location.reset();
    }
    if (has_arguments == (location != nullptr)) invalid_bind_request();
    if (!location) return std::nullopt;

    auto ref = std::dynamic_pointer_cast<arrow::BinaryArray>(location->GetFieldByName("ref"));
    auto size = std::dynamic_pointer_cast<arrow::Int64Array>(location->GetFieldByName("size"));
    auto sha256 = std::dynamic_pointer_cast<arrow::StringArray>(location->GetFieldByName("sha256"));
    if (!ref || !size || !sha256 || ref->IsNull(0) || size->IsNull(0) || sha256->IsNull(0)) {
        invalid_ref("arguments_location must carry ref (binary), size (int64) and sha256 (utf8)");
    }
    Location out;
    out.ref = ref->GetString(0);
    out.size = size->Value(0);
    out.sha256 = sha256->GetString(0);
    if (out.size < 1) invalid_ref("arguments_location.size must be a positive byte count");
    if (!is_lower_hex_sha256(out.sha256)) {
        invalid_ref("arguments_location.sha256 must be 64 lowercase hex digits");
    }
    return out;
}

std::string resolve(const std::shared_ptr<arrow::RecordBatch>& bind_call,
                    vgi_rpc::CallContext* context) {
    const auto location = check(bind_call);
    if (!location) return wire::get_optional_binary(bind_call, kArgumentsField).value_or("");
    if (!context) invalid_ref("argument references need the request's context to resolve");
    try {
        const auto data = context->resolve_argument_ref(location->ref, location->size,
                                                        location->sha256);
        return *data;
    } catch (const vgi_rpc::ArgumentsRefMissingError&) {
        throw vgi_rpc::KindedError(kMissingKind, "NotFoundError", kMissingMessage,
                                   vgi_rpc::Code::NOT_FOUND);
    } catch (const vgi_rpc::InvalidArgumentsRefError& error) {
        throw vgi_rpc::KindedError(error.kind(), error.exception_type(),
                                   error.kind() + ": " + error.what(), error.code());
    } catch (const vgi_rpc::ArgumentsRefIntegrityError& error) {
        throw vgi_rpc::KindedError(error.kind(), error.exception_type(),
                                   error.kind() + ": " + error.what(), error.code());
    }
}

std::string fingerprint_input(const std::shared_ptr<arrow::RecordBatch>& bind_call) {
    if (const auto location = check(bind_call)) {
        const auto digest = vgi_rpc::crypto::hex_decode(location->sha256);
        std::string out(1, '\x01');
        out += u64le(static_cast<uint64_t>(location->size));
        out.append(reinterpret_cast<const char*>(digest->data()), digest->size());
        return out;
    }
    const auto raw = wire::get_optional_binary(bind_call, kArgumentsField).value_or("");
    std::string out(1, '\x00');
    out += u64le(raw.size());
    out += raw;
    return out;
}

}  // namespace vgi::arguments_ref

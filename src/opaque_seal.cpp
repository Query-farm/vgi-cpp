// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "opaque_seal.h"

#include <utility>
#include <vector>

#include <arrow/array.h>
#include <arrow/builder.h>
#include <vgi_rpc/arrow_utils.h>
#include <vgi_rpc/crypto.h>
#include <vgi_rpc/errors.h>

#include "wire.h"

namespace vgi::opaque {

namespace {

namespace crypto = vgi_rpc::crypto;

constexpr char kAttachDomain[] = "vgi.attach_opaque_data.v1";
constexpr char kTransactionDomain[] = "vgi.transaction_opaque_data.v1";
constexpr const char* kAttachField = "attach_opaque_data";
constexpr const char* kTransactionField = "transaction_opaque_data";

thread_local Call* g_current = nullptr;

// The domain literal's terminating NUL is part of the AAD.
std::string domain_prefix(const char* domain, size_t size) {
    return std::string(domain, size);
}

std::string seal(const Key& key, uint8_t version, const std::string& plaintext,
                 const std::string& aad) {
    std::string out(1, static_cast<char>(version));
    out += crypto::aead_seal(key, plaintext, aad);
    return out;
}

std::string open(const Key& key, uint8_t version, const std::string& envelope,
                 const std::string& aad, const char* field) {
    if (envelope.size() < 1 + crypto::kAeadNonceBytes + crypto::kAeadTagBytes ||
        static_cast<uint8_t>(envelope[0]) != version) {
        reject(field);
    }
    auto opened = crypto::aead_open(key, envelope.substr(1), aad);
    if (!opened) reject(field);
    return std::move(*opened);
}

std::shared_ptr<arrow::Array> one_binary(const std::string& value) {
    arrow::BinaryBuilder builder;
    VGI_RPC_THROW_NOT_OK(builder.Append(value));
    return vgi_rpc::unwrap(builder.Finish());
}

bool is_binary_column(const std::shared_ptr<arrow::RecordBatch>& batch, int i) {
    return batch->column(i)->type_id() == arrow::Type::BINARY;
}

std::optional<std::string> binary_at(const std::shared_ptr<arrow::RecordBatch>& batch, int i) {
    const auto& column = batch->column(i);
    if (column->IsNull(0)) return std::nullopt;
    return std::static_pointer_cast<arrow::BinaryArray>(column)->GetString(0);
}

}  // namespace

std::string identity_tail(const vgi_rpc::AuthContext& auth) {
    if (!auth.authenticated) return std::string("\0anonymous", 10);
    return std::string("\x01", 1) + auth.domain + std::string(1, '\0') +
           auth.principal.value_or("");
}

std::string attach_aad(const vgi_rpc::AuthContext& auth) {
    return domain_prefix(kAttachDomain, sizeof(kAttachDomain)) + identity_tail(auth);
}

std::string transaction_aad(const vgi_rpc::AuthContext& auth, const std::string& attach_envelope) {
    return domain_prefix(kTransactionDomain, sizeof(kTransactionDomain)) + identity_tail(auth) +
           std::string(1, '\0') + attach_envelope;
}

std::string seal_attach(const Key& key, const std::string& plaintext,
                        const vgi_rpc::AuthContext& auth) {
    return seal(key, kAttachEnvelopeVersion, plaintext, attach_aad(auth));
}

std::string open_attach(const Key& key, const std::string& envelope,
                        const vgi_rpc::AuthContext& auth) {
    return open(key, kAttachEnvelopeVersion, envelope, attach_aad(auth), kAttachField);
}

std::string seal_transaction(const Key& key, const std::string& plaintext,
                             const vgi_rpc::AuthContext& auth, const std::string& attach_envelope) {
    return seal(key, kTransactionEnvelopeVersion, plaintext,
                transaction_aad(auth, attach_envelope));
}

std::string open_transaction(const Key& key, const std::string& envelope,
                             const vgi_rpc::AuthContext& auth, const std::string& attach_envelope) {
    return open(key, kTransactionEnvelopeVersion, envelope, transaction_aad(auth, attach_envelope),
                kTransactionField);
}

void reject(const std::string& field) {
    // Classified, and identical for every failure mode: the message names
    // only the field, and there are no details.
    throw vgi_rpc::KindedError("opaque_data_not_recognized", "ValueError",
                               field + " not recognized", vgi_rpc::Code::INVALID_ARGUMENT);
}

std::string short_hash(const std::string& value) {
    // Over the value's lowercase hex text, as vgi_rpc.sentry.short_hash does,
    // so one value carries one token in every log that names it.
    const auto text =
        crypto::hex_encode(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    const auto digest = crypto::sha256(reinterpret_cast<const uint8_t*>(text.data()), text.size());
    return crypto::hex_encode(digest.data(), digest.size()).substr(0, 12);
}

std::string random_id() {
    const auto bytes = crypto::random_bytes(16);
    return crypto::hex_encode(bytes.data(), bytes.size());
}

Call::Call(const std::optional<Key>& key, const vgi_rpc::AuthContext& auth)
    : key_(key), auth_(auth), previous_(g_current) {
    g_current = this;
}

Call::~Call() {
    g_current = previous_;
}

vgi_rpc::Request Call::open(const vgi_rpc::Request& request) {
    if (!key_ || !request.batch()) return request;
    auto opened = open_batch(request.batch(), std::nullopt);
    if (opened == request.batch()) return request;
    return vgi_rpc::Request(std::move(opened), request.metadata());
}

std::shared_ptr<arrow::RecordBatch> Call::open_batch(
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const std::optional<std::string>& outer_attach) {
    if (!batch) return batch;
    const auto& schema = batch->schema();
    // Every request record is one row of declared types. Anything else that
    // carries an opaque field is not something this worker issued, and must
    // not slip past the open.
    for (const char* field : {kAttachField, kTransactionField}) {
        const int index = schema->GetFieldIndex(field);
        if (index >= 0 && (batch->num_rows() != 1 || !is_binary_column(batch, index))) {
            reject(field);
        }
    }
    if (batch->num_rows() != 1) return batch;

    // The attach value at this level, exactly as sent: a transaction at this
    // level is bound to it (or, absent one, to the enclosing level's).
    std::optional<std::string> attach_here = outer_attach;
    const int attach_index = schema->GetFieldIndex(kAttachField);
    if (attach_index >= 0 && is_binary_column(batch, attach_index)) {
        if (auto sent = binary_at(batch, attach_index)) {
            attach_here = sent;
            if (!attach_as_sent_) attach_as_sent_ = sent;
        }
    }

    std::vector<std::shared_ptr<arrow::Array>> columns;
    bool changed = false;
    for (int i = 0; i < batch->num_columns(); ++i) {
        std::shared_ptr<arrow::Array> column = batch->column(i);
        const auto& name = schema->field(i)->name();
        if (is_binary_column(batch, i)) {
            const auto value = binary_at(batch, i);
            if (value && name == kAttachField) {
                column = one_binary(open_attach(*key_, *value, auth_));
                changed = true;
            } else if (value && name == kTransactionField) {
                // A transaction with no attach beside it has nothing to be
                // bound to, so it cannot be one this worker issued.
                if (!attach_here) reject(kTransactionField);
                column = one_binary(open_transaction(*key_, *value, auth_, *attach_here));
                changed = true;
            } else if (value && (name == "request" || name == "bind_call")) {
                // Records nested as IPC: the request dataclass, and the bind
                // call an init / planning request carries.
                const auto inner = wire::decode_ipc(*value);
                const auto opened = open_batch(inner, attach_here);
                if (opened != inner) {
                    column = one_binary(wire::encode_ipc(opened));
                    changed = true;
                }
            }
        }
        columns.push_back(std::move(column));
    }
    if (!changed) return batch;
    return arrow::RecordBatch::Make(schema, 1, std::move(columns));
}

std::string Call::seal_attach(const std::string& plaintext) const {
    if (!key_) return plaintext;
    return opaque::seal_attach(*key_, plaintext, auth_);
}

std::string Call::seal_transaction(const std::string& plaintext) const {
    if (!key_) return plaintext;
    // Bound to the attach envelope this call carried.
    return opaque::seal_transaction(*key_, plaintext, auth_, attach_as_sent_.value_or(""));
}

const Call* current() {
    return g_current;
}

std::string echo_attach(const std::string& plaintext) {
    if (g_current && g_current->attach_as_sent()) return *g_current->attach_as_sent();
    return plaintext;
}

std::string seal_attach_out(const std::string& plaintext) {
    return g_current ? g_current->seal_attach(plaintext) : plaintext;
}

std::string seal_transaction_out(const std::string& plaintext) {
    return g_current ? g_current->seal_transaction(plaintext) : plaintext;
}

}  // namespace vgi::opaque

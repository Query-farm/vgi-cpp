// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "memory_catalog.h"
#include "opaque_seal.h"

#include <arrow/array.h>
#include <arrow/util/key_value_metadata.h>
#include <algorithm>
#include <cctype>
#include <random>
#include <stdexcept>
#include <utility>

#include "catalog_contents.h"
#include "column_metadata.h"
#include "column_expressions.h"
#include "methods.h"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"
#include <vgi_rpc/errors.h>

namespace vgi {

namespace {

namespace gen = ::vgi::generated;

void require_unconstrained_field(const std::shared_ptr<arrow::Field>& field, bool top_level) {
    const auto& metadata = field->metadata();
    if (top_level && metadata && metadata->Contains("is_row_id")) return;
    bool logical = field->type()->id() == arrow::Type::EXTENSION;
    if (metadata) {
        for (const auto& key : metadata->keys()) {
            logical = logical || key.rfind("ARROW:extension:", 0) == 0 || key.rfind("vgi.logical_type.", 0) == 0;
        }
    }
    if (logical || !field->nullable()) {
        throw vgi_rpc::KindedError(vgi_rpc::ERROR_KIND_METHOD_NOT_IMPLEMENTED, "MethodNotImplementedError",
            "catalog_table_create: nonnullable or logical columns require supported canonical semantics",
            vgi_rpc::Code::UNIMPLEMENTED);
    }
    if (field->type()->id() == arrow::Type::STRUCT) {
        for (const auto& child : field->type()->fields()) require_unconstrained_field(child, false);
    }
}

std::map<std::string, std::string> view_map(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    auto values = std::dynamic_pointer_cast<arrow::MapArray>(wire::column(batch, field));
    if (!values) throw std::invalid_argument("view metadata must be a string map");
    std::map<std::string, std::string> result;
    if (values->IsNull(0)) return result;
    auto keys = std::dynamic_pointer_cast<arrow::StringArray>(values->keys());
    auto items = std::dynamic_pointer_cast<arrow::StringArray>(values->items());
    if (!keys || !items) throw std::invalid_argument("view metadata must be a string map");
    const auto begin = values->value_offset(0);
    const auto end = begin + values->value_length(0);
    for (auto i = begin; i < end; ++i) {
        if (keys->IsNull(i) || items->IsNull(i)) throw std::invalid_argument("view metadata cannot contain null entries");
        result.emplace(keys->GetString(i), items->GetString(i));
    }
    return result;
}

std::vector<std::pair<std::string, std::string>> view_pairs(const std::shared_ptr<arrow::RecordBatch>& batch, const std::string& field) {
    const auto values = view_map(batch, field);
    return {values.begin(), values.end()};
}

std::string path_key(const SchemaPath& path) {
    std::string key;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) key.push_back('\0');
        key += path[i];
    }
    return key;
}

std::string path_display(const SchemaPath& path) {
    std::string out;
    for (size_t i = 0; i < path.size(); ++i) {
        if (i) out.push_back('.');
        out += path[i];
    }
    return out;
}

std::string lower(std::string value) {
    for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

std::string upper(std::string value) {
    for (auto& c : value) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return value;
}

// From the OS CSPRNG: the token makes an ATTACH private, so it must not be
// predictable.
std::string random_token() {
    return opaque::random_id();
}

// CREATE's on_conflict for an existing object: replace it (true), keep it
// (false), or fail.
bool on_conflict_replaces(const char* kind, const std::string& name,
                          const std::string& on_conflict) {
    const auto action = upper(on_conflict);
    if (action == "REPLACE") return true;
    if (action == "IGNORE") return false;
    throw std::invalid_argument(std::string(kind) + " with name \"" + name + "\" already exists");
}

std::shared_ptr<arrow::RecordBatch> items_payload(const std::string& method,
                                                  std::vector<std::string> items) {
    return wire::ResultBuilder(payload_schema_of(method))
        .set_binary_list("items", std::move(items))
        .finish();
}

std::vector<std::string> object_items(const std::map<std::string, std::string>& sorted_items) {
    std::vector<std::string> out;
    out.reserve(sorted_items.size());
    for (const auto& [key, item] : sorted_items) out.push_back(item);
    return out;
}

// The parameters of a request-wrapped method (`{request: binary}`), or the
// parameter batch itself.
std::shared_ptr<arrow::RecordBatch> unwrap(const std::shared_ptr<arrow::RecordBatch>& params) {
    if (params && params->schema()->GetFieldIndex("request") >= 0 &&
        params->schema()->GetFieldIndex("attach_opaque_data") < 0) {
        if (auto inner = wire::get_ipc(params, "request")) return inner;
    }
    return params;
}

}  // namespace

MemoryCatalog::MemoryCatalog(MemoryCatalogOptions options) : options_(std::move(options)) {
    if (options_.name.empty()) throw std::invalid_argument("MemoryCatalog: name is required");
}

std::optional<std::string> MemoryCatalog::catalog_of(const std::string& attach_opaque_data) {
    const auto separator = attach_opaque_data.find('\0');
    if (separator == std::string::npos || separator == 0) return std::nullopt;
    return attach_opaque_data.substr(0, separator);
}

std::string MemoryCatalog::catalog_info() const {
    return wire::encode_ipc(wire::ResultBuilder(gen::CatalogInfoSchema())
                                .set_string("name", options_.name)
                                .fill_defaults()
                                .finish());
}

MemoryCatalog::State& MemoryCatalog::state(const std::string& attach) {
    const auto found = attaches_.find(attach);
    if (found == attaches_.end()) {
        throw std::invalid_argument(
            options_.name +
            ": not attached (its state lives in the worker process that attached it)");
    }
    return found->second;
}

int64_t MemoryCatalog::reported_version(const State& state) const {
    return options_.unversioned ? 0 : state.version;
}

std::vector<const MemoryCatalog::Schema*> MemoryCatalog::sorted_schemas(const State& state) const {
    std::vector<const Schema*> out;
    for (const auto& [key, schema] : state.schemas) out.push_back(&schema);
    // Parents first, then by path: the map is already in path order.
    std::stable_sort(out.begin(), out.end(), [](const Schema* a, const Schema* b) {
        return a->path.size() < b->path.size();
    });
    return out;
}

std::string MemoryCatalog::schema_item(const std::string& attach, const Schema& schema) const {
    auto builder = wire::ResultBuilder(gen::SchemaInfoSchema())
                       .set_string_list("path", schema.path)
                       .set_binary("attach_opaque_data", opaque::echo_attach(attach));
    builder.set_optional_string("comment", schema.comment);
    builder.set_string_map("tags", {schema.tags.begin(), schema.tags.end()});
    return wire::encode_ipc(builder.fill_defaults().finish());
}

std::vector<SchemaContents> MemoryCatalog::snapshot(const std::string& attach) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto& st = state(attach);
    std::vector<SchemaContents> out;
    for (const auto* schema : sorted_schemas(st)) {
        SchemaContents row;
        row.path = schema->path;
        row.schema = schema_item(attach, *schema);
        for (const auto& [key, object] : schema->tables) row.tables.push_back(object.item);
        for (const auto& [key, object] : schema->views) row.views.push_back(object.item);
        out.push_back(std::move(row));
    }
    return out;
}

std::shared_ptr<arrow::RecordBatch> MemoryCatalog::attach(
    const std::shared_ptr<arrow::RecordBatch>& params) {
    const auto request = unwrap(params);
    const auto name = wire::get_string(request, "name");
    if (name != options_.name) {
        throw std::invalid_argument("Unknown catalog: '" + name + "'. Available: " + options_.name);
    }
    // The dispatcher's five-field seal shape, catalog first, so a request is
    // routed by its first field; the token makes every ATTACH private.
    const std::string attach = options_.name + std::string(3, '\0') + random_token() + '\0';
    int64_t version = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        State st;
        st.schemas[path_key({"main"})] = Schema{{"main"}, std::nullopt, {}, {}, {}};
        version = reported_version(st);
        attaches_[attach] = std::move(st);
    }
    auto builder = wire::ResultBuilder(payload_schema_of("catalog_attach"));
    builder.set_binary("attach_opaque_data", opaque::seal_attach_out(attach))
        .set_int64("catalog_version", version)
        .set_bool("catalog_version_frozen", false)
        .set_bool("attach_opaque_data_required", true)
        .set_string("default_schema", "main")
        .set_bool("supports_catalog_contents", options_.supports_catalog_contents);
    builder.set_optional_string("comment", options_.comment);
    return builder.fill_defaults().finish();
}

std::shared_ptr<arrow::RecordBatch> MemoryCatalog::contents(
    const std::string& attach, const std::shared_ptr<arrow::RecordBatch>& params) {
    int64_t version = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        version = reported_version(state(attach));
    }
    CatalogContentsCall call;
    call.catalog_name = options_.name;
    call.catalog_version = version;
    call.if_none_match = wire::get_optional_string(params, "if_none_match");
    call.contents = [this, attach]() { return snapshot(attach); };
    return encode_catalog_contents(version,
                                   answer_catalog_contents(options_.catalog_contents_handler,
                                                           options_.catalog_contents_etag, call));
}

void MemoryCatalog::schema_create(const std::string& attach,
                                  const std::shared_ptr<arrow::RecordBatch>& params) {
    const auto path = wire::get_schema_path(params, "path");
    if (path.empty()) throw std::invalid_argument("schema path is empty");
    std::lock_guard<std::mutex> lock(mutex_);
    auto& st = state(attach);
    if (st.schemas.count(path_key(path)) &&
        !on_conflict_replaces("Schema", path_display(path),
                              wire::get_optional_enum(params, "on_conflict").value_or(""))) {
        return;
    }
    if (path.size() > 1) {
        const SchemaPath parent(path.begin(), path.end() - 1);
        if (!st.schemas.count(path_key(parent))) {
            throw std::invalid_argument("Schema " + path_display(parent) + " does not exist");
        }
    }
    st.schemas[path_key(path)] = Schema{path, wire::get_optional_string(params, "comment"), view_map(params, "tags"), {}, {}};
    ++st.version;
}

void MemoryCatalog::schema_drop(const std::string& attach,
                                const std::shared_ptr<arrow::RecordBatch>& params) {
    const auto path = wire::get_schema_path(params, "path");
    std::lock_guard<std::mutex> lock(mutex_);
    auto& st = state(attach);
    const auto key = path_key(path);
    const auto found = st.schemas.find(key);
    if (found == st.schemas.end()) {
        if (wire::get_optional_bool(params, "ignore_not_found").value_or(false)) return;
        throw std::invalid_argument("Schema " + path_display(path) + " does not exist");
    }
    std::vector<std::string> children;
    for (const auto& [other, schema] : st.schemas) {
        if (other.size() > key.size() && other.compare(0, key.size(), key) == 0 &&
            other[key.size()] == '\0') {
            children.push_back(other);
        }
    }
    const bool empty =
        found->second.tables.empty() && found->second.views.empty() && children.empty();
    if (!empty && !wire::get_optional_bool(params, "cascade").value_or(false)) {
        throw std::invalid_argument("Schema " + path_display(path) + " is not empty; use CASCADE");
    }
    for (const auto& child : children) st.schemas.erase(child);
    st.schemas.erase(key);
    ++st.version;
}

void MemoryCatalog::put_object(const std::string& attach, const SchemaPath& path,
                               const std::string& name, const char* kind,
                               const std::string& on_conflict, bool view, std::string item) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& st = state(attach);
    const auto schema = st.schemas.find(path_key(path));
    if (schema == st.schemas.end()) {
        throw std::invalid_argument("Schema " + path_display(path) + " does not exist");
    }
    auto& objects = view ? schema->second.views : schema->second.tables;
    const auto key = lower(name);
    if (objects.count(key) && !on_conflict_replaces(kind, name, on_conflict)) return;
    objects[key] = Object{name, std::move(item)};
    ++st.version;
}

void MemoryCatalog::drop_object(const std::string& attach, const SchemaPath& path,
                                const std::string& name, const char* kind, bool ignore_not_found,
                                bool view) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& st = state(attach);
    if (const auto schema = st.schemas.find(path_key(path)); schema != st.schemas.end()) {
        auto& objects = view ? schema->second.views : schema->second.tables;
        if (objects.erase(lower(name))) {
            ++st.version;
            return;
        }
    }
    if (ignore_not_found) return;
    throw std::invalid_argument(std::string(kind) + " with name " + name + " does not exist");
}

std::optional<std::shared_ptr<arrow::RecordBatch>> MemoryCatalog::serve(
    const std::string& method, const std::shared_ptr<arrow::RecordBatch>& raw_params) {
    if (method == "catalog_attach") return attach(raw_params);

    const auto params = unwrap(raw_params);
    const auto attach = wire::get_binary(params, "attach_opaque_data");

    // Lifecycle and version.
    if (method == "catalog_detach") {
        std::lock_guard<std::mutex> lock(mutex_);
        attaches_.erase(attach);
        return std::nullopt;
    }
    if (method == "catalog_version") {
        std::lock_guard<std::mutex> lock(mutex_);
        return wire::ResultBuilder(payload_schema_of(method))
            .set_int64("version", reported_version(state(attach)))
            .finish();
    }
    if (method == "catalog_contents") return contents(attach, params);
    // No transactions of its own: every DDL applies at once.
    if (method == "catalog_transaction_begin") {
        return wire::ResultBuilder(payload_schema_of(method)).fill_defaults().finish();
    }
    if (method == "catalog_transaction_commit" || method == "catalog_transaction_rollback") {
        return std::nullopt;
    }

    // Discovery.
    if (method == "catalog_schemas") {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> items;
        for (const auto* schema : sorted_schemas(state(attach))) {
            items.push_back(schema_item(attach, *schema));
        }
        return items_payload(method, std::move(items));
    }
    if (method == "catalog_schema_get") {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& st = state(attach);
        std::vector<std::string> items;
        const auto found = st.schemas.find(path_key(wire::get_schema_path(params, "path")));
        if (found != st.schemas.end()) items.push_back(schema_item(attach, found->second));
        return items_payload(method, std::move(items));
    }
    if (method == "catalog_schema_contents_tables" || method == "catalog_schema_contents_views") {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& st = state(attach);
        std::map<std::string, std::string> items;
        const auto found = st.schemas.find(path_key(wire::get_schema_path(params, "path")));
        if (found != st.schemas.end()) {
            const auto& objects = method == "catalog_schema_contents_views" ? found->second.views
                                                                            : found->second.tables;
            for (const auto& [key, object] : objects) items[key] = object.item;
        }
        return items_payload(method, object_items(items));
    }
    if (method == "catalog_table_get" || method == "catalog_view_get") {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& st = state(attach);
        std::vector<std::string> items;
        const auto found = st.schemas.find(path_key(wire::get_schema_path(params, "schema_path")));
        if (found != st.schemas.end()) {
            const auto& objects =
                method == "catalog_view_get" ? found->second.views : found->second.tables;
            if (const auto object = objects.find(lower(wire::get_string(params, "name")));
                object != objects.end()) {
                items.push_back(object->second.item);
            }
        }
        return items_payload(method, std::move(items));
    }
    // A memory catalog holds no functions, macros, indexes or COPY formats.
    if (method == "catalog_schema_contents_functions" ||
        method == "catalog_schema_contents_macros" || method == "catalog_schema_contents_indexes" ||
        method == "catalog_macro_get" || method == "catalog_index_get" ||
        method == "catalog_copy_from_formats") {
        return items_payload(method, {});
    }

    // DDL.
    if (method == "catalog_schema_create") {
        if (wire::get_optional_binary(params, "options").has_value()) {
            throw vgi_rpc::KindedError(vgi_rpc::ERROR_KIND_METHOD_NOT_IMPLEMENTED, "MethodNotImplementedError",
                "catalog_schema_create requires negotiated vgi.schema.options.create.v1", vgi_rpc::Code::UNIMPLEMENTED);
        }
        schema_create(attach, params);
        return std::nullopt;
    }
    if (method == "catalog_schema_drop") {
        schema_drop(attach, params);
        return std::nullopt;
    }
    if (method == "catalog_table_create") {
        if (!wire::get_binary_list(params, "constraint_definitions").empty()) {
            throw vgi_rpc::KindedError(vgi_rpc::ERROR_KIND_METHOD_NOT_IMPLEMENTED, "MethodNotImplementedError",
                "catalog_table_create: canonical constraint enforcement is not supported", vgi_rpc::Code::UNIMPLEMENTED);
        }
        for (const char* field : {"comment", "tags", "options", "layout"}) {
            if (params->schema()->GetFieldIndex(field) >= 0 && !wire::column(params, field)->IsNull(0)) {
                throw vgi_rpc::KindedError(vgi_rpc::ERROR_KIND_METHOD_NOT_IMPLEMENTED, "MethodNotImplementedError",
                    "catalog_table_create: table metadata/options/layout profiles are not supported",
                    vgi_rpc::Code::UNIMPLEMENTED);
            }
        }
        const auto name = wire::get_string(params, "name");
        const auto path = wire::get_schema_path(params, "schema_path");
        if (wire::get_binary(params, "columns").empty()) {
            throw std::invalid_argument("catalog_table_create: missing columns schema");
        }
        const auto columns_schema = wire::decode_schema(wire::get_binary(params, "columns"));
        validate_column_metadata(columns_schema);
        validate_column_expressions(columns_schema, path, false);
        for (const auto& field : columns_schema->fields()) {
            require_unconstrained_field(field, true);
        }
        auto info = wire::ResultBuilder(gen::TableInfoSchema())
                        .set_string("name", name)
                        .set_string_list("schema_path", path)
                        .set_binary("columns", wire::get_binary(params, "columns"));
        info.set_binary_list("constraints", {}).set_null("definition_revision");
        put_object(attach, path, name, "Table",
                   wire::get_optional_enum(params, "on_conflict").value_or(""), false,
                   wire::encode_ipc(info.fill_defaults().finish()));
        return std::nullopt;
    }
    if (method == "catalog_table_drop" || method == "catalog_view_drop") {
        const bool view = method == "catalog_view_drop";
        drop_object(attach, wire::get_schema_path(params, "schema_path"),
                    wire::get_string(params, "name"), view ? "View" : "Table",
                    wire::get_optional_bool(params, "ignore_not_found").value_or(false), view);
        return std::nullopt;
    }
    if (method == "catalog_view_comment_set" || method == "catalog_view_column_comment_set") {
        const bool column = method == "catalog_view_column_comment_set";
        std::vector<std::string> column_path;
        if (column) {
            column_path = wire::get_string_list(params, "column_path");
            if (column_path.size() != 1 || column_path[0].empty() || column_path[0].find('\0') != std::string::npos)
                throw std::invalid_argument("view column comments require one raw output identifier");
        }
        const auto name = wire::get_string(params, "name");
        const auto path = wire::get_schema_path(params, "schema_path");
        const auto comment = wire::get_optional_string(params, "comment");
        const bool ignore = wire::get_bool(params, "ignore_not_found");
        std::lock_guard<std::mutex> lock(mutex_);
        auto& current = state(attach);
        auto schema = current.schemas.find(path_key(path));
        Object* view = nullptr;
        if (schema != current.schemas.end()) {
            auto found = schema->second.views.find(lower(name));
            if (found != schema->second.views.end()) view = &found->second;
        }
        if (!view) {
            if (ignore) return std::nullopt;
            throw std::invalid_argument("View with name " + name + " does not exist");
        }
        auto batch = wire::decode_ipc(view->item);
        auto replacement = wire::ResultBuilder(gen::ViewInfoSchema());
        std::string field = "comment";
        if (column) {
            field = "column_comments";
            auto comments = view_map(batch, field);
            if (comment) comments[column_path[0]] = *comment;
            else comments.erase(column_path[0]);
            replacement.set_string_map(field, {comments.begin(), comments.end()});
        } else replacement.set_optional_string(field, comment);
        auto values = replacement.fill_defaults().finish()->GetColumnByName(field);
        const auto index = batch->schema()->GetFieldIndex(field);
        auto updated = batch->SetColumn(index, batch->schema()->field(index), values);
        if (!updated.ok()) throw std::runtime_error(updated.status().ToString());
        view->item = wire::encode_ipc(*updated);
        current.version++;
        return std::nullopt;
    }
    if (method == "catalog_view_create") {
        if (wire::get_string(params, "security") != "regular") {
            throw std::runtime_error("view security mode is unsupported");
        }
        const auto name = wire::get_string(params, "name");
        const auto path = wire::get_schema_path(params, "schema_path");
        const auto column_comments = view_pairs(params, "column_comments");
        for (const auto& entry : column_comments) {
            if (entry.first.empty() || entry.first.find('\0') != std::string::npos)
                throw std::invalid_argument("view column comments require raw output identifiers");
        }
        auto info = wire::ResultBuilder(gen::ViewInfoSchema())
                        .set_string("name", name)
                        .set_string_list("schema_path", path)
                        .set_string("definition", wire::get_string(params, "definition"))
                        .set_string_list("aliases", wire::get_string_list(params, "aliases"))
                        .set_string("security", "regular")
                        .set_optional_string("comment", wire::get_optional_string(params, "comment"))
                        .set_string_map("tags", view_pairs(params, "tags"))
                        .set_string_map("column_comments", column_comments);
        put_object(attach, path, name, "View",
                   wire::get_optional_enum(params, "on_conflict").value_or(""), true,
                   wire::encode_ipc(info.fill_defaults().finish()));
        return std::nullopt;
    }
    throw std::invalid_argument(options_.name + ": " + method +
                                " is not supported by an in-memory catalog");
}

}  // namespace vgi

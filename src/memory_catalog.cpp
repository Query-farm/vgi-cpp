// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "memory_catalog.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <stdexcept>
#include <utility>

#include "catalog_contents.h"
#include "methods.h"
#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"

namespace vgi {

namespace {

namespace gen = ::vgi::generated;

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

std::string random_token() {
    static thread_local std::mt19937_64 engine{std::random_device{}()};
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 2; ++i) {
        auto bits = engine();
        for (int j = 0; j < 16; ++j, bits >>= 4) out.push_back(kHex[bits & 0xf]);
    }
    return out;
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
                       .set_binary("attach_opaque_data", attach);
    builder.set_optional_string("comment", schema.comment);
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
        st.schemas[path_key({"main"})] = Schema{{"main"}, std::nullopt, {}, {}};
        version = reported_version(st);
        attaches_[attach] = std::move(st);
    }
    auto builder = wire::ResultBuilder(payload_schema_of("catalog_attach"));
    builder.set_binary("attach_opaque_data", attach)
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
    st.schemas[path_key(path)] = Schema{path, wire::get_optional_string(params, "comment"), {}, {}};
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
        schema_create(attach, params);
        return std::nullopt;
    }
    if (method == "catalog_schema_drop") {
        schema_drop(attach, params);
        return std::nullopt;
    }
    if (method == "catalog_table_create") {
        const auto name = wire::get_string(params, "name");
        const auto path = wire::get_schema_path(params, "schema_path");
        if (wire::get_binary(params, "columns").empty()) {
            throw std::invalid_argument("catalog_table_create: missing columns schema");
        }
        auto info = wire::ResultBuilder(gen::TableInfoSchema())
                        .set_string("name", name)
                        .set_string_list("schema_path", path)
                        .set_binary("columns", wire::get_binary(params, "columns"));
        for (const char* constraint :
             {"not_null_constraints", "unique_constraints", "check_constraints",
              "primary_key_constraints", "foreign_key_constraints"}) {
            if (params->schema()->GetFieldIndex(constraint) >= 0) {
                info.set_array(constraint, wire::column(params, constraint));
            }
        }
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
    if (method == "catalog_view_create") {
        const auto name = wire::get_string(params, "name");
        const auto path = wire::get_schema_path(params, "schema_path");
        auto info = wire::ResultBuilder(gen::ViewInfoSchema())
                        .set_string("name", name)
                        .set_string_list("schema_path", path)
                        .set_string("definition", wire::get_string(params, "definition"));
        put_object(attach, path, name, "View",
                   wire::get_optional_enum(params, "on_conflict").value_or(""), true,
                   wire::encode_ipc(info.fill_defaults().finish()));
        return std::nullopt;
    }
    throw std::invalid_argument(options_.name + ": " + method +
                                " is not supported by an in-memory catalog");
}

}  // namespace vgi

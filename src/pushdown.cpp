// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/pushdown.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <regex>
#include <stdexcept>
#include <unordered_set>

#include <arrow/array.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/concatenate.h>
#include <arrow/compute/api.h>
#include <arrow/compute/initialize.h>
#include <arrow/scalar.h>
#include <arrow/util/key_value_metadata.h>
#include <nlohmann/json.hpp>

#include "wire.h"

namespace vgi {

struct PushdownFilters::Spec {
    std::string kind;
    std::string column_name;
    size_t column_index = 0;
    size_t field_index = 0;
    std::string field_name;
    std::string op;
    std::string function;
    std::shared_ptr<arrow::DataType> data_type;
    std::shared_ptr<arrow::Array> value;
    bool negated = false;
    bool advisory = false;
    std::string id;
    uint64_t revision = 0;
    std::vector<std::shared_ptr<Spec>> children;
};

using Spec = PushdownFilters::Spec;
using json = nlohmann::json;

namespace {

constexpr size_t kMaxPayloadBytes = 16U << 20U;
constexpr size_t kMaxJsonBytes = 1U << 20U;
constexpr size_t kMaxDepth = 64;
constexpr size_t kMaxNodes = 10000;
constexpr size_t kMaxPredicates = 1024;
constexpr size_t kMaxPredicateIds = 4096;

[[noreturn]] void invalid(const std::string& message) {
    throw std::invalid_argument("Filter Encoding v2: " + message);
}

template <class T>
T value_or_throw(arrow::Result<T> result, const std::string& context) {
    if (!result.ok()) throw std::runtime_error(context + ": " + result.status().ToString());
    return std::move(result).ValueUnsafe();
}

void require_keys(const json& object, std::initializer_list<const char*> required,
                  std::initializer_list<const char*> optional, const std::string& where) {
    if (!object.is_object()) invalid(where + " must be an object");
    std::unordered_set<std::string> allowed;
    for (const auto* key : required) {
        allowed.insert(key);
        if (!object.contains(key)) invalid(where + " is missing '" + key + "'");
    }
    for (const auto* key : optional) allowed.insert(key);
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (!allowed.count(it.key())) invalid(where + " has unknown property '" + it.key() + "'");
    }
}

std::string required_string(const json& object, const char* key, const std::string& where) {
    const auto& value = object.at(key);
    if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        invalid(where + "." + key + " must be a nonempty string");
    }
    return value.get<std::string>();
}

uint64_t required_uint(const json& object, const char* key, const std::string& where) {
    const auto& value = object.at(key);
    if (!value.is_number_unsigned()) invalid(where + "." + key + " must be uint64");
    return value.get<uint64_t>();
}

bool canonical_payload_name(const std::string& name) {
    for (const auto* prefix : {"value_", "type_", "artifact_"}) {
        if (name.rfind(prefix, 0) != 0) continue;
        const auto suffix = name.substr(std::char_traits<char>::length(prefix));
        if (suffix.empty() || (suffix.size() > 1 && suffix.front() == '0')) return false;
        return std::all_of(suffix.begin(), suffix.end(),
                           [](char c) { return c >= '0' && c <= '9'; });
    }
    return false;
}

std::string metadata_value(const std::shared_ptr<arrow::Schema>& schema, const std::string& key) {
    if (!schema->metadata()) invalid("missing schema metadata '" + key + "'");
    auto value = schema->metadata()->Get(key);
    if (!value.ok()) invalid("missing schema metadata '" + key + "'");
    return value.ValueUnsafe();
}

std::string validate_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (!batch || batch->num_rows() != 1)
        invalid("filter RecordBatch must contain exactly one row");
    if (batch->num_columns() == 0) invalid("filter RecordBatch has no filter_spec field");
    const auto& first = batch->schema()->field(0);
    if (first->name() != "filter_spec" || first->type()->id() != arrow::Type::STRING ||
        first->nullable()) {
        invalid("first field must be filter_spec: utf8 not null");
    }
    auto text = std::dynamic_pointer_cast<arrow::StringArray>(batch->column(0));
    if (!text || text->IsNull(0)) invalid("filter_spec value must not be NULL");
    if (text->value_length(0) > static_cast<int64_t>(kMaxJsonBytes))
        invalid("filter JSON exceeds 1 MiB");
    std::unordered_set<std::string> names{"filter_spec"};
    for (int i = 1; i < batch->num_columns(); ++i) {
        const auto& field = batch->schema()->field(i);
        if (!names.insert(field->name()).second || !canonical_payload_name(field->name())) {
            invalid("noncanonical or duplicate payload field '" + field->name() + "'");
        }
        if (field->name().rfind("type_", 0) == 0 && !batch->column(i)->IsNull(0)) {
            invalid("type payload must contain NULL");
        }
    }
    if (metadata_value(batch->schema(), "vgi_filter_encoding") != "vgi.filters.v2" ||
        metadata_value(batch->schema(), "vgi_filter_version") != "2") {
        invalid("unsupported filter encoding/version");
    }
    const auto context = metadata_value(batch->schema(), "vgi_evaluation_context");
    if (context != "vgi.none.v1")
        invalid("evaluation context '" + context + "' was not advertised");
    for (const auto* key :
         {"vgi_time_zone", "vgi_calendar", "vgi_default_collation", "vgi_ieee_floating_point_ops",
          "vgi_integer_division", "vgi_context_provider_fingerprint"}) {
        if (batch->schema()->metadata() && batch->schema()->metadata()->Contains(key)) {
            invalid("vgi.none.v1 forbids session-context metadata");
        }
    }
    return context;
}

void validate_arrow_extensions(const std::shared_ptr<arrow::Field>& field) {
    static const std::unordered_set<std::string> known = {
        "arrow.bool8",
        "arrow.json",
        "arrow.uuid",
        "geoarrow.linestring",
        "geoarrow.multilinestring",
        "geoarrow.multipoint",
        "geoarrow.multipolygon",
        "geoarrow.point",
        "geoarrow.polygon",
        "geoarrow.wkb",
    };
    if (field->metadata() && field->metadata()->Contains("ARROW:extension:name")) {
        const auto name = field->metadata()->Get("ARROW:extension:name");
        if (!name.ok() || !known.count(name.ValueUnsafe()))
            invalid("unknown Arrow extension type on field '" + field->name() + "'");
    }
    if (field->type()->id() == arrow::Type::EXTENSION) {
        const auto& extension = static_cast<const arrow::ExtensionType&>(*field->type());
        if (!known.count(extension.extension_name()))
            invalid("unknown Arrow extension type '" + extension.extension_name() + "'");
    }
    if (field->type()->id() == arrow::Type::STRUCT) {
        for (const auto& child : static_cast<const arrow::StructType&>(*field->type()).fields())
            validate_arrow_extensions(child);
    } else if (field->type()->id() == arrow::Type::LIST) {
        validate_arrow_extensions(
            static_cast<const arrow::ListType&>(*field->type()).value_field());
    } else if (field->type()->id() == arrow::Type::LARGE_LIST) {
        validate_arrow_extensions(
            static_cast<const arrow::LargeListType&>(*field->type()).value_field());
    } else if (field->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
        validate_arrow_extensions(
            static_cast<const arrow::FixedSizeListType&>(*field->type()).value_field());
    }
}

std::shared_ptr<arrow::Array> payload(const std::shared_ptr<arrow::RecordBatch>& batch,
                                      const std::string& prefix, uint64_t reference) {
    const auto name = prefix + "_" + std::to_string(reference);
    const auto indices = batch->schema()->GetAllFieldIndices(name);
    if (indices.size() != 1) invalid("missing or duplicate payload field '" + name + "'");
    const auto position = indices.front();
    const auto& field = batch->schema()->field(position);
    if (prefix != "artifact") validate_arrow_extensions(field);
    return batch->column(position);
}

std::string root_column(const std::shared_ptr<Spec>& expression) {
    if (!expression) return {};
    if (expression->kind == "column") return expression->column_name;
    if (expression->kind == "field" && !expression->children.empty())
        return root_column(expression->children[0]);
    return {};
}

bool is_direct_column(const std::shared_ptr<Spec>& expression, const std::string& column) {
    return expression && expression->kind == "column" && expression->column_name == column;
}

std::shared_ptr<arrow::DataType> logical_type(std::shared_ptr<arrow::DataType> type) {
    while (type && type->id() == arrow::Type::DICTIONARY) {
        type = static_cast<const arrow::DictionaryType&>(*type).value_type();
    }
    return type;
}

bool same_logical_type(const std::shared_ptr<arrow::DataType>& left,
                       const std::shared_ptr<arrow::DataType>& right) {
    const auto left_type = logical_type(left);
    const auto right_type = logical_type(right);
    return left_type && right_type && left_type->Equals(right_type);
}

bool boolean_type(const std::shared_ptr<arrow::DataType>& type) {
    const auto logical = logical_type(type);
    return logical && logical->id() == arrow::Type::BOOL;
}

bool numeric_type(const std::shared_ptr<arrow::DataType>& type) {
    const auto logical = logical_type(type);
    if (!logical) return false;
    switch (logical->id()) {
        case arrow::Type::UINT8:
        case arrow::Type::INT8:
        case arrow::Type::UINT16:
        case arrow::Type::INT16:
        case arrow::Type::UINT32:
        case arrow::Type::INT32:
        case arrow::Type::UINT64:
        case arrow::Type::INT64:
        case arrow::Type::HALF_FLOAT:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        case arrow::Type::DECIMAL128:
        case arrow::Type::DECIMAL256: return true;
        default: return false;
    }
}

bool exact_context_free_cast(const std::shared_ptr<arrow::DataType>& source,
                             const std::shared_ptr<arrow::DataType>& target) {
    if (!source || !target) return false;
    if (source->Equals(target)) return true;
    if (arrow::is_integer(source->id()) && arrow::is_integer(target->id())) return true;
    return source->id() == arrow::Type::FLOAT && target->id() == arrow::Type::DOUBLE;
}

bool contextual_type(const std::shared_ptr<arrow::DataType>& type) {
    const auto logical = logical_type(type);
    if (!logical) return false;
    switch (logical->id()) {
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::DATE32:
        case arrow::Type::DATE64:
        case arrow::Type::TIME32:
        case arrow::Type::TIME64:
        case arrow::Type::TIMESTAMP: return true;
        default: return false;
    }
}

void validate_identity(const json& identity, const std::string& where) {
    require_keys(identity, {"namespace", "name", "version"}, {}, where);
    const auto identity_namespace = required_string(identity, "namespace", where);
    const auto identity_name = required_string(identity, "name", where);
    const auto identity_version = required_uint(identity, "version", where);
    static const std::regex namespace_pattern(R"([a-z][a-z0-9]*(?:\.[a-z][a-z0-9_]*)*)");
    static const std::regex name_pattern(R"([a-z][a-z0-9_]*)");
    if (!std::regex_match(identity_namespace, namespace_pattern) ||
        !std::regex_match(identity_name, name_pattern) || identity_version == 0) {
        invalid(where + " has a noncanonical identity");
    }
}

struct Parser {
    std::shared_ptr<arrow::RecordBatch> batch;
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& join_keys;
    std::shared_ptr<arrow::Schema> output_schema;
    size_t nodes = 0;

    std::shared_ptr<Spec> expression(const json& node, size_t depth, bool root = false) {
        if (depth > kMaxDepth) invalid("expression exceeds nesting-depth limit");
        if (++nodes > kMaxNodes) invalid("document exceeds expression-node limit");
        if (!node.is_object() || !node.contains("node") || !node.at("node").is_string()) {
            invalid("expression must have a string node");
        }
        const auto kind = node.at("node").get<std::string>();
        auto result = std::make_shared<Spec>();
        if (kind == "column_ref") {
            require_keys(node, {"node", "column_index", "column_name"}, {}, "column_ref");
            result->kind = "column";
            result->column_index =
                static_cast<size_t>(required_uint(node, "column_index", "column_ref"));
            result->column_name = required_string(node, "column_name", "column_ref");
            if (!output_schema ||
                result->column_index >= static_cast<size_t>(output_schema->num_fields())) {
                invalid("column_ref index is outside the authoritative output schema");
            }
            const auto& field = output_schema->field(static_cast<int>(result->column_index));
            validate_arrow_extensions(field);
            if (field->name() != result->column_name)
                invalid("column_ref name does not match authoritative index");
            result->data_type = field->type();
            return result;
        }
        if (kind == "field_ref") {
            require_keys(node, {"node", "expression", "field_index", "field_name"}, {},
                         "field_ref");
            result->kind = "field";
            result->children.push_back(expression(node.at("expression"), depth + 1));
            result->field_index =
                static_cast<size_t>(required_uint(node, "field_index", "field_ref"));
            result->field_name = required_string(node, "field_name", "field_ref");
            const auto parent = result->children[0]->data_type;
            if (!parent || parent->id() != arrow::Type::STRUCT)
                invalid("field_ref input must be STRUCT");
            const auto& fields = static_cast<const arrow::StructType&>(*parent).fields();
            if (result->field_index >= fields.size() ||
                fields[result->field_index]->name() != result->field_name) {
                invalid("field_ref name/index does not match authoritative struct");
            }
            validate_arrow_extensions(fields[result->field_index]);
            result->column_name = root_column(result->children[0]);
            result->data_type = fields[result->field_index]->type();
            return result;
        }
        if (kind == "literal") {
            require_keys(node, {"node", "value_ref"}, {}, "literal");
            result->kind = "literal";
            result->value = payload(batch, "value", required_uint(node, "value_ref", "literal"));
            result->data_type = result->value->type();
            return result;
        }
        if (kind == "comparison") {
            require_keys(node, {"node", "op", "left", "right"}, {}, "comparison");
            result->kind = "constant";
            result->op = required_string(node, "op", "comparison");
            if (result->op != "eq" && result->op != "ne" && result->op != "lt" &&
                result->op != "le" && result->op != "gt" && result->op != "ge" &&
                result->op != "distinct_from" && result->op != "not_distinct_from")
                invalid("unknown comparison operator '" + result->op + "'");
            result->children = {expression(node.at("left"), depth + 1),
                                expression(node.at("right"), depth + 1)};
            if (!same_logical_type(result->children[0]->data_type,
                                   result->children[1]->data_type)) {
                invalid("comparison operands have incompatible types");
            }
            result->column_name = root_column(result->children[0]);
            if (result->column_name.empty()) result->column_name = root_column(result->children[1]);
            if (result->children[1]->kind == "literal") result->value = result->children[1]->value;
            result->data_type = arrow::boolean();
            return result;
        }
        if (kind == "and" || kind == "or") {
            require_keys(node, {"node", "children"}, {}, kind);
            if (!node.at("children").is_array() || node.at("children").size() < 2)
                invalid(kind + " requires at least two children");
            result->kind = kind;
            for (const auto& child : node.at("children"))
                result->children.push_back(expression(child, depth + 1));
            for (const auto& child : result->children) {
                if (!boolean_type(child->data_type)) invalid(kind + " children must be BOOLEAN");
            }
            result->data_type = arrow::boolean();
            return result;
        }
        if (kind == "not") {
            require_keys(node, {"node", "expression"}, {}, "not");
            result->kind = "not";
            result->children.push_back(expression(node.at("expression"), depth + 1));
            if (!boolean_type(result->children[0]->data_type))
                invalid("not operand must be BOOLEAN");
            result->data_type = arrow::boolean();
            return result;
        }
        if (kind == "is_null") {
            require_keys(node, {"node", "expression", "negated"}, {}, "is_null");
            if (!node.at("negated").is_boolean()) invalid("is_null.negated must be Boolean");
            result->negated = node.at("negated").get<bool>();
            result->kind = result->negated ? "is_not_null" : "is_null";
            result->children.push_back(expression(node.at("expression"), depth + 1));
            result->column_name = root_column(result->children[0]);
            result->data_type = arrow::boolean();
            return result;
        }
        if (kind == "in") {
            require_keys(node, {"node", "expression", "set", "negated"}, {}, "in");
            if (!node.at("negated").is_boolean()) invalid("in.negated must be Boolean");
            result->kind = "in";
            result->negated = node.at("negated").get<bool>();
            result->children.push_back(expression(node.at("expression"), depth + 1));
            result->column_name = root_column(result->children[0]);
            const auto& set = node.at("set");
            if (!set.is_object() || !set.contains("kind") || !set.at("kind").is_string())
                invalid("in.set must have a string kind");
            if (set.at("kind") == "literal") {
                require_keys(set, {"kind", "value_ref"}, {}, "in.set");
                auto list = payload(batch, "value", required_uint(set, "value_ref", "in.set"));
                if (list->type_id() == arrow::Type::LIST) {
                    const auto& values = static_cast<const arrow::ListArray&>(*list);
                    if (values.IsNull(0)) invalid("literal IN list must not be NULL");
                    result->value = values.value_slice(0);
                } else if (list->type_id() == arrow::Type::LARGE_LIST) {
                    const auto& values = static_cast<const arrow::LargeListArray&>(*list);
                    if (values.IsNull(0)) invalid("literal IN list must not be NULL");
                    result->value = values.value_slice(0);
                } else
                    invalid("literal IN payload must be a list scalar");
            } else if (set.at("kind") == "external") {
                require_keys(set, {"kind", "batch_index", "column_index", "column_name"}, {},
                             "in.set");
                const auto bi = static_cast<size_t>(required_uint(set, "batch_index", "in.set"));
                const auto ci = static_cast<size_t>(required_uint(set, "column_index", "in.set"));
                const auto name = required_string(set, "column_name", "in.set");
                if (bi >= join_keys.size() ||
                    ci >= static_cast<size_t>(join_keys[bi]->num_columns()))
                    invalid("external IN batch/column index is unavailable");
                if (join_keys[bi]->schema()->field(static_cast<int>(ci))->name() != name)
                    invalid("external IN column name does not match authoritative index");
                validate_arrow_extensions(join_keys[bi]->schema()->field(static_cast<int>(ci)));
                result->value = join_keys[bi]->column(static_cast<int>(ci));
            } else
                invalid("in.set has an unknown kind");
            if (!same_logical_type(result->children[0]->data_type, result->value->type()))
                invalid("IN expression and set have incompatible types");
            result->data_type = arrow::boolean();
            return result;
        }
        if (kind == "cast") {
            require_keys(node, {"node", "expression", "type_ref"}, {}, "cast");
            result->kind = "cast";
            result->children.push_back(expression(node.at("expression"), depth + 1));
            result->value = payload(batch, "type", required_uint(node, "type_ref", "cast"));
            if (!result->value->IsNull(0)) invalid("cast type payload must contain NULL");
            result->data_type = result->value->type();
            if (contextual_type(result->children[0]->data_type) &&
                contextual_type(result->data_type)) {
                invalid("context-dependent cast requires vgi.duckdb.session.v1");
            }
            if (!exact_context_free_cast(result->children[0]->data_type, result->data_type))
                invalid("cast has no exact context-free standard-v1 evaluator");
            return result;
        }
        if (kind == "arithmetic") {
            require_keys(node, {"node", "op", "left", "right"}, {}, "arithmetic");
            result->kind = "arithmetic";
            result->op = required_string(node, "op", "arithmetic");
            if (result->op != "add" && result->op != "subtract" && result->op != "multiply" &&
                result->op != "divide" && result->op != "modulo")
                invalid("unknown arithmetic operator");
            if (result->op == "divide" || result->op == "modulo")
                invalid("context-dependent arithmetic requires vgi.duckdb.session.v1");
            result->children = {expression(node.at("left"), depth + 1),
                                expression(node.at("right"), depth + 1)};
            if (!numeric_type(result->children[0]->data_type) ||
                !same_logical_type(result->children[0]->data_type,
                                   result->children[1]->data_type)) {
                invalid("arithmetic operands require one exact numeric type");
            }
            result->data_type = result->children[0]->data_type;
            return result;
        }
        if (kind == "negate") {
            require_keys(node, {"node", "expression"}, {}, "negate");
            result->kind = "negate";
            result->children.push_back(expression(node.at("expression"), depth + 1));
            if (!numeric_type(result->children[0]->data_type))
                invalid("negate operand must be numeric");
            result->data_type = result->children[0]->data_type;
            return result;
        }
        if (kind == "call") {
            require_keys(node, {"node", "function", "arguments"}, {"options"}, "call");
            if (!node.at("function").is_string()) {
                validate_identity(node.at("function"), "call.function");
                invalid("extension filter functions were not advertised");
            }
            result->function = node.at("function").get<std::string>();
            if (result->function != "starts_with" && result->function != "ends_with" &&
                result->function != "contains" && result->function != "list_contains")
                invalid("unknown standard filter function");
            if (!node.at("arguments").is_array() || node.at("arguments").size() != 2)
                invalid("standard filter functions require exactly two arguments");
            if (node.contains("options") && !node.at("options").empty())
                invalid("standard filter functions do not accept options");
            result->kind = "call";
            for (const auto& argument : node.at("arguments"))
                result->children.push_back(expression(argument, depth + 1));
            if (result->function == "starts_with" || result->function == "ends_with" ||
                result->function == "contains") {
                for (const auto& argument : result->children) {
                    if (!argument->data_type || argument->data_type->id() != arrow::Type::STRING)
                        invalid("string function arguments must be UTF8");
                }
            } else {
                const auto list_type = result->children[0]->data_type;
                if (!list_type || list_type->id() != arrow::Type::LIST)
                    invalid("list_contains first argument must be LIST");
                const auto element_type =
                    static_cast<const arrow::ListType&>(*list_type).value_type();
                if (!same_logical_type(element_type, result->children[1]->data_type))
                    invalid("list_contains element type mismatch");
            }
            result->data_type = arrow::boolean();
            return result;
        }
        if (kind == "runtime_filter") {
            require_keys(node, {"node", "algorithm", "input", "artifact_ref", "null_handling"}, {},
                         "runtime_filter");
            if (!root) invalid("runtime_filter may appear only at a predicate root");
            const auto& algorithm = node.at("algorithm");
            validate_identity(algorithm, "runtime_filter.algorithm");
            const auto algorithm_namespace =
                required_string(algorithm, "namespace", "runtime_filter.algorithm");
            const auto algorithm_name =
                required_string(algorithm, "name", "runtime_filter.algorithm");
            const auto algorithm_version =
                required_uint(algorithm, "version", "runtime_filter.algorithm");
            if (algorithm_namespace != "duckdb.runtime_filter" ||
                (algorithm_name != "bloom" && algorithm_name != "prefix_range") ||
                algorithm_version != 1) {
                invalid("unknown runtime-filter algorithm");
            }
            const auto null_handling = required_string(node, "null_handling", "runtime_filter");
            if (null_handling != "pass" && null_handling != "reject")
                invalid("runtime_filter.null_handling must be pass or reject");
            result->kind = "runtime_filter";
            result->children.push_back(expression(node.at("input"), depth + 1));
            (void)payload(batch, "artifact", required_uint(node, "artifact_ref", "runtime_filter"));
            return result;
        }
        invalid("unknown expression node '" + kind + "'");
    }
};

json document_for(const std::shared_ptr<arrow::RecordBatch>& batch) {
    const auto text = std::static_pointer_cast<arrow::StringArray>(batch->column(0))->GetString(0);
    if (std::getenv("VGI_FILTER_DEBUG")) std::fprintf(stderr, "[vgi-filter] %s\n", text.c_str());
    try {
        bool duplicate = false;
        std::vector<std::unordered_set<std::string>> object_keys;
        auto callback = [&](int depth, json::parse_event_t event, json& parsed) {
            if (event == json::parse_event_t::object_start) {
                if (object_keys.size() <= static_cast<size_t>(depth))
                    object_keys.resize(static_cast<size_t>(depth) + 1);
                object_keys[static_cast<size_t>(depth)].clear();
            } else if (event == json::parse_event_t::key && depth > 0) {
                auto& keys = object_keys[static_cast<size_t>(depth - 1)];
                if (!keys.insert(parsed.get<std::string>()).second) duplicate = true;
            }
            return true;
        };
        auto document = json::parse(text, callback);
        if (duplicate) invalid("invalid filter JSON: duplicate object key");
        return document;
    } catch (const json::exception& error) {
        invalid(std::string("invalid filter JSON: ") + error.what());
    }
}

void validate_header(const json& document, const char* kind, const char* member) {
    require_keys(document, {"encoding", "semantics", "kind", member}, {}, "filter document");
    if (document.at("encoding") != "vgi.filters.v2")
        invalid("document encoding must be vgi.filters.v2");
    if (document.at("semantics") != "vgi.duckdb.standard.v1")
        invalid("unsupported filter semantics");
    if (document.at("kind") != kind)
        invalid(std::string("expected a ") + kind + " filter document");
    if (!document.at(member).is_array()) invalid(std::string(member) + " must be an array");
}

std::shared_ptr<Spec> parse_predicate(Parser& parser, const json& item, bool delta) {
    require_keys(item, {"id", "revision", "mode", "source", "expression"},
                 delta ? std::initializer_list<const char*>{"operation"}
                       : std::initializer_list<const char*>{},
                 "predicate");
    auto expression = parser.expression(item.at("expression"), 1, true);
    expression->id = required_string(item, "id", "predicate");
    if (expression->id.size() > 128) invalid("predicate ID exceeds 128 bytes");
    expression->revision = required_uint(item, "revision", "predicate");
    const auto mode = required_string(item, "mode", "predicate");
    if (mode != "required" && mode != "advisory") invalid("unknown predicate mode");
    if (delta && mode != "advisory") invalid("delta upserts must be advisory");
    expression->advisory = mode == "advisory";
    const auto source = required_string(item, "source", "predicate");
    if (source != "query" && source != "join" && source != "top_n" &&
        source != "split_refinement" && source != "other")
        invalid("unknown predicate source");
    if (expression->kind == "runtime_filter" && !expression->advisory) {
        invalid("runtime_filter predicates must be advisory");
    }
    if (expression->kind != "runtime_filter" && !boolean_type(expression->data_type))
        invalid("predicate root must resolve to BOOLEAN");
    return expression;
}

Filter::Kind kind_of(const std::string& kind) {
    if (kind == "constant") return Filter::Kind::Constant;
    if (kind == "in") return Filter::Kind::In;
    if (kind == "is_null") return Filter::Kind::IsNull;
    if (kind == "is_not_null") return Filter::Kind::IsNotNull;
    return Filter::Kind::Other;
}

void collect_columns(const std::shared_ptr<Spec>& spec, std::vector<std::string>& columns) {
    if (spec->kind == "column") columns.push_back(spec->column_name);
    for (const auto& child : spec->children) collect_columns(child, columns);
}

void flatten(const std::shared_ptr<Spec>& spec, std::vector<Filter>& out) {
    if (spec->kind == "and" || spec->kind == "or") {
        for (const auto& child : spec->children) flatten(child, out);
        return;
    }
    std::vector<std::string> columns;
    collect_columns(spec, columns);
    if (columns.empty()) out.push_back({kind_of(spec->kind), spec->column_name, spec->op});
    for (const auto& column : columns) out.push_back({kind_of(spec->kind), column, spec->op});
}

std::shared_ptr<arrow::Array> array_from_datum(const arrow::Datum& datum, int64_t length) {
    if (datum.is_array()) {
        auto array = datum.make_array();
        if (array->length() != length)
            throw std::runtime_error("filter expression length mismatch");
        return array;
    }
    if (!datum.is_scalar())
        throw std::runtime_error("filter expression did not yield an array or scalar");
    return value_or_throw(arrow::MakeArrayFromScalar(*datum.scalar(), length),
                          "broadcast filter scalar");
}

arrow::Datum decoded_dictionary(arrow::Datum value) {
    if (!value.is_array() || value.type()->id() != arrow::Type::DICTIONARY) return value;
    const auto& dictionary = static_cast<const arrow::DictionaryType&>(*value.type());
    return value_or_throw(arrow::compute::Cast(value, dictionary.value_type()),
                          "decode dictionary filter input");
}

arrow::Datum call(const std::string& name, std::vector<arrow::Datum> arguments) {
    static const auto initialized = arrow::compute::Initialize();
    if (!initialized.ok()) throw std::runtime_error(initialized.ToString());
    for (auto& argument : arguments) argument = decoded_dictionary(std::move(argument));
    return value_or_throw(arrow::compute::CallFunction(name, std::move(arguments)),
                          "evaluate " + name);
}

bool floating_type(const std::shared_ptr<arrow::DataType>& type) {
    return type->id() == arrow::Type::HALF_FLOAT || type->id() == arrow::Type::FLOAT ||
           type->id() == arrow::Type::DOUBLE;
}

bool comparison_type_supported(const std::shared_ptr<arrow::DataType>& type) {
    switch (type->id()) {
        case arrow::Type::NA:
        case arrow::Type::LIST:
        case arrow::Type::LARGE_LIST:
        case arrow::Type::FIXED_SIZE_LIST:
        case arrow::Type::LIST_VIEW:
        case arrow::Type::LARGE_LIST_VIEW:
        case arrow::Type::STRUCT:
        case arrow::Type::MAP:
        case arrow::Type::SPARSE_UNION:
        case arrow::Type::DENSE_UNION:
        case arrow::Type::RUN_END_ENCODED:
        case arrow::Type::EXTENSION: return false;
        default: return true;
    }
}

bool floating_compare(double left, double right, const std::string& op) {
    int ordering = 0;
    if (std::isnan(left)) {
        ordering = std::isnan(right) ? 0 : 1;
    } else if (std::isnan(right)) {
        ordering = -1;
    } else if (left < right) {
        ordering = -1;
    } else if (left > right) {
        ordering = 1;
    }
    if (op == "eq" || op == "not_distinct_from") return ordering == 0;
    if (op == "ne" || op == "distinct_from") return ordering != 0;
    if (op == "lt") return ordering < 0;
    if (op == "le") return ordering <= 0;
    if (op == "gt") return ordering > 0;
    if (op == "ge") return ordering >= 0;
    throw std::runtime_error("unsupported comparison operator");
}

arrow::Datum compare(arrow::Datum left, arrow::Datum right, const std::string& op, int64_t length) {
    left = decoded_dictionary(std::move(left));
    right = decoded_dictionary(std::move(right));
    auto left_values = array_from_datum(left, length);
    auto right_values = array_from_datum(right, length);
    if (!left_values->type()->Equals(right_values->type()) ||
        !comparison_type_supported(left_values->type())) {
        throw std::runtime_error("comparison type has no exact standard-v1 evaluator");
    }

    const bool distinct = op == "distinct_from" || op == "not_distinct_from";
    if (floating_type(left_values->type()) || floating_type(right_values->type())) {
        left_values =
            array_from_datum(value_or_throw(arrow::compute::Cast(left_values, arrow::float64()),
                                            "cast floating comparison input"),
                             length);
        right_values =
            array_from_datum(value_or_throw(arrow::compute::Cast(right_values, arrow::float64()),
                                            "cast floating comparison input"),
                             length);
        const auto& left_floats = static_cast<const arrow::DoubleArray&>(*left_values);
        const auto& right_floats = static_cast<const arrow::DoubleArray&>(*right_values);
        arrow::BooleanBuilder output;
        auto status = output.Reserve(length);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        for (int64_t row = 0; row < length; ++row) {
            const bool left_null = left_floats.IsNull(row);
            const bool right_null = right_floats.IsNull(row);
            if (left_null || right_null) {
                if (distinct) {
                    const bool value = left_null != right_null;
                    (void)output.Append(op == "distinct_from" ? value : !value);
                } else {
                    (void)output.AppendNull();
                }
                continue;
            }
            (void)output.Append(
                floating_compare(left_floats.Value(row), right_floats.Value(row), op));
        }
        std::shared_ptr<arrow::Array> result;
        status = output.Finish(&result);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return arrow::Datum(result);
    }

    if (distinct) {
        auto equals = array_from_datum(call("equal", {left_values, right_values}), length);
        const auto& equal_values = static_cast<const arrow::BooleanArray&>(*equals);
        arrow::BooleanBuilder output;
        auto status = output.Reserve(length);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        for (int64_t row = 0; row < length; ++row) {
            const bool left_null = left_values->IsNull(row);
            const bool right_null = right_values->IsNull(row);
            const bool value =
                left_null || right_null ? left_null != right_null : !equal_values.Value(row);
            (void)output.Append(op == "distinct_from" ? value : !value);
        }
        std::shared_ptr<arrow::Array> result;
        status = output.Finish(&result);
        if (!status.ok()) throw std::runtime_error(status.ToString());
        return arrow::Datum(result);
    }

    static const std::map<std::string, std::string> kernels = {
        {"eq", "equal"},      {"ne", "not_equal"}, {"lt", "less"},
        {"le", "less_equal"}, {"gt", "greater"},   {"ge", "greater_equal"},
    };
    const auto found = kernels.find(op);
    if (found == kernels.end()) throw std::runtime_error("unsupported comparison operator");
    return call(found->second, {left_values, right_values});
}

std::optional<std::string> string_at(const std::shared_ptr<arrow::Array>& values, int64_t index) {
    if (values->IsNull(index)) return std::nullopt;
    if (values->type_id() == arrow::Type::STRING) {
        return static_cast<const arrow::StringArray&>(*values).GetString(index);
    }
    if (values->type_id() == arrow::Type::LARGE_STRING) {
        return static_cast<const arrow::LargeStringArray&>(*values).GetString(index);
    }
    throw std::runtime_error("standard string filter requires UTF8 arguments");
}

double floating_scalar_value(const std::shared_ptr<arrow::Scalar>& value) {
    auto array =
        value_or_throw(arrow::MakeArrayFromScalar(*value, 1), "materialize floating scalar");
    auto casted =
        value_or_throw(arrow::compute::Cast(array, arrow::float64()), "cast floating scalar");
    return std::static_pointer_cast<arrow::DoubleArray>(casted.make_array())->Value(0);
}

bool nested_scalar_equal(const std::shared_ptr<arrow::Scalar>& left,
                         const std::shared_ptr<arrow::Scalar>& right) {
    if (!left || !right) throw std::runtime_error("nested equality received no scalar");
    if (!left->is_valid || !right->is_valid) return left->is_valid == right->is_valid;
    if (!same_logical_type(left->type, right->type)) return false;
    const auto decode_dictionary = [](std::shared_ptr<arrow::Scalar> value) {
        if (value->type->id() != arrow::Type::DICTIONARY) return value;
        auto array =
            value_or_throw(arrow::MakeArrayFromScalar(*value, 1), "materialize dictionary scalar");
        const auto target = logical_type(value->type);
        auto decoded =
            value_or_throw(arrow::compute::Cast(array, target), "decode dictionary scalar");
        return value_or_throw(decoded.make_array()->GetScalar(0), "read dictionary scalar");
    };
    const auto left_value = decode_dictionary(left);
    const auto right_value = decode_dictionary(right);
    const auto type = left_value->type;
    if (floating_type(type)) {
        return floating_compare(floating_scalar_value(left_value),
                                floating_scalar_value(right_value), "eq");
    }
    if (type->id() == arrow::Type::STRUCT) {
        const auto& left_struct = static_cast<const arrow::StructScalar&>(*left_value);
        const auto& right_struct = static_cast<const arrow::StructScalar&>(*right_value);
        if (left_struct.value.size() != right_struct.value.size()) return false;
        for (size_t i = 0; i < left_struct.value.size(); ++i) {
            if (!nested_scalar_equal(left_struct.value[i], right_struct.value[i])) return false;
        }
        return true;
    }
    switch (type->id()) {
        case arrow::Type::LIST:
        case arrow::Type::LARGE_LIST:
        case arrow::Type::FIXED_SIZE_LIST:
        case arrow::Type::LIST_VIEW:
        case arrow::Type::LARGE_LIST_VIEW:
        case arrow::Type::MAP: {
            const auto& left_list = static_cast<const arrow::BaseListScalar&>(*left_value);
            const auto& right_list = static_cast<const arrow::BaseListScalar&>(*right_value);
            if (left_list.value->length() != right_list.value->length()) return false;
            for (int64_t i = 0; i < left_list.value->length(); ++i) {
                const auto left_child =
                    value_or_throw(left_list.value->GetScalar(i), "read nested list value");
                const auto right_child =
                    value_or_throw(right_list.value->GetScalar(i), "read nested list value");
                if (!nested_scalar_equal(left_child, right_child)) return false;
            }
            return true;
        }
        default: return left_value->Equals(*right_value);
    }
}

arrow::Datum evaluate(const std::shared_ptr<Spec>& spec,
                      const std::shared_ptr<arrow::RecordBatch>& batch);

arrow::Datum evaluate_standard_call(const Spec& spec,
                                    const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto left = array_from_datum(evaluate(spec.children[0], batch), batch->num_rows());
    auto right = array_from_datum(evaluate(spec.children[1], batch), batch->num_rows());
    arrow::BooleanBuilder output;
    auto status = output.Reserve(batch->num_rows());
    if (!status.ok()) throw std::runtime_error(status.ToString());
    for (int64_t row = 0; row < batch->num_rows(); ++row) {
        if (left->IsNull(row) || right->IsNull(row)) {
            (void)output.AppendNull();
            continue;
        }
        bool matched = false;
        if (spec.function == "list_contains") {
            std::shared_ptr<arrow::Array> values;
            if (left->type_id() == arrow::Type::LIST) {
                values = static_cast<const arrow::ListArray&>(*left).value_slice(row);
            } else if (left->type_id() == arrow::Type::LARGE_LIST) {
                values = static_cast<const arrow::LargeListArray&>(*left).value_slice(row);
            } else {
                throw std::runtime_error("list_contains requires a list argument");
            }
            const auto needle = value_or_throw(right->GetScalar(row), "read list_contains needle");
            for (int64_t i = 0; i < values->length(); ++i) {
                if (values->IsNull(i)) continue;
                const auto candidate =
                    value_or_throw(values->GetScalar(i), "read list_contains value");
                if (nested_scalar_equal(candidate, needle)) {
                    matched = true;
                    break;
                }
            }
        } else {
            const auto haystack = *string_at(left, row);
            const auto needle = *string_at(right, row);
            if (spec.function == "starts_with") matched = haystack.rfind(needle, 0) == 0;
            if (spec.function == "ends_with") {
                matched =
                    haystack.size() >= needle.size() &&
                    haystack.compare(haystack.size() - needle.size(), needle.size(), needle) == 0;
            }
            if (spec.function == "contains") matched = haystack.find(needle) != std::string::npos;
        }
        (void)output.Append(matched);
    }
    std::shared_ptr<arrow::Array> result;
    status = output.Finish(&result);
    if (!status.ok()) throw std::runtime_error(status.ToString());
    return arrow::Datum(result);
}

arrow::Datum evaluate(const std::shared_ptr<Spec>& spec,
                      const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (spec->kind == "column") {
        int index = -1;
        if (spec->column_index < static_cast<size_t>(batch->num_columns()) &&
            batch->schema()->field(static_cast<int>(spec->column_index))->name() ==
                spec->column_name) {
            index = static_cast<int>(spec->column_index);
        } else {
            const auto indices = batch->schema()->GetAllFieldIndices(spec->column_name);
            if (indices.size() == 1) index = indices.front();
        }
        if (index < 0)
            throw std::runtime_error("filter column '" + spec->column_name + "' is absent");
        if (!batch->schema()->field(index)->type()->Equals(spec->data_type))
            throw std::runtime_error("filter column '" + spec->column_name + "' changed type");
        return arrow::Datum(batch->column(index));
    }
    if (spec->kind == "field") {
        auto parent = array_from_datum(evaluate(spec->children[0], batch), batch->num_rows());
        auto values = std::dynamic_pointer_cast<arrow::StructArray>(parent);
        if (!values || spec->field_index >= static_cast<size_t>(values->num_fields())) {
            throw std::runtime_error("filter field_ref input is not the authoritative struct");
        }
        auto child =
            decoded_dictionary(arrow::Datum(values->field(static_cast<int>(spec->field_index))));
        if (values->null_count() == 0) return child;
        return call("if_else", {call("is_valid", {parent}), child,
                                arrow::Datum(arrow::MakeNullScalar(child.type()))});
    }
    if (spec->kind == "literal") {
        return arrow::Datum(value_or_throw(spec->value->GetScalar(0), "read filter literal"));
    }
    if (spec->kind == "constant") {
        return compare(evaluate(spec->children[0], batch), evaluate(spec->children[1], batch),
                       spec->op, batch->num_rows());
    }
    if (spec->kind == "and" || spec->kind == "or") {
        auto result = evaluate(spec->children[0], batch);
        for (size_t i = 1; i < spec->children.size(); ++i) {
            result = call(spec->kind == "and" ? "and_kleene" : "or_kleene",
                          {std::move(result), evaluate(spec->children[i], batch)});
        }
        return result;
    }
    if (spec->kind == "not") return call("invert", {evaluate(spec->children[0], batch)});
    if (spec->kind == "is_null" || spec->kind == "is_not_null") {
        return call(spec->kind == "is_null" ? "is_null" : "is_valid",
                    {evaluate(spec->children[0], batch)});
    }
    if (spec->kind == "in") {
        auto input = decoded_dictionary(evaluate(spec->children[0], batch));
        auto values = decoded_dictionary(arrow::Datum(spec->value)).make_array();
        auto result = arrow::Datum(value_or_throw(
            arrow::MakeArrayFromScalar(arrow::BooleanScalar(false), batch->num_rows()),
            "initialize IN result"));
        for (int64_t index = 0; index < values->length(); ++index) {
            auto candidate = value_or_throw(values->GetScalar(index), "read IN value");
            result = call("or_kleene", {std::move(result), compare(input, arrow::Datum(candidate),
                                                                   "eq", batch->num_rows())});
        }
        return spec->negated ? call("invert", {std::move(result)}) : result;
    }
    if (spec->kind == "cast") {
        const auto source = spec->children[0]->data_type;
        const auto target = spec->data_type;
        if (!exact_context_free_cast(source, target))
            throw std::runtime_error("cast has no exact context-free standard-v1 evaluator");
        return value_or_throw(
            arrow::compute::Cast(evaluate(spec->children[0], batch), spec->data_type),
            "evaluate cast");
    }
    if (spec->kind == "arithmetic") {
        return call(spec->op + "_checked",
                    {evaluate(spec->children[0], batch), evaluate(spec->children[1], batch)});
    }
    if (spec->kind == "negate") return call("negate_checked", {evaluate(spec->children[0], batch)});
    if (spec->kind == "call") {
        return evaluate_standard_call(*spec, batch);
    }
    throw std::runtime_error("runtime filter has no negotiated evaluator");
}

std::string format_scalar(const std::shared_ptr<arrow::Array>& array, int64_t index) {
    if (!array || index >= array->length() || array->IsNull(index)) return "NULL";
    if (array->type_id() == arrow::Type::STRING) {
        return "'" + static_cast<const arrow::StringArray&>(*array).GetString(index) + "'";
    }
    if (array->type_id() == arrow::Type::LARGE_STRING) {
        return "'" + static_cast<const arrow::LargeStringArray&>(*array).GetString(index) + "'";
    }
    if (array->type_id() == arrow::Type::BOOL) {
        return static_cast<const arrow::BooleanArray&>(*array).Value(index) ? "True" : "False";
    }
    auto casted = arrow::compute::Cast(*array->Slice(index, 1), arrow::utf8());
    if (!casted.ok()) return {};
    return std::static_pointer_cast<arrow::StringArray>(casted.MoveValueUnsafe())->GetString(0);
}

const char* op_symbol(const std::string& op) {
    if (op == "eq") return "=";
    if (op == "ne") return "!=";
    if (op == "lt") return "<";
    if (op == "le") return "<=";
    if (op == "gt") return ">";
    if (op == "ge") return ">=";
    if (op == "distinct_from") return "IS DISTINCT FROM";
    if (op == "not_distinct_from") return "IS NOT DISTINCT FROM";
    return "?";
}

std::string render(const std::shared_ptr<Spec>& spec) {
    if (spec->kind == "column") return spec->column_name;
    if (spec->kind == "field") return render(spec->children[0]) + "." + spec->field_name;
    if (spec->kind == "literal") return format_scalar(spec->value, 0);
    if (spec->kind == "constant") {
        return render(spec->children[0]) + " " + op_symbol(spec->op) + " " +
               render(spec->children[1]);
    }
    if (spec->kind == "and" || spec->kind == "or") {
        std::string result = "(";
        for (size_t i = 0; i < spec->children.size(); ++i) {
            if (i) result += spec->kind == "and" ? " AND " : " OR ";
            result += render(spec->children[i]);
        }
        return result + ")";
    }
    if (spec->kind == "not") return "NOT (" + render(spec->children[0]) + ")";
    if (spec->kind == "is_null" || spec->kind == "is_not_null") {
        return render(spec->children[0]) + (spec->kind == "is_null" ? " IS NULL" : " IS NOT NULL");
    }
    if (spec->kind == "in") {
        std::string values;
        if (spec->value && spec->value->length() > 20) {
            values = std::to_string(spec->value->length()) + " values";
        } else {
            for (int64_t i = 0; spec->value && i < spec->value->length(); ++i) {
                if (i) values += ", ";
                values += format_scalar(spec->value, i);
            }
        }
        return render(spec->children[0]) + (spec->negated ? " NOT IN (" : " IN (") + values + ")";
    }
    return "(expression)";
}

std::string render_repr(const std::shared_ptr<Spec>& spec) {
    if (spec->kind == "constant") return "ConstantFilter(" + render(spec) + ")";
    if (spec->kind == "and" || spec->kind == "or") {
        std::string result = spec->kind == "and" ? "AndFilter([" : "OrFilter([";
        for (size_t i = 0; i < spec->children.size(); ++i) {
            if (i) result += ", ";
            result += render_repr(spec->children[i]);
        }
        return result + "])";
    }
    return render(spec);
}

bool mentions(const std::shared_ptr<Spec>& spec, const std::string& column) {
    std::vector<std::string> columns;
    collect_columns(spec, columns);
    return std::find(columns.begin(), columns.end(), column) != columns.end();
}

std::shared_ptr<arrow::Array> discrete_values(const std::shared_ptr<Spec>& spec,
                                              const std::string& column) {
    if (spec->kind == "constant" && spec->op == "eq" && spec->children.size() == 2 &&
        is_direct_column(spec->children[0], column) && spec->children[1]->kind == "literal") {
        return spec->children[1]->value;
    }
    if (spec->kind == "in" && !spec->negated && !spec->children.empty() &&
        is_direct_column(spec->children[0], column))
        return spec->value;
    if (spec->kind == "and") {
        for (const auto& child : spec->children) {
            if (auto found = discrete_values(child, column)) return found;
        }
        return nullptr;
    }
    if (spec->kind == "or") {
        arrow::ArrayVector arrays;
        for (const auto& child : spec->children) {
            auto found = discrete_values(child, column);
            if (!found) return nullptr;
            arrays.push_back(std::move(found));
        }
        if (arrays.empty()) return nullptr;
        auto combined = arrow::Concatenate(arrays);
        if (!combined.ok()) return nullptr;
        auto unique = arrow::compute::Unique(combined.MoveValueUnsafe());
        return unique.ok() ? unique.MoveValueUnsafe() : nullptr;
    }
    return nullptr;
}

struct Bounds {
    std::optional<int64_t> min;
    std::optional<int64_t> max;
};

std::optional<int64_t> scalar_int64(const std::shared_ptr<arrow::Array>& value) {
    if (!value || value->length() == 0 || value->IsNull(0)) return std::nullopt;
    if (!arrow::is_integer(logical_type(value->type())->id())) return std::nullopt;
    auto casted = arrow::compute::Cast(*value->Slice(0, 1), arrow::int64());
    if (!casted.ok()) return std::nullopt;
    return std::static_pointer_cast<arrow::Int64Array>(casted.MoveValueUnsafe())->Value(0);
}

Bounds intersect(Bounds left, const Bounds& right) {
    if (right.min) left.min = left.min ? std::max(*left.min, *right.min) : right.min;
    if (right.max) left.max = left.max ? std::min(*left.max, *right.max) : right.max;
    return left;
}

Bounds unite(Bounds left, const Bounds& right) {
    if (left.min && right.min)
        left.min = std::min(*left.min, *right.min);
    else
        left.min.reset();
    if (left.max && right.max)
        left.max = std::max(*left.max, *right.max);
    else
        left.max.reset();
    return left;
}

std::optional<Bounds> bounds_for(const std::shared_ptr<Spec>& spec, const std::string& column) {
    if (spec->kind == "constant" && spec->children.size() == 2) {
        auto op = spec->op;
        std::shared_ptr<Spec> literal;
        if (is_direct_column(spec->children[0], column) && spec->children[1]->kind == "literal") {
            literal = spec->children[1];
        } else if (is_direct_column(spec->children[1], column) &&
                   spec->children[0]->kind == "literal") {
            literal = spec->children[0];
            if (op == "gt")
                op = "lt";
            else if (op == "ge")
                op = "le";
            else if (op == "lt")
                op = "gt";
            else if (op == "le")
                op = "ge";
        }
        auto value = literal ? scalar_int64(literal->value) : std::nullopt;
        if (!value) return std::nullopt;
        Bounds result;
        if (op == "eq") result.min = result.max = *value;
        if (op == "gt")
            result.min = *value == std::numeric_limits<int64_t>::max() ? *value : *value + 1;
        if (op == "ge") result.min = *value;
        if (op == "lt")
            result.max = *value == std::numeric_limits<int64_t>::min() ? *value : *value - 1;
        if (op == "le") result.max = *value;
        if (!result.min && !result.max) return std::nullopt;
        return result;
    }
    if (spec->kind == "in" && !spec->negated && is_direct_column(spec->children[0], column)) {
        if (!arrow::is_integer(logical_type(spec->value->type())->id())) return std::nullopt;
        auto casted = arrow::compute::Cast(*spec->value, arrow::int64());
        if (!casted.ok()) return std::nullopt;
        const auto values = std::static_pointer_cast<arrow::Int64Array>(casted.MoveValueUnsafe());
        Bounds result;
        for (int64_t i = 0; i < values->length(); ++i) {
            if (values->IsNull(i)) continue;
            result.min = result.min ? std::min(*result.min, values->Value(i)) : values->Value(i);
            result.max = result.max ? std::max(*result.max, values->Value(i)) : values->Value(i);
        }
        return result.min ? std::optional<Bounds>(result) : std::nullopt;
    }
    if (spec->kind == "and" || spec->kind == "or") {
        std::optional<Bounds> result;
        for (const auto& child : spec->children) {
            auto current = bounds_for(child, column);
            if (!current) {
                if (spec->kind == "or") return std::nullopt;
                continue;
            }
            result = result ? (spec->kind == "and" ? intersect(*result, *current)
                                                   : unite(*result, *current))
                            : current;
        }
        return result;
    }
    return std::nullopt;
}

}  // namespace

PushdownFilters PushdownFilters::parse(const std::string& ipc_bytes,
                                       const std::vector<std::string>& join_key_batches,
                                       std::shared_ptr<arrow::Schema> output_schema) {
    PushdownFilters filters;
    filters.output_schema_ = std::move(output_schema);
    for (const auto& bytes : join_key_batches) {
        auto batch = wire::decode_ipc(bytes);
        if (!batch) invalid("join-key IPC stream contains no batch");
        filters.join_keys_.push_back(std::move(batch));
    }
    if (ipc_bytes.empty()) return filters;
    if (ipc_bytes.size() > kMaxPayloadBytes) invalid("filter payload exceeds 16 MiB");
    auto batch = wire::decode_ipc(ipc_bytes);
    filters.evaluation_context_ = validate_batch(batch);
    auto document = document_for(batch);
    validate_header(document, "snapshot", "predicates");
    if (document.at("predicates").size() > kMaxPredicates)
        invalid("snapshot exceeds predicate limit");
    if (!document.at("predicates").empty() && !filters.output_schema_) {
        invalid("non-empty snapshot requires the authoritative unprojected bind output schema");
    }
    Parser parser{batch, filters.join_keys_, filters.output_schema_};
    for (const auto& item : document.at("predicates")) {
        auto predicate = parse_predicate(parser, item, false);
        if (predicate->revision != 0) invalid("snapshot predicate revisions must be zero");
        if (!filters.revisions_.emplace(predicate->id, 0).second) invalid("duplicate predicate ID");
        if (!predicate->advisory) filters.required_ids_.insert(predicate->id);
        filters.specs_.push_back(std::move(predicate));
    }
    for (const auto& spec : filters.specs_) flatten(spec, filters.filters_);
    return filters;
}

void PushdownFilters::apply_delta(const std::string& ipc_bytes) {
    if (ipc_bytes.empty()) invalid("dynamic filter delta is empty");
    if (ipc_bytes.size() > kMaxPayloadBytes) invalid("filter payload exceeds 16 MiB");
    auto batch = wire::decode_ipc(ipc_bytes);
    const auto context = validate_batch(batch);
    if (context != evaluation_context_) invalid("evaluation context changed within one scan");
    auto document = document_for(batch);
    validate_header(document, "delta", "updates");
    PushdownFilters next = *this;
    Parser parser{batch, next.join_keys_, next.output_schema_};
    std::unordered_set<std::string> seen;
    for (const auto& update : document.at("updates")) {
        if (!update.is_object()) invalid("delta update must be an object");
        if (!update.contains("operation") || !update.contains("id") ||
            !update.contains("revision")) {
            invalid("delta update is missing required properties");
        }
        const auto operation = required_string(update, "operation", "update");
        const auto id = required_string(update, "id", "update");
        const auto revision = required_uint(update, "revision", "update");
        if (id.size() > 128) invalid("predicate ID exceeds 128 bytes");
        if (!seen.insert(id).second) invalid("duplicate delta predicate ID");
        if (required_ids_.count(id)) invalid("delta targets required predicate");
        std::shared_ptr<Spec> parsed;
        if (operation == "remove") {
            require_keys(update, {"operation", "id", "revision"}, {}, "remove update");
        } else if (operation == "upsert") {
            require_keys(update, {"operation", "id", "revision", "mode", "source", "expression"},
                         {}, "upsert update");
            parsed = parse_predicate(parser, update, true);
        } else
            invalid("delta operation must be remove or upsert");
        const auto old = next.revisions_.find(id);
        if (old != next.revisions_.end() && revision <= old->second) continue;
        next.specs_.erase(std::remove_if(next.specs_.begin(), next.specs_.end(),
                                         [&](const auto& spec) { return spec->id == id; }),
                          next.specs_.end());
        if (parsed) next.specs_.push_back(std::move(parsed));
        next.revisions_[id] = revision;
    }
    if (next.revisions_.size() > kMaxPredicateIds) invalid("delta exceeds predicate-ID limit");
    next.filters_.clear();
    for (const auto& spec : next.specs_) flatten(spec, next.filters_);
    *this = std::move(next);
}

std::shared_ptr<arrow::Array> PushdownFilters::values_for(const Spec& spec) const {
    return spec.value;
}

std::vector<Filter> PushdownFilters::column_filters(const std::string& column) const {
    std::vector<Filter> result;
    for (const auto& filter : filters_)
        if (filter.column_name == column) result.push_back(filter);
    return result;
}

bool PushdownFilters::has_filter_for_column(const std::string& column) const {
    return std::any_of(specs_.begin(), specs_.end(),
                       [&](const auto& spec) { return mentions(spec, column); });
}

std::vector<std::string> PushdownFilters::filtered_columns() const {
    std::vector<std::string> result;
    for (const auto& spec : specs_) collect_columns(spec, result);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

ColumnBounds PushdownFilters::column_bounds(const std::string& column) const {
    std::optional<Bounds> result;
    for (const auto& spec : specs_) {
        auto current = bounds_for(spec, column);
        if (current) result = result ? intersect(*result, *current) : current;
    }
    return result ? ColumnBounds{result->min, result->max} : ColumnBounds{};
}

std::vector<std::shared_ptr<Spec>> PushdownFilters::column_specs(const std::string& column) const {
    std::vector<std::shared_ptr<Spec>> result;
    for (const auto& spec : specs_)
        if (mentions(spec, column)) result.push_back(spec);
    return result;
}

std::shared_ptr<arrow::Array> PushdownFilters::or_column_values(const Spec& spec,
                                                                const std::string& column) const {
    return discrete_values(std::make_shared<Spec>(spec), column);
}

std::shared_ptr<arrow::Array> PushdownFilters::column_values(const std::string& column) const {
    for (const auto& spec : specs_)
        if (auto values = discrete_values(spec, column)) return values;
    return nullptr;
}

std::string PushdownFilters::format() const {
    if (specs_.empty()) return "(none)";
    std::string result;
    for (const auto& spec : specs_) {
        if (!result.empty()) result += " AND ";
        result += render(spec);
    }
    return result;
}

std::string PushdownFilters::format_repr() const {
    if (specs_.empty()) return "(none)";
    std::string result = "PushdownFilters([";
    for (size_t i = 0; i < specs_.size(); ++i) {
        if (i) result += ", ";
        result += render_repr(specs_[i]);
    }
    return result + "])";
}

std::shared_ptr<arrow::RecordBatch> PushdownFilters::apply(
    const std::shared_ptr<arrow::RecordBatch>& batch) const {
    if (!batch || specs_.empty()) return batch;
    auto surviving = batch;
    for (const auto& predicate : specs_) {
        if (predicate->kind == "runtime_filter") continue;
        try {
            auto mask = array_from_datum(evaluate(predicate, surviving), surviving->num_rows());
            if (mask->type_id() != arrow::Type::BOOL)
                throw std::runtime_error("predicate did not evaluate to BOOLEAN");
            arrow::compute::FilterOptions options(
                arrow::compute::FilterOptions::NullSelectionBehavior::DROP);
            auto filtered = arrow::compute::Filter(surviving, mask, options);
            if (!filtered.ok()) throw std::runtime_error(filtered.status().ToString());
            surviving = filtered.MoveValueUnsafe().record_batch();
        } catch (const std::exception&) {
            if (predicate->advisory) continue;
            throw;
        }
    }
    return surviving;
}

}  // namespace vgi

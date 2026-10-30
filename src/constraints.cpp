// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Canonical constraint records: the public builder, its decoder, and the
// serve-time checks that mirror the DuckDB extension's ATTACH contract.

#include "vgi/constraints.h"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <utility>

#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "wire.h"

namespace vgi {

namespace {

namespace gen = ::vgi::generated;

constexpr std::array<std::pair<ConstraintKind, const char*>, 5> kKinds{{
    {ConstraintKind::NotNull, "NOT_NULL"},
    {ConstraintKind::PrimaryKey, "PRIMARY_KEY"},
    {ConstraintKind::Unique, "UNIQUE"},
    {ConstraintKind::Check, "CHECK"},
    {ConstraintKind::ForeignKey, "FOREIGN_KEY"},
}};
constexpr std::array<std::pair<ConstraintEnforcement, const char*>, 2> kEnforcements{{
    {ConstraintEnforcement::Enforced, "ENFORCED"},
    {ConstraintEnforcement::MetadataOnly, "METADATA_ONLY"},
}};
constexpr std::array<std::pair<ConstraintValidation, const char*>, 3> kValidations{{
    {ConstraintValidation::AllRows, "ALL_ROWS"},
    {ConstraintValidation::FutureWrites, "FUTURE_WRITES"},
    {ConstraintValidation::Unvalidated, "UNVALIDATED"},
}};

template <class E, size_t N>
const char* spell(const std::array<std::pair<E, const char*>, N>& table, E value) {
    for (const auto& [key, name] : table)
        if (key == value) return name;
    throw std::invalid_argument("unknown constraint enum value");
}

template <class E, size_t N>
E parse(const std::array<std::pair<E, const char*>, N>& table, const std::string& value,
        const char* what) {
    for (const auto& [key, name] : table)
        if (value == name) return key;
    throw std::invalid_argument(std::string("unknown constraint ") + what + ": " + value);
}

std::string path_string(const ColumnPath& path) {
    std::string out;
    for (const auto& component : path) {
        if (!out.empty()) out += '.';
        out += component;
    }
    return out;
}

void require_name(const std::string& value, const char* what) {
    if (value.empty() || value.find('\0') != std::string::npos)
        throw std::invalid_argument(std::string(what) + " must be nonempty without NUL");
}

void require_paths(const std::vector<ColumnPath>& paths) {
    std::set<ColumnPath> seen;
    for (const auto& path : paths) {
        if (path.empty()) throw std::invalid_argument("constraint column path must be nonempty");
        for (const auto& component : path) require_name(component, "column path component");
        if (!seen.insert(path).second)
            throw std::invalid_argument("duplicate constraint column path: " + path_string(path));
    }
}

std::string object_ref(const char* kind, const SchemaPath& schema_path, const std::string& name,
                       const std::optional<std::string>& owner) {
    return wire::encode_ipc(wire::ResultBuilder(gen::CatalogObjectRefSchema())
                                .set_string("kind", kind)
                                .set_string_list("schema_path", schema_path)
                                .set_string("name", name)
                                .set_optional_string("owner_table", owner)
                                .finish());
}

std::shared_ptr<arrow::RecordBatch> decode_record(const std::string& bytes,
                                                  const std::shared_ptr<arrow::Schema>& schema,
                                                  const char* what) {
    std::shared_ptr<arrow::RecordBatch> batch;
    try {
        batch = wire::decode_ipc(bytes);
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("invalid ") + what + " record: " + error.what());
    }
    if (!batch || batch->num_rows() != 1 || !batch->schema()->Equals(*schema, false))
        throw std::invalid_argument(std::string("invalid ") + what + " record");
    return batch;
}

std::vector<ColumnPath> get_paths(const std::shared_ptr<arrow::RecordBatch>& batch,
                                  const std::string& field) {
    auto outer = std::static_pointer_cast<arrow::ListArray>(wire::column(batch, field));
    if (outer->IsNull(0)) throw std::invalid_argument("null " + field);
    auto inner = std::static_pointer_cast<arrow::ListArray>(outer->value_slice(0));
    std::vector<ColumnPath> paths;
    for (int64_t i = 0; i < inner->length(); ++i) {
        if (inner->IsNull(i)) throw std::invalid_argument("null element in " + field);
        auto components = std::static_pointer_cast<arrow::StringArray>(inner->value_slice(i));
        ColumnPath path;
        for (int64_t j = 0; j < components->length(); ++j) {
            if (components->IsNull(j)) throw std::invalid_argument("null element in " + field);
            path.push_back(components->GetString(j));
        }
        paths.push_back(std::move(path));
    }
    return paths;
}

// An ordinary STRUCT is one the extension descends into: not an extension or
// VGI logical type wearing a struct as its storage.
const arrow::FieldVector* ordinary_struct_children(const arrow::Field& field) {
    if (field.type()->id() != arrow::Type::STRUCT) return nullptr;
    if (const auto& metadata = field.metadata())
        for (const auto& key : metadata->keys())
            if (key == "ARROW:extension:name" || key.rfind("vgi.logical_type.", 0) == 0)
                return nullptr;
    return &field.type()->fields();
}

}  // namespace

const char* to_string(ConstraintKind kind) { return spell(kKinds, kind); }
const char* to_string(ConstraintEnforcement enforcement) {
    return spell(kEnforcements, enforcement);
}
const char* to_string(ConstraintValidation validation) { return spell(kValidations, validation); }

std::string constraint_record(const ConstraintRecordSpec& spec) {
    if (spec.schema_path.empty()) throw std::invalid_argument("constraint schema_path is empty");
    for (const auto& component : spec.schema_path) require_name(component, "schema path component");
    require_name(spec.table, "constraint table");
    require_name(spec.id, "constraint id");
    if (!spec.enforcement || !spec.validation)
        throw std::invalid_argument("constraint '" + spec.id +
                                    "': enforcement and validation must be stated explicitly");
    const bool check = spec.kind == ConstraintKind::Check;
    const bool foreign = spec.kind == ConstraintKind::ForeignKey;
    if (check != spec.check_sql.has_value())
        throw std::invalid_argument("check_sql is required for CHECK and only CHECK");
    if (check && spec.check_sql->empty()) throw std::invalid_argument("check_sql is empty");
    if (check ? !spec.columns.empty() : spec.columns.empty())
        throw std::invalid_argument(check ? std::string("CHECK takes no column paths")
                                          : std::string(to_string(spec.kind)) +
                                                " requires column paths");
    if (spec.kind == ConstraintKind::NotNull && spec.columns.size() != 1)
        throw std::invalid_argument("NOT_NULL requires exactly one column path");
    require_paths(spec.columns);
    if (foreign != spec.references.has_value())
        throw std::invalid_argument("references is required for FOREIGN_KEY and only FOREIGN_KEY");
    if (spec.declared_name) require_name(*spec.declared_name, "declared_name");

    std::optional<std::string> check_expression;
    if (check) {
        auto context = wire::ResultBuilder(gen::QueryEvaluationContextSchema())
                           .set_string("timezone", "UTC")
                           .set_string("collation", "binary")
                           .set_string("default_order", "asc")
                           .set_string("default_null_order", "nulls_last")
                           .set_bool("preserve_insertion_order", true)
                           .finish();
        auto context_struct =
            arrow::StructArray::Make(context->columns(), context->schema()->fields());
        if (!context_struct.ok())
            throw std::runtime_error(context_struct.status().ToString());
        check_expression =
            wire::encode_ipc(wire::ResultBuilder(gen::SqlExpressionSchema())
                                 .set_string("sql", *spec.check_sql)
                                 .set_string("profile", SCALAR_EXPRESSION_READ)
                                 .set_string_list("schema_path", spec.schema_path)
                                 .set_array("evaluation_context", context_struct.MoveValueUnsafe())
                                 .finish());
    }

    std::optional<std::string> foreign_key, referenced_constraint;
    if (foreign) {
        const auto& ref = *spec.references;
        if (ref.schema_path.empty())
            throw std::invalid_argument("foreign key schema_path is empty");
        require_name(ref.table, "foreign key referenced table");
        require_name(ref.constraint_id, "foreign key referenced constraint_id");
        require_paths(ref.columns);
        if (ref.columns.size() != spec.columns.size())
            throw std::invalid_argument("foreign key local and referenced arity must match");
        foreign_key = wire::encode_ipc(
            wire::ResultBuilder(gen::ForeignKeyDefinitionSchema())
                .set_binary("referenced_table",
                            object_ref("table", ref.schema_path, ref.table, std::nullopt))
                .set_string_list_list("referenced_column_paths", ref.columns)
                .set_enum("match", "SIMPLE")
                .set_enum("on_update", "NO_ACTION")
                .set_enum("on_delete", "NO_ACTION")
                .finish());
        referenced_constraint =
            object_ref("constraint", ref.schema_path, ref.constraint_id, ref.table);
    }

    auto definition = wire::ResultBuilder(gen::ConstraintDefinitionSchema());
    definition.set_enum("kind", to_string(spec.kind))
        .set_string_list_list("column_paths", spec.columns)
        .set_optional_string("declared_name", spec.declared_name);
    if (check_expression)
        definition.set_binary("check_expression", *check_expression);
    else
        definition.set_null("check_expression");
    if (foreign_key)
        definition.set_binary("foreign_key", *foreign_key);
    else
        definition.set_null("foreign_key");

    auto info = wire::ResultBuilder(gen::ConstraintInfoSchema());
    info.set_binary("ref", object_ref("constraint", spec.schema_path, spec.id, spec.table))
        .set_binary("definition", wire::encode_ipc(definition.finish()))
        .set_enum("enforcement", to_string(*spec.enforcement))
        .set_enum("validation", to_string(*spec.validation));
    if (referenced_constraint)
        info.set_binary("referenced_constraint", *referenced_constraint);
    else
        info.set_null("referenced_constraint");
    return wire::encode_ipc(info.finish());
}

ConstraintRecordView read_constraint_record(const std::string& bytes) {
    auto info = decode_record(bytes, gen::ConstraintInfoSchema(), "ConstraintInfo");
    auto ref = decode_record(wire::get_binary(info, "ref"), gen::CatalogObjectRefSchema(),
                             "CatalogObjectRef");
    if (wire::get_string(ref, "kind") != "constraint")
        throw std::invalid_argument("ConstraintInfo.ref is not a constraint reference");
    auto definition = decode_record(wire::get_binary(info, "definition"),
                                    gen::ConstraintDefinitionSchema(), "ConstraintDefinition");

    ConstraintRecordView view;
    view.schema_path = wire::get_string_list(ref, "schema_path");
    view.table = wire::get_optional_string(ref, "owner_table").value_or("");
    view.id = wire::get_string(ref, "name");
    view.kind = parse(kKinds, wire::get_enum(definition, "kind"), "kind");
    view.columns = get_paths(definition, "column_paths");
    view.enforcement =
        parse(kEnforcements, wire::get_enum(info, "enforcement"), "enforcement");
    view.validation = parse(kValidations, wire::get_enum(info, "validation"), "validation");
    view.declared_name = wire::get_optional_string(definition, "declared_name");
    if (auto check = wire::get_optional_binary(definition, "check_expression")) {
        auto expression = decode_record(*check, gen::SqlExpressionSchema(), "SqlExpression");
        view.check_sql = wire::get_string(expression, "sql");
    }
    if (auto foreign = wire::get_optional_binary(definition, "foreign_key")) {
        auto fk = decode_record(*foreign, gen::ForeignKeyDefinitionSchema(),
                                "ForeignKeyDefinition");
        auto target = decode_record(wire::get_binary(fk, "referenced_table"),
                                    gen::CatalogObjectRefSchema(), "CatalogObjectRef");
        ConstraintReference reference;
        reference.schema_path = wire::get_string_list(target, "schema_path");
        reference.table = wire::get_string(target, "name");
        reference.columns = get_paths(fk, "referenced_column_paths");
        if (auto key = wire::get_optional_binary(info, "referenced_constraint")) {
            auto target_key =
                decode_record(*key, gen::CatalogObjectRefSchema(), "CatalogObjectRef");
            reference.constraint_id = wire::get_string(target_key, "name");
        }
        view.references = std::move(reference);
    }
    return view;
}

void validate_constraint_records(const std::string& table, const arrow::Schema& columns,
                                 const std::vector<std::string>& constraints) {
    const std::string prefix = "Table '" + table + "': ";
    std::set<ColumnPath> required;
    for (const auto& bytes : constraints) {
        ConstraintRecordView record;
        try {
            record = read_constraint_record(bytes);
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument(prefix + error.what());
        }
        // DuckDB reads only facts that hold for every row; anything else fails ATTACH.
        if (record.enforcement != ConstraintEnforcement::Enforced ||
            record.validation != ConstraintValidation::AllRows)
            throw std::invalid_argument(prefix + "constraint '" + record.id + "' is " +
                                        to_string(record.enforcement) + "/" +
                                        to_string(record.validation) +
                                        "; DuckDB accepts only ENFORCED + ALL_ROWS records");
        if (record.kind == ConstraintKind::NotNull || record.kind == ConstraintKind::PrimaryKey)
            required.insert(record.columns.begin(), record.columns.end());
    }

    std::set<ColumnPath> seen;
    auto walk = [&](auto&& self, const arrow::FieldVector& fields, const ColumnPath& parent)
        -> void {
        for (const auto& field : fields) {
            // The extension excludes the synthetic row-id column from the rule.
            if (parent.empty() && field->metadata() && field->metadata()->Contains("is_row_id"))
                continue;
            auto path = parent;
            path.push_back(field->name());
            seen.insert(path);
            const bool named = required.count(path) != 0;
            if (field->nullable() == named) {
                throw std::invalid_argument(
                    field->nullable()
                        ? prefix + "column " + path_string(path) +
                              " is in a NOT_NULL/PRIMARY_KEY record but its Arrow field is "
                              "nullable"
                        : prefix + "column " + path_string(path) +
                              " is non-nullable in Arrow but no NOT_NULL/PRIMARY_KEY record "
                              "names it; declare one with vgi::constraint_record() or make the "
                              "field nullable");
            }
            if (const auto* children = ordinary_struct_children(*field))
                self(self, *children, path);
        }
    };
    walk(walk, columns.fields(), {});
    for (const auto& path : required)
        if (!seen.count(path))
            throw std::invalid_argument(prefix + "NOT_NULL/PRIMARY_KEY column " +
                                        path_string(path) +
                                        " is not a column (or an ordinary STRUCT member) of the "
                                        "table");
}

}  // namespace vgi

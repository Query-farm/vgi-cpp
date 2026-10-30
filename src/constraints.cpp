// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Canonical constraint records: the public builder, its decoder, and the
// serve-time checks that mirror the DuckDB extension's ATTACH contract.

#include "vgi/constraints.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <regex>
#include <set>
#include <stdexcept>
#include <utility>

#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

#include "vgi/generated/vgi_protocol_schemas.hpp"
#include "sql_scalar.h"
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

std::shared_ptr<arrow::RecordBatch> capability_record(
    const std::string& bytes, const std::shared_ptr<arrow::Schema>& schema) {
    if (bytes.empty() || bytes.size() > 8 * 1024 * 1024)
        throw std::invalid_argument("invalid catalog capability record size");
    auto source = std::make_shared<arrow::io::BufferReader>(arrow::Buffer::FromString(bytes));
    auto opened = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!opened.ok()) throw std::invalid_argument(opened.status().ToString());
    auto reader = opened.MoveValueUnsafe();
    std::shared_ptr<arrow::RecordBatch> batch;
    auto status = reader->ReadNext(&batch);
    if (!status.ok()) throw std::invalid_argument(status.ToString());
    if (!batch || batch->num_rows() != 1 || !batch->schema()->Equals(*schema, true) ||
        !batch->ValidateFull().ok())
        throw std::invalid_argument("invalid catalog capability record");
    for (int i = 0; i < batch->num_columns(); ++i)
        if (!schema->field(i)->nullable() && batch->column(i)->IsNull(0))
            throw std::invalid_argument("null required catalog capability field");
    const auto tail = bytes.substr(static_cast<size_t>(source->Tell().ValueOrDie()));
    if (!tail.empty() && tail != std::string(4, '\0') &&
        tail != std::string("\xff\xff\xff\xff\0\0\0\0", 8))
        throw std::invalid_argument("trailing catalog capability data");
    return batch;
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

const char* to_string(ConstraintKind kind) {
    return spell(kKinds, kind);
}
const char* to_string(ConstraintEnforcement enforcement) {
    return spell(kEnforcements, enforcement);
}
const char* to_string(ConstraintValidation validation) {
    return spell(kValidations, validation);
}

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
        if (!context_struct.ok()) throw std::runtime_error(context_struct.status().ToString());
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

namespace {

// Everything a record says, including what ConstraintRecordView leaves out but
// the extension still checks.
struct DecodedRecord {
    ConstraintRecordView view;
    // The referenced key's own CatalogObjectRef (FOREIGN_KEY only).
    SchemaPath key_schema_path;
    std::string key_owner;
    bool has_key = false;
    std::string match, on_update, on_delete;
    // The CHECK SqlExpression envelope.
    std::string check_profile;
    SchemaPath check_schema_path;
    bool default_context = false;
};

DecodedRecord decode_constraint_record(const std::string& bytes) {
    auto info = decode_record(bytes, gen::ConstraintInfoSchema(), "ConstraintInfo");
    auto ref = decode_record(wire::get_binary(info, "ref"), gen::CatalogObjectRefSchema(),
                             "CatalogObjectRef");
    if (wire::get_string(ref, "kind") != "constraint")
        throw std::invalid_argument("ConstraintInfo.ref is not a constraint reference");
    auto definition = decode_record(wire::get_binary(info, "definition"),
                                    gen::ConstraintDefinitionSchema(), "ConstraintDefinition");

    DecodedRecord out;
    auto& view = out.view;
    view.schema_path = wire::get_string_list(ref, "schema_path");
    view.table = wire::get_optional_string(ref, "owner_table").value_or("");
    view.id = wire::get_string(ref, "name");
    view.kind = parse(kKinds, wire::get_enum(definition, "kind"), "kind");
    view.columns = get_paths(definition, "column_paths");
    view.enforcement = parse(kEnforcements, wire::get_enum(info, "enforcement"), "enforcement");
    view.validation = parse(kValidations, wire::get_enum(info, "validation"), "validation");
    view.declared_name = wire::get_optional_string(definition, "declared_name");
    if (auto check = wire::get_optional_binary(definition, "check_expression")) {
        auto expression = decode_record(*check, gen::SqlExpressionSchema(), "SqlExpression");
        view.check_sql = wire::get_string(expression, "sql");
        out.check_profile = wire::get_string(expression, "profile");
        out.check_schema_path = wire::get_string_list(expression, "schema_path");
        auto context = std::static_pointer_cast<arrow::StructArray>(
            wire::column(expression, "evaluation_context"));
        auto text = [&](const char* name) {
            auto values =
                std::static_pointer_cast<arrow::StringArray>(context->GetFieldByName(name));
            return values && !values->IsNull(0) ? values->GetString(0) : std::string();
        };
        auto order = std::static_pointer_cast<arrow::BooleanArray>(
            context->GetFieldByName("preserve_insertion_order"));
        out.default_context = !context->IsNull(0) && text("timezone") == "UTC" &&
                              text("collation") == "binary" && text("default_order") == "asc" &&
                              text("default_null_order") == "nulls_last" && order &&
                              !order->IsNull(0) && order->Value(0);
    }
    if (auto foreign = wire::get_optional_binary(definition, "foreign_key")) {
        auto fk =
            decode_record(*foreign, gen::ForeignKeyDefinitionSchema(), "ForeignKeyDefinition");
        auto target = decode_record(wire::get_binary(fk, "referenced_table"),
                                    gen::CatalogObjectRefSchema(), "CatalogObjectRef");
        if (wire::get_string(target, "kind") != "table" ||
            wire::get_optional_string(target, "owner_table"))
            throw std::invalid_argument("foreign key referenced_table is not a table reference");
        ConstraintReference reference;
        reference.schema_path = wire::get_string_list(target, "schema_path");
        reference.table = wire::get_string(target, "name");
        reference.columns = get_paths(fk, "referenced_column_paths");
        out.match = wire::get_enum(fk, "match");
        out.on_update = wire::get_enum(fk, "on_update");
        out.on_delete = wire::get_enum(fk, "on_delete");
        if (auto key = wire::get_optional_binary(info, "referenced_constraint")) {
            auto target_key =
                decode_record(*key, gen::CatalogObjectRefSchema(), "CatalogObjectRef");
            if (wire::get_string(target_key, "kind") != "constraint")
                throw std::invalid_argument("referenced_constraint is not a constraint reference");
            reference.constraint_id = wire::get_string(target_key, "name");
            out.key_schema_path = wire::get_string_list(target_key, "schema_path");
            out.key_owner = wire::get_optional_string(target_key, "owner_table").value_or("");
            out.has_key = true;
        }
        view.references = std::move(reference);
    } else if (wire::get_optional_binary(info, "referenced_constraint")) {
        throw std::invalid_argument("referenced_constraint is set on a non-FOREIGN_KEY record");
    }
    return out;
}

bool row_id(const arrow::Field& field) {
    return field.metadata() && field.metadata()->Contains("is_row_id");
}

// The paths a record set makes non-nullable: every NOT_NULL and PRIMARY_KEY
// column path.
std::set<ColumnPath> required_paths(const std::vector<DecodedRecord>& records) {
    std::set<ColumnPath> required;
    for (const auto& record : records)
        if (record.view.kind == ConstraintKind::NotNull ||
            record.view.kind == ConstraintKind::PrimaryKey)
            required.insert(record.view.columns.begin(), record.view.columns.end());
    return required;
}

std::shared_ptr<arrow::Field> serve_field(const std::shared_ptr<arrow::Field>& field,
                                          const ColumnPath& path,
                                          const std::set<ColumnPath>& required) {
    auto served = field->WithNullable(required.count(path) == 0);
    if (const auto* children = ordinary_struct_children(*field)) {
        arrow::FieldVector rebuilt;
        for (const auto& child : *children) {
            auto child_path = path;
            child_path.push_back(child->name());
            rebuilt.push_back(serve_field(child, child_path, required));
        }
        served = served->WithType(arrow::struct_(rebuilt));
    }
    return served;
}

std::shared_ptr<arrow::Schema> serve_schema(const arrow::Schema& columns,
                                            const std::set<ColumnPath>& required) {
    arrow::FieldVector fields;
    for (const auto& field : columns.fields())
        fields.push_back(row_id(*field) ? field : serve_field(field, {field->name()}, required));
    return arrow::schema(fields, columns.metadata());
}

// The top-level field `name` names exactly, refusing the row-id column and a
// name two fields share.
const arrow::Field* top_level(const arrow::Schema& columns, const std::string& name) {
    const auto indices = columns.GetAllFieldIndices(name);
    if (indices.size() != 1 || row_id(*columns.field(indices[0]))) return nullptr;
    return columns.field(indices[0]).get();
}

const arrow::Field* nested(const arrow::Field& parent, const std::string& name) {
    const arrow::Field* found = nullptr;
    for (const auto& child : parent.type()->fields()) {
        if (child->name() != name) continue;
        if (found) return nullptr;
        found = child.get();
    }
    return found;
}

void validate_records(const std::optional<SchemaPath>& owner_schema, const std::string& owner,
                      const std::string& display, const arrow::Schema& raw_columns,
                      const std::vector<std::string>& constraints) {
    const std::string prefix = "Table '" + display + "': ";
    auto fail = [&](const std::string& message) { throw std::invalid_argument(prefix + message); };
    if (constraints.size() > 1024) fail("more than 1024 constraint records");
    size_t total = 0;
    std::vector<DecodedRecord> records;
    for (const auto& bytes : constraints) {
        total += bytes.size();
        if (bytes.empty() || total > 8 * 1024 * 1024)
            fail("constraint records must be nonempty and total at most 8 MiB");
        try {
            records.push_back(decode_constraint_record(bytes));
        } catch (const std::invalid_argument& error) {
            fail(error.what());
        }
    }
    const auto columns = serve_schema(raw_columns, required_paths(records));

    std::set<std::string> ids, labels;
    bool primary = false;
    for (const auto& decoded : records) {
        const auto& record = decoded.view;
        const std::string where = "constraint '" + record.id + "' ";
        auto bad = [&](const std::string& message) { fail(where + message); };
        auto identifier = [&](const std::string& value, const char* what) {
            if (value.empty() || value.size() > 4096 || value.find('\0') != std::string::npos)
                bad(std::string("has an invalid ") + what);
        };
        identifier(record.id, "id");
        if (record.schema_path.size() != 1)
            bad("must name a one-component schema path; DuckDB reads no other");
        identifier(record.schema_path[0], "schema name");
        identifier(record.table, "owner table");
        if (owner_schema && (record.schema_path != *owner_schema || record.table != owner))
            bad("belongs to " + path_string(record.schema_path) + "." + record.table +
                ", not this table");
        if (!ids.insert(record.id).second) bad("duplicates another constraint id");
        if (record.declared_name) {
            identifier(*record.declared_name, "declared name");
            std::string folded = *record.declared_name;
            for (auto& c : folded)
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (!labels.insert(folded).second)
                bad("reuses declared name '" + *record.declared_name +
                    "' (names are compared case-insensitively)");
        }
        // DuckDB reads only facts that hold for every row; anything else fails ATTACH.
        if (record.enforcement != ConstraintEnforcement::Enforced ||
            record.validation != ConstraintValidation::AllRows)
            throw std::invalid_argument(prefix + "constraint '" + record.id + "' is " +
                                        to_string(record.enforcement) + "/" +
                                        to_string(record.validation) +
                                        "; DuckDB accepts only ENFORCED + ALL_ROWS records");
        if (record.columns.size() > 1024) bad("has more than 1024 column paths");
        for (const auto& path : record.columns) {
            if (path.empty() || path.size() > 64) bad("has an empty or oversized column path");
            for (const auto& component : path) identifier(component, "column path component");
        }
        switch (record.kind) {
            case ConstraintKind::NotNull: {
                if (record.columns.size() != 1) bad("NOT_NULL requires exactly one column path");
                const auto& path = record.columns[0];
                const arrow::Field* field = top_level(*columns, path[0]);
                for (size_t depth = 1; field && depth < path.size(); ++depth)
                    field =
                        ordinary_struct_children(*field) ? nested(*field, path[depth]) : nullptr;
                if (!field)
                    bad("NOT_NULL column " + path_string(path) +
                        " is not a column (or an ordinary STRUCT member) of the table");
                break;
            }
            case ConstraintKind::PrimaryKey:
            case ConstraintKind::Unique:
            case ConstraintKind::ForeignKey: {
                for (const auto& path : record.columns) {
                    if (path.size() != 1)
                        bad(std::string(to_string(record.kind)) + " column " + path_string(path) +
                            " is nested; DuckDB reads nested paths only for NOT_NULL");
                    if (!top_level(*columns, path[0]))
                        bad(std::string(to_string(record.kind)) + " column " + path[0] +
                            " is not a column of the table");
                }
                if (record.kind == ConstraintKind::PrimaryKey) {
                    if (primary) bad("is a second PRIMARY_KEY; a table has at most one");
                    primary = true;
                }
                if (record.kind != ConstraintKind::ForeignKey) break;
                const auto& ref = *record.references;
                if (ref.schema_path != record.schema_path || !decoded.has_key ||
                    decoded.key_schema_path != ref.schema_path || decoded.key_owner != ref.table)
                    bad("must reference an identified key of a table in the same schema");
                identifier(ref.table, "referenced table");
                identifier(ref.constraint_id, "referenced constraint id");
                if (ref.columns.size() != record.columns.size())
                    bad("local and referenced column counts differ");
                for (const auto& path : ref.columns)
                    if (path.size() != 1) bad("referenced column paths must be top-level");
                if (decoded.match != "SIMPLE") bad("must be MATCH SIMPLE");
                for (const auto* action : {&decoded.on_update, &decoded.on_delete})
                    if (*action != "NO_ACTION" && *action != "RESTRICT")
                        bad("uses referential action " + *action +
                            "; DuckDB reads only NO_ACTION and RESTRICT");
                break;
            }
            case ConstraintKind::Check: {
                if (decoded.check_profile != SCALAR_EXPRESSION_READ)
                    bad("CHECK expression profile must be " + std::string(SCALAR_EXPRESSION_READ));
                if (decoded.check_schema_path != record.schema_path)
                    bad("CHECK expression belongs to another schema");
                if (!decoded.default_context)
                    bad("CHECK requires the default evaluation context (UTC, binary, asc, "
                        "nulls_last, preserve_insertion_order)");
                try {
                    sql::validate_numeric_check(*record.check_sql, *columns);
                } catch (const std::invalid_argument& error) {
                    bad("CHECK (" + *record.check_sql + "): " + error.what());
                }
                break;
            }
        }
    }

    // Holds by construction after serve_schema; kept as the extension's own rule.
    const auto required = required_paths(records);
    auto walk = [&](auto&& self, const arrow::FieldVector& fields,
                    const ColumnPath& parent) -> void {
        for (const auto& field : fields) {
            if (parent.empty() && row_id(*field)) continue;
            auto path = parent;
            path.push_back(field->name());
            if (field->nullable() != (required.count(path) == 0))
                fail("column " + path_string(path) +
                     " nullability disagrees with its NOT_NULL/PRIMARY_KEY records");
            if (const auto* children = ordinary_struct_children(*field))
                self(self, *children, path);
        }
    };
    walk(walk, columns->fields(), {});
}

void validate_profiles(const std::vector<std::string>& profiles) {
    static const std::regex identifier("^[a-z][a-z0-9_]*(\\.[a-z][a-z0-9_]*)*\\.v[1-9][0-9]*$");
    std::set<std::string> seen;
    if (profiles.size() > 256) throw std::invalid_argument("too many catalog capability profiles");
    for (const auto& profile : profiles)
        if (profile.size() > 256 || !std::regex_match(profile, identifier) ||
            !seen.insert(profile).second)
            throw std::invalid_argument("invalid or duplicate catalog capability profile: " +
                                        profile);
}

}  // namespace

ConstraintRecordView read_constraint_record(const std::string& bytes) {
    return decode_constraint_record(bytes).view;
}

std::shared_ptr<arrow::Schema> served_columns(const arrow::Schema& columns,
                                              const std::vector<std::string>& constraints) {
    std::vector<DecodedRecord> records;
    for (const auto& bytes : constraints) records.push_back(decode_constraint_record(bytes));
    return serve_schema(columns, required_paths(records));
}

void validate_constraint_records(const SchemaPath& schema_path, const std::string& table,
                                 const arrow::Schema& columns,
                                 const std::vector<std::string>& constraints) {
    validate_records(schema_path, table, path_string(schema_path) + "." + table, columns,
                     constraints);
}

void validate_constraint_records(const std::string& table, const arrow::Schema& columns,
                                 const std::vector<std::string>& constraints) {
    validate_records(std::nullopt, "", table, columns, constraints);
}

std::string encode_catalog_capabilities(const CatalogCapabilityProfiles& profiles) {
    for (const auto* list : {&profiles.read, &profiles.ddl, &profiles.write, &profiles.transaction})
        validate_profiles(*list);
    return wire::encode_ipc(wire::ResultBuilder(gen::CatalogCapabilitiesSchema())
                                .set_string_list("read", profiles.read)
                                .set_string_list("ddl", profiles.ddl)
                                .set_string_list("write", profiles.write)
                                .set_string_list("transaction", profiles.transaction)
                                .finish());
}

CatalogCapabilityProfiles decode_catalog_capabilities(const std::string& bytes) {
    auto record = capability_record(bytes, gen::CatalogCapabilitiesSchema());
    CatalogCapabilityProfiles out;
    for (auto [category, list] :
         {std::pair{"read", &out.read}, std::pair{"ddl", &out.ddl}, std::pair{"write", &out.write},
          std::pair{"transaction", &out.transaction}}) {
        auto values = std::static_pointer_cast<arrow::ListArray>(wire::column(record, category));
        if (values->IsNull(0) || values->value_slice(0)->null_count() != 0)
            throw std::invalid_argument("null catalog capability profile");
        *list = wire::get_string_list(record, category);
        validate_profiles(*list);
    }
    return out;
}

CatalogCapabilityProfiles client_catalog_offer(
    const std::optional<std::string>& client_capabilities) {
    if (!client_capabilities) return {};
    auto client = capability_record(*client_capabilities, gen::ClientCapabilitiesSchema());
    auto catalog = wire::get_optional_binary(client, "catalog");
    return catalog ? decode_catalog_capabilities(*catalog) : CatalogCapabilityProfiles{};
}

CatalogCapabilityProfiles select_catalog_capabilities(const CatalogCapabilityProfiles& available,
                                                      const CatalogCapabilityProfiles& offered) {
    auto intersect = [](const std::vector<std::string>& mine,
                        const std::vector<std::string>& theirs) {
        std::vector<std::string> out;
        for (const auto& profile : mine)
            if (std::find(theirs.begin(), theirs.end(), profile) != theirs.end())
                out.push_back(profile);
        return out;
    };
    auto drop = [](std::vector<std::string>& list, const char* profile) {
        list.erase(std::remove(list.begin(), list.end(), profile), list.end());
    };
    auto has = [](const std::vector<std::string>& list, const char* profile) {
        return std::find(list.begin(), list.end(), profile) != list.end();
    };
    CatalogCapabilityProfiles out{intersect(available.read, offered.read),
                                  intersect(available.ddl, offered.ddl),
                                  intersect(available.write, offered.write),
                                  intersect(available.transaction, offered.transaction)};
    // These describe or act on canonical constraints, so none outlives the
    // records profile, and CREATE needs an authority that enforces.
    if (!has(out.read, CONSTRAINT_RECORDS_READ)) {
        drop(out.read, SCALAR_EXPRESSION_READ);
        drop(out.ddl, CONSTRAINT_CREATE);
        drop(out.write, CONSTRAINT_ENFORCE);
    }
    if (!has(out.write, CONSTRAINT_ENFORCE)) drop(out.ddl, CONSTRAINT_CREATE);
    return out;
}

}  // namespace vgi

// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "constraint_fixtures.h"

#include <set>
#include <string_view>
#include <variant>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <arrow/array/builder_primitive.h>

namespace example::constraint_fixtures {
namespace {
template <class Builder, class T>
std::shared_ptr<arrow::Array> values(std::initializer_list<T> input) {
    Builder builder;
    for (const auto& value : input) {
        auto status = builder.Append(value);
        if (!status.ok()) throw std::runtime_error(status.ToString());
    }
    return builder.Finish().ValueOrDie();
}

std::shared_ptr<arrow::Array> integers(std::initializer_list<int64_t> input) {
    return values<arrow::Int64Builder>(input);
}
std::shared_ptr<arrow::Array> strings(std::initializer_list<const char*> input) {
    return values<arrow::StringBuilder>(input);
}

const arrow::Array& column(const Columns& columns, std::string_view name) {
    for (const auto& [key, array] : columns)
        if (key == name) return *array;
    throw std::invalid_argument("constraint fixture missing column: " + std::string(name));
}

// The fixtures key only on top-level columns.
std::vector<std::string> column_names(const std::vector<vgi::ColumnPath>& paths) {
    std::vector<std::string> names;
    for (const auto& path : paths) {
        if (path.size() != 1) throw std::invalid_argument("fixture needs top-level key");
        names.push_back(path.front());
    }
    return names;
}

using Key = std::vector<std::variant<int64_t, std::string>>;
std::optional<Key> key(const Columns& columns, const std::vector<std::string>& names, int64_t row) {
    Key result;
    for (const auto& name : names) {
        const auto& array = column(columns, name);
        if (array.IsNull(row)) return std::nullopt;
        if (array.type_id() == arrow::Type::INT64)
            result.emplace_back(static_cast<const arrow::Int64Array&>(array).Value(row));
        else if (array.type_id() == arrow::Type::STRING)
            result.emplace_back(static_cast<const arrow::StringArray&>(array).GetString(row));
        else
            throw std::invalid_argument("unsupported fixture key type");
    }
    return result;
}
}  // namespace

Columns rows(const std::string& table, int64_t version) {
    if (table == "departments")
        return {{"id", integers({1, 2, 3})},
                {"name", strings({"Engineering", "Sales", "HR"})},
                {"budget", values<arrow::DoubleBuilder>({500000.0, 300000.0, 200000.0})}};
    if (table == "products")
        return {{"id", integers({1, 2, 3})},
                {"name", strings({"Widget", "Gadget", "Doohickey"})},
                {"quantity", integers({100, 50, 200})},
                {"price", values<arrow::DoubleBuilder>({9.99, 24.99, 4.99})}};
    if (table == "employees")
        return {{"id", integers({1, 2, 3, 4, 5})},
                {"name", strings({"Alice", "Bob", "Carol", "Dave", "Eve"})},
                {"email", strings({"alice@co.com", "bob@co.com", "carol@co.com", "dave@co.com",
                                   "eve@co.com"})},
                {"department_id", integers({1, 1, 2, 2, 3})}};
    if (table == "projects")
        return {{"department_id", integers({1, 1, 2})},
                {"project_code", strings({"P001", "P002", "P003"})},
                {"title", strings({"Backend API", "Frontend UI", "Sales Portal"})}};
    if (table == "versioned_constraints" && version >= 1 && version <= 3) {
        if (version == 1) return {{"id", integers({1, 2})}, {"name", strings({"Alice", "Bob"})}};
        Columns result = {{"id", integers({1, 2, 3})},
                          {"name", strings({"Alice", "Bob", "Carol"})},
                          {"email", strings({"a@co", "b@co", "c@co"})}};
        if (version == 3) result.emplace_back("department_id", integers({1, 2, 1}));
        return result;
    }
    throw std::invalid_argument("unknown immutable constraint fixture/version");
}

void validate_rows(const std::string& table, const Columns& columns, int64_t version) {
    if (columns.empty() || !columns.front().second)
        throw std::invalid_argument("empty constraint fixture schema");
    const auto size = columns.front().second->length();
    std::set<std::string> seen;
    for (const auto& [name, array] : columns) {
        if (!array || array->length() != size || !seen.insert(name).second)
            throw std::invalid_argument("invalid constraint fixture columns");
        auto status = array->ValidateFull();
        if (!status.ok()) throw std::invalid_argument(status.ToString());
    }
    // Validate every published definition, refusing a new unsupported kind
    // instead of silently trusting it. Keys use typed values, not SQL text.
    for (const auto& bytes : definitions(table, version)) {
        const auto record = vgi::read_constraint_record(bytes);
        const auto kind = record.kind;
        const std::string kind_name = vgi::to_string(kind);
        const auto names = record.kind == vgi::ConstraintKind::Check ? std::vector<std::string>{}
                                                                     : column_names(record.columns);
        if (kind == vgi::ConstraintKind::NotNull) {
            for (const auto& name : names)
                if (column(columns, name).null_count() != 0)
                    throw std::invalid_argument("fixture NOT NULL violated: " + name);
        } else if (kind == vgi::ConstraintKind::PrimaryKey || kind == vgi::ConstraintKind::Unique) {
            std::set<Key> keys;
            for (int64_t row = 0; row < size; ++row) {
                auto value = key(columns, names, row);
                if (!value && kind == vgi::ConstraintKind::PrimaryKey)
                    throw std::invalid_argument("fixture PRIMARY KEY contains NULL");
                if (value && !keys.insert(*value).second)
                    throw std::invalid_argument("fixture " + kind_name + " contains duplicate key");
            }
        } else if (kind == vgi::ConstraintKind::ForeignKey) {
            const auto& target = record.references;
            if (!target || target->table != "departments" ||
                target->schema_path != std::vector<std::string>{"data"})
                throw std::invalid_argument("unsupported fixture foreign key");
            const auto referenced = rows("departments");
            validate_rows("departments", referenced);
            const auto target_names = column_names(target->columns);
            if (target_names != std::vector<std::string>{"id"} || names.size() != 1)
                throw std::invalid_argument("unsupported fixture referenced key");
            std::set<Key> keys;
            for (int64_t row = 0; row < referenced.front().second->length(); ++row)
                keys.insert(*key(referenced, target_names, row));
            for (int64_t row = 0; row < size; ++row) {
                auto value = key(columns, names, row);
                if (value && !keys.contains(*value))
                    throw std::invalid_argument("fixture FOREIGN KEY missing referenced row");
            }
        } else if (kind == vgi::ConstraintKind::Check) {
            if (record.check_sql != kBudgetCheck)
                throw std::invalid_argument("unsupported fixture CHECK expression");
            const auto& array = column(columns, "budget");
            if (array.type_id() != arrow::Type::DOUBLE)
                throw std::invalid_argument("unexpected fixture budget type");
            for (int64_t row = 0; row < size; ++row)
                if (!array.IsNull(row) &&
                    !(static_cast<const arrow::DoubleArray&>(array).Value(row) >= 0))
                    throw std::invalid_argument("fixture CHECK budget >= 0 violated");
        } else {
            throw std::invalid_argument("unsupported fixture constraint kind: " + kind_name);
        }
    }
}

std::shared_ptr<arrow::Schema> required_schema(const std::shared_ptr<arrow::Schema>& schema,
                                               const std::string& table, int64_t version) {
    std::set<std::string> required;
    for (const auto& bytes : definitions(table, version)) {
        const auto record = vgi::read_constraint_record(bytes);
        if (record.kind == vgi::ConstraintKind::NotNull ||
            record.kind == vgi::ConstraintKind::PrimaryKey)
            for (const auto& name : column_names(record.columns)) required.insert(name);
    }
    auto fields = schema->fields();
    for (auto& field : fields)
        if (required.erase(field->name())) field = field->WithNullable(false);
    if (!required.empty())
        throw std::invalid_argument("fixture required column missing from schema");
    return arrow::schema(std::move(fields), schema->metadata());
}
}  // namespace example::constraint_fixtures

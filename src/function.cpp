// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/function.h"

#include <stdexcept>

#include <arrow/type.h>

namespace vgi {

ArgSpec ArgSpec::column(std::string name, int index, std::string type, std::string description) {
    ArgSpec s;
    s.name = std::move(name);
    s.index = index;
    s.type = std::move(type);
    s.description = std::move(description);
    return s;
}

ArgSpec ArgSpec::constant_arg(std::string name, int index, std::string type,
                              std::string description) {
    ArgSpec s = column(std::move(name), index, std::move(type), std::move(description));
    s.constant = true;
    return s;
}

ArgSpec& ArgSpec::with_varargs() {
    varargs = true;
    return *this;
}

ArgSpec ArgSpec::constant_typed(std::string name, int index, std::shared_ptr<arrow::DataType> type,
                                std::string description) {
    ArgSpec spec = constant_arg(std::move(name), index, "", std::move(description));
    spec.arrow_type = std::move(type);
    return spec;
}

ArgSpec ArgSpec::table(std::string name, int index, std::string description) {
    return column(std::move(name), index, "table", std::move(description));
}

ArgSpec ArgSpec::column_typed(std::string name, int index, std::shared_ptr<arrow::DataType> type,
                              std::string description) {
    ArgSpec spec = column(std::move(name), index, "", std::move(description));
    spec.arrow_type = std::move(type);
    return spec;
}

ArgSpec ArgSpec::any_column(std::string name, int index, std::string description) {
    return column(std::move(name), index, "any", std::move(description));
}

ArgSpec& ArgSpec::with_range(std::optional<double> low, std::optional<double> high) {
    // Inclusive, which is what every fixture that declares a range wants. The
    // exclusive fields stay available for the cases that need them.
    ge = low;
    le = high;
    return *this;
}

ArgSpec& ArgSpec::with_bound(TypeBound bound) {
    type_bound = std::move(bound);
    return *this;
}

ArgSpec ArgSpec::named(std::string name, std::string type, std::string description) {
    ArgSpec s;
    s.name = std::move(name);
    s.type = std::move(type);
    s.description = std::move(description);
    s.required = false;
    return s;
}

std::shared_ptr<arrow::DataType> BindParams::input_type(size_t index) const {
    if (auto type = arguments.positional_type(index)) {
        // A polymorphic parameter is advertised as null and resolved by the
        // engine to the call site's real type, which arrives in the input
        // schema rather than here.
        if (type->id() != arrow::Type::NA) return type;
    }
    if (input_schema && static_cast<int>(index) < input_schema->num_fields()) {
        return input_schema->field(static_cast<int>(index))->type();
    }
    return nullptr;
}

std::shared_ptr<arrow::Schema> ScalarFunction::bind(const BindParams&) const {
    auto md = metadata();
    if (!md.return_type) {
        // A function that leaves return_type empty is declaring that its
        // result depends on its arguments, which means it owes an override.
        throw std::runtime_error("scalar function '" + name() +
                                 "' declares no fixed return_type and does not override bind()");
    }
    // The engine names the single output column "result"; see the canonical
    // Python worker's scalar path.
    return arrow::schema({arrow::field("result", md.return_type, /*nullable=*/true)});
}

}  // namespace vgi

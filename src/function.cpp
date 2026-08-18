// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/function.h"

#include <stdexcept>

#include <arrow/type.h>

namespace vgi {

ArgSpec ArgSpec::column(std::string name, int index, std::string type,
                        std::string description) {
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

ArgSpec ArgSpec::named(std::string name, std::string type, std::string description) {
    ArgSpec s;
    s.name = std::move(name);
    s.type = std::move(type);
    s.description = std::move(description);
    s.required = false;
    return s;
}

std::shared_ptr<arrow::Schema> ScalarFunction::bind(const BindParams&) const {
    auto md = metadata();
    if (!md.return_type) {
        // A function that leaves return_type empty is declaring that its
        // result depends on its arguments, which means it owes an override.
        throw std::runtime_error(
            "scalar function '" + name() +
            "' declares no fixed return_type and does not override bind()");
    }
    // The engine names the single output column "result"; see the canonical
    // Python worker's scalar path.
    return arrow::schema({arrow::field("result", md.return_type, /*nullable=*/true)});
}

}  // namespace vgi

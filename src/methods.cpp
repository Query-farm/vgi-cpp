// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "methods.h"

#include <arrow/type.h>

#include <stdexcept>
#include <variant>

#include "generated/vgi_service.hpp"

namespace vgi {
namespace {

namespace gen = ::vgi::generated;

MethodKind kind_of(const gen::VgiMethod& method) {
    if (std::holds_alternative<gen::VgiMethod::Stream>(method.handler)) return MethodKind::Stream;
    if (std::holds_alternative<gen::VgiMethod::Void>(method.handler)) return MethodKind::Void;
    return method.payload ? MethodKind::Result : MethodKind::Binary;
}

}  // namespace

const std::shared_ptr<arrow::Schema>& envelope_schema() {
    return gen::OptionalResultEnvelopeSchema();
}

const std::shared_ptr<arrow::Schema>& payload_schema_of(const std::string& method) {
    static const std::shared_ptr<arrow::Schema> none;
    for (const auto& spec : protocol_methods()) {
        if (spec.name == method) return spec.payload ? spec.payload : none;
    }
    throw std::runtime_error("no such protocol method: " + method);
}

const std::vector<MethodSpec>& protocol_methods() {
    static const std::vector<MethodSpec> methods = [] {
        std::vector<MethodSpec> out;
        out.reserve(gen::VGI_METHODS.size());
        for (const auto& method : gen::VGI_METHODS) {
            out.push_back({std::string(method.name), kind_of(method), method.params(),
                           method.payload ? method.payload() : nullptr,
                           method.result && method.result()->field(0)->nullable()});
        }
        return out;
    }();
    return methods;
}

}  // namespace vgi

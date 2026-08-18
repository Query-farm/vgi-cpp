// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/copy_to.h"

#include <stdexcept>

#include "dispatcher.h"

namespace vgi {
namespace {

// Exposes a CopyToFunction as a TableBufferingFunction, so the whole sink
// path is reused rather than duplicated.
//
// There is no source phase: `combine` returns an empty finalize list, and
// `finalize_producer` is therefore never called — the close happens in
// `combine`, which is the last call the engine makes.
class CopyToSink : public TableBufferingFunction {
public:
    explicit CopyToSink(std::shared_ptr<CopyToFunction> writer) : writer_(std::move(writer)) {}

    std::string name() const override { return writer_->handler_name(); }
    FunctionMetadata metadata() const override { return writer_->metadata(); }
    std::vector<ArgSpec> argument_specs() const override { return writer_->argument_specs(); }

    std::shared_ptr<arrow::Schema> bind(const BindParams& params) const override {
        // A COPY writer produces no rows, so its "output" is the input it is
        // given. Calling it directly as a table function is a user error worth
        // naming, since the failure would otherwise be an empty result.
        if (!params.input_schema) {
            throw std::invalid_argument(
                writer_->handler_name() + " is a COPY TO format writer; invoke it via COPY " +
                "<source> TO '<path>' (FORMAT " + writer_->format() + "), not directly.");
        }
        return params.input_schema;
    }

    std::string process(const ProcessParams& params,
                        const std::shared_ptr<arrow::RecordBatch>& batch) override {
        if (batch && batch->num_rows() > 0) writer_->write(params, batch);
        return params.execution_id;
    }

    std::vector<std::string> combine(const ProcessParams& params,
                                     const std::vector<std::string>&) override {
        // Closed here, and nothing is returned to finalize: the destination is
        // complete once the last shard has been written.
        writer_->close(params);
        return {};
    }

    std::unique_ptr<TableProducer> finalize_producer(const ProcessParams&,
                                                     const std::string&) override {
        // Unreachable — combine returns no finalize ids.
        return nullptr;
    }

private:
    std::shared_ptr<CopyToFunction> writer_;
};

}  // namespace

void Dispatcher::register_copy_to(std::shared_ptr<CopyToFunction> writer) {
    if (!writer) throw std::invalid_argument("register_copy_to: null writer");
    copy_to_.push_back(writer);
    register_buffering(std::make_shared<CopyToSink>(std::move(writer)));
}

}  // namespace vgi

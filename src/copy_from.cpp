// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

#include "vgi/copy_from.h"

#include <stdexcept>
#include <utility>

#include "dispatcher.h"

namespace vgi {
namespace {

// Drains the batches one read() materialized. Separate from the function
// because the function is shared and immutable while a scan has a position.
class CopyFromProducer : public TableProducer {
public:
    explicit CopyFromProducer(std::vector<std::shared_ptr<arrow::RecordBatch>> batches)
        : batches_(std::move(batches)) {}

    std::shared_ptr<arrow::RecordBatch> next_batch() override {
        if (next_ >= batches_.size()) return nullptr;
        return batches_[next_++];
    }

private:
    std::vector<std::shared_ptr<arrow::RecordBatch>> batches_;
    size_t next_ = 0;
};

// Exposes a CopyFromFunction as a TableFunction, so the whole bind/init/scan
// path is reused rather than duplicated.
class CopyFromScan : public TableFunction {
public:
    explicit CopyFromScan(std::shared_ptr<CopyFromFunction> reader)
        : reader_(std::move(reader)) {}

    std::string name() const override { return reader_->handler_name(); }
    FunctionMetadata metadata() const override { return reader_->metadata(); }
    std::vector<ArgSpec> argument_specs() const override { return reader_->argument_specs(); }

    std::shared_ptr<arrow::Schema> bind(const BindParams& params) const override {
        // DuckDB forces the scan's columns to the COPY target's, so the only
        // schema this may answer with is the one it was handed.
        if (!params.copy_from_schema) throw std::invalid_argument(misuse_message());
        return params.copy_from_schema;
    }

    std::vector<SecretLookup> secret_lookups(const BindParams& params) const override {
        // Forwarded, because only the reader knows to scope its request to the
        // COPY source path.
        return reader_->secret_lookups(params);
    }

    std::unique_ptr<TableProducer> init(const ProcessParams& params) const override {
        if (!params.copy_from_path) throw std::invalid_argument(misuse_message());
        return std::make_unique<CopyFromProducer>(reader_->read(params));
    }

private:
    // Calling the handler directly as a table function is a user error worth
    // naming, since the failure would otherwise be an empty result.
    std::string misuse_message() const {
        return reader_->handler_name() + " is a COPY FROM format reader; invoke it via COPY " +
               "<table> FROM '<path>' (FORMAT " + reader_->format() + "), not directly.";
    }

    std::shared_ptr<CopyFromFunction> reader_;
};

}  // namespace

void Dispatcher::register_copy_from(std::shared_ptr<CopyFromFunction> reader) {
    if (!reader) throw std::invalid_argument("register_copy_from: null reader");
    copy_from_.push_back(reader);
    register_table(std::make_shared<CopyFromScan>(std::move(reader)));
}

}  // namespace vgi

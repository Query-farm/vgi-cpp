// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// `ticket_probe`: the cross-SDK fixture catalog for attach tickets.
//
// Every SDK's fixture worker serves it identically (vgi-python
// docs/protocol/vgi-attach-tickets.md §7), and the extension's
// `attach_ticket/*.test` run against each:
//
//   - catalog `ticket_probe`, default schema `main`;
//   - attach options, in order: `region` VARCHAR (default 'us-east-1') and
//     `api_key` VARCHAR, required and secret;
//   - table `main.probe`, backed by the table function `main.ticket_probe` (no
//     arguments): one row, `region` and `api_key_sha256` -- the first 12
//     lowercase hex of SHA-256(UTF-8(api_key)). The key itself never leaves.
//
// So a reattach with nothing but `vgi_attach_ticket` reading the same row
// proves the secret took effect without travelling again, and one without the
// ticket fails for want of the required `api_key`.

#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>
#include <vgi_rpc/crypto.h>

#include <vgi/worker.h>

#include "registry.h"

namespace example {
namespace {

constexpr const char* kCatalog = "ticket_probe";
constexpr const char* kDefaultRegion = "us-east-1";

std::shared_ptr<arrow::Array> one_string(const std::string& value) {
    arrow::StringBuilder builder;
    (void)builder.Append(value);
    std::shared_ptr<arrow::Array> array;
    (void)builder.Finish(&array);
    return array;
}

std::shared_ptr<arrow::Schema> probe_columns() {
    return arrow::schema({arrow::field("region", arrow::utf8(), /*nullable=*/true),
                          arrow::field("api_key_sha256", arrow::utf8(), /*nullable=*/true)});
}

std::string option_text(const std::shared_ptr<arrow::RecordBatch>& options,
                        const std::string& name) {
    if (!options) return {};
    const auto column = options->GetColumnByName(name);
    if (!column || column->IsNull(0) || column->type_id() != arrow::Type::STRING) return {};
    return std::static_pointer_cast<arrow::StringArray>(column)->GetString(0);
}

class TicketProbe : public vgi::TableFunction {
public:
    std::string name() const override { return "ticket_probe"; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            "Report the attach options of this ticket_probe attach (the api_key only as a digest)";
        md.categories = {"generator", "testing"};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override { return {}; }

    std::shared_ptr<arrow::Schema> bind(const vgi::BindParams&) const override {
        return probe_columns();
    }

    vgi::TableCardinality cardinality(const vgi::ProcessParams&) const override { return {1, 1}; }

    std::unique_ptr<vgi::TableProducer> init(const vgi::ProcessParams& params) const override {
        if (!params.attach_options) {
            throw std::runtime_error(
                "ticket_probe must be read through an attach of the ticket_probe catalog");
        }
        auto region = option_text(params.attach_options, "region");
        if (region.empty()) region = kDefaultRegion;
        const auto api_key = option_text(params.attach_options, "api_key");
        const auto digest = vgi_rpc::crypto::sha256(
            reinterpret_cast<const uint8_t*>(api_key.data()), api_key.size());
        const auto hex = vgi_rpc::crypto::hex_encode(digest.data(), digest.size()).substr(0, 12);

        // Built in the bound order, since the engine may have narrowed it.
        std::vector<std::shared_ptr<arrow::Array>> columns;
        for (const auto& field : params.output_schema->fields()) {
            columns.push_back(one_string(field->name() == "region" ? region : hex));
        }
        return std::make_unique<OneRow>(arrow::RecordBatch::Make(params.output_schema, 1, columns));
    }

private:
    class OneRow : public vgi::TableProducer {
    public:
        explicit OneRow(std::shared_ptr<arrow::RecordBatch> batch) : batch_(std::move(batch)) {}

        std::shared_ptr<arrow::RecordBatch> next_batch() override {
            auto batch = batch_;
            batch_ = nullptr;
            return batch;
        }

    private:
        std::shared_ptr<arrow::RecordBatch> batch_;
    };
};

}  // namespace

void register_ticket_probe(vgi::Worker& worker) {
    auto& model = worker.catalog(kCatalog);
    model.comment = "Attach-ticket probe: one plain and one secret attach option";
    vgi::AttachOptionSpec region{"region", "Region the probe reports back", arrow::utf8(),
                                 one_string(kDefaultRegion), /*required=*/false};
    vgi::AttachOptionSpec api_key{"api_key", "API key; only its digest is ever returned",
                                  arrow::utf8(), nullptr, /*required=*/true};
    api_key.secret = true;
    model.attach_options = {region, api_key};
    worker.register_table_in(kCatalog, "main", std::make_shared<TicketProbe>());

    vgi::CatalogTable probe;
    probe.name = "probe";
    probe.scan_function = "ticket_probe";
    probe.columns = probe_columns();
    probe.comment = "The options this attach was made with";
    probe.cardinality = 1;
    model.schema("main").tables.push_back(std::move(probe));
}

}  // namespace example

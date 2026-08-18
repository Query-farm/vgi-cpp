// © Copyright 2025-2026, Query.Farm LLC - https://query.farm
// SPDX-License-Identifier: Apache-2.0

// A toy delimited-text `COPY … TO` writer.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

// Shards are appended here and read back at close.
constexpr const char* kShardNamespace = "copy_to_shard";

std::string encode_batch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto sink = arrow::io::BufferOutputStream::Create().ValueOrDie();
    auto writer = arrow::ipc::MakeStreamWriter(sink, batch->schema()).ValueOrDie();
    (void)writer->WriteRecordBatch(*batch);
    (void)writer->Close();
    return sink->Finish().ValueOrDie()->ToString();
}

std::shared_ptr<arrow::RecordBatch> decode_batch(const std::string& bytes) {
    if (bytes.empty()) return nullptr;
    auto buffer = arrow::Buffer::FromString(bytes);
    auto source = std::make_shared<arrow::io::BufferReader>(buffer);
    auto reader = arrow::ipc::RecordBatchStreamReader::Open(source);
    if (!reader.ok()) return nullptr;
    std::shared_ptr<arrow::RecordBatch> batch;
    (void)reader.ValueUnsafe()->ReadNext(&batch);
    return batch;
}

struct Options {
    std::string null_string;
    std::string delimiter = ",";
    bool header = false;
    int64_t header_repeat = 1;
    std::string on_exists = "overwrite";
    std::string fail_on_value;
};

class ExampleLines : public vgi::CopyToFunction {
public:
    ExampleLines(std::string format, std::string handler, std::string comment,
                 std::string description, bool ordered)
        : format_(std::move(format)),
          handler_(std::move(handler)),
          comment_(std::move(comment)),
          description_(std::move(description)),
          ordered_(ordered) {}

    std::string format() const override { return format_; }
    std::string handler_name() const override { return handler_; }
    std::optional<std::string> comment() const override { return comment_; }
    bool ordered() const override { return ordered_; }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = description_;
        md.tags = {{"category", "copy_to"}, {"stability", "test"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        // Named, because COPY options are written `(FORMAT x, delimiter '|')`.
        // The destination path is not among them: it comes from the COPY
        // statement itself.
        return {
            vgi::ArgSpec::named("null_string", "varchar", "Token written for SQL NULL"),
            vgi::ArgSpec::named("delimiter", "varchar", "Field separator"),
            vgi::ArgSpec::named("header", "boolean", "Write a header row of column names"),
            vgi::ArgSpec::named("header_repeat", "int64",
                                "When header=true, write the header line this many times"),
            vgi::ArgSpec::named("on_exists", "varchar",
                                "Behavior when the destination file already exists"),
            vgi::ArgSpec::named("fail_on_value", "varchar",
                                "If non-empty, fail mid-write when a cell equals this value"),
        };
    }

    void write(const vgi::ProcessParams& params,
               const std::shared_ptr<arrow::RecordBatch>& batch) override {
        // Validated on every batch, not only at close: a missing required
        // option should surface on the first one, not after the whole COPY
        // has run.
        const auto options = parse_options(params.arguments);
        if (!options.fail_on_value.empty()) {
            for (int i = 0; i < batch->num_columns(); ++i) {
                auto text = cast_to(batch->column(i), arrow::utf8());
                const auto& values = static_cast<const arrow::StringArray&>(*text);
                for (int64_t row = 0; row < values.length(); ++row) {
                    if (!values.IsNull(row) && values.GetString(row) == options.fail_on_value) {
                        // The message must name the option: the test matches
                        // on `fail_on_value`, which is also the only thing
                        // that tells a user which knob caused the failure.
                        throw std::invalid_argument(format_ + ": fail_on_value matched " +
                                                    options.fail_on_value);
                    }
                }
            }
        }
        // Appended to execution-scoped storage rather than held: the sink runs
        // in several processes and close runs in another.
        params.storage->append(params.execution_id, kShardNamespace, "", encode_batch(batch));
    }

    int64_t close(const vgi::ProcessParams& params) override {
        const auto options = parse_options(params.arguments);
        const auto path = params.copy_to_path.value_or("");
        if (path.empty()) throw std::runtime_error(format_ + ": no destination path");

        if (options.on_exists == "error" && std::filesystem::exists(path)) {
            // "already exists" is the phrase the test matches on, and the
            // one a user recognizes.
            throw std::runtime_error(format_ + ": destination already exists: " + path);
        }

        auto shards = params.storage->scan(params.execution_id, kShardNamespace, "", 0,
                                           SIZE_MAX);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error(format_ + ": cannot open " + path);

        int64_t written = 0;
        bool wrote_header = false;
        for (const auto& [id, bytes] : shards) {
            (void)id;
            auto batch = decode_batch(bytes);
            if (!batch) continue;

            if (options.header && !wrote_header) {
                wrote_header = true;
                for (int64_t repeat = 0; repeat < options.header_repeat; ++repeat) {
                    for (int i = 0; i < batch->num_columns(); ++i) {
                        if (i) out << options.delimiter;
                        out << batch->schema()->field(i)->name();
                    }
                    out << '\n';
                }
            }

            std::vector<std::shared_ptr<arrow::Array>> text;
            text.reserve(static_cast<size_t>(batch->num_columns()));
            for (int i = 0; i < batch->num_columns(); ++i) {
                text.push_back(cast_to(batch->column(i), arrow::utf8()));
            }
            for (int64_t row = 0; row < batch->num_rows(); ++row) {
                for (size_t i = 0; i < text.size(); ++i) {
                    if (i) out << options.delimiter;
                    const auto& values = static_cast<const arrow::StringArray&>(*text[i]);
                    out << (values.IsNull(row) ? options.null_string : values.GetString(row));
                }
                out << '\n';
                ++written;
            }
        }

        // Headers still go out for an empty COPY: close is called even when
        // nothing was written, and a header-only file is the right result.
        if (options.header && !wrote_header) {
            for (int64_t repeat = 0; repeat < options.header_repeat; ++repeat) out << '\n';
        }
        out.flush();
        if (!out) throw std::runtime_error(format_ + ": write failed for " + path);
        return written;
    }

private:
    Options parse_options(const vgi::Arguments& arguments) const {
        Options options;
        auto null_string = arguments.named_string("null_string");
        if (!null_string) {
            throw std::invalid_argument(format_ + ": required option 'null_string' is missing");
        }
        options.null_string = *null_string;

        options.delimiter = arguments.named_string("delimiter").value_or(",");
        if (options.delimiter.empty()) {
            throw std::invalid_argument(format_ + ": 'delimiter' must not be empty");
        }
        if (auto header = arguments.named("header")) {
            auto values = cast_to(header, arrow::boolean());
            options.header = !values->IsNull(0) &&
                             static_cast<const arrow::BooleanArray&>(*values).Value(0);
        }
        options.header_repeat = arguments.named_int64("header_repeat").value_or(1);
        if (options.header_repeat < 0 || options.header_repeat > 3) {
            throw std::invalid_argument(format_ + ": 'header_repeat' must be between 0 and 3");
        }
        options.on_exists = arguments.named_string("on_exists").value_or("overwrite");
        if (options.on_exists != "overwrite" && options.on_exists != "error") {
            throw std::invalid_argument(format_ +
                                        ": 'on_exists' must be one of ['overwrite', 'error']");
        }
        options.fail_on_value = arguments.named_string("fail_on_value").value_or("");
        return options;
    }

    std::string format_;
    std::string handler_;
    std::string comment_;
    std::string description_;
    bool ordered_;
};

// A writer that forwards a `CREATE SECRET` credential, so the destination-
// scoped secret path is exercised end to end.
class SecretLines : public vgi::CopyToFunction {
public:
    std::string format() const override { return "secret_lines_out"; }
    std::string handler_name() const override { return "secret_lines_writer"; }

    std::optional<std::string> comment() const override {
        return "Writer that forwards a CREATE SECRET credential (test fixture)";
    }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Write the resolved secret's api_key + row count to the destination";
        md.tags = {{"category", "copy_to"}, {"stability", "test"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("secret_type", "varchar",
                                    "Secret type to fetch, scoped by the destination path")};
    }

    std::vector<vgi::SecretLookup> secret_lookups(
        const vgi::BindParams& params) const override {
        if (!params.copy_to_path) return {};
        // Scoped to the destination: a cloud write wants the credential for
        // the bucket it is writing to, not any credential of that type.
        return {{secret_type(params.arguments), *params.copy_to_path, std::nullopt}};
    }

    void write(const vgi::ProcessParams& params,
               const std::shared_ptr<arrow::RecordBatch>& batch) override {
        params.storage->append(params.execution_id, kSecretShardNamespace, "",
                               std::to_string(batch->num_rows()));
    }

    int64_t close(const vgi::ProcessParams& params) override {
        const auto type = secret_type(params.arguments);
        const auto api_key = params.secrets.field(type, "api_key").value_or("NONE");

        int64_t rows = 0;
        for (const auto& [id, value] : params.storage->scan(params.execution_id,
                                                            kSecretShardNamespace, "", 0,
                                                            SIZE_MAX)) {
            (void)id;
            rows += std::strtoll(value.c_str(), nullptr, 10);
        }

        const auto path = params.copy_to_path.value_or("");
        if (path.empty()) throw std::runtime_error("secret_lines_out: no destination path");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << api_key << "," << rows << '\n';
        out.flush();
        return rows;
    }

private:
    static constexpr const char* kSecretShardNamespace = "copy_to_secret_shard";

    static std::string secret_type(const vgi::Arguments& arguments) {
        return arguments.named_string("secret_type").value_or("vgi_example");
    }
};

}  // namespace

void register_copy_to(vgi::Worker& worker) {
    worker.register_copy_to(std::make_shared<SecretLines>());
    worker.register_copy_to(std::make_shared<ExampleLines>(
        "example_lines_out", "example_lines_writer", "Toy delimited-text writer for tests",
        "Write the COPY source to a delimited text file", /*ordered=*/false));
    worker.register_copy_to(std::make_shared<ExampleLines>(
        "example_lines_ordered_out", "example_lines_ordered_writer",
        "Toy delimited-text writer (ordered, single-thread sink)",
        "Write the COPY source to a delimited file, preserving source order",
        /*ordered=*/true));
}

}  // namespace example

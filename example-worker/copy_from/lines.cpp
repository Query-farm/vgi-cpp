// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// A toy delimited-text `COPY … FROM` reader, the counterpart of the
// `example_lines_out` writer.

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_binary.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

struct Options {
    std::string null_string;
    std::string delimiter = ",";
    int64_t skip_rows = 0;
    std::string on_error = "fail";
};

std::vector<std::string> read_lines(const std::string& path, const std::string& format) {
    std::ifstream source(path, std::ios::binary);
    if (!source) throw std::runtime_error(format + ": cannot read " + path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(source, line)) {
        // The suite writes these files with DuckDB's CSV writer, which is
        // free to use CRLF; a stray CR would otherwise land in the last cell.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(std::move(line));
    }
    return lines;
}

std::vector<std::string> split(const std::string& line, const std::string& delimiter) {
    std::vector<std::string> cells;
    size_t start = 0;
    for (size_t at = line.find(delimiter, start); at != std::string::npos;
         at = line.find(delimiter, start)) {
        cells.push_back(line.substr(start, at - start));
        start = at + delimiter.size();
    }
    cells.push_back(line.substr(start));
    return cells;
}

class ExampleLines : public vgi::CopyFromFunction {
public:
    std::string format() const override { return "example_lines"; }
    std::string handler_name() const override { return "example_lines_copy_reader"; }

    std::optional<std::string> comment() const override {
        return "Toy delimited-text reader for tests";
    }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Read a delimited text file into the COPY target table";
        md.tags = {{"category", "copy_from"}, {"stability", "test"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        // Named, because COPY options are written `(FORMAT x, delimiter '|')`.
        // The source path is not among them: it comes from the COPY statement.
        return {
            vgi::ArgSpec::named("null_string", "varchar", "Token parsed as SQL NULL"),
            vgi::ArgSpec::named("delimiter", "varchar", "Field separator"),
            vgi::ArgSpec::named("skip_rows", "int64", "Leading lines to skip before data"),
            vgi::ArgSpec::named("on_error", "varchar",
                                "Behavior on a row whose column count does not match the target"),
        };
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> read(
        const vgi::ProcessParams& params) const override {
        const auto options = parse_options(params.arguments);
        const auto& schema = params.output_schema;
        const int columns = schema->num_fields();

        auto lines = read_lines(params.copy_from_path.value_or(""), format());
        std::vector<std::vector<std::string>> rows;
        for (size_t line = static_cast<size_t>(options.skip_rows); line < lines.size(); ++line) {
            if (lines[line].empty()) continue;
            auto cells = split(lines[line], options.delimiter);
            if (static_cast<int>(cells.size()) != columns) {
                if (options.on_error == "skip") continue;
                throw std::invalid_argument(format() + ": row has " + std::to_string(cells.size()) +
                                            " fields, expected " + std::to_string(columns) + ": '" +
                                            lines[line] + "'");
            }
            rows.push_back(std::move(cells));
        }

        // One string column per target field, then cast: DuckDB inserts no
        // cast between this scan and the INSERT, so the batch has to arrive in
        // the target's own types.
        std::vector<std::shared_ptr<arrow::Array>> values;
        values.reserve(static_cast<size_t>(columns));
        for (int column = 0; column < columns; ++column) {
            arrow::StringBuilder builder;
            (void)builder.Reserve(static_cast<int64_t>(rows.size()));
            for (const auto& row : rows) {
                const auto& cell = row[static_cast<size_t>(column)];
                if (cell == options.null_string) {
                    (void)builder.AppendNull();
                } else {
                    (void)builder.Append(cell);
                }
            }
            std::shared_ptr<arrow::Array> text;
            (void)builder.Finish(&text);
            values.push_back(cast_to(text, schema->field(column)->type()));
        }
        return {arrow::RecordBatch::Make(schema, static_cast<int64_t>(rows.size()), values)};
    }

private:
    Options parse_options(const vgi::Arguments& arguments) const {
        Options options;
        auto null_string = arguments.named_string("null_string");
        if (!null_string) {
            throw std::invalid_argument(format() + ": required option 'null_string' is missing");
        }
        options.null_string = *null_string;

        options.delimiter = arguments.named_string("delimiter").value_or(",");
        if (options.delimiter.empty()) {
            throw std::invalid_argument(format() + ": 'delimiter' must not be empty");
        }
        options.skip_rows = arguments.named_int64("skip_rows").value_or(0);
        if (options.skip_rows < 0) {
            throw std::invalid_argument(format() + ": 'skip_rows' must be >= 0");
        }
        options.on_error = arguments.named_string("on_error").value_or("fail");
        if (options.on_error != "fail" && options.on_error != "skip") {
            throw std::invalid_argument(format() + ": 'on_error' must be one of ['fail', 'skip']");
        }
        return options;
    }
};

// A reader that forwards a `CREATE SECRET` credential, so the source-scoped
// secret path is exercised end to end.
class SecretLines : public vgi::CopyFromFunction {
public:
    std::string format() const override { return "secret_lines_in"; }
    std::string handler_name() const override { return "secret_lines_reader"; }

    std::optional<std::string> comment() const override {
        return "Reader that forwards a CREATE SECRET credential (test fixture)";
    }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = "Emit the resolved secret's api_key as a single VARCHAR row";
        md.tags = {{"category", "copy_from"}, {"stability", "test"}};
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::named("secret_type", "varchar",
                                    "Secret type to fetch, scoped by the source path")};
    }

    std::vector<vgi::SecretLookup> secret_lookups(const vgi::BindParams& params) const override {
        if (!params.copy_from_path) return {};
        // Scoped to the source: a cloud read wants the credential for the
        // bucket it is reading from, not any credential of that type.
        return {{secret_type(params.arguments), *params.copy_from_path, std::nullopt}};
    }

    std::vector<std::shared_ptr<arrow::RecordBatch>> read(
        const vgi::ProcessParams& params) const override {
        const auto& schema = params.output_schema;
        if (schema->num_fields() != 1) {
            throw std::invalid_argument(format() + ": expected a single-column target, got " +
                                        std::to_string(schema->num_fields()));
        }
        const auto path = params.copy_from_path.value_or("");
        // By scope and type, never by name: the user chose the secret's name
        // in CREATE SECRET, so the fixture cannot know it.
        std::string api_key = "NONE";
        if (const auto* secret =
                params.secrets.for_scope_of_type(path, secret_type(params.arguments))) {
            if (auto found = secret->find("api_key"); found != secret->end()) {
                api_key = found->second;
            }
        }

        arrow::StringBuilder builder;
        (void)builder.Append(api_key);
        std::shared_ptr<arrow::Array> text;
        (void)builder.Finish(&text);
        return {arrow::RecordBatch::Make(schema, 1, {cast_to(text, schema->field(0)->type())})};
    }

private:
    static std::string secret_type(const vgi::Arguments& arguments) {
        return arguments.named_string("secret_type").value_or("vgi_example");
    }
};

}  // namespace

void register_copy_from(vgi::Worker& worker) {
    worker.register_copy_from(std::make_shared<ExampleLines>());
    worker.register_copy_from(std::make_shared<SecretLines>());
}

}  // namespace example

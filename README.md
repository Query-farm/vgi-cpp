# vgi-c++

The **C++ worker SDK for VGI** (Vector Gateway Interface) — write DuckDB
functions and catalogs in C++, ship them as one binary, no DuckDB extension to
compile and no version coupling.

A worker is an ordinary binary that DuckDB launches and talks to over Apache
Arrow IPC. It can expose scalar / table / aggregate functions and whole
catalogs that behave like native DuckDB objects. This repo is the C++ sibling
of the [Rust](https://github.com/Query-farm/vgi-rust) and
[Python](https://github.com/Query-farm/vgi-python) SDKs, and is built on the
C++ [vgi-rpc](https://github.com/Query-farm/vgi-rpc-cpp) port.

The example worker passes the VGI integration suite in
[`vgi`](https://github.com/Query-farm/vgi) over the subprocess transport. What
is and is not implemented is written down in
[`docs/roadmap.md`](docs/roadmap.md), and the suite — not that file — is the
specification.

## Build

```bash
cmake --preset default   # Debug + tests
cmake --build build
ctest --test-dir build
```

`vgi-rpc-c++` is built from a sibling checkout by default, since the two move
together. Override with `-DVGI_RPC_SOURCE_DIR=<path>`, or use an installed
copy with `-DVGI_USE_INSTALLED_RPC=ON`.

## A worker

```cpp
#include <vgi/worker.h>

class UpperCase : public vgi::ScalarFunction {
    std::string name() const override { return "upper_case"; }
    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.return_type = arrow::utf8();
        return md;
    }
    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column("value", 0, "varchar")};
    }
    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override;
};

int main(int argc, char** argv) {
    vgi::Worker worker;
    worker.register_scalar(std::make_shared<UpperCase>());
    worker.run(argc, argv);  // stdio, --unix <path>, or --http
}
```

`example-worker/` is the full fixture set the integration suite runs against —
several hundred functions across every shape the protocol has, and the best
place to look for how any one of them is meant to behave.

## Running the suite

```bash
scripts/run_tests.sh                 # the whole in-scope suite
scripts/run_tests.sh scalar          # one category
scripts/run_tests.sh test/sql/integration/scalar/upper_case.test
```

It drives `~/Development/vgi`'s `unittest` binary against the release worker.
`VGI_EXT` points at the extension checkout, `VGI_CPP_BUILD` at the build
directory, and `VGI_CPP_TEST_CACHE` at a scratch directory — the last matters
when two runs go at once, since they would otherwise share one log.

## Licence

Apache-2.0.

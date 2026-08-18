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

> **Status: early.** The build, the generated protocol layer, and the SDK's
> public shape exist, and a worker built with it answers `__describe__` as
> protocol `vgi` 1.3.0. No VGI method is implemented yet, so it cannot yet be
> attached from DuckDB. See [`docs/roadmap.md`](docs/roadmap.md).

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

`example-worker/` is the full fixture set the integration suite runs against.

## Licence

Apache-2.0.

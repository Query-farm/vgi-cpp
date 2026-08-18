// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// Geospatial fixtures. Each function exists in three shapes — struct{lat,lon},
// list<double>, and fixed_size_list<double,2> — because a DuckDB point literal
// can arrive as any of them and the three are separate overloads to the
// engine, not one polymorphic function.

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <arrow/array.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>

#include <vgi/worker.h>

#include "scalar/util.h"

namespace example {
namespace {

std::shared_ptr<arrow::DataType> point_struct_type() {
    return arrow::struct_({arrow::field("lat", arrow::float64(), /*nullable=*/true),
                           arrow::field("lon", arrow::float64(), /*nullable=*/true)});
}

std::shared_ptr<arrow::DataType> point_list_type() {
    return arrow::list(arrow::field("item", arrow::float64(), /*nullable=*/true));
}

std::shared_ptr<arrow::DataType> point_fixed_type() {
    return arrow::fixed_size_list(arrow::field("item", arrow::float64(), /*nullable=*/true), 2);
}

using Coordinates = std::vector<std::optional<double>>;

// Coordinates arrive as decimal, integer, or float depending on how the SQL
// literal was written, so cast before reading. A cast that fails yields all
// nulls rather than throwing: a malformed point is a null result, not an error.
Coordinates to_doubles(const std::shared_ptr<arrow::Array>& array) {
    Coordinates out(static_cast<size_t>(array->length()));
    std::shared_ptr<arrow::Array> casted;
    try {
        casted = cast_to(array, arrow::float64());
    } catch (const std::exception&) {
        return out;
    }
    const auto& values = static_cast<const arrow::DoubleArray&>(*casted);
    for (int64_t i = 0; i < values.length(); ++i) {
        if (!values.IsNull(i)) out[static_cast<size_t>(i)] = values.Value(i);
    }
    return out;
}

// Split a point column of any supported shape into parallel lat/lon vectors.
std::pair<Coordinates, Coordinates> lat_lon(const std::shared_ptr<arrow::Array>& array) {
    const auto rows = static_cast<size_t>(array->length());

    if (array->type()->id() == arrow::Type::STRUCT) {
        const auto& points = static_cast<const arrow::StructArray&>(*array);
        auto lat = points.GetFieldByName("lat");
        auto lon = points.GetFieldByName("lon");
        if (!lat || !lon) throw std::runtime_error("point struct has no lat/lon");
        return {to_doubles(lat), to_doubles(lon)};
    }

    // Both list shapes read the same way: element 0 is lat, element 1 lon.
    const auto read_pairs = [rows](auto&& value_at) {
        Coordinates lat(rows);
        Coordinates lon(rows);
        for (size_t i = 0; i < rows; ++i) {
            auto coordinates = value_at(static_cast<int64_t>(i));
            if (!coordinates) continue;
            auto values = to_doubles(coordinates);
            if (values.size() > 0) lat[i] = values[0];
            if (values.size() > 1) lon[i] = values[1];
        }
        return std::pair{lat, lon};
    };

    if (array->type()->id() == arrow::Type::LIST) {
        const auto& lists = static_cast<const arrow::ListArray&>(*array);
        return read_pairs([&](int64_t i) -> std::shared_ptr<arrow::Array> {
            return lists.IsNull(i) ? nullptr : lists.value_slice(i);
        });
    }
    if (array->type()->id() == arrow::Type::FIXED_SIZE_LIST) {
        const auto& lists = static_cast<const arrow::FixedSizeListArray&>(*array);
        return read_pairs([&](int64_t i) -> std::shared_ptr<arrow::Array> {
            return lists.IsNull(i) ? nullptr : lists.value_slice(i);
        });
    }
    throw std::runtime_error("unsupported point type " + array->type()->ToString());
}

// The three shapes differ only in the declared argument type, so one class
// covers all of them and the constructor picks the shape.
enum class Shape { Struct, List, Fixed };

const char* shape_suffix(Shape shape) {
    switch (shape) {
        case Shape::Struct: return "struct";
        case Shape::List: return "list";
        case Shape::Fixed: return "fixed";
    }
    return "fixed";
}

const char* shape_phrase(Shape shape) {
    return shape == Shape::Fixed ? "fixed-size list" : shape_suffix(shape);
}

std::shared_ptr<arrow::DataType> shape_type(Shape shape) {
    switch (shape) {
        case Shape::Struct: return point_struct_type();
        case Shape::List: return point_list_type();
        case Shape::Fixed: return point_fixed_type();
    }
    return point_fixed_type();
}

class GeoDistance : public vgi::ScalarFunction {
public:
    explicit GeoDistance(Shape shape) : shape_(shape) {}

    std::string name() const override {
        return std::string("geo_distance_") + shape_suffix(shape_);
    }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description =
            std::string("Euclidean distance between two ") + shape_phrase(shape_) + " points";
        md.return_type = arrow::float64();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        return {vgi::ArgSpec::column_typed("p1", 0, shape_type(shape_), "point"),
                vgi::ArgSpec::column_typed("p2", 1, shape_type(shape_), "point")};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        auto [lat1, lon1] = lat_lon(batch->column(0));
        auto [lat2, lon2] = lat_lon(batch->column(1));

        arrow::DoubleBuilder out;
        (void)out.Reserve(static_cast<int64_t>(lat1.size()));
        for (size_t i = 0; i < lat1.size(); ++i) {
            if (!lat1[i] || !lon1[i] || !lat2[i] || !lon2[i]) {
                (void)out.AppendNull();
                continue;
            }
            const double dy = *lat2[i] - *lat1[i];
            const double dx = *lon2[i] - *lon1[i];
            (void)out.Append(std::sqrt(dy * dy + dx * dx));
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }

private:
    Shape shape_;
};

// `geo_centroid_<shape>(points...)` — the mean of N points, whatever shape
// they arrive in, always returned as a struct.
class GeoCentroid : public vgi::ScalarFunction {
public:
    explicit GeoCentroid(Shape shape) : shape_(shape) {}

    std::string name() const override {
        return std::string("geo_centroid_") + shape_suffix(shape_);
    }

    vgi::FunctionMetadata metadata() const override {
        vgi::FunctionMetadata md;
        md.description = std::string("Centroid of N ") + shape_phrase(shape_) + " points";
        md.return_type = point_struct_type();
        return md;
    }

    std::vector<vgi::ArgSpec> argument_specs() const override {
        auto points = vgi::ArgSpec::column_typed("points", 0, shape_type(shape_), "point");
        points.with_varargs();
        return {points};
    }

    std::shared_ptr<arrow::RecordBatch> process(
        const vgi::ProcessParams& params,
        const std::shared_ptr<arrow::RecordBatch>& batch) const override {
        std::vector<std::pair<Coordinates, Coordinates>> columns;
        columns.reserve(static_cast<size_t>(batch->num_columns()));
        for (int i = 0; i < batch->num_columns(); ++i) {
            columns.push_back(lat_lon(batch->column(i)));
        }

        auto lat_builder = std::make_shared<arrow::DoubleBuilder>();
        auto lon_builder = std::make_shared<arrow::DoubleBuilder>();
        arrow::StructBuilder out(point_struct_type(), arrow::default_memory_pool(),
                                 {lat_builder, lon_builder});

        for (int64_t row = 0; row < batch->num_rows(); ++row) {
            double lat_sum = 0;
            double lon_sum = 0;
            int64_t counted = 0;
            bool any_null = false;
            for (const auto& [lat, lon] : columns) {
                const auto i = static_cast<size_t>(row);
                if (!lat[i] || !lon[i]) {
                    any_null = true;
                    break;
                }
                lat_sum += *lat[i];
                lon_sum += *lon[i];
                ++counted;
            }
            if (any_null || counted == 0) {
                (void)out.AppendNull();
                continue;
            }
            (void)out.Append();
            (void)lat_builder->Append(lat_sum / static_cast<double>(counted));
            (void)lon_builder->Append(lon_sum / static_cast<double>(counted));
        }
        std::shared_ptr<arrow::Array> array;
        (void)out.Finish(&array);
        return result(params, array);
    }

private:
    Shape shape_;
};

}  // namespace

void register_geo(vgi::Worker& worker) {
    for (auto shape : {Shape::Struct, Shape::List, Shape::Fixed}) {
        worker.register_scalar(std::make_shared<GeoDistance>(shape));
        worker.register_scalar(std::make_shared<GeoCentroid>(shape));
    }
}

}  // namespace example

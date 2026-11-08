// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
#include "vgi/scan_planning.h"

#include <utility>

namespace vgi {

ScanPlanningFilters::ScanPlanningFilters(std::string raw,
                                         std::shared_ptr<arrow::Schema> table_columns)
    : raw_(std::move(raw)), table_columns_(std::move(table_columns)) {}

const PushdownFilters& ScanPlanningFilters::decode() const {
    if (!decoded_) {
        decoded_ = std::make_shared<const PushdownFilters>(
            PushdownFilters::parse(raw_, {}, table_columns_));
    }
    return *decoded_;
}

bool ScanPlanningFilters::may_match(const std::vector<ColumnStatistics>& statistics) const {
    return decode().may_match(statistics);
}

}  // namespace vgi

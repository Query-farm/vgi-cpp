// © Copyright 2025, 2026 Query Farm LLC - https://query.farm
// A recognizer for the bounded scalar SQL the DuckDB extension reads in CHECK
// constraints and VIRTUAL generated columns. See sql_scalar.h.

#include "sql_scalar.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>

namespace vgi::sql {

namespace {

std::string lower(std::string value) {
    for (auto& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

[[noreturn]] void refuse(const std::string& message) {
    throw std::invalid_argument(message);
}

enum class Tok { Ident, Quoted, String, Number, Op, End };

struct Token {
    Tok kind;
    std::string text;
};

std::vector<Token> tokenize(const std::string& sql) {
    std::vector<Token> out;
    size_t i = 0;
    const size_t n = sql.size();
    auto digit = [&](size_t at) {
        return at < n && std::isdigit(static_cast<unsigned char>(sql[at]));
    };
    while (i < n) {
        const char c = sql[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else if (c == '-' && i + 1 < n && sql[i + 1] == '-') {
            while (i < n && sql[i] != '\n') ++i;
        } else if (c == '/' && i + 1 < n && sql[i + 1] == '*') {
            const auto end = sql.find("*/", i + 2);
            if (end == std::string::npos) refuse("unterminated comment");
            i = end + 2;
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t j = i;
            while (j < n && (std::isalnum(static_cast<unsigned char>(sql[j])) || sql[j] == '_'))
                ++j;
            if (j < n && (sql[j] == '\'' || sql[j] == '$'))
                refuse("prefixed string literals and parameters are unsupported");
            out.push_back({Tok::Ident, sql.substr(i, j - i)});
            i = j;
        } else if (c == '"' || c == '\'') {
            std::string text;
            size_t j = i + 1;
            for (;; ++j) {
                if (j >= n) refuse("unterminated quoted text");
                if (sql[j] == c) {
                    if (j + 1 < n && sql[j + 1] == c) {
                        text.push_back(c);
                        ++j;
                        continue;
                    }
                    break;
                }
                text.push_back(sql[j]);
            }
            if (c == '"' && text.empty()) refuse("empty quoted identifier");
            out.push_back({c == '"' ? Tok::Quoted : Tok::String, text});
            i = j + 1;
        } else if (digit(i) || (c == '.' && digit(i + 1))) {
            size_t j = i;
            while (digit(j)) ++j;
            if (j < n && sql[j] == '.') {
                ++j;
                while (digit(j)) ++j;
            }
            if (j < n && (sql[j] == 'e' || sql[j] == 'E')) {
                size_t k = j + 1;
                if (k < n && (sql[k] == '+' || sql[k] == '-')) ++k;
                if (!digit(k)) refuse("malformed numeric literal");
                j = k;
                while (digit(j)) ++j;
            }
            if (j < n && (std::isalpha(static_cast<unsigned char>(sql[j])) || sql[j] == '_'))
                refuse("malformed numeric literal");
            out.push_back({Tok::Number, sql.substr(i, j - i)});
            i = j;
        } else {
            static const char* const kOps[] = {"::", "||", "<=", ">=", "<>", "!=", "==", "=", "<",
                                               ">",  "+",  "-",  "*",  "/",  "%",  "(",  ")", ","};
            bool matched = false;
            for (const char* op : kOps) {
                const std::string text(op);
                if (sql.compare(i, text.size(), text) == 0) {
                    out.push_back({Tok::Op, text});
                    i += text.size();
                    matched = true;
                    break;
                }
            }
            if (!matched) refuse(std::string("unsupported character '") + c + "'");
        }
    }
    out.push_back({Tok::End, ""});
    return out;
}

enum class Kind {
    Constant,
    Column,
    Cast,
    Comparison,
    Conjunction,
    Not,
    IsNull,
    Coalesce,
    Function
};
enum class Literal { Integer, Bigint, Hugeint, Decimal, Double, Varchar, Boolean, Null };

struct Node {
    Kind kind = Kind::Constant;
    Literal literal = Literal::Null;
    // Column name, function name, comparison operator, conjunction, or the
    // cast's normalized target type.
    std::string text;
    std::vector<Node> children;
};

Literal classify_number(const std::string& text) {
    if (text.find_first_of("eE") != std::string::npos) return Literal::Double;
    if (text.find('.') != std::string::npos) return Literal::Decimal;
    const bool negative = !text.empty() && text[0] == '-';
    std::string digits = text.substr(negative ? 1 : 0);
    digits.erase(0, std::min(digits.find_first_not_of('0'), digits.size() - 1));
    // Compare magnitudes as strings: the literal may exceed any machine type.
    auto fits = [&](const std::string& limit) {
        return digits.size() < limit.size() || (digits.size() == limit.size() && digits <= limit);
    };
    if (fits(negative ? "2147483648" : "2147483647")) return Literal::Integer;
    if (fits(negative ? "9223372036854775808" : "9223372036854775807")) return Literal::Bigint;
    return Literal::Hugeint;
}

bool numeric(const Node& node) {
    return node.kind == Kind::Constant &&
           (node.literal == Literal::Integer || node.literal == Literal::Bigint ||
            node.literal == Literal::Hugeint || node.literal == Literal::Decimal ||
            node.literal == Literal::Double);
}

class Parser {
public:
    explicit Parser(const std::string& sql) : tokens_(tokenize(sql)) {}

    Node parse() {
        auto node = expression();
        if (peek().kind != Tok::End)
            refuse("unexpected '" + peek().text + "' after the expression");
        return node;
    }

private:
    const Token& peek(size_t ahead = 0) const {
        return tokens_[std::min(pos_ + ahead, tokens_.size() - 1)];
    }
    Token next() { return tokens_[std::min(pos_++, tokens_.size() - 1)]; }
    bool op(const char* text) const { return peek().kind == Tok::Op && peek().text == text; }
    bool keyword(const char* word, size_t ahead = 0) const {
        return peek(ahead).kind == Tok::Ident && lower(peek(ahead).text) == word;
    }
    void expect(const char* text) {
        if (!op(text)) refuse(std::string("expected '") + text + "'");
        ++pos_;
    }
    struct Depth {
        explicit Depth(int& depth) : depth_(depth) {
            if (++depth_ > 256) refuse("expression is too deeply nested");
        }
        ~Depth() { --depth_; }
        int& depth_;
    };

    static Node make(Kind kind, std::string text, std::vector<Node> children) {
        Node node;
        node.kind = kind;
        node.text = std::move(text);
        node.children = std::move(children);
        return node;
    }

    Node expression() {
        Depth guard(depth_);
        auto left = conjunction();
        while (keyword("or")) {
            ++pos_;
            left = make(Kind::Conjunction, "or", {std::move(left), conjunction()});
        }
        return left;
    }
    Node conjunction() {
        auto left = negation();
        while (keyword("and")) {
            ++pos_;
            left = make(Kind::Conjunction, "and", {std::move(left), negation()});
        }
        return left;
    }
    Node negation() {
        Depth guard(depth_);
        if (keyword("not")) {
            ++pos_;
            return make(Kind::Not, "not", {negation()});
        }
        return null_test();
    }
    Node null_test() {
        auto node = comparison();
        for (;;) {
            if (keyword("isnull") || keyword("notnull")) {
                const bool is_not = keyword("notnull");
                ++pos_;
                node = make(Kind::IsNull, is_not ? "is not null" : "is null", {std::move(node)});
            } else if (keyword("is")) {
                ++pos_;
                bool is_not = false;
                if (keyword("not")) {
                    ++pos_;
                    is_not = true;
                }
                if (!keyword("null")) refuse("only IS [NOT] NULL is supported");
                ++pos_;
                node = make(Kind::IsNull, is_not ? "is not null" : "is null", {std::move(node)});
            } else {
                return node;
            }
        }
    }
    Node comparison() {
        auto left = concatenation();
        for (const char* text : {"=", "==", "<>", "!=", "<", ">", "<=", ">="}) {
            if (!op(text)) continue;
            ++pos_;
            std::string normalized = text;
            if (normalized == "==") normalized = "=";
            if (normalized == "!=") normalized = "<>";
            auto node = make(Kind::Comparison, normalized, {std::move(left), concatenation()});
            for (const char* again : {"=", "==", "<>", "!=", "<", ">", "<=", ">="})
                if (op(again)) refuse("comparisons do not chain");
            return node;
        }
        return left;
    }
    Node concatenation() {
        auto left = additive();
        while (op("||")) {
            ++pos_;
            left = make(Kind::Function, "||", {std::move(left), additive()});
        }
        return left;
    }
    Node additive() {
        auto left = multiplicative();
        while (op("+") || op("-")) {
            auto name = next().text;
            left = make(Kind::Function, name, {std::move(left), multiplicative()});
        }
        return left;
    }
    Node multiplicative() {
        auto left = unary();
        while (op("*") || op("/") || op("%")) {
            auto name = next().text;
            left = make(Kind::Function, name, {std::move(left), unary()});
        }
        return left;
    }
    Node unary() {
        Depth guard(depth_);
        if (op("-")) {
            ++pos_;
            auto child = unary();
            // DuckDB's grammar folds a negated numeric literal into the constant.
            if (numeric(child)) {
                child.text = child.text[0] == '-' ? child.text.substr(1) : "-" + child.text;
                child.literal = classify_number(child.text);
                return child;
            }
            return make(Kind::Function, "-", {std::move(child)});
        }
        if (op("+")) {
            ++pos_;
            return make(Kind::Function, "+", {unary()});
        }
        return postfix();
    }
    Node postfix() {
        auto node = primary();
        while (op("::")) {
            ++pos_;
            node = make(Kind::Cast, type_name(), {std::move(node)});
        }
        return node;
    }
    // A cast target, normalized to DuckDB's type id spelling, or the raw text
    // for a type the grammars never accept.
    std::string type_name() {
        if (peek().kind != Tok::Ident) refuse("expected a type name");
        std::string name = lower(next().text);
        if (name == "double" && keyword("precision")) ++pos_;
        if (name == "character" && keyword("varying")) {
            ++pos_;
            name = "varchar";
        }
        if (op("(")) {
            ++pos_;
            for (;;) {
                if (peek().kind != Tok::Number) refuse("expected a type modifier");
                ++pos_;
                if (op(")")) break;
                expect(",");
            }
            ++pos_;
        }
        static const std::map<std::string, std::string> kTypes{
            {"boolean", "BOOLEAN"}, {"bool", "BOOLEAN"},   {"logical", "BOOLEAN"},
            {"tinyint", "TINYINT"}, {"int1", "TINYINT"},   {"smallint", "SMALLINT"},
            {"int2", "SMALLINT"},   {"short", "SMALLINT"}, {"integer", "INTEGER"},
            {"int", "INTEGER"},     {"int4", "INTEGER"},   {"signed", "INTEGER"},
            {"bigint", "BIGINT"},   {"int8", "BIGINT"},    {"long", "BIGINT"},
            {"varchar", "VARCHAR"}, {"text", "VARCHAR"},   {"string", "VARCHAR"},
            {"char", "VARCHAR"},    {"bpchar", "VARCHAR"}, {"character", "VARCHAR"},
            {"float", "FLOAT"},     {"float4", "FLOAT"},   {"real", "FLOAT"},
            {"double", "DOUBLE"},   {"float8", "DOUBLE"},  {"decimal", "DECIMAL"},
            {"numeric", "DECIMAL"},
        };
        const auto found = kTypes.find(name);
        return found == kTypes.end() ? name : found->second;
    }
    std::vector<Node> arguments() {
        expect("(");
        std::vector<Node> args;
        if (op(")")) {
            ++pos_;
            return args;
        }
        if (keyword("distinct") || keyword("all") || op("*"))
            refuse("DISTINCT, ALL and * arguments are unsupported");
        for (;;) {
            args.push_back(expression());
            if (op(")")) break;
            expect(",");
        }
        ++pos_;
        return args;
    }
    Node primary() {
        const auto& token = peek();
        if (op("(")) {
            ++pos_;
            auto inner = expression();
            expect(")");
            return inner;
        }
        if (token.kind == Tok::Number) {
            Node node;
            node.kind = Kind::Constant;
            node.text = next().text;
            node.literal = classify_number(node.text);
            return node;
        }
        if (token.kind == Tok::String) {
            Node node;
            node.kind = Kind::Constant;
            node.literal = Literal::Varchar;
            node.text = next().text;
            return node;
        }
        if (token.kind == Tok::Quoted) {
            auto name = next().text;
            if (op("(")) return make(Kind::Function, name, arguments());
            return make(Kind::Column, name, {});
        }
        if (token.kind != Tok::Ident) refuse("unexpected '" + token.text + "'");
        const auto word = lower(token.text);
        if (word == "true" || word == "false" || word == "null") {
            ++pos_;
            Node node;
            node.kind = Kind::Constant;
            node.literal = word == "null" ? Literal::Null : Literal::Boolean;
            node.text = word;
            return node;
        }
        if (word == "try_cast") refuse("TRY_CAST is unsupported");
        if (word == "cast") {
            ++pos_;
            expect("(");
            auto child = expression();
            if (!keyword("as")) refuse("expected AS in CAST");
            ++pos_;
            auto type = type_name();
            expect(")");
            return make(Kind::Cast, type, {std::move(child)});
        }
        static const std::set<std::string> kReserved{
            "and",    "or",   "not",   "is",     "isnull",   "notnull", "as",   "case",
            "when",   "then", "else",  "end",    "in",       "between", "like", "ilike",
            "select", "from", "where", "exists", "distinct", "all",     "any",  "some"};
        if (kReserved.count(word)) refuse("unexpected keyword " + token.text);
        auto name = next().text;
        if (op("(")) {
            if (word == "coalesce") return make(Kind::Coalesce, "coalesce", arguments());
            return make(Kind::Function, name, arguments());
        }
        return make(Kind::Column, name, {});
    }

    std::vector<Token> tokens_;
    size_t pos_ = 0;
    int depth_ = 0;
};

Node parse(const std::string& sql) {
    if (sql.empty() || sql.size() > 1024 * 1024 || sql.find('\0') != std::string::npos)
        refuse("invalid SQL size or text");
    return Parser(sql).parse();
}

bool has_extension_name(const arrow::Field& field) {
    return field.type()->id() == arrow::Type::EXTENSION ||
           (field.metadata() && field.metadata()->Contains("ARROW:extension:name"));
}

bool is_row_id(const arrow::Field& field) {
    return field.metadata() && field.metadata()->Contains("is_row_id");
}

// The index of the top-level, non-row-id column `name` names, matched the way
// DuckDB binds an unquoted reference: case-insensitively.
int resolve(const arrow::Schema& columns, const std::string& name, const char* what) {
    int index = -1;
    for (int i = 0; i < columns.num_fields(); ++i) {
        if (lower(columns.field(i)->name()) != lower(name)) continue;
        if (index >= 0) refuse(std::string(what) + " column reference '" + name + "' is ambiguous");
        index = i;
    }
    if (index < 0 || is_row_id(*columns.field(index)))
        refuse(std::string(what) + " references a missing or synthetic column '" + name + "'");
    return index;
}

}  // namespace

void validate_numeric_check(const std::string& sql, const arrow::Schema& columns) {
    const auto node = parse(sql);
    if (node.kind != Kind::Comparison)
        refuse("CHECK supports only a Boolean comparison of numeric columns and constants");
    for (const auto& operand : node.children) {
        if (operand.kind == Kind::Constant) {
            if (!numeric(operand)) refuse("CHECK requires numeric operands");
            continue;
        }
        if (operand.kind != Kind::Column)
            refuse("CHECK supports only numeric column/constant comparisons");
        const auto& field = *columns.field(resolve(columns, operand.text, "CHECK"));
        if (has_extension_name(field)) refuse("CHECK extension-typed operands are unsupported");
        switch (field.type()->id()) {
            case arrow::Type::INT8:
            case arrow::Type::INT16:
            case arrow::Type::INT32:
            case arrow::Type::INT64:
            case arrow::Type::FLOAT:
            case arrow::Type::DOUBLE: break;
            default:
                refuse("CHECK requires primitive signed numeric columns; '" + field.name() +
                       "' is " + field.type()->ToString());
        }
    }
}

namespace {

// The generated grammar: vgi_column_expression.cpp Scalar() with
// using_expression = true and generated = true.
void generated_node(const Node& node, size_t depth, size_t& nodes, std::set<std::string>& refs) {
    if (depth > 64 || ++nodes > 4096) refuse("scalar expression is too complex");
    static const std::set<std::string> kPrimitive{"BOOLEAN", "TINYINT", "SMALLINT",
                                                  "INTEGER", "BIGINT",  "VARCHAR"};
    switch (node.kind) {
        case Kind::Constant:
            if (node.literal == Literal::Hugeint || node.literal == Literal::Decimal ||
                node.literal == Literal::Double)
                refuse("unsupported scalar literal type in '" + node.text +
                       "' (generated columns take Boolean, signed integer or string literals)");
            break;
        case Kind::Column:
            if (node.text.size() > 4096) refuse("invalid column name");
            refs.insert(lower(node.text));
            break;
        case Kind::Cast:
            if (!kPrimitive.count(node.text))
                refuse("unsupported scalar cast to " + node.text +
                       " (generated columns cast only to BOOLEAN, signed integer or VARCHAR)");
            break;
        case Kind::Comparison:
        case Kind::Conjunction:
        case Kind::Not:
        case Kind::IsNull:
        case Kind::Coalesce: break;
        case Kind::Function: {
            static const std::set<std::string> kAllowed{"+",   "-",     "*",     "%",
                                                        "abs", "lower", "upper", "||"};
            if (!kAllowed.count(node.text == "||" ? node.text : lower(node.text)))
                refuse("unsupported scalar function '" + node.text +
                       "' (generated columns allow + - * % || abs lower upper)");
            break;
        }
    }
    for (const auto& child : node.children) generated_node(child, depth + 1, nodes, refs);
}

bool generated_operand_type(const arrow::Field& field) {
    if (has_extension_name(field)) return false;
    switch (field.type()->id()) {
        case arrow::Type::BOOL:
        case arrow::Type::INT8:
        case arrow::Type::INT16:
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::STRING_VIEW: return true;
        default: return false;
    }
}

}  // namespace

void validate_generated_columns(const arrow::Schema& columns,
                                const std::vector<std::string>& generated_sql) {
    std::map<int, std::set<int>> dependencies;
    bool physical = false;
    for (int i = 0; i < columns.num_fields(); ++i) {
        const auto& field = *columns.field(i);
        const auto sql = i < static_cast<int>(generated_sql.size()) ? generated_sql[i] : "";
        if (sql.empty()) {
            physical = physical || !is_row_id(field);
            continue;
        }
        const auto where = "generated column '" + field.name() + "': ";
        try {
            if (!field.nullable())
                refuse(
                    "generated columns must be nullable (a NOT_NULL or PRIMARY_KEY record names "
                    "it)");
            if (has_extension_name(field))
                refuse("generated columns cannot have an extension type");
            switch (field.type()->id()) {
                case arrow::Type::BOOL:
                case arrow::Type::INT8:
                case arrow::Type::INT16:
                case arrow::Type::INT32:
                case arrow::Type::INT64:
                case arrow::Type::STRING: break;
                default:
                    refuse(
                        "generated columns require a signed integer, Boolean or utf8 type, not " +
                        field.type()->ToString());
            }
            size_t nodes = 0;
            std::set<std::string> refs;
            generated_node(parse(sql), 0, nodes, refs);
            for (const auto& name : refs) {
                const int index = resolve(columns, name, "generated expression");
                if (!generated_operand_type(*columns.field(index)))
                    refuse("unsupported generated operand type: column '" +
                           columns.field(index)->name() + "' is " +
                           columns.field(index)->type()->ToString());
                dependencies[i].insert(index);
            }
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument(where + error.what());
        }
    }
    const bool any = std::any_of(generated_sql.begin(), generated_sql.end(),
                                 [](const std::string& sql) { return !sql.empty(); });
    if (!any) return;
    if (!physical) refuse("a table with generated columns requires a physical column");
    // A generated column may read another, but never itself through any chain.
    std::map<int, int> state;
    std::function<void(int)> visit = [&](int column) {
        auto& mark = state[column];
        if (mark == 2) return;
        if (mark == 1)
            refuse("generated column '" + columns.field(column)->name() + "' depends on itself");
        mark = 1;
        for (int next : dependencies[column]) visit(next);
        state[column] = 2;
    };
    for (const auto& [column, _] : dependencies) visit(column);
}

}  // namespace vgi::sql

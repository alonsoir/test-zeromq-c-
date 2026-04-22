// test_config_parser_traversal.cpp
//
// ACCEPTANCE TEST — DEBT-SAFE-PATH-TEST-PRODUCTION-001 (rag-ingester)
// RED->GREEN: path validation gate.
// Usa el config real de produccion — el JSON es la ley.

#include <gtest/gtest.h>
#include "common/config_parser.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// Path al config real de produccion (disponible en la VM via /vagrant)
static const std::string REAL_CONFIG = "/vagrant/rag-ingester/config/rag-ingester.json";

// RED: path inexistente debe lanzar excepcion
TEST(ConfigParserTraversal, RejectNonExistentPath) {
    EXPECT_THROW(
        rag_ingester::ConfigParser::load("/nonexistent/path/config.json"),
        std::runtime_error
    ) << "ConfigParser MUST throw for non-existent path";
}

// RED: path vacio debe lanzar excepcion
TEST(ConfigParserTraversal, RejectEmptyPath) {
    EXPECT_THROW(
        rag_ingester::ConfigParser::load(""),
        std::exception
    ) << "ConfigParser MUST throw for empty path";
}

// GREEN: config real de produccion debe cargarse correctamente
TEST(ConfigParserTraversal, ProductionConfigLoadsCorrectly) {
    if (!fs::exists(REAL_CONFIG)) {
        GTEST_SKIP() << "Production config not available at " << REAL_CONFIG;
    }
    EXPECT_NO_THROW({
        auto config = rag_ingester::ConfigParser::load(REAL_CONFIG);
        EXPECT_EQ(config.service.id, "rag-ingester-default");
    }) << "ConfigParser MUST accept production config";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  DEBT-SAFE-PATH-TEST-PRODUCTION-001 — rag-ingester   \n";
    std::cout << "  RED->GREEN: path validation gate                     \n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    return RUN_ALL_TESTS();
}

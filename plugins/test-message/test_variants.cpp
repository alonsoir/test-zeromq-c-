// ============================================================================
// test_plugin_message_variants.cpp — TEST-INTEG-4a-PLUGIN
// ============================================================================
// Verifica las 3 variantes del plugin test-message.
// Gate:
//   Variante A → PLUGIN_OK, stats.errors == 0
//   Variante B → D8 violation detectada, stats.errors >= 1
//   Variante C → PLUGIN_ERROR registrado, stats.errors >= 1, no crash
// ============================================================================

#include "plugin_loader/plugin_loader.hpp"
#include "plugin_loader/plugin_api.h"
#include <cstdio>
#include <cstring>
#include <cassert>

using namespace ml_defender;

// Payload de prueba
static const uint8_t TEST_PAYLOAD[] = {0x01, 0x02, 0x03, 0x04, 0x05};

static MessageContext make_ctx() {
    MessageContext ctx{};
    ctx.payload     = TEST_PAYLOAD;
    ctx.payload_len = sizeof(TEST_PAYLOAD);
    ctx.src_ip      = 0xC0A80001; // 192.168.0.1
    ctx.dst_ip      = 0xC0A80002;
    ctx.src_port    = 12345;
    ctx.dst_port    = 80;
    ctx.protocol    = 6; // TCP
    ctx.direction   = 0; // RX
    ctx.nonce       = nullptr; // test mode — NULL permitido (ver plugin_api.h)
    ctx.tag         = nullptr;
    ctx.result_code = 0;
    memset(ctx.annotation, 0, sizeof(ctx.annotation));
    memset(ctx.reserved,   0, sizeof(ctx.reserved));
    return ctx;
}

int main() {
    int failures = 0;

    // ----------------------------------------------------------------
    // Variante A: comportamiento correcto
    // ----------------------------------------------------------------
    fprintf(stderr, "\n=== TEST VARIANT A ===\n");
    setenv("MLD_TEST_VARIANT", "A", 1);
    {
        PluginLoader loader("/vagrant/plugins/test-message/test_config.json");
        loader.load_plugins();
        if (loader.loaded_count() == 0) {
            fprintf(stderr, "SKIP: no plugins loaded (check config)\n");
        } else {
            MessageContext ctx = make_ctx();
            loader.invoke_all(ctx);
            const auto& stats = loader.stats();
            bool ok = (stats[0].errors == 0 && ctx.result_code == 0);
            fprintf(stderr, "Variant A: errors=%zu result_code=%d → %s\n",
                    stats[0].errors, ctx.result_code, ok ? "PASS" : "FAIL");
            if (!ok) ++failures;
        }
    }

    // ----------------------------------------------------------------
    // Variante B: D8 violation
    // ----------------------------------------------------------------
    fprintf(stderr, "\n=== TEST VARIANT B ===\n");
    setenv("MLD_TEST_VARIANT", "B", 1);
    {
        PluginLoader loader("/vagrant/plugins/test-message/test_config.json");
        loader.load_plugins();
        if (loader.loaded_count() > 0) {
            MessageContext ctx = make_ctx();
            loader.invoke_all(ctx);
            const auto& stats = loader.stats();
            bool ok = (stats[0].errors >= 1);
            fprintf(stderr, "Variant B: errors=%zu → %s (expect D8 VIOLATION log above)\n",
                    stats[0].errors, ok ? "PASS" : "FAIL");
            if (!ok) ++failures;
        }
    }

    // ----------------------------------------------------------------
    // Variante C: result_code anomaly, no crash
    // ----------------------------------------------------------------
    fprintf(stderr, "\n=== TEST VARIANT C ===\n");
    setenv("MLD_TEST_VARIANT", "C", 1);
    {
        PluginLoader loader("/vagrant/plugins/test-message/test_config.json");
        loader.load_plugins();
        if (loader.loaded_count() > 0) {
            MessageContext ctx = make_ctx();
            loader.invoke_all(ctx);
            const auto& stats = loader.stats();
            bool ok = (stats[0].errors >= 1);
            fprintf(stderr, "Variant C: errors=%zu → %s (no crash = OK)\n",
                    stats[0].errors, ok ? "PASS" : "FAIL");
            if (!ok) ++failures;
        }
    }

    fprintf(stderr, "\n=== TEST-INTEG-4a-PLUGIN: %s (%d failures) ===\n",
            failures == 0 ? "PASSED" : "FAILED", failures);
    return failures == 0 ? 0 : 1;
}

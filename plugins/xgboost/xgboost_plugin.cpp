/**
 * xgboost_plugin.cpp — ADR-026 Track 1
 *
 * Plugin de inferencia XGBoost para ml-detector.
 * Carga modelo pre-entrenado en CTU-13 Neris (.json XGBoost format).
 * Implementa plugin_process_message(MessageContext&) — PLUGIN_MODE_NORMAL.
 *
 * Gate de merge (docs/XGBOOST-VALIDATION.md):
 *   Precision >= 0.99 | F1 >= 0.9985 | CTU-13 Neris | 4 runs mínimo
 *
 * Fail-closed: std::terminate() si el modelo no carga (ADR-025 D8-pre).
 *
 * DAY 119 — Alonso Isidoro Román
 */

#include <plugin_loader/plugin_api.h>
#include <xgboost/c_api.h>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

// ── Configuración ────────────────────────────────────────────────
#ifndef MLD_XGBOOST_MODEL_PATH
#define MLD_XGBOOST_MODEL_PATH "/etc/ml-defender/models/xgboost_ctu13.json"
#endif

// ── Estado global del plugin ─────────────────────────────────────
static BoosterHandle g_booster = nullptr;
static bool          g_loaded  = false;

// ── Helpers ──────────────────────────────────────────────────────
static void check_xgb(int ret, const char* op) {
    if (ret != 0) {
        fprintf(stderr, "[plugin_xgboost] XGBoost error in %s: %s\n",
                op, XGBGetLastError());
        std::terminate();
    }
}

// ── plugin_init ──────────────────────────────────────────────────
// @requires: config != nullptr
// @ensures: g_loaded == true || std::terminate()
// @invariant: no side effects si falla
extern "C" PluginResult plugin_init(const PluginConfig* config) {
    (void)config;  // No se usa en v0.1 — modelo en ruta compilada

    fprintf(stderr, "[plugin_xgboost] Loading model: %s\n",
            MLD_XGBOOST_MODEL_PATH);

    int ret = XGBoosterCreate(nullptr, 0, &g_booster);
    if (ret != 0) {
        fprintf(stderr, "[plugin_xgboost] FATAL: XGBoosterCreate failed: %s\n",
                XGBGetLastError());
        std::terminate();  // fail-closed
    }

    ret = XGBoosterLoadModel(g_booster, MLD_XGBOOST_MODEL_PATH);
    if (ret != 0) {
        fprintf(stderr, "[plugin_xgboost] FATAL: model not found at %s: %s\n",
                MLD_XGBOOST_MODEL_PATH, XGBGetLastError());
        XGBoosterFree(g_booster);
        g_booster = nullptr;
        std::terminate();  // fail-closed — ADR-025 D8-pre
    }

    g_loaded = true;
    fprintf(stderr, "[plugin_xgboost] Model loaded OK (fail-closed active)\n");
    return PLUGIN_OK;
}

// ── plugin_process_packet ────────────────────────────────────────
// Obligatorio por API — xgboost opera sobre MessageContext, no PacketContext
extern "C" PluginResult plugin_process_packet(PacketContext* ctx) {
    (void)ctx;
    return PLUGIN_SKIP;  // XGBoost no opera sobre PacketContext
}

// ── plugin_process_message ───────────────────────────────────────
// @requires: ctx != nullptr && ctx->payload != nullptr
// @ensures: return PLUGIN_OK || std::terminate() (fail-closed)
// @invariant: no side effects on MessageContext si falla
extern "C" PluginResult plugin_process_message(MessageContext* ctx) {
    if (!g_loaded || g_booster == nullptr) {
        fprintf(stderr, "[plugin_xgboost] FATAL: invoked without loaded model\n");
        std::terminate();
    }

    if (ctx == nullptr) {
        fprintf(stderr, "[plugin_xgboost] FATAL: null MessageContext\n");
        std::terminate();  // ADR-023 FIX-C
    }

    // TODO (Fase 2 DAY 119+): extraer features del payload de ctx
    // ml-detector pre-procesa float32[] — Opción B unanimidad Consejo DAY 118
    // Fase 2 implementará el feature extraction completo desde MessageContext

    fprintf(stderr, "[plugin_xgboost] invoke OK — model ready, inference pending (Fase 2)\n");
    return PLUGIN_OK;
}

// ── plugin_shutdown ──────────────────────────────────────────────
extern "C" void plugin_shutdown() {
    if (g_booster != nullptr) {
        XGBoosterFree(g_booster);
        g_booster = nullptr;
    }
    g_loaded = false;
    fprintf(stderr, "[plugin_xgboost] shutdown OK\n");
}

// ── plugin_name ──────────────────────────────────────────────────
extern "C" const char* plugin_name() {
    return "plugin_xgboost";
}

// ── plugin_version ───────────────────────────────────────────────
extern "C" const char* plugin_version() {
    return "0.1.0-adr026-track1";
}

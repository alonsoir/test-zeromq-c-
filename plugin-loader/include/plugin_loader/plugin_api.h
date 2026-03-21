#pragma once
// ============================================================================
// plugin_api.h — ML Defender Plugin Contract (C ABI)
// ============================================================================
// API C pura para máxima portabilidad y ABI stability entre compilaciones.
// Todo plugin DEBE exportar los 6 símbolos definidos al final de este header.
//
// Restricciones PHASE 1 (ADR-012):
//   - Plugins: SOLO feature extraction
//   - Decisión de bloqueo: NUNCA en un plugin
//   - Crash isolation (proceso separado + watchdog): PHASE 2
//   - Autenticación de plugins: PHASE 2 (seed-client, DAY 95-96)
//
// Versionado: si se modifica esta API, incrementar PLUGIN_API_VERSION.
// El loader rechaza plugins con versión distinta (warning, no abort).
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#define PLUGIN_API_VERSION 1

// ----------------------------------------------------------------------------
// PluginConfig — pasado a plugin_init()
// ----------------------------------------------------------------------------
typedef struct PluginConfig {
    const char* name;         // nombre del plugin (ej: "hello", "ja4")
    const char* config_json;  // sección JSON del plugin como string (puede ser "{}")
} PluginConfig;

// ----------------------------------------------------------------------------
// PacketContext — contexto por paquete, pasado a plugin_process_packet()
// El plugin PUEDE escribir en threat_hint.
// El plugin NO debe castear features ni alert_queue sin conocer la versión del host.
// ----------------------------------------------------------------------------
typedef struct PacketContext {
    // Read-only: datos del flujo
    const uint8_t* raw_bytes;
    size_t         length;
    uint32_t       src_ip;
    uint32_t       dst_ip;
    uint16_t       src_port;
    uint16_t       dst_port;
    uint8_t        protocol;

    // Write: enriquecimiento de features (puntero opaco al FlowFeatures del host)
    void*          features;

    // Write: cola de alertas del componente host (puntero opaco)
    void*          alert_queue;

    // Write: sugerencia de amenaza del plugin
    //   0 = benign, 1 = suspicious, 2 = malicious
    // La DECISIÓN FINAL de bloqueo pertenece siempre al core (ADR-012).
    int            threat_hint;
} PacketContext;

// ----------------------------------------------------------------------------
// PluginResult — código de retorno de las funciones del plugin
// ----------------------------------------------------------------------------
typedef enum {
    PLUGIN_OK    = 0,  // operación correcta
    PLUGIN_ERROR = 1,  // error — el loader incrementa error counter
    PLUGIN_SKIP  = 2   // plugin decide no procesar este paquete/evento
} PluginResult;

// ----------------------------------------------------------------------------
// Símbolos OBLIGATORIOS que todo plugin debe exportar
// ----------------------------------------------------------------------------

// Identificación
const char*  plugin_name();           // ej: "hello"
const char*  plugin_version();        // ej: "0.1.0"
int          plugin_api_version();    // debe retornar PLUGIN_API_VERSION

// Ciclo de vida
PluginResult plugin_init(const PluginConfig* config);
PluginResult plugin_process_packet(PacketContext* ctx);
void         plugin_shutdown();

#ifdef __cplusplus
}
#endif

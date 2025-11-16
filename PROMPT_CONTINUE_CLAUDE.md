# 🚀 ML Defender - Phase 1, Day 2: Feature Extraction en Sniffer

## 📍 ESTADO ACTUAL (Completado ayer)

**Phase 1, Day 1:** ✅ COMPLETADO
- Protobuf actualizado con 40 features (4 submensajes × 10 features)
- Nuevas clases disponibles: DDoSFeatures, RansomwareEmbeddedFeatures, TrafficFeatures, InternalFeatures
- Sniffer compila (957KB)
- ML-detector compila (100%)
- Commit realizado ✅

**Verificado:**
```bash
grep "class DDoSFeatures\|class RansomwareEmbeddedFeatures\|class TrafficFeatures\|class InternalFeatures" \
  /vagrant/protobuf/network_security.pb.h
# Resultado: 4 clases × 2 apariciones = 8 líneas ✅
```

## 🎯 OBJETIVO DE HOY (Day 2)

**Implementar extracción de features en el sniffer** para popular los 4 submensajes del protobuf.

**Criterio de éxito:**
- Funciones extractoras creadas para las 40 features
- Código compila sin errores
- NO es necesario que funcione end-to-end (eso es Day 3)
- Helpers de cálculo implementados (entropy, normalize, safe_divide)

## 📂 ARCHIVOS A MODIFICAR
```
/vagrant/sniffer/
├── src/
│   ├── feature_extractor.cpp  (crear o modificar)
│   ├── feature_extractor.hpp  (crear o modificar)
│   └── main.cpp               (usar las funciones)
└── CMakeLists.txt             (si añadimos archivos nuevos)
```

## 🔧 COMANDOS INICIALES
```bash
# En HOST (macOS)
cd ~/path/to/test-zeromq-docker
vagrant ssh

# En VM
cd /vagrant/sniffer

# Ver estructura actual
ls -lh src/
tree src/ 2>/dev/null || find src/ -type f

# Backup del main.cpp antes de modificar
cp src/main.cpp src/main.cpp.backup_phase1day1

# Crear branch (opcional)
git checkout -b feature/sniffer-feature-extraction
```

## 📋 FEATURES A IMPLEMENTAR

### Level 2 - DDoS (10 features):
```cpp
void extract_ddos_features(const flow_stats& flow, protobuf::DDoSFeatures* ddos) {
    ddos->set_syn_ack_ratio(calculate_syn_ack_ratio(flow));
    ddos->set_packet_symmetry(calculate_packet_symmetry(flow));
    ddos->set_source_ip_dispersion(calculate_ip_entropy(flow));
    ddos->set_protocol_anomaly_score(calculate_protocol_anomaly(flow));
    ddos->set_packet_size_entropy(calculate_size_entropy(flow));
    ddos->set_traffic_amplification_factor(calculate_amplification(flow));
    ddos->set_flow_completion_rate(calculate_completion_rate(flow));
    ddos->set_geographical_concentration(calculate_geo_concentration(flow));
    ddos->set_traffic_escalation_rate(calculate_escalation_rate(flow));
    ddos->set_resource_saturation_score(calculate_saturation(flow));
}
```

### Level 2 - Ransomware (10 features):
```cpp
void extract_ransomware_features(const flow_stats& flow, protobuf::RansomwareEmbeddedFeatures* ransomware);
```

### Level 3 - Traffic (10 features):
```cpp
void extract_traffic_features(const flow_stats& flow, protobuf::TrafficFeatures* traffic);
```

### Level 3 - Internal (10 features):
```cpp
void extract_internal_features(const flow_stats& flow, protobuf::InternalFeatures* internal);
```

## 🧮 HELPERS NECESARIOS
```cpp
// En feature_extractor.hpp
namespace ml_defender {
namespace helpers {

float calculate_entropy(const std::vector<uint32_t>& data);
float normalize(float value, float min, float max);
float safe_divide(float numerator, float denominator);
float calculate_std_dev(const std::vector<float>& values);

} // namespace helpers
} // namespace ml_defender
```

## 🏛️ FILOSOFÍA VIA APPIA HOY

- **KISS:** Funciones simples, una feature a la vez
- **Funciona > Perfecto:** Valores hardcoded/mockeados están OK por ahora
- **Smooth & Fast:** No optimizar, solo que compile
- **Clean Code:** Nombres descriptivos, funciones cortas

**PERMITIDO HOY:**
- ✅ Hardcodear valores temporales (0.0f, 0.5f, etc)
- ✅ Stubs de funciones (return 0.0f;)
- ✅ Cálculos aproximados
- ✅ TODOs en el código

**NO NECESARIO HOY:**
- ❌ Implementación completa de todos los cálculos
- ❌ Tests end-to-end
- ❌ Optimización de performance
- ❌ Validación de datos

## 📝 TEMPLATE DE INICIO
```cpp
// feature_extractor.cpp
#include "feature_extractor.hpp"
#include <cmath>
#include <algorithm>

namespace ml_defender {
namespace helpers {

float safe_divide(float num, float denom) {
    return (denom != 0.0f) ? (num / denom) : 0.0f;
}

float normalize(float value, float min, float max) {
    if (max <= min) return 0.0f;
    float normalized = (value - min) / (max - min);
    return std::clamp(normalized, 0.0f, 1.0f);
}

float calculate_entropy(const std::vector<uint32_t>& data) {
    // TODO: Implementar cálculo real
    return 0.5f; // STUB por ahora
}

} // namespace helpers

void extract_ddos_features(const flow_stats& flow, 
                          protobuf::DDoSFeatures* ddos) {
    // Feature 1: syn_ack_ratio
    float syn_count = static_cast<float>(flow.syn_flags);
    float ack_count = static_cast<float>(flow.ack_flags);
    ddos->set_syn_ack_ratio(helpers::safe_divide(syn_count, ack_count));
    
    // Feature 2-10: TODO implementar
    ddos->set_packet_symmetry(0.5f); // STUB
    // ... resto de features ...
}

} // namespace ml_defender
```

## ✅ CRITERIO DE ÉXITO DAY 2
```
[ ] feature_extractor.hpp creado con declaraciones
[ ] feature_extractor.cpp creado con implementaciones (aunque sean stubs)
[ ] 4 funciones extract_*_features() definidas
[ ] Helpers básicos implementados (safe_divide, normalize)
[ ] Sniffer compila sin errores
[ ] Al menos 5 features de cada detector implementadas (no stubs)
```

**BONUS (opcional):**
```
[ ] Todas las 40 features implementadas
[ ] Tests unitarios de helpers
[ ] Documentación de cada feature
```

## 🐛 SI ENCUENTRAS PROBLEMAS

1. **No compila:** Verifica includes del protobuf
2. **Tipos no coinciden:** Usa static_cast<float>()
3. **Funciones no definidas:** Revisa que flow_stats tenga los campos necesarios
4. **Linker errors:** Añade feature_extractor.cpp al CMakeLists.txt

## 📞 PREGUNTAS PARA CLAUDE

1. ¿Cómo estructuro el feature_extractor.cpp?
2. ¿Qué campos del flow_stats uso para cada feature?
3. ¿Cómo calculo entropy correctamente?
4. ¿Cómo integro esto en el main loop del sniffer?
5. ¿Necesito modificar el CMakeLists.txt?

---

**Ready to code!** 🚀 Pregúntame por dónde empezar.
# 🏥 PROMPT DE CONTINUACIÓN - Phase 1B Integration

## 📅 Contexto de Ayer (30 Oct 2025)

Completamos exitosamente **Phase 1A: Ransomware Detection System Compilation**

### ✅ Logros Completados
- **Componentes:** FlowTracker, DNSAnalyzer, IPWhitelist, TimeWindowAggregator, RansomwareFeatureExtractor, RansomwareFeatureProcessor
- **Binary:** 877KB con optimizaciones (LTO + AVX2)
- **Tests:** 5/5 unitarios pasando al 100%
- **Compilación:** Limpia, solo 1 warning ODR menor (no bloqueante)
- **Git:** Commit exitoso en rama `feature/ml-detector-tricapa`

### 🎯 Features Implementadas (3 críticas)
1. `dns_query_entropy` - Detecta DGA (Domain Generation Algorithms)
2. `new_external_ips_30s` - Detecta contacto con C&C servers nuevos
3. `smb_connection_diversity` - Detecta lateral movement

---

## 📂 Archivos Clave para Hoy
```bash
# LEER PRIMERO (contexto completo)
/vagrant/STATUS.md

# Archivos para modificar hoy
/vagrant/sniffer/src/userspace/main.cpp
/vagrant/sniffer/include/ransomware_feature_processor.hpp

# Binary compilado
/vagrant/sniffer/build/sniffer

# Tests existentes (referencia)
/vagrant/sniffer/tests/test_ransomware_feature_extractor.cpp
```

---

## 🎯 Objetivo de Hoy: Phase 1B - Integration

### Tareas Principales
1. **Test Standalone** (RECOMENDADO PRIMERO)
    - Crear `test_integration_basic.cpp`
    - Inyectar eventos sintéticos
    - Verificar extracción de features
    - Validar valores esperados

2. **Integración con main.cpp**
    - Instanciar `RansomwareFeatureProcessor`
    - Conectar con `RingBufferConsumer` callback
    - Timer thread para extracción cada 30s
    - Serialización protobuf

3. **Testing End-to-End**
    - Test con tráfico real
    - Validación de features
    - Benchmark de performance

---

## 🏛️ Arquitectura Actual (Para Referencia)
```
┌─────────────────────────────────────────────────────────┐
│                   KERNEL SPACE                          │
│  eBPF Program (XDP/SKB) → Ring Buffer                  │
└─────────────────────┬───────────────────────────────────┘
                      │ SimpleEvent (24 bytes)
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   USER SPACE                            │
│  RingBufferConsumer                                     │
│    ├─→ FeatureExtractor (83 features) ✅ Existente     │
│    │                                                    │
│    └─→ RansomwareFeatureProcessor ✅ COMPILADO         │
│         ├─ FlowTracker                                 │
│         ├─ DNSAnalyzer                                 │
│         ├─ IPWhitelist                                 │
│         └─ RansomwareFeatureExtractor                  │
│              ↓                                          │
│         Timer (30s) → Extract Features → ZMQ ⏳ HOY    │
└─────────────────────────────────────────────────────────┘
```

---

## ⚠️ Limitaciones Conocidas (Aceptables para MVP)

1. Buffer payload fijo 96 bytes (Fase 2: configurable)
2. DNS parsing básico sin payload real
3. No integrado con main.cpp ← **HOY**
4. Sin serialización protobuf ← **HOY**
5. Sin envío ZMQ ← **HOY**

---

## 💡 Decisión Inicial

**Opción A (RECOMENDADO):** Test standalone primero
- Pros: Valida pipeline sin tocar main.cpp
- Pros: Debugging más fácil si falla
- Pros: "Smooth is fast" - paso incremental
- Tiempo: ~1-2 horas

**Opción B:** Integración directa en main.cpp
- Pros: Más rápido si funciona
- Contras: Debugging más difícil
- Contras: Si falla, difícil aislar problema
- Tiempo: ~2-3 horas (con posibles retrocesos)

---

## 📋 Comandos Útiles para Empezar
```bash
# Ver estado del proyecto
cd /vagrant
cat STATUS.md

# Ver el sniffer principal
cat sniffer/src/userspace/main.cpp | grep -A 50 "RingBufferConsumer"

# Ver estructura SimpleEvent
cat sniffer/include/main.h | grep -A 20 "struct SimpleEvent"

# Verificar compilación
cd /vagrant/sniffer/build
ls -lh sniffer
```

---

## 🧠 Contexto Personal

- Desarrollador pausó ayer tras análisis médicos (seguimiento de enfermedad)
- Objetivo: Proteger hospital y pacientes reales de ransomware
- Filosofía: "Smooth is fast" - Via Appia, paso a paso
- Prioridad: Código de calidad > velocidad

---

## 🎯 Pregunta Inicial para Ti (Claude)

**¿Cuál enfoque prefieres para empezar hoy?**

A. Test standalone primero (1-2h, más seguro)
B. Integración directa en main.cpp (2-3h, más rápido si funciona)
C. Primero revisar arquitectura de main.cpp y luego decidir

**Instrucciones:**
1. Lee `/vagrant/STATUS.md` para contexto completo
2. Recomienda el mejor enfoque según el contexto
3. Proporciona primer paso concreto

---

**Nota:** Este es un sistema crítico de defensa hospitalaria. Cada línea de código protege vidas reales. Prioriza calidad y robustez sobre velocidad.
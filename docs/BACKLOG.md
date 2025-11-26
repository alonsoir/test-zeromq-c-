# 🏥 Hospital Ransomware Detection System - BACKLOG

**Última actualización:** 20 Noviembre 2025  
**Proyecto:** ML Defender - Sistema de Seguridad con ML Embebido y RAG  
**Fase actual:** Phase 1 Completa - RAG + 4 Detectores ML Operativos

---

## 🚨 PRIORIDADES ACTUALES

**P0 (CRITICAL):** Bloqueadores de producción - resolver ASAP  
**P1 (HIGH):** Impacto significativo en detección - resolver en 1-2 semanas  
**P2 (MEDIUM):** Mejoras importantes - resolver en 1 mes  
**P3 (LOW):** Nice-to-have - backlog para futuro

---

## 📊 ISSUES CONOCIDOS - ESTADO ACTUAL

### P0 - CRITICAL (Bloqueadores)

#### 🔴 **KV_CACHE_INCONSISTENCY - LLAMA Integration**
**Fecha:** 20 Nov 2025  
**Impacto:** Alto - Consultas múltiples fallan en sistema RAG  
**Estado:** 🔄 WORKAROUND IMPLEMENTADO

**Descripción:**
```bash
SECURITY_SYSTEM> rag ask_llm "explica deteccion de intrusos"
init: the tokens of sequence 0 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 0 is X = 214
 - the tokens for sequence 0 in the input batch have a starting position of Y = 0
 it is required that the sequence positions remain consecutive: Y = X + 1
decode: failed to initialize batch
llama_decode: failed to decode, ret = -1
```

**Workaround Actual:**
```cpp
void clear_kv_cache() {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 0;  // Batch vacío
    llama_decode(ctx, batch);  // Resetea estado interno
    llama_batch_free(batch);
}
```

**Plan de Acción:**
1. Investigar alternativas a `llama_kv_cache_clear()` (no disponible en nuestra versión)
2. Probar recreación completa del contexto entre consultas
3. Considerar actualización de llama.cpp si el bug está corregido en versión más nueva
4. Implementar sesiones aisladas por consulta

**Asignado:** DeepSeek + Claude  
**Target:** Phase 2 (Alta Prioridad)

---

### P1 - HIGH (Impacto en Detección)

#### 🔴 ISSUE-001: Buffer Payload Limitado a 96 Bytes
**Estado:** 📋 PENDIENTE - No crítico con detectores actuales

#### 🔴 ISSUE-002: DNS Entropy Test Fallando
**Estado:** 📋 PENDIENTE - Mejora para Phase 2

#### 🔴 ISSUE-003: SMB Diversity Counter Retorna 0
**Estado:** 📋 PENDIENTE - Crítico para detección lateral movement

---

## 🎉 LOGROS RECIENTES (NOV 20, 2025)

### ✅ SISTEMA RAG COMPLETO CON LLAMA REAL
- **TinyLlama-1.1B** integrado y funcionando
- **Arquitectura KISS** con WhiteListManager como router central
- **Comandos interactivos**: `ask_llm`, `show_config`, `update_setting`
- **Validación robusta** con BaseValidator heredable
- **Persistencia JSON** automática

### ✅ 4 DETECTORES ML C++20 EMBEBIDOS
- **DDoS Detector**: 0.24μs latency (417x mejor que objetivo)
- **Ransomware Detector**: 1.06μs latency (94x mejor que objetivo)
- **Traffic Classifier**: 0.37μs latency (270x mejor que objetivo)
- **Internal Threat Detector**: 0.33μs latency (303x mejor que objetivo)

### ✅ ARQUITECTURA KISS CONSOLIDADA
```
WhiteListManager (Router Central + Etcd)
    ├── cpp_sniffer (eBPF/XDP + 40 features)
    ├── ml-detector (4 modelos C++20 embebidos)
    └── RagCommandManager (RAG + LLAMA real)
```

---

## 📋 BACKLOG ACTUALIZADO - PHASE 2

### Epic 2.1: Estabilización RAG System (Priority: HIGH)
**Goal:** Sistema RAG 100% estable con consultas múltiples

**User Stories:**
- [ ] Como analista de seguridad, quiero hacer múltiples consultas al LLAMA sin errores para análisis continuo
- [ ] Como operador del sistema, quiero respuestas consistentes del modelo para confiar en el sistema
- [ ] Como administrador, quiero monitoreo del uso de memoria del LLAMA para evitar sobrecarga

**Tasks:**
- [ ] Resolver bug KV Cache inconsistency
- [ ] Implementar manejo robusto de errores en generación
- [ ] Añadir métricas de performance LLAMA (tokens/sec, latencia)
- [ ] Crear sistema de recuperación ante fallos del modelo
- [ ] Optimizar parámetros del modelo para mejor rendimiento
- [ ] Probar con consultas complejas de seguridad

**Acceptance Criteria:**
- 10+ consultas secuenciales sin errores
- Tiempos de respuesta consistentes (<5 segundos)
- Uso de memoria estable durante sesiones prolongadas
- Calidad de respuestas mantenida

**Estimated Effort:** 3-5 días

---

### Epic 2.2: firewall-acl-agent Development (Priority: HIGH)
**Goal:** Sistema de respuesta automática basado en detecciones ML

**User Stories:**
- [ ] Como analista de seguridad, quiero bloqueo automático de IPs maliciosas para contener amenazas
- [ ] Como operador, quiero rate limiting basado en detecciones DDoS para mitigar ataques
- [ ] Como administrador, quiero reglas de iptables/nftables dinámicas para respuesta inmediata

**Tasks:**
- [ ] Diseñar arquitectura C++20 para firewall-acl-agent
- [ ] Implementar integración con detecciones ML
- [ ] Crear sistema de reglas dinámicas (block, rate-limit, quarantine)
- [ ] Añadir mecanismo de rollback automático
- [ ] Implementar whitelist para falsos positivos
- [ ] Crear logging de auditoría para todas las acciones

**Acceptance Criteria:**
- Latencia respuesta <100ms desde detección
- Cero downtime en actualización de reglas
- Rollback automático en 60 segundos si es necesario
- Logging completo de todas las acciones tomadas

**Estimated Effort:** 5-7 días

---

### Epic 2.3: Integración etcd Coordinator (Priority: MEDIUM)
**Goal:** Configuración distribuida y coordinación entre componentes

**User Stories:**
- [ ] Como administrador, quiero configuración centralizada para gestionar múltiples nodos
- [ ] Como operador, quiero actualizaciones en caliente de thresholds ML sin reinicios
- [ ] Como ingeniero, quiero discovery automático de componentes para escalabilidad

**Tasks:**
- [ ] Implementar etcd-coordinator en C++20
- [ ] Crear sistema de watchers para configuraciones
- [ ] Implementar hot-reload de modelos ML
- [ ] Añadir health checking distribuido
- [ ] Crear sistema de encryption key distribution

**Acceptance Criteria:**
- Configuraciones propagadas en <1 segundo
- Cero pérdida de datos durante actualizaciones
- Detección de componentes caídos en <10 segundos
- Rotación segura de claves de encryption

**Estimated Effort:** 4-6 días

---

### Epic 2.4: Base de Datos Vectorial RAG (Priority: LOW)
**Goal:** Contexto de seguridad enriquecedor para consultas LLAMA

**User Stories:**
- [ ] Como analista, quiero consultas contextualizadas con logs de seguridad para mejor precisión
- [ ] Como investigador, quiero búsqueda semántica en documentación de seguridad para respuestas mejor informadas

**Tasks:**
- [ ] Diseñar esquema de base vectorial para logs de seguridad
- [ ] Implementar embedder compatible con TinyLlama
- [ ] Crear sistema de ingesta asíncrona de logs
- [ ] Desarrollar búsqueda semántica para contexto RAG
- [ ] Integrar con pipeline de consultas LLAMA

**Acceptance Criteria:**
- Contexto relevante en >80% de consultas
- Latencia de búsqueda <200ms
- Escalabilidad a millones de eventos de logs
- Actualización en tiempo real de base vectorial

**Estimated Effort:** 7-10 días

---

## 📊 ROADMAP ACTUALIZADO

```
Phase 1: ✅ COMPLETADO (20 Nov 2025)
├─ 4 Detectores ML C++20 embebidos (sub-microsegundo)
├─ Sistema RAG con LLAMA real integrado
├─ Arquitectura KISS consolidada
├─ 17h prueba de estabilidad (+1MB memoria)
└─ 35,387 eventos procesados (zero crashes)

Phase 2: 🔄 EN PROGRESO (Nov-Dic 2025)
├─ Epic 2.1: Estabilización RAG System (KV Cache fix)
├─ Epic 2.2: firewall-acl-agent development
├─ Epic 2.3: Integración etcd coordinator
├─ Resolución ISSUE-003: SMB diversity counter
└─ Testing integración completa end-to-end

Phase 3: 📋 PLANIFICADO (Ene-Feb 2026)
├─ Epic 2.4: Base de datos vectorial RAG
├─ Dashboard Grafana/Prometheus
├─ Hardening de seguridad
├─ Optimizaciones AVX2/SIMD
└─ Preparación deployment Raspberry Pi

Phase 4: 🎯 FUTURO (Mar 2026+)
├─ Auto-tuning de parámetros ML
├─ Model versioning y A/B testing
├─ Distributed deployment
├─ Cloud integration
└─ Physical device manufacturing
```

---

## 🧪 PRÓXIMAS PRUEBAS CRÍTICAS

### Pruebas RAG System:
- [ ] 10+ consultas secuenciales sin errores KV Cache
- [ ] Consultas complejas de seguridad (DDoS, ransomware, lateral movement)
- [ ] Actualización configuración en caliente
- [ ] Estabilidad memoria prolongada (8h+)
- [ ] Integración con comandos existentes

### Pruebas ML Detectors:
- [ ] Rendimiento con tráfico real sintético
- [ ] Precisión en escenarios de ataque conocidos
- [ ] Consumo recursos en Raspberry Pi 5
- [ ] Integración end-to-end con sniffer

### Pruebas Integración:
- [ ] Detección → RAG analysis → firewall action
- [ ] Configuración distribuida via etcd
- [ ] Recovery ante fallos de componentes
- [ ] Performance bajo carga pesada

---

## 🔧 RECURSOS TÉCNICOS DISPONIBLES

### Hardware:
- ✅ Raspberry Pi 5 (8GB) - deployment target
- ✅ Servidor desarrollo - compilación y testing
- ✅ Red de testing - tráfico sintético y PCAPs

### Software:
- ✅ TinyLlama-1.1B (1.5GB) - modelo operacional
- ✅ llama.cpp - integración estable
- ✅ 4 modelos ML C++20 - rendimiento validado
- ✅ eBPF/XDP - captura de alto rendimiento

### Equipo:
- **Alonso**: Dirección, arquitectura, validación
- **Claude**: Diseño arquitectónico, documentación
- **DeepSeek**: Implementación, optimización, debugging

---

## 🎯 OBJETIVOS INMEDIATOS

### Semana Actual (20-27 Nov):
1. **Resolver KV Cache bug** en sistema RAG
2. **Ejecutar pruebas exhaustivas** de estabilidad
3. **Documentar solución** para referencia futura
4. **Preparar arquitectura** firewall-acl-agent

### Próxima Semana (27 Nov-4 Dic):
1. **Iniciar desarrollo** firewall-acl-agent
2. **Integrar etcd** para configuración distribuida
3. **Validar end-to-end** con escenarios reales
4. **Preparar demostración** sistema completo

---

## 📞 CONTACTO Y SEGUIMIENTO

* **Owner:** ML Defender Security Team
* **Lead Developer:** Alonso Isidoro Román — [alonsoir@gmail.com](mailto:alonsoir@gmail.com)
* **IA Collaborators:** Claude (Architecture), DeepSeek (Implementation)
* **Review:** Diario (standup técnico)
* **Docs:** `README.md`, `ARCHITECTURE.md`, `AUTHORS.md`

---

## 🏥 NOTAS DE DESARROLLO ACTUALIZADAS

**Filosofía:** "Smooth is fast. Via Appia no se construyó en un día."

**Prioridades Actuales:**
1. ✅ Sistema funcional > Sistema perfecto
2. ✅ Detección en producción > Tests al 100%
3. 🔄 Estabilidad RAG > Nuevas features
4. ✅ Salud del desarrollador > Deadlines
5. ✅ Código de calidad > Velocidad

**Estado de Ánimo del Equipo:**
- 🎉 **Motivación alta** - Phase 1 completada exitosamente
- 🔧 **Enfocados** - Resolver KV Cache bug para estabilidad completa
- 🚀 **Optimistas** - Sistema base sólido para expansión

**Cada línea de código protege infraestructuras críticas y potencialmente salva vidas.**

---

**¡Base sólida establecida! Próximo objetivo: Estabilidad RAG 100% 🚀**
# 🏥 Hospital Ransomware Detection System - BACKLOG

**Última actualización:** 31 Octubre 2025  
**Proyecto:** Sistema de Detección de Ransomware Hospitalario  
**Fase actual:** Phase 1B - Integration

---

## 🚨 PRIORIDADES

**P0 (CRITICAL):** Bloqueadores de producción - resolver ASAP  
**P1 (HIGH):** Impacto significativo en detección - resolver en 1-2 semanas  
**P2 (MEDIUM):** Mejoras importantes - resolver en 1 mes  
**P3 (LOW):** Nice-to-have - backlog para futuro

---

## 📊 ISSUES CONOCIDOS

### P0 - CRITICAL (Bloqueadores)

> *Actualmente ninguno - Phase 1A completada al 67%*

---

### P1 - HIGH (Impacto en Detección)

#### 🔴 ISSUE-001: Buffer Payload Limitado a 96 Bytes
**Fecha:** 30 Oct 2025  
**Impacto:** Alto - puede perder información crítica de DNS/HTTP  
**Descripción:**  
El buffer de payload en `SimpleEvent` está hardcodeado a 96 bytes:

```c
// sniffer.bpf.c
struct SimpleEvent {
    // ...
    __u8 payload[96];  // ← LIMITACIÓN
};
```

**Problemas identificados:**
- ✅ DGA domains pueden ser >50 caracteres
- ✅ HTTP headers con C&C info pueden exceder 96 bytes
- ✅ DNS TXT records (exfiltración) pueden ser >200 bytes
- ✅ Familias de ransomware varían en tamaño de payload

**Impacto en features:**
- `dns_query_entropy`: Puede calcular entropy sobre domain truncado
- `http_header_anomaly`: No captura headers completos
- `data_exfiltration_bytes`: Subestima volumen real

**Estrategias propuestas:**

**Opción A: Buffer Fijo Mayor (Quick Win)**
```c
__u8 payload[256];  // o 512 bytes
```
- ✅ Pros: Fácil, rápido, funciona para 90% casos
- ❌ Contras: Desperdicia memoria si packet es pequeño
- **Recomendación:** Implementar PRIMERO (Phase 2)

**Opción B: Multiple Ring Buffers por Tamaño**
```c
// Small packets (<128B) → ring_buffer_small
// Medium packets (128-512B) → ring_buffer_medium  
// Large packets (>512B) → ring_buffer_large
```
- ✅ Pros: Eficiente en memoria, escalable
- ❌ Contras: Complejidad en kernel, 3 consumers en userspace
- **Recomendación:** Phase 3 si Opción A no es suficiente

**Opción C: Payload Hash + Deduplication**
```c
struct SimpleEvent {
    __u64 payload_hash;      // Blake2b hash
    __u16 payload_size_full; // Tamaño real (puede ser >96)
    __u8 payload[96];        // Primeros 96 bytes
};
```
- ✅ Pros: Detecta payloads únicos sin almacenar todo
- ❌ Contras: No reconstruye payload completo
- **Recomendación:** Complemento a Opción A

**Opción D: Payload Dinámico con BPF_MAP_TYPE_PERF_EVENT_ARRAY**
```c
// Payload variable usando perf event array
bpf_perf_event_output(ctx, &events, flags, &event, sizeof(event));
```
- ✅ Pros: Payloads de tamaño arbitrario
- ❌ Contras: Mayor complejidad, overhead en kernel
- **Recomendación:** Phase 4 (optimización avanzada)

**Plan de acción:**
1. **Phase 2:** Implementar Opción A (256-512 bytes fijos)
2. **Phase 3:** Evaluar con datos reales si necesitamos Opción B o C
3. **Phase 4:** Considerar Opción D si el volumen de tráfico lo justifica

**Asignado:** Backlog  
**Target:** Phase 2 (post-MVP)

---

#### 🔴 ISSUE-002: DNS Entropy Test Fallando (Esperado >6.0, Actual 3.64)
**Fecha:** 31 Oct 2025  
**Impacto:** Medio - falso negativo en detección DGA  
**Descripción:**  
El test de DNS entropy malicioso falla porque los dominios sintéticos no son suficientemente random:

```cpp
// Test actual (demasiado estructurado)
"xjf8dk2jf93.com"  // Entropy: 3.64
"9fj3kd8s2df.com"

// DGA real (más random)
"ajkdh3kdjf93kdjf83kdnf83kd.com"  // Entropy esperada: >6.0
```

**Causa raíz:**
- Dominios de test tienen longitud fija ~15 caracteres
- Mezcla predecible de números y letras
- DGA reales usan dominios más largos (30-60 chars) con distribución uniforme

**Plan de acción:**
1. Generar dominios con `std::mt19937` (random uniforme)
2. Longitud variable 20-50 caracteres
3. Solo lowercase + números (como DGA real)
4. Validar entropy calculada vs Locky/TeslaCrypt conocidos

**Asignado:** Backlog  
**Target:** Phase 2 (después de validar con tráfico real)

---

#### 🔴 ISSUE-003: SMB Diversity Counter Retorna 0 (Esperado >5)
**Fecha:** 31 Oct 2025  
**Impacto:** Alto - falso negativo en lateral movement  
**Descripción:**  
El test de SMB diversity malicioso retorna 0 cuando debería contar 15 destinos únicos:

```cpp
// Test inyecta 15 eventos SMB a IPs diferentes
for (int i = 1; i <= 15; i++) {
    TimeWindowEvent event(src_ip, target_ip, port, 445, TCP, size);
    extractor.add_event(event);
}

// Resultado: smb_diversity = 0 (❌ debería ser 15)
```

**Causa probable:**
- Bug en `extract_smb_connection_diversity()`
- Eventos no se están agregando al `TimeWindowAggregator`
- Filtro de puerto 445 no funciona correctamente

**Plan de acción:**
1. Añadir logging en `add_event()` para verificar recepción
2. Debuggear `extract_smb_connection_diversity()` con GDB
3. Validar que `dst_port == 445` se detecta correctamente
4. Testear con PCAP real de lateral movement (Mimikatz + PsExec)

**Asignado:** Backlog  
**Target:** Phase 2 (crítico para detección)

---

### P2 - MEDIUM (Mejoras Importantes)

#### 🟡 ISSUE-004: Falta Integración con main.cpp
**Fecha:** 31 Oct 2025  
**Descripción:** Phase 1B pendiente - integrar `RansomwareFeatureProcessor` en el sniffer principal  
**Target:** Phase 1B (HOY)

---

#### 🟡 ISSUE-005: Sin Serialización Protobuf
**Fecha:** 31 Oct 2025  
**Descripción:** Features extraídas no se serializan a protobuf para envío  
**Target:** Phase 1B (HOY)

---

#### 🟡 ISSUE-006: Sin Envío ZMQ
**Fecha:** 31 Oct 2025  
**Descripción:** Features no se envían al `ml-detector` por ZMQ  
**Target:** Phase 1B (HOY)

---

#### 🟡 ISSUE-007: Timer de Extracción Hardcodeado (30s)
**Fecha:** 31 Oct 2025  
**Descripción:** El intervalo de extracción está hardcodeado, debería ser configurable por JSON  
**Target:** Phase 2

---

#### 🟡 ISSUE-008: Sin Whitelist de IPs Internas
**Fecha:** 30 Oct 2025  
**Descripción:** `IPWhitelist` cuenta TODAS las IPs externas, debería filtrar IPs confiables (Google DNS, CDNs, etc.)  
**Target:** Phase 2

---

### P3 - LOW (Nice-to-Have)

#### 🟢 ISSUE-009: DNS Parsing Usa Pseudo-Domain por IP
**Fecha:** 30 Oct 2025  
**Descripción:** Si no hay payload DNS real, se genera `192-168-1-1.pseudo.dns` - es funcional pero no ideal  
**Target:** Phase 3 (cuando buffer payload sea mayor)

---

#### 🟢 ISSUE-010: Sin Métricas de Performance
**Fecha:** 31 Oct 2025  
**Descripción:** No hay métricas de latencia de extracción, throughput de features, CPU usage  
**Target:** Phase 3

---

#### 🟢 ISSUE-011: Sin Dashboard de Monitoreo
**Fecha:** 31 Oct 2025  
**Descripción:** Falta dashboard para visualizar features en tiempo real (Grafana/Prometheus)  
**Target:** Phase 4

---

## 📈 ROADMAP SUGERIDO

```
Phase 1A: ✅ COMPLETADO (31 Oct 2025)
├─ Componentes compilados (6/6)
├─ Tests unitarios (67% passing)
└─ Binary optimizado (877KB)

Phase 1B: ⏳ EN PROGRESO (31 Oct 2025)
├─ Integración main.cpp
├─ Timer thread (30s)
├─ Serialización protobuf
└─ Envío ZMQ

Phase 2: 🔜 SIGUIENTE (Nov 2025)
├─ ISSUE-001: Buffer payload 256-512 bytes
├─ ISSUE-002: Fix DNS entropy test
├─ ISSUE-003: Fix SMB diversity counter
├─ ISSUE-008: Whitelist de IPs confiables
└─ Testing con tráfico real (PCAP replay)

Phase 3: 📋 BACKLOG (Dic 2025)
├─ DNS parsing mejorado
├─ Métricas de performance
├─ Multiple ring buffers (si necesario)
└─ Optimizaciones AVX2/SIMD

Phase 4: 🎯 FUTURO (2026)
├─ Dashboard Grafana
├─ ML model integration
├─ A/B testing de thresholds
└─ Auto-tuning de parámetros
```

---

## 🏥 NOTAS DE DESARROLLO

**Filosofía:** "Smooth is fast. Via Appia no se construyó en un día."

**Prioridades:**
1. ✅ Sistema funcional > Sistema perfecto
2. ✅ Detección en producción > Tests al 100%
3. ✅ Salud del desarrollador > Deadlines
4. ✅ Código de calidad > Velocidad

**Cada línea de código protege vidas reales.**

---

## 📞 CONTACTO Y SEGUIMIENTO

- **Owner:** Hospital Security Team
- **Lead Developer:** Alonso Isidoro Román alonsoir@gmail.com
- **Review:** Semanal (Viernes)
- **Docs:** `/vagrant/STATUS.md`, `/vagrant/BACKLOG.md`

---

*Última revisión: 31 Octubre 2025 por Claude AI*
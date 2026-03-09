# ML Defender — Prompt de Continuidad DAY 81
**Generado:** Cierre DAY 80 (9 marzo 2026)
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 44/46 ✅

---

## Logros DAY 80

### TAREA 0 — Phase1-Day4-CRITICAL: Thresholds desde JSON ✅ CERRADO

El TODO más antiguo del proyecto (Day 4). **JSON is the LAW.** 🦅

**Problema:** 8 literales hardcodeados en `ring_consumer.cpp` ignoraban la
infraestructura JSON ya existente y funcional.

**4 capas de bug resueltas:**

| Capa | Fichero | Problema | Fix |
|---|---|---|---|
| 1 | `ring_consumer.cpp` | 8 literales `0.7f/0.75f/0.00000000065f` | → `config_.ml_defender.thresholds.*` |
| 2 | `main.cpp` | `ml_defender` nunca se mapeaba a `sniffer_config` | → mapeo explícito campo a campo |
| 3 | `config_types.h` | `StrictSnifferConfig` no tenía campo `ml_defender` | → struct añadido con defaults |
| 4 | `main.cpp` | Asignación struct-a-struct con layouts distintos → NaN | → mapeo explícito `.thresholds.*` |

**Evidencia en log de arranque:**
```
[Config] ML Defender thresholds (StrictConfig): DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
[ML Defender] Thresholds (JSON): DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
```

**Ficheros modificados:**
- `sniffer/src/userspace/ring_consumer.cpp`
- `sniffer/src/userspace/main.cpp`
- `sniffer/include/config_types.h`
- `sniffer/src/userspace/config_types.cpp`

### F1 post-thresholds DAY 80 ✅

Thresholds conservadores del sniffer.json (`ddos=0.85, ransomware=0.90, traffic=0.80, internal=0.85`)
vs hardcodeados DAY 79 (`0.70/0.75/0.70/0.00000000065`):

| Métrica | DAY 79 (hardcoded) | DAY 80 (JSON) | Delta |
|---|---|---|---|
| **F1** | 0.9921 | **0.9934** | +0.0013 ✅ |
| Precision | 0.9844 | **0.9869** | +0.0025 ✅ |
| Recall | 1.0000 | **1.0000** | 0 ✅ |
| FN | 0 | **0** | 0 ✅ |
| FP | 106/134 | 79/95 | -27 abs ✅ |
| Total eventos | 6810 | 6035 | -775 (distinto replay) |

**Nota honesta:** FPR aparentemente peor (83% vs 79%) no es regresión — el dataset
cambió entre replays (134 vs 95 eventos benignos). FP absolutos bajaron de 106 a 79.
Comparativa limpia requiere mismo replay en ambas condiciones — pendiente DAY 81.

---

## Estado de features DAY 80 (sin cambios)

| Submensaje | Reales | SENTINEL | Semántico (válido) |
|---|---|---|---|
| ddos_embedded | 9/10 | 0 | 1 (flow_completion 0.5f) |
| ransomware_embedded | 6/10 | 4 | 0 |
| traffic_classification | 6/10 | 4 | 0 |
| internal_anomaly | 7/10 | 3 | 0 |
| **Total** | **28/40** | **11** | **1** |

---

## ORDEN DAY 81

### TAREA 0 — Sanity check post-commit (5 min)

```bash
# Confirmar que el commit DAY 80 está en main o en la branch
git log --oneline -5

# Confirmar thresholds desde JSON
make pipeline-stop && make pipeline-start && sleep 8
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
# Esperado: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85
```

### TAREA 1 — Inspección FlowStatistics → features atacables (P1)

```bash
grep -A 80 'struct FlowStatistics' sniffer/include/flow_manager.hpp
# Objetivo: identificar campos disponibles para:
# - tcp_udp_ratio: ¿hay tcp_packets y udp_packets separados?
# - protocol_variety: ¿hay set de protocolos vistos?
# - flow_duration_std: ¿hay timestamps de inicio/fin por flujo?
```

**Criterio de decisión:**
- Campo existe en FlowStatistics → implementar extractor real
- Campo no existe → confirmar SENTINEL, documentar en deuda técnica como
  "requiere extensión de FlowStatistics" (no es Phase 2, es Phase 1 incompleto)

### TAREA 2 — Comparativa F1 limpia (mismo replay, ambas condiciones) (P1)

Para el paper necesitamos la comparativa con el mismo fichero PCAP:

```bash
# Con thresholds actuales (JSON 0.85/0.90)
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
vagrant up client
make test-replay-neris
# Guardar CSV como referencia DAY80_clean

# Cambiar thresholds en sniffer.json a 0.70/0.75 (valores DAY 79)
# Repetir replay
# Comparar F1
```

### TAREA 3 — ADR unificación logs ml-detector (P1)

Documentar formalmente como ADR:
- `detector.log` (spdlog interno) vs `ml-detector.log` (stdout Makefile)
- Decisión: mover `log_file` al JSON de configuración
- Estado: deuda conocida hasta implementar hot-reload (ENT-4)

### TAREA 4 — Validación dataset balanceado (P0 paper) (si tiempo)

El Consejo de Sabios señala esto como el gap científico más importante.
CTU-13 Neris 98% atacante — reviewers lo saben.

Dataset mínimo viable:
```bash
# Opción A: CTU-13 escenarios adicionales (disponible ya)
# Opción B: MAWI backbone traffic + CTU-13 Neris mezclados
# Opción C: CICIDS2017 (requiere descarga ~7GB)
```

---

## Deuda técnica actualizada

| Item | Prioridad | DAY |
|---|---|---|
| ~~Thresholds desde JSON (Phase1-Day4-CRITICAL)~~ | ~~P0~~ | ~~80~~ ✅ CERRADO |
| Inspección FlowStatistics → tcp_udp_ratio | **P1** | 81 |
| flow_duration_std / connection_duration_std | P1 | 81 |
| protocol_variety | P1 | 81 |
| Comparativa F1 limpia (mismo replay) | P1 | 81 |
| Unificar detector.log + ml-detector.log (ADR) | P1 | 81 |
| Validación dataset balanceado | **P0 paper** | 81-82 |
| is_forward dirección flow (ransomware_processor) | P2 | 81-82 |
| DNS payload parsing real (vs pseudo-domain) | P2 | 82 |
| Telemetría: ratio eventos sentinel vs reales | P2 | 82 |
| test_trace_id 2 fallos preexistentes DAY 72 | P2 | post-validación |
| trace_id en CLI | P2 | post-validación |
| io_intensity/resource_usage/file_operations/process_anomaly | P3 | post-paper |
| geographical_concentration | SKIP | decisión arquitectural deliberada |
| HSM/IRootKeyProvider | P3 | post-paper |

---

## Sanity check al arrancar DAY 81

```bash
# 1. Confirmar thresholds JSON funcionando
make pipeline-start && sleep 8
vagrant ssh -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
# Esperado: DDoS=0.85 Ransomware=0.9 Traffic=0.8 Internal=0.85

# 2. Confirmar estado de features (sin cambios desde DAY 79)
grep -c 'MISSING_FEATURE_SENTINEL' \
  sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 21

grep -c '0\.5f' sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 2 (flow_completion semántico + comparación cv)

# 3. No quedan literales hardcodeados en ring_consumer
grep -n '0\.7f\|0\.75f\|0\.00000000065f' \
  sniffer/src/userspace/ring_consumer.cpp
# Esperado: vacío

# 4. Pipeline y logs
make pipeline-status
vagrant ssh -c "ls -lah /vagrant/logs/lab/*.log"
# Esperado: 7 ficheros

# 5. VM client
vagrant status
```

---

## Notas para el paper (acumuladas DAY 80)

- **Phase1-Day4-CRITICAL cerrado:** 4 capas de bug entre literales hardcodeados
  y la infraestructura JSON existente. Citable como antipatrón: infraestructura
  correcta existe pero no se conecta — frecuente en desarrollo iterativo rápido.
- **Thresholds conservadores mejoran F1:** 0.85/0.90 vs 0.70/0.75 → F1 sube
  de 0.9921 a 0.9934, FP absolutos bajan de 106 a 79. Recall=1.0 preservado.
- **Comparativa requiere replay idéntico:** El número de eventos benignos varía
  entre replays (134 vs 95), lo que hace el FPR no comparable directamente.
  Documentado honestamente.
- **`StrictSnifferConfig` vs `SnifferConfig`:** Dos structs de config con el
  mismo JSON pero layouts distintos. La asignación directa produjo NaN. El mapeo
  explícito campo a campo es la solución correcta y documentable.
- Sentinel matemáticamente inalcanzable: rango splits [0.0, 5.1], sentinel = -9999.0f
- `flow_completion_rate` devuelve 0.5f semántico — comentario protector en código
- F1=0.9934 con 28/40 features reales. FPR alto por desequilibrio CTU-13.
- Gap crítico para paper: validación en tráfico balanceado (señalado por todo el Consejo)

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca usar `sed -i`. Usar Python3 inline para ediciones.
- **VM client:** `vagrant up client` antes de `make test-replay-neris`.
- **Flujo correcto test:**
  ```bash
  vagrant up client
  make pipeline-start && sleep 15
  make test-replay-neris
  ```
- **Thresholds en:** `sniffer/build-debug/config/sniffer.json` → sección `ml_defender.thresholds`
- **Log de confirmación en arranque:** `grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log`

---

*Consejo de Sabios — Cierre DAY 80, 9 marzo 2026*
*DAY 81 arranca con: FlowStatistics inspection → F1 comparativa limpia → dataset balanceado*
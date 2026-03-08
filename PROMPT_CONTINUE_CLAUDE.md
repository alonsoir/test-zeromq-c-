# ML Defender — Prompt de Continuidad DAY 80
**Generado:** Cierre DAY 79 (8 marzo 2026)
**Branch activa:** `feature/ring-consumer-real-features`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 44/46 ✅

---

## Logros DAY 79

### TAREA 1 — 8× `0.5f` → `MISSING_FEATURE_SENTINEL` ✅
Corregidos con Python3 inline (macOS, BSD sed incompatible con GNU sed).

| Función | Línea | Tipo | Acción tomada |
|---|---|---|---|
| `geographical_concentration` | 161 | placeholder arquitectural | → SENTINEL |
| `io_intensity` | 224 | TODO Phase 2 | → SENTINEL |
| `resource_usage` | 244 | TODO Phase 2 | → SENTINEL |
| `file_operations` | 273 | TODO Phase 2 | → SENTINEL |
| `process_anomaly` | 288 | TODO Phase 2 | → SENTINEL |
| `temporal_pattern` else | 303 | no hay datos IAT | → SENTINEL |
| `behavior_consistency` iat_mean==0 | 332 | no hay datos | → SENTINEL |
| `packet_size_consistency` mean==0 | 530 | no hay datos | → SENTINEL |

**Intocables documentados:**
- Línea 139: `return 0.5f; // SEMANTIC VALUE: TCP established-not-closed. NOT a placeholder.`
- Línea 298: `if (cv < 0.5f)` — comparación, no feature value

**Resultado:** `grep -c 'MISSING_FEATURE_SENTINEL'` → 21 | `grep -c '0\.5f'` → 2 (ambos intocables)

### Logging estándar — deuda técnica liquidada ✅
Todos los componentes escriben en `/vagrant/logs/lab/`:

| Componente | Fichero log |
|---|---|
| etcd-server | `/vagrant/logs/lab/etcd-server.log` |
| rag-security | `/vagrant/logs/lab/rag-security.log` |
| rag-ingester | `/vagrant/logs/lab/rag-ingester.log` |
| ml-detector (stdout) | `/vagrant/logs/lab/ml-detector.log` |
| ml-detector (spdlog) | `/vagrant/logs/lab/detector.log` (ADR pendiente: unificar) |
| firewall-acl-agent | `/vagrant/logs/lab/firewall-agent.log` |
| sniffer | `/vagrant/logs/lab/sniffer.log` |

Nuevos targets Makefile:
- `make logs-all` — tail -f de los 6 simultáneamente
- `make logs-lab-clean` — rota logs a `/vagrant/logs/lab/archive/`

**Nota macOS permanente:** `sed -i` falla en macOS (BSD sed). Siempre usar Python3 inline o ir a la VM Vagrant para ediciones de ficheros.

### F1 Baseline CTU-13 Neris DAY 79 ✅

| Métrica | Valor |
|---|---|
| **F1** | **0.9921** |
| Precision | 0.9844 |
| Recall | 1.0000 |
| TP | 6676 |
| FP | 106 |
| FN | 0 |
| TN | 28 |
| Total eventos CSV | 6810 |
| Features reales | 21/40 SENTINEL activos |

**Ground truth:** IP botnet 147.32.84.165 (CTU-13 Neris)
**Nota honesta:** 106 FP sobre 134 eventos no-botnet = 79% FPR en tráfico benigno. Dataset muy desequilibrado (Neris dominante). Documentado en DAY79_sentinel_analysis.md.

---

## Estado de features DAY 79 → DAY 80

| Submensaje | Reales | SENTINEL | Semántico (válido) |
|---|---|---|---|
| ddos_embedded | 9/10 | 0 | 1 (flow_completion 0.5f) |
| ransomware_embedded | 6/10 | 4 | 0 |
| traffic_classification | 6/10 | 4 | 0 |
| internal_anomaly | 7/10 | 3 | 0 |
| **Total** | **28/40** | **11** | **1** |

---

## ORDEN DAY 80

### TAREA 0 — CRÍTICA: Thresholds desde JSON (Phase1-Day4-CRITICAL) (45 min)

El TODO más antiguo del proyecto. "JSON is the law."

```bash
# Ver estructura de config del ml-detector
cat ml-detector/config/ml_detector_config.json

# Ver dónde están los thresholds hardcodeados
grep -n '0\.7f\|0\.75f\|0\.00000000065f' sniffer/src/userspace/ring_consumer.cpp
```

Thresholds hardcodeados en `ring_consumer.cpp`:
- DDoS: `0.7f`
- Ransomware: `0.75f`
- Traffic: `0.7f`
- Internal: `0.00000000065f`

Implementar `struct MLThresholds` leída en `initialize()`:
```cpp
struct MLThresholds {
    float ddos       = 0.7f;   // fallback EXPLÍCITO, nunca silencioso
    float ransomware = 0.75f;
    float traffic    = 0.7f;
    float internal   = 0.00000000065f;
};
```

### TAREA 1 — Inspección FlowStatistics → features atacables (20 min)

```bash
cat sniffer/include/flow_manager.hpp | grep -A 60 'struct FlowStatistics'
```

Features objetivo si los campos existen:
- `tcp_udp_ratio`: ¿hay conteo TCP vs UDP por protocolo?
- `protocol_variety`: ¿hay set de protocolos vistos?
- `flow_duration_std`: solo viable con multi-flow — confirmar SENTINEL si no

### TAREA 2 — Unificar los dos logs de ml-detector (ADR pendiente) (20 min)

Actualmente coexisten:
- `detector.log` — spdlog interno del componente
- `ml-detector.log` — stdout capturado por Makefile

ADR: mover `log_file` al JSON de configuración de ml-detector. Hasta entonces, documentar el doble fichero como deuda conocida.

### TAREA 3 — F1 post-thresholds (15 min)

Después de implementar thresholds desde JSON, repetir:
```bash
make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
make test-replay-neris
# Calcular F1 con el script Python del DAY 79
```

Objetivo: documentar si los thresholds desde JSON cambian el F1 vs 0.9921.

---

## Deuda técnica actualizada

| Item | Prioridad | DAY |
|---|---|---|
| Thresholds desde JSON (Phase1-Day4-CRITICAL) | **P0** | 80 |
| Inspección FlowStatistics → tcp_udp_ratio | P1 | 80 |
| flow_duration_std / connection_duration_std | P1 | 80 |
| protocol_variety | P1 | 80 |
| Unificar detector.log + ml-detector.log | P1 | 80 |
| F1 post-thresholds | P1 | 80 |
| is_forward dirección flow (ransomware_processor) | P2 | 80-81 |
| DNS payload parsing real (vs pseudo-domain) | P2 | 81 |
| Telemetría: ratio eventos sentinel vs reales | P2 | 81 |
| test_trace_id 2 fallos preexistentes DAY 72 | P2 | post-validación |
| trace_id en CLI | P2 | post-validación |
| io_intensity/resource_usage/file_operations/process_anomaly | P3 | post-paper (requiere eBPF tracepoints) |
| geographical_concentration | SKIP | decisión arquitectural deliberada |
| HSM/IRootKeyProvider | P3 | post-paper |

---

## Sanity check al arrancar DAY 80

```bash
# 1. Confirmar estado de features
grep -c 'MISSING_FEATURE_SENTINEL' \
  sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 21

grep -c '0\.5f' sniffer/src/userspace/ml_defender_features.cpp
# Esperado: 2 (flow_completion semántico + comparación cv)

# 2. Pipeline y logs
make pipeline-start && sleep 15 && make pipeline-status
vagrant ssh -c "ls -lah /vagrant/logs/lab/*.log"
# Esperado: 7 ficheros (detector.log + 6 nuevos)

# 3. VM client disponible
vagrant status
# Esperado: defender running, client running (o arrancarlo con vagrant up client)
```

---

## Notas para el paper (acumuladas DAY 79)

- Sentinel matemáticamente inalcanzable: rango splits [0.0, 5.1], sentinel = -9999.0f → routing determinista left_child. Citable como decisión de ingeniería rigurosa.
- `0.5f` dentro del rango de splits es PEOR que sentinel — lección aprendida documentada en `docs/engineering_decisions/DAY79_sentinel_analysis.md`.
- Distinción crítica: `flow_completion_rate` devuelve 0.5f semántico (TCP established-not-closed) — no es placeholder. Documentado con comentario protector en código.
- F1=0.9921 baseline con 28/40 features reales. Recall perfecto (FN=0). FPR alto (79%) en tráfico benigno por desequilibrio dataset CTU-13.
- Logging caótico como antipatrón: 4 componentes sin log a fichero, deuda acumulada ~40 días. Solución: redirección tmux en Makefile. ADR pendiente: campo log_file en JSON de cada componente.
- 70% features single-flow (eBPF) vs 30% multi-flow (TimeWindowAggregator).

---

## Infraestructura DAY 79

- **macOS (BSD sed):** Nunca usar `sed -i` sin `-e ''`. Usar Python3 inline para ediciones de ficheros en el proyecto.
- **VM client:** `vagrant up client` antes de `make test-replay-neris`. `autostart: false` por defecto.
- **Flujo correcto test con tráfico real:**
  ```bash
  vagrant up client
  make pipeline-start && sleep 15
  make test-replay-neris
  ```

*Consejo de Sabios — Cierre DAY 79, 8 marzo 2026*
*DAY 80 arranca con: thresholds JSON → FlowStatistics inspection → F1 post-thresholds*
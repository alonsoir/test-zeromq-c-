# ML Defender — Prompt de Continuidad DAY 85
**Generado:** Cierre DAY 84 (12 marzo 2026)
**Branch activa:** `main`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id **46/46** ✅

---

## Logros DAY 84

### TAREA 1 — ARCHITECTURE.md v5.1.0 ✅
- 815 líneas, 10 diagramas Mermaid, totalmente en inglés
- Resultados validados: F1=1.0000, FPR=0.0049%, latencia 0.24–1.06μs
- Executive Summary para CTO incluido
- PDF generado en `/mnt/user-data/outputs/ARCHITECTURE.pdf`
- Commit pendiente: `git add docs/ARCHITECTURE.md && git commit -m "docs: ARCHITECTURE.md v5.1.0"`

### TAREA 2 — Fix trace_id 46/46 ✅ CERRADO DAY 84
**Deuda preexistente desde DAY 72 — cerrada.**

**FAIL 1 (bug en test):** `ts_bucket0_start = 1000000ULL` no alineado al bucket de
`ransomware` (window=60000ms). `1000000 / 60000 = 16`, `1059999 / 60000 = 17` → buckets
distintos → fallo intermitente. Fix: `60000000ULL` (múltiplo exacto de 60000).

**FAIL 2 (bug en implementación):** En `generate_trace_id_with_metadata()`:
```cpp
// ANTES (incorrecto):
if (eff_src == "0.0.0.0" || eff_dst == "0.0.0.0") fallback = true;
// disparaba fallback=true para "0.0.0.0" real (IP válida de wildcard listener)

// DESPUÉS (correcto):
auto is_empty_or_ws = [](const std::string& s) -> bool {
    if (s.empty()) return true;
    for (unsigned char c : s) if (!std::isspace(c)) return false;
    return true;
};
if (is_empty_or_ws(src_ip)) fallback = true;
if (is_empty_or_ws(dst_ip)) fallback = true;
// inspecciona el INPUT original, no el resultado normalizado
```

**Ficheros modificados:**
- `rag-ingester/include/utils/trace_id_generator.hpp`
- `rag-ingester/tests/test_trace_id.cpp`

**Resultado:** `make test` → 100% passed, 0 failed. Suite completa limpia. ✅

### Decisiones estratégicas DAY 84
- **Producción:** bare-metal Linux, dual NIC, sin Vagrant. Kernel ≥ 5.8. Hardware ~150-200€.
- **Modelo de negocio:** open-core. MIT para hospitales/escuelas. Enterprise comercial.
  Tier gratuito enterprise para organizaciones que genuinamente no pueden pagar.
- **Paper:** arXiv como "Independent Researcher, Extremadura, Spain". No requiere afiliación
  universitaria. Endorser objetivo: Sebastian Garcia (CTU Prague, autor CTU-13).
- **Repo cleanup:** POST-paper. Freeze en `snapshot/day84-pre-cleanup` antes de limpiar.
- **Vagrant:** solo para reproducibilidad de experimentos, nunca en producción.
  Añadir `make test-replay-neris-full` que hace `vagrant up` automático.

---

## Estado del sistema (sin cambios desde DAY 83)

**Branch:** `main` — tag `v0.83.0-day83-main`
**Pipeline:** 6/6 RUNNING ✅
**F1:** 1.0000 (CTU-13 Neris, 19,135 flows, thresholds 0.85/0.90/0.80/0.85)
**FPR ML:** 0.0049% (2 FP / 40,467 flows bigFlows benigno)
**Reducción FP vs Fast Detector:** ~15,500x

**Features:** 28/40 reales | 11 sentinel (-9999.0f) | 1 semántico (0.5f TCP half-open)

---

## ORDEN DAY 85

### TAREA 0 — Sanity check (5 min)
```bash
vagrant status
git branch  # confirmar main
make test 2>&1 | grep -E '(tests passed|tests failed)'
```

### TAREA 1 — Commit DAY 84 completo (10 min)
```bash
# Fix trace_id ya aplicado en VM — commit pendiente en macOS tras vagrant halt
git add rag-ingester/include/utils/trace_id_generator.hpp
git add rag-ingester/tests/test_trace_id.cpp
git commit -m "fix: trace_id 46/46 — bucket alignment + fallback_applied semantics

FAIL 1 (test bug): ts_bucket0_start=1000000 not bucket-aligned for
window=60000ms. Fixed: 60000000ULL (exact multiple of window).

FAIL 2 (impl bug): fallback_applied triggered for real '0.0.0.0' input.
Fixed: is_empty_or_ws() checks original input, not normalized result.

Test suite: 44/46 → 46/46. Pre-existing since DAY 72 — closed DAY 84.

Co-authored-by: Claude (Anthropic) <claude@anthropic.com>"

# ARCHITECTURE.md si no se hizo DAY 84
git add docs/ARCHITECTURE.md
git commit -m "docs: ARCHITECTURE.md v5.1.0 — Consejo de Sabios review"

git push origin main
```

### TAREA 2 — Abstract arXiv (P0)
**Este es el objetivo principal de DAY 85.**

Datos disponibles:
- F1=1.0000 | Precision=1.0000 | Recall=1.0000 (CTU-13 Neris, 19,135 flows)
- F1=0.9976 con thresholds conservadores (comparativa DAY 81)
- FPR ML = 0.0049% (2 FP / 40,467 flows benignos)
- FPR Fast Detector = 76.8% — ML reduce FPs ~15,500x
- Latencia: DDoS 0.24μs | Ransomware 1.06μs | Traffic 0.37μs | Internal 0.33μs
- 6 componentes C++20, eBPF/XDP, embedded RandomForest
- Afiliación: "Independent Researcher, Extremadura, Spain"

Estructura propuesta del paper:
1. Abstract
2. Introduction — motivación (ransomware hospitalario), gap en literatura
3. Architecture — pipeline 6 componentes, dual-score design
4. Implementation — C++20, eBPF/XDP, embedded RandomForest
5. Evaluation — CTU-13 Neris F1=1.0000, bigFlows FPR=0.0049%
6. Limitations — 28/40 features, DEBT-FD-001, single-node
7. Future Work — PHASE2, Enterprise features
8. Conclusion
9. Acknowledgments — Consejo de Sabios

Venue: arXiv (cs.CR) → RAID 2026 / USENIX Security Fall 2026

### TAREA 3 — DNS payload parsing real (P2, si sobra tiempo)
Actualmente usa pseudo-domain. Implementar parsing real desde el paquete.

---

## Deuda técnica actualizada

| Item | Prioridad | Estado |
|---|---|---|
| Paper arXiv — redacción | **P0** | DAY 85+ |
| `make test-replay-neris-full` (Vagrant auto) | P2 | DAY 85-86 |
| DNS payload parsing real | P2 | DAY 85-86 |
| DEBT-FD-001: FastDetector Path A → JSON | P1-PHASE2 | post-paper |
| Ring Consumer 28/40 → 40/40 features | P2-PHASE2 | post-paper |
| ADR-007: Consenso AND firewall | P1-PHASE2 | post-paper |
| ~~Fix trace_id 2 fallos DAY 72~~ | ~~P1~~ | ✅ CERRADO DAY 84 |

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca `sed -i`. Usar Python3 inline **desde dentro de la VM** para evitar corrupción de escape sequences por SSH/heredoc.
- **JSON sniffer:** `sniffer/config/sniffer.json` (NO build-debug/)
- **JSON ml-detector:** `ml-detector/config/ml_detector_config.json`
- **level1_attack threshold:** 0.65
- **VM:** `defender` (no `server`)
- **CSV ml-detector:** `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
- **CSV firewall:** `/vagrant/logs/firewall_logs/firewall_blocks.csv`
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events N`
- **Fuente de verdad F1:** `docs/experiments/f1_replay_log.csv`
- **Flujo correcto test:**
```bash
  vagrant status
  make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
  vagrant ssh defender -c "grep 'Thresholds (JSON)' /vagrant/logs/lab/sniffer.log"
  vagrant up client  # solo si aborted
  make test-replay-neris
  python3 scripts/calculate_f1_neris.py /vagrant/logs/lab/sniffer.log --total-events 19135
```

---

*Consejo de Sabios — Cierre DAY 84, 12 marzo 2026*
*DAY 85 arranca con: commit DAY 84 + abstract arXiv*
*Test suite: 100% ✅ — sin deuda técnica bloqueante*
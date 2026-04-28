#!/bin/bash
# DAY 134 — Integración quirúrgica ADR-040 + ADR-041
# Ejecutar desde macOS en la raíz del repo.

# ─────────────────────────────────────────────────────────────────────────────
# BACKLOG.md
# ─────────────────────────────────────────────────────────────────────────────

vagrant ssh -- python3 << 'PYEOF'
BACKLOG = "/vagrant/docs/BACKLOG.md"
with open(BACKLOG, "r") as f:
    content = f.read()

# ── 1. DEUDA ABIERTA: añadir ADR-040 + ADR-041 después de DEBT-PROD-FALCO-RULES-EXTENDED-001
new_debts = """
---

### DEBT-ADR040-001 a 012 — ML Plugin Retraining Contract *(nuevas — DAY 134)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** post-FEDER (implementación Año 1)
**Origen:** ADR-040 v2 — Consejo 8/8 DAY 134 (17 enmiendas, aprobado unánime)

| ID | Descripción | Target |
|----|-------------|--------|
| DEBT-ADR040-001 | Golden set v1 (≥50K flows, 70/30, Parquet, SHA-256 embebido en plugin) | v1.0 — pre-FEDER si posible |
| DEBT-ADR040-002 | Verificar que ml-detector emite `confidence_score ∈ [0,1]` en salida ZeroMQ | v1.0 |
| DEBT-ADR040-003 | `walk_forward_split.py` — `--split-field timestamp_first_packet`, mín. 3 ventanas, KS drift | v1.1 |
| DEBT-ADR040-004 | `check_guardrails.py` — Recall −0.5pp / F1 −2pp / FPR +1pp / latencia p99 +10% → exit 1 | v1.1 |
| DEBT-ADR040-005 | Integrar guardrail en proceso de firma Ed25519 (ADR-025) — `prod-sign` invoca guardrail | v1.1 |
| DEBT-ADR040-006 | IPW + uncertainty sampling (P≈0.5) en rag-ingester, ratio adaptativo [3%-10%] por drift | v1.2 |
| DEBT-ADR040-007 | Interfaz web revisión exploración en rag-security — etiquetado manual del 5% (Año 1) | v1.2 |
| DEBT-ADR040-008 | Informe diversidad por ciclo: Shannon entropy, MITRE ATT&CK coverage %, novelty score | v1.2 |
| DEBT-ADR040-009 | Competición algoritmos: XGBoost vs CatBoost vs LightGBM vs RF (multicriterio, una vez) | pre-lock-in |
| DEBT-ADR040-010 | Dataset lineage en metadatos del plugin (hash dataset + golden set + git commits) | v1.1 |
| DEBT-ADR040-011 | Canary deployment: 5-10% tráfico 24h antes de 100% (manual Año 1, flota Año 2) | v1.2 |
| DEBT-ADR040-012 | `docs/GOLDEN-SET-REGISTRY.md` con hash v1 + proceso evolución controlada | v1.0 |

**Prerequisito crítico (enmienda Claude, Consejo DAY 134):** IPW no es implementable sin `confidence_score`. DEBT-ADR040-002 debe resolverse antes de DEBT-ADR040-006. Sin este dato, la Regla 4 queda en deuda técnica real, no solo en papel.

**Test de cierre DEBT-ADR040-004:** `make retrain-eval PLUGIN=candidate.ubj` → exit 1 ante regresión en cualquiera de los 4 umbrales. `make prod-sign` no ejecuta si guardrail falla.

---

### DEBT-ADR041-001 a 006 — Hardware Acceptance Metrics FEDER *(nuevas — DAY 134)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** pre-FEDER, deadline 22 sep 2026
**Origen:** ADR-041 — Consejo 8/8 DAY 134

| ID | Descripción | Estado |
|----|-------------|--------|
| DEBT-ADR041-001 | Subconjunto pcap CTU-13 benchmark versionado con SHA-256 (`ctu13-neris-benchmark.pcap`) | ⏳ PENDIENTE |
| DEBT-ADR041-002 | `make golden-set-eval ARCH=$(uname -m)` — exit 0 dentro de tolerancia, exit 1 regresión | ⏳ PENDIENTE (depende ADR-040) |
| DEBT-ADR041-003 | `make feder-demo` — suite completa desde VM fría, <30 min, sin trucos pregrabados | ⏳ PENDIENTE |
| DEBT-ADR041-004 | Compra hardware x86 (NUC/mini-PC ~300€, NIC con soporte XDP nativo) | ⏳ post-métricas definidas |
| DEBT-ADR041-005 | Compra Raspberry Pi 4/5 | ⏳ post-métricas definidas |
| DEBT-ADR041-006 | Primera ejecución protocolo completo en hardware físico | ⏳ post-compra hardware |

**Nota crítica hardware (DeepSeek):** Verificar driver NIC antes de comprar x86. XDP nativo requiere driver compatible (mlx5, i40e, ixgbe, virtio_net con XDP). Sin XDP nativo, Variant A degrada a generic-skb-mode y el delta científico A/B se distorsiona.

**Nota ARM (DeepSeek):** Temperatura máxima ≤75°C sin ventilador. Crítica para despliegue pasivo 24/7 en armarios hospitalarios.

**Tolerancias ML por arquitectura:**
- x86: TOLERANCE = 0.0000 (mismo resultado esperado)
- ARM: TOLERANCE = 0.0005 (±0.05% diferencias NEON vs AVX2)
"""

anchor = "### DEBT-PROD-FALCO-RULES-EXTENDED-001 *(nueva — DAY 133)*"
assert anchor in content, f"ANCHOR NOT FOUND: {anchor}"
insert_after = content.find(anchor)
# Find end of that block (next --- or next ###)
block_end = content.find("\n---\n\n## 🔵", insert_after)
if block_end == -1:
    block_end = content.find("\n\n## 🔵", insert_after)
content = content[:block_end] + "\n" + new_debts + content[block_end:]
print("✓ Deudas ADR-040 + ADR-041 añadidas en sección DEUDA ABIERTA")

# ── 2. Decisiones de diseño: añadir filas DAY 134 antes del cierre de la tabla
new_decisions = """| **Walk-forward obligatorio** | K-fold prohibido. Split sobre `timestamp_first_packet` ordenado. Mín. 3 ventanas. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Golden set inmutable** | ≥50K flows, SHA-256 embebido en plugin firmado. Evolución controlada con solapamiento 6 meses. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Guardrail asimétrico Ed25519** | Recall −0.5pp (más restrictivo). F1 −2pp. FPR +1pp. Latencia p99 +10%. Exit 1 = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |
| **IPW + uncertainty sampling** | 5% exploración (P≈0.5). Ratio adaptativo [3%-10%] por drift. Memory replay buffer. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Competición algoritmos pre-lock-in** | Multicriterio: Recall 40% + F1 25% + latencia 20% + tamaño 10% + carga 5%. Una sola vez. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Dataset lineage obligatorio** | Hash dataset + golden set + features_version + git commits en metadatos. Sin lineage = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |
| **Niveles despliegue FEDER** | Nivel 1 (RPi4/5, ≤50 usuarios) + Nivel 2 (x86, 50-200). Demo mínima: ambos simultáneos. | ADR-041 · Consejo 8/8 · DAY 134 |
| **Latencia end-to-end como métrica primaria** | Captura → alerta → iptables efectiva. Más relevante operacionalmente que latencia de detección. | ADR-041 · DeepSeek · DAY 134 |
| **Temperatura ARM como gate aceptación** | ≤75°C sin ventilador. Crítica para armarios hospitalarios 24/7. | ADR-041 · DeepSeek · DAY 134 |
| **Pipeline evaluación: híbrido A+B** | Scripts versionados en repo (local Vagrant/Makefile). CI GitHub Actions = mismo código, 2 entradas. | ADR-040 · Consejo 6/7 · DAY 134 |
"""
# Insert before last row of decisiones table (before "---" that follows the table)
decisions_anchor = "| **\"Fuzzing misses nothing\" ELIMINADA**"
assert decisions_anchor in content, f"ANCHOR NOT FOUND: {decisions_anchor}"
# Find end of that row
pos = content.find(decisions_anchor)
row_end = content.find("\n", pos) + 1
content = content[:row_end] + new_decisions + content[row_end:]
print("✓ Decisiones DAY 134 añadidas en tabla de decisiones")

# ── 3. Estado global: añadir nuevas deudas antes del cierre del bloque ```
new_status_lines = """ADR-040 ML Retraining Contract (definición):      100% ✅  DAY 134 (Consejo 8/8, 17 enmiendas)
ADR-041 HW Acceptance Metrics (definición):        100% ✅  DAY 134 (Consejo 8/8)
DEBT-ADR040-001 (golden set v1):                     0% ⏳  v1.0 — post-FEDER
DEBT-ADR040-002 (confidence_score ml-detector):      0% ⏳  v1.0
DEBT-ADR040-003 (walk_forward_split.py):             0% ⏳  v1.1
DEBT-ADR040-004 (check_guardrails.py):               0% ⏳  v1.1
DEBT-ADR040-005 (guardrail integrado Ed25519):       0% ⏳  v1.1
DEBT-ADR040-006 (IPW + uncertainty sampling):        0% ⏳  v1.2
DEBT-ADR040-007 (interfaz web exploración):          0% ⏳  v1.2 Año 1
DEBT-ADR040-008 (informe diversidad):                0% ⏳  v1.2
DEBT-ADR040-009 (competición algoritmos):            0% ⏳  pre-lock-in XGBoost
DEBT-ADR040-010 (dataset lineage):                   0% ⏳  v1.1
DEBT-ADR040-011 (canary deployment):                 0% ⏳  Año 2 flota
DEBT-ADR040-012 (GOLDEN-SET-REGISTRY.md):            0% ⏳  v1.0
DEBT-ADR041-001 (pcap CTU-13 benchmark versionado):  0% ⏳  pre-FEDER
DEBT-ADR041-002 (make golden-set-eval):              0% ⏳  depende ADR-040
DEBT-ADR041-003 (make feder-demo):                   0% ⏳  pre-FEDER
DEBT-ADR041-004 (compra hardware x86 NIC XDP):       0% ⏳  post-métricas
DEBT-ADR041-005 (compra Raspberry Pi 4/5):           0% ⏳  post-métricas
DEBT-ADR041-006 (ejecución en hardware físico):      0% ⏳  post-compra
"""

# Insert before the closing ``` of the status block
status_anchor = "ADR-031 aRGus-seL4:                       0% ⏳  branch independiente"
assert status_anchor in content, f"ANCHOR NOT FOUND: {status_anchor}"
pos = content.find(status_anchor)
row_end = content.find("\n", pos) + 1
content = content[:row_end] + new_status_lines + content[row_end:]
print("✓ Estado global actualizado con ADR-040 + ADR-041")

# ── 4. Consejo DAY 134: añadir nota antes del cierre del fichero
consejo_day134 = """
---

## 📝 Notas del Consejo de Sabios — DAY 134 (8/8)

> "DAY 134 — ADR-040 + ADR-041: contratos de calidad ML y métricas de aceptación hardware.
>
> ADR-040 — 17 enmiendas, aprobado 8/8:
> D1: Walk-forward obligatorio. K-fold prohibido en NDR temporal.
> D2: Golden set inmutable con SHA-256 embebido en plugin firmado (Gemini).
> D3: Guardrail asimétrico — Recall más restrictivo que F1 (correcto para infraestructura crítica).
> D4: IPW + uncertainty sampling (P≈0.5), no exploración aleatoria pura (Gemini).
> D5: Ratio exploración adaptativo [3%-10%] por drift detectado (ChatGPT-5).
> D6: Memory replay buffer como complemento al golden set (Grok).
> D7: Competición algoritmos multicriterio — XGBoost no asumido ganador a priori.
> D8: Dataset lineage obligatorio — prerequisito de firma Ed25519.
> D9: Canary 5-10% / 24h antes de despliegue completo (ChatGPT-5).
> D10: Pipeline evaluación híbrido — mismo código, dos entradas (local + CI).
>
> Enmienda crítica (Claude): confidence_score es prerequisito de IPW.
> Sin DEBT-ADR040-002 resuelto, Regla 4 no es implementable.
>
> ADR-041 — aprobado 8/8:
> D1: Niveles 1-2-3 con métricas proporcionales (Qwen).
> D2: Latencia end-to-end (→ iptables) como métrica operacional primaria (DeepSeek).
> D3: Temperatura ARM ≤75°C — gate no negociable para armarios hospitalarios (DeepSeek).
> D4: Delta XDP/libpcap es contribución científica independiente publicable.
> D5: Demo FEDER sin trucos pregrabados — protocolo reproducible por evaluador externo.
>
> Pregunta abierta (arquitecto): Opción A (Vagrant/Makefile) recomendada para demo FEDER.
> Opción B (CI/CD) como evolución post-FEDER cuando pipeline de reentrenamiento esté activo.
>
> 'El contrato de calidad ML no termina en el deploy. Termina cuando el modelo
>  aprende sin olvidar, sin retroalimentarse y sin regresionar en silencio.' — Consejo (8/8)"
> — Consejo de Sabios (8/8) · DAY 134

"""

# Insert before the last line (the italic footer)
footer_anchor = "*DAY 133 — 27 Abril 2026 · Commit c6e0c9f1 · feature/adr030-variant-a*"
assert footer_anchor in content, f"ANCHOR NOT FOUND: {footer_anchor}"
pos = content.find(footer_anchor)
content = content[:pos] + consejo_day134 + content[pos:]
print("✓ Nota Consejo DAY 134 añadida")

# ── 5. Update BACKLOG-FEDER-001 gate: añadir checkboxes ADR-041
feder_gate_anchor = "- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)"
if feder_gate_anchor in content:
    new_feder_items = """- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)
- [ ] ADR-041 protocolo hardware: métricas mínimas validadas en x86 + ARM (make feder-demo)
- [ ] Golden set v1 creado y versionado (DEBT-ADR040-001)"""
    content = content.replace(feder_gate_anchor, new_feder_items)
    print("✓ BACKLOG-FEDER-001 gates actualizados")

# ── 6. Update footer date
content = content.replace(
    "*DAY 133 — 27 Abril 2026 · Commit c6e0c9f1 · feature/adr030-variant-a*",
    "*DAY 134 — 28 Abril 2026 · ADR-040 + ADR-041 integrados · feature/adr030-variant-a*"
)
print("✓ Footer actualizado a DAY 134")

with open(BACKLOG, "w") as f:
    f.write(content)

print(f"\n✅ BACKLOG.md actualizado. Líneas: {len(content.splitlines())}")
PYEOF

echo "✓ BACKLOG.md listo"

# ─────────────────────────────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────────────────────────────

vagrant ssh -- python3 << 'PYEOF'
README = "/vagrant/README.md"
with open(README, "r") as f:
    content = f.read()

# ── 1. Badges: insertar tras el badge BSR
badge_bsr = "[![BSR](https://img.shields.io/badge/BSR-cap__bpf_ADR--039-brightgreen)]()"
new_badges = """[![BSR](https://img.shields.io/badge/BSR-cap__bpf_ADR--039-brightgreen)]()
[![ADR-040](https://img.shields.io/badge/ADR--040-ML_Retraining_Contract-blue)](docs/adr/ADR-040-ml-plugin-retraining-contract.md)
[![ADR-041](https://img.shields.io/badge/ADR--041-FEDER_HW_Metrics-orange)](docs/adr/ADR-041-hardware-acceptance-metrics-feder.md)"""
assert badge_bsr in content, f"BADGE ANCHOR NOT FOUND"
content = content.replace(badge_bsr, new_badges)
print("✓ Badges ADR-040 + ADR-041 añadidos")

# ── 2. Tabla deuda técnica: añadir ADR-040 + ADR-041 al final
debt_row_anchor = "| DEBT-CRYPTO-003a | ⏳ | feature/crypto-hardening |"
new_debt_rows = """| DEBT-CRYPTO-003a | ⏳ | feature/crypto-hardening |
| DEBT-ADR040-001..012 | ⏳ post-FEDER | ADR-040 ML Retraining (12 deudas) |
| DEBT-ADR041-001..006 | ⏳ pre-FEDER | ADR-041 HW Acceptance Metrics (6 tareas) |"""
assert debt_row_anchor in content, f"DEBT TABLE ANCHOR NOT FOUND"
content = content.replace(debt_row_anchor, new_debt_rows)
print("✓ Deudas ADR-040 + ADR-041 añadidas en tabla de deuda técnica")

# ── 3. Estado actual: actualizar header a DAY 134
content = content.replace(
    "## Estado actual — DAY 133 (2026-04-27)",
    "## Estado actual — DAY 134 (2026-04-28)"
)
# Actualizar descripción tag
content = content.replace(
    "**Tag activo:** `v0.5.2-hardened` | **Commit:** `c6e0c9f1` | **Branch activa:** `feature/adr030-variant-a`\n**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`\n**Paper:** arXiv:2604.04952 · Draft v18 en GitHub (pre-arXiv, pendiente tabla fuzzing §6.8)",
    "**Tag activo:** `v0.5.2-hardened` | **Commit:** `c6e0c9f1` | **Branch activa:** `feature/adr030-variant-a`\n**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`\n**Paper:** arXiv:2604.04952 · Draft v18 en GitHub (pre-arXiv, pendiente tabla fuzzing §6.8)\n**ADR-040:** ML Plugin Retraining Contract — PROPUESTO v2 (Consejo 8/8, 17 enmiendas)\n**ADR-041:** Hardware Acceptance Metrics FEDER — PROPUESTO (Consejo 8/8)"
)
print("✓ Estado actual actualizado a DAY 134")

# ── 4. Hitos DAY 133: añadir DAY 134 milestone
milestone_anchor = "- ✅ DAY 133: **ADR-030 Variant A — cap_bpf · AppArmor 6/6 · Falco 10 reglas · Paper v18** 🎉"
new_milestone = """- ✅ DAY 133: **ADR-030 Variant A — cap_bpf · AppArmor 6/6 · Falco 10 reglas · Paper v18** 🎉
- ✅ DAY 134: **ADR-040 ML Retraining Contract (8/8, 17 enmiendas) · ADR-041 FEDER HW Metrics (8/8)** 🎉"""
assert milestone_anchor in content, f"MILESTONE ANCHOR NOT FOUND"
content = content.replace(milestone_anchor, new_milestone)
print("✓ Milestone DAY 134 añadido")

# ── 5. Roadmap DONE DAY 133: añadir bloque DONE DAY 134
roadmap_done_anchor = "### ✅ DONE — DAY 132 (26 Apr 2026)"
new_roadmap_done = """### ✅ DONE — DAY 134 (28 Apr 2026) — ADR-040 + ADR-041 🎉
- [x] ADR-040 ML Plugin Retraining Contract v2 — 7 reglas, 12 deudas, Consejo 8/8 (17 enmiendas)
- [x] ADR-041 Hardware Acceptance Metrics FEDER — 3 niveles despliegue, 10 métricas, Consejo 8/8
- [x] BACKLOG.md + README.md actualizados con ADR-040 + ADR-041

### ✅ DONE — DAY 132 (26 Apr 2026)"""
assert roadmap_done_anchor in content, f"ROADMAP ANCHOR NOT FOUND"
content = content.replace(roadmap_done_anchor, new_roadmap_done)
print("✓ Roadmap DAY 134 añadido")

# ── 6. Actualizar "✅ main" banner
content = content.replace(
    "✅ `main` is tagged `v0.5.2-hardened`. Branch activa: `feature/adr030-variant-a` — ADR-030 Variant A infraestructura completa (DAY 133).",
    "✅ `main` is tagged `v0.5.2-hardened`. Branch activa: `feature/adr030-variant-a` — ADR-030 Variant A completa (DAY 133) · ADR-040 + ADR-041 aprobados (DAY 134)."
)
print("✓ Banner principal actualizado")

with open(README, "w") as f:
    f.write(content)

print(f"\n✅ README.md actualizado. Líneas: {len(content.splitlines())}")
PYEOF

echo "✓ README.md listo"

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "═══ VERIFICACIÓN ═══"
vagrant ssh -c 'echo "BACKLOG.md — ADR-040:" && grep -c "ADR-040" /vagrant/docs/BACKLOG.md && echo "BACKLOG.md — ADR-041:" && grep -c "ADR-041" /vagrant/docs/BACKLOG.md && echo "BACKLOG.md — DEBT-ADR040:" && grep -c "DEBT-ADR040" /vagrant/docs/BACKLOG.md && echo "BACKLOG.md — DEBT-ADR041:" && grep -c "DEBT-ADR041" /vagrant/docs/BACKLOG.md && echo "BACKLOG.md — DAY 134:" && grep -c "DAY 134" /vagrant/docs/BACKLOG.md && echo "README.md — ADR-040 badge:" && grep -c "ADR--040" /vagrant/README.md && echo "README.md — ADR-041 badge:" && grep -c "ADR--041" /vagrant/README.md && echo "README.md — DAY 134:" && grep -c "DAY 134" /vagrant/README.md'
#!/usr/bin/env python3
# DAY 134 — Integración quirúrgica ADR-040 + ADR-041
# Ejecutar desde la raíz del repo:  python3 day134_patch_docs.py

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent  # ajusta si lo mueves fuera de la raíz

BACKLOG = REPO_ROOT / "docs" / "BACKLOG.md"
README  = REPO_ROOT / "README.md"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def assert_anchor(content, anchor, label):
    if anchor not in content:
        print(f"❌ ANCHOR NOT FOUND [{label}]: {repr(anchor[:80])}")
        sys.exit(1)

def insert_after_line(content, anchor, insertion):
    pos = content.find(anchor)
    row_end = content.find("\n", pos) + 1
    return content[:row_end] + insertion + content[row_end:]

def replace_exact(content, old, new, label):
    assert_anchor(content, old, label)
    return content.replace(old, new, 1)

# ─────────────────────────────────────────────────────────────────────────────
# BACKLOG.md
# ─────────────────────────────────────────────────────────────────────────────

print("── BACKLOG.md ──────────────────────────────────────────────────────────")
content = BACKLOG.read_text()

# 1. Deudas ADR-040 + ADR-041 — después de DEBT-PROD-FALCO-RULES-EXTENDED-001
NEW_DEBTS = """

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

**Prerequisito crítico (enmienda Claude, DAY 134):** IPW no es implementable sin `confidence_score`. DEBT-ADR040-002 debe resolverse antes de DEBT-ADR040-006.

**Test de cierre DEBT-ADR040-004:** `make retrain-eval PLUGIN=candidate.ubj` → exit 1 ante regresión. `make prod-sign` no ejecuta si guardrail falla.

---

### DEBT-ADR041-001 a 006 — Hardware Acceptance Metrics FEDER *(nuevas — DAY 134)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** pre-FEDER, deadline 22 sep 2026
**Origen:** ADR-041 — Consejo 8/8 DAY 134

| ID | Descripción | Estado |
|----|-------------|--------|
| DEBT-ADR041-001 | Subconjunto pcap CTU-13 benchmark versionado con SHA-256 (`ctu13-neris-benchmark.pcap`) | ⏳ PENDIENTE |
| DEBT-ADR041-002 | `make golden-set-eval ARCH=$(uname -m)` — exit 0 dentro de tolerancia, exit 1 regresión | ⏳ PENDIENTE (depende ADR-040) |
| DEBT-ADR041-003 | `make feder-demo` — suite completa desde VM fría, <30 min, sin trucos pregrabados | ⏳ PENDIENTE |
| DEBT-ADR041-004 | Compra hardware x86 (NUC/mini-PC ~300€, NIC con soporte XDP nativo — mlx5/i40e/ixgbe) | ⏳ post-métricas definidas |
| DEBT-ADR041-005 | Compra Raspberry Pi 4/5 | ⏳ post-métricas definidas |
| DEBT-ADR041-006 | Primera ejecución protocolo completo en hardware físico | ⏳ post-compra hardware |

**Nota DeepSeek:** Verificar driver NIC antes de comprar x86. Sin XDP nativo el delta científico A/B se distorsiona.
**Nota DeepSeek:** Temperatura ARM ≤75°C sin ventilador — gate no negociable para armarios hospitalarios 24/7.
**Tolerancias ML:** x86 TOLERANCE=0.0000 · ARM TOLERANCE=0.0005 (NEON vs AVX2).
"""

ANCHOR_DEBTS = "### DEBT-PROD-FALCO-RULES-EXTENDED-001 *(nueva — DAY 133)*"
assert_anchor(content, ANCHOR_DEBTS, "deudas")
# Insertar después del bloque FALCO EXTENDED (buscar el siguiente bloque de nivel 2)
pos = content.find(ANCHOR_DEBTS)
next_section = content.find("\n\n## ", pos)
content = content[:next_section] + "\n" + NEW_DEBTS + content[next_section:]
print("✓ Deudas ADR-040 + ADR-041 añadidas")

# 2. Decisiones de diseño — añadir filas DAY 134
NEW_DECISIONS = '| **"Fuzzing misses nothing" ELIMINADA** | Frase incorrecta. Fuzzing es estocástico, no exhaustivo. | Consejo 8/8 · DAY 133 |\n| **Walk-forward obligatorio (ADR-040)** | K-fold prohibido. Split sobre `timestamp_first_packet` ordenado. Mín. 3 ventanas. | ADR-040 · Consejo 8/8 · DAY 134 |\n| **Golden set inmutable (ADR-040)** | ≥50K flows, SHA-256 embebido en plugin firmado. Evolución controlada, solapamiento 6 meses. | ADR-040 · Consejo 8/8 · DAY 134 |\n| **Guardrail asimétrico Ed25519 (ADR-040)** | Recall −0.5pp (más restrictivo). F1 −2pp. FPR +1pp. Latencia p99 +10%. Exit 1 = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |\n| **IPW + uncertainty sampling (ADR-040)** | 5% exploración (P≈0.5). Ratio adaptativo [3%-10%] por drift. Memory replay buffer. | ADR-040 · Consejo 8/8 · DAY 134 |\n| **Competición algoritmos pre-lock-in (ADR-040)** | Multicriterio: Recall 40% + F1 25% + latencia 20% + tamaño 10% + carga 5%. Una sola vez. | ADR-040 · Consejo 8/8 · DAY 134 |\n| **Dataset lineage obligatorio (ADR-040)** | Hash dataset + golden set + features_version + git commits. Sin lineage = no firma. | ADR-040 · Consejo 8/8 · DAY 134 |\n| **Niveles despliegue FEDER (ADR-041)** | Nivel 1 (RPi4/5, ≤50 usuarios) + Nivel 2 (x86, 50-200). Demo mínima: ambos simultáneos. | ADR-041 · Consejo 8/8 · DAY 134 |\n| **Latencia end-to-end como métrica primaria (ADR-041)** | Captura → alerta → iptables efectiva. Más relevante que latencia de detección aislada. | ADR-041 · DeepSeek · DAY 134 |\n| **Temperatura ARM como gate (ADR-041)** | ≤75°C sin ventilador. Crítica para armarios hospitalarios 24/7. | ADR-041 · DeepSeek · DAY 134 |\n| **Pipeline evaluación híbrido (ADR-040)** | Scripts en repo (local Vagrant). CI = mismo código, segunda entrada. Opción A recomendada FEDER. | ADR-040 · Consejo 6/7 · DAY 134 |'

OLD_DECISION = '| **"Fuzzing misses nothing" ELIMINADA** | Frase incorrecta. Fuzzing es estocástico, no exhaustivo. | Consejo 8/8 · DAY 133 |'
content = replace_exact(content, OLD_DECISION, NEW_DECISIONS, "decisiones")
print("✓ Decisiones DAY 134 añadidas")

# 3. Estado global — añadir antes del cierre ```
NEW_STATUS = 'ADR-031 aRGus-seL4:                       0% ⏳  branch independiente\nADR-040 ML Retraining Contract (def.):    100% ✅  DAY 134 (Consejo 8/8, 17 enmiendas)\nADR-041 HW Acceptance Metrics (def.):     100% ✅  DAY 134 (Consejo 8/8)\nDEBT-ADR040-001 (golden set v1):            0% ⏳  v1.0 post-FEDER\nDEBT-ADR040-002 (confidence_score):         0% ⏳  v1.0\nDEBT-ADR040-003 (walk_forward_split.py):    0% ⏳  v1.1\nDEBT-ADR040-004 (check_guardrails.py):      0% ⏳  v1.1\nDEBT-ADR040-005 (guardrail + Ed25519):      0% ⏳  v1.1\nDEBT-ADR040-006 (IPW + uncertainty):        0% ⏳  v1.2\nDEBT-ADR040-007 (interfaz web exploración): 0% ⏳  v1.2 Año 1\nDEBT-ADR040-008 (informe diversidad):       0% ⏳  v1.2\nDEBT-ADR040-009 (competición algoritmos):   0% ⏳  pre-lock-in XGBoost\nDEBT-ADR040-010 (dataset lineage):          0% ⏳  v1.1\nDEBT-ADR040-011 (canary deployment):        0% ⏳  Año 2 flota\nDEBT-ADR040-012 (GOLDEN-SET-REGISTRY.md):  0% ⏳  v1.0\nDEBT-ADR041-001 (pcap CTU-13 versionado):   0% ⏳  pre-FEDER\nDEBT-ADR041-002 (make golden-set-eval):     0% ⏳  depende ADR-040\nDEBT-ADR041-003 (make feder-demo):          0% ⏳  pre-FEDER\nDEBT-ADR041-004 (compra hardware x86):      0% ⏳  post-métricas\nDEBT-ADR041-005 (compra Raspberry Pi 4/5):  0% ⏳  post-métricas\nDEBT-ADR041-006 (ejecución hw físico):      0% ⏳  post-compra'

OLD_STATUS = "ADR-031 aRGus-seL4:                       0% ⏳  branch independiente"
content = replace_exact(content, OLD_STATUS, NEW_STATUS, "estado global")
print("✓ Estado global actualizado")

# 4. Nota Consejo DAY 134
CONSEJO_DAY134 = '''
---

## 📝 Notas del Consejo de Sabios — DAY 134 (8/8)

> "DAY 134 — ADR-040 + ADR-041: contratos de calidad ML y métricas de aceptación hardware.
>
> ADR-040 — 17 enmiendas, aprobado 8/8:
> D1: Walk-forward obligatorio. K-fold prohibido en NDR temporal.
> D2: Golden set inmutable con SHA-256 embebido en plugin firmado (Gemini).
> D3: Guardrail asimétrico — Recall más restrictivo que F1 (infraestructura crítica).
> D4: IPW + uncertainty sampling (P≈0.5), no exploración aleatoria pura (Gemini).
> D5: Ratio exploración adaptativo [3%-10%] por drift detectado (ChatGPT-5).
> D6: Memory replay buffer como complemento al golden set (Grok).
> D7: Competición algoritmos multicriterio — XGBoost no asumido ganador a priori.
> D8: Dataset lineage obligatorio — prerequisito de firma Ed25519.
> D9: Canary 5-10% / 24h antes de despliegue completo (ChatGPT-5).
> D10: Pipeline evaluación híbrido — mismo código, dos entradas (local + CI).
> Enmienda crítica (Claude): confidence_score es prerequisito de IPW.
>
> ADR-041 — aprobado 8/8:
> D1: Tres niveles despliegue con métricas proporcionales (Qwen).
> D2: Latencia end-to-end (→ iptables) como métrica operacional primaria (DeepSeek).
> D3: Temperatura ARM ≤75°C — gate no negociable para armarios hospitalarios (DeepSeek).
> D4: Delta XDP/libpcap es contribución científica independiente publicable.
> D5: Demo FEDER reproducible por evaluador externo — sin trucos pregrabados.
> Pregunta abierta: Opción A (Vagrant) recomendada demo FEDER. Opción B (CI) post-FEDER.
>
> 'El contrato de calidad ML no termina en el deploy. Termina cuando el modelo
>  aprende sin olvidar, sin retroalimentarse y sin regresionar en silencio.' — Consejo (8/8)"
> — Consejo de Sabios (8/8) · DAY 134

'''
FOOTER = "*DAY 133 — 27 Abril 2026 · Commit c6e0c9f1 · feature/adr030-variant-a*"
assert_anchor(content, FOOTER, "footer consejo")
content = content.replace(FOOTER, CONSEJO_DAY134 + FOOTER, 1)
print("✓ Nota Consejo DAY 134 añadida")

# 5. BACKLOG-FEDER-001 gates
OLD_GATE = "- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)"
NEW_GATE = """- [ ] Demo técnica grabable < 10 minutos (`scripts/feder-demo.sh`)
- [ ] ADR-041 protocolo hardware: métricas validadas en x86 + ARM (`make feder-demo`)
- [ ] Golden set v1 creado y versionado (DEBT-ADR040-001)"""
content = replace_exact(content, OLD_GATE, NEW_GATE, "feder gates")
print("✓ BACKLOG-FEDER-001 gates actualizados")

# 6. Footer
content = replace_exact(
    content,
    "*DAY 133 — 27 Abril 2026 · Commit c6e0c9f1 · feature/adr030-variant-a*",
    "*DAY 134 — 28 Abril 2026 · ADR-040 + ADR-041 integrados · feature/adr030-variant-a*",
    "footer date"
)
print("✓ Footer actualizado a DAY 134")

BACKLOG.write_text(content)
print(f"✅ BACKLOG.md guardado ({len(content.splitlines())} líneas)\n")

# ─────────────────────────────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────────────────────────────

print("── README.md ───────────────────────────────────────────────────────────")
content = README.read_text()

# 1. Badges
OLD_BADGE = "[![BSR](https://img.shields.io/badge/BSR-cap__bpf_ADR--039-brightgreen)]()"
NEW_BADGE = """[![BSR](https://img.shields.io/badge/BSR-cap__bpf_ADR--039-brightgreen)]()
[![ADR-040](https://img.shields.io/badge/ADR--040-ML_Retraining_Contract-blue)](docs/adr/ADR-040-ml-plugin-retraining-contract.md)
[![ADR-041](https://img.shields.io/badge/ADR--041-FEDER_HW_Metrics-orange)](docs/adr/ADR-041-hardware-acceptance-metrics-feder.md)"""
content = replace_exact(content, OLD_BADGE, NEW_BADGE, "badges")
print("✓ Badges ADR-040 + ADR-041 añadidos")

# 2. Tabla deuda técnica
OLD_DEBT = "| DEBT-CRYPTO-003a | ⏳ | feature/crypto-hardening |"
NEW_DEBT = """| DEBT-CRYPTO-003a | ⏳ | feature/crypto-hardening |
| DEBT-ADR040-001..012 | ⏳ post-FEDER | ADR-040 ML Retraining (12 deudas — ver BACKLOG.md) |
| DEBT-ADR041-001..006 | ⏳ pre-FEDER | ADR-041 HW Acceptance Metrics (6 tareas — ver BACKLOG.md) |"""
content = replace_exact(content, OLD_DEBT, NEW_DEBT, "tabla deuda")
print("✓ Deudas añadidas en tabla de deuda técnica")

# 3. Estado actual header + referencias
OLD_HEADER = "## Estado actual — DAY 133 (2026-04-27)"
content = replace_exact(content, OLD_HEADER, "## Estado actual — DAY 134 (2026-04-28)", "header estado")
OLD_PAPER_LINE = "**Paper:** arXiv:2604.04952 · Draft v18 en GitHub (pre-arXiv, pendiente tabla fuzzing §6.8)"
NEW_PAPER_LINE = """**Paper:** arXiv:2604.04952 · Draft v18 en GitHub (pre-arXiv, pendiente tabla fuzzing §6.8)
**ADR-040:** ML Plugin Retraining Contract — PROPUESTO v2 (Consejo 8/8, 17 enmiendas)
**ADR-041:** Hardware Acceptance Metrics FEDER — PROPUESTO (Consejo 8/8)"""
content = replace_exact(content, OLD_PAPER_LINE, NEW_PAPER_LINE, "paper line")
print("✓ Estado actual actualizado a DAY 134")

# 4. Milestone DAY 134
OLD_MILESTONE = "- ✅ DAY 133: **ADR-030 Variant A — cap_bpf · AppArmor 6/6 · Falco 10 reglas · Paper v18** 🎉"
NEW_MILESTONE = """- ✅ DAY 133: **ADR-030 Variant A — cap_bpf · AppArmor 6/6 · Falco 10 reglas · Paper v18** 🎉
- ✅ DAY 134: **ADR-040 ML Retraining Contract (8/8, 17 enmiendas) · ADR-041 FEDER HW Metrics (8/8)** 🎉"""
content = replace_exact(content, OLD_MILESTONE, NEW_MILESTONE, "milestone")
print("✓ Milestone DAY 134 añadido")

# 5. Roadmap DONE DAY 134
OLD_ROADMAP = "### ✅ DONE — DAY 132 (26 Apr 2026)"
NEW_ROADMAP = """### ✅ DONE — DAY 134 (28 Apr 2026) — ADR-040 + ADR-041 🎉
- [x] ADR-040 ML Plugin Retraining Contract v2 — 7 reglas, 12 deudas, Consejo 8/8 (17 enmiendas)
- [x] ADR-041 Hardware Acceptance Metrics FEDER — 3 niveles, 10 métricas, Consejo 8/8
- [x] BACKLOG.md + README.md actualizados con ADR-040 + ADR-041

### ✅ DONE — DAY 132 (26 Apr 2026)"""
content = replace_exact(content, OLD_ROADMAP, NEW_ROADMAP, "roadmap")
print("✓ Roadmap DAY 134 añadido")

# 6. Banner
OLD_BANNER = "✅ `main` is tagged `v0.5.2-hardened`. Branch activa: `feature/adr030-variant-a` — ADR-030 Variant A infraestructura completa (DAY 133)."
NEW_BANNER = "✅ `main` is tagged `v0.5.2-hardened`. Branch activa: `feature/adr030-variant-a` — ADR-030 Variant A completa (DAY 133) · ADR-040 + ADR-041 aprobados (DAY 134)."
content = replace_exact(content, OLD_BANNER, NEW_BANNER, "banner")
print("✓ Banner principal actualizado")

README.write_text(content)
print(f"✅ README.md guardado ({len(content.splitlines())} líneas)\n")

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

print("── VERIFICACIÓN ────────────────────────────────────────────────────────")
bl = BACKLOG.read_text()
rm = README.read_text()

checks = [
    (bl, "DEBT-ADR040-001", "BACKLOG ADR-040 deudas"),
    (bl, "DEBT-ADR041-001", "BACKLOG ADR-041 deudas"),
    (bl, "DAY 134 — 28 Abril", "BACKLOG footer DAY 134"),
    (bl, "Consejo de Sabios — DAY 134", "BACKLOG nota Consejo DAY 134"),
    (bl, "make feder-demo", "BACKLOG FEDER gate ADR-041"),
    (rm, "ADR--040", "README badge ADR-040"),
    (rm, "ADR--041", "README badge ADR-041"),
    (rm, "DAY 134 (2026-04-28)", "README header DAY 134"),
    (rm, "DONE — DAY 134", "README roadmap DAY 134"),
    (rm, "ADR-040 ML Retraining Contract (8/8", "README milestone DAY 134"),
]

all_ok = True
for text, needle, label in checks:
    found = needle in text
    symbol = "✓" if found else "✗"
    print(f"  {symbol} {label}")
    if not found:
        all_ok = False

print()
if all_ok:
    print("✅ Todas las verificaciones OK — listo para git add + commit")
else:
    print("❌ Algunas verificaciones fallaron — revisar manualmente")
    sys.exit(1)
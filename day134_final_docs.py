#!/usr/bin/env python3
# DAY 134 — Actualizar BACKLOG.md + README.md con síntesis del Consejo
# Ejecutar desde la raíz del repo: python3 day134_final_docs.py

import sys
from pathlib import Path

REPO = Path(__file__).parent
BACKLOG = REPO / "docs" / "BACKLOG.md"
README  = REPO / "README.md"

def assert_anchor(content, anchor, label):
    if anchor not in content:
        print(f"❌ ANCHOR NOT FOUND [{label}]: {repr(anchor[:80])}")
        sys.exit(1)

def replace_exact(content, old, new, label):
    assert_anchor(content, old, label)
    return content.replace(old, new, 1)

# ─────────────────────────────────────────────────────────────────────────────
# BACKLOG.md
# ─────────────────────────────────────────────────────────────────────────────
print("── BACKLOG.md ──────────────────────────────────────────────────────────")
content = BACKLOG.read_text()

# 1. Añadir REGLAS PERMANENTES DAY 134 (Consejo síntesis)
NEW_RULES = """- **REGLA PERMANENTE (DAY 133 — Consejo 8/8):** `cap_sys_admin` está prohibida en imágenes de producción si el kernel es ≥5.8. Usar `cap_bpf` para operaciones eBPF. Documentar fallback con DEBT-KERNEL-COMPAT-001 si necesario.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** `make hardened-full` es el EMECAS sagrado de la hardened VM — siempre incluye `vagrant destroy -f`. Para iteración de desarrollo usar `make hardened-redeploy` (sin destroy). Los gates `check-prod-all` se ejecutan siempre en ambos modos.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** Las semillas criptográficas NO se transfieren en el procedimiento EMECAS. La hardened VM arranca sin seeds. Target `prod-deploy-seeds` explícito para el momento del deploy real. Los WARNs de `seed.bin no existe` en `check-prod-permissions` son estado correcto por diseño.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** Falco .deb y artefactos binarios de terceros van en `dist/vendor/` (gitignored). El hash SHA-256 se committea en `dist/vendor/CHECKSUMS`. `make vendor-download` descarga y verifica. Si hash no coincide → abort.
- **REGLA PERMANENTE (DAY 134 — Consejo 8/8):** DEBT-ADR040-002 (`confidence_score` en ml-detector) es prerequisito bloqueante de DEBT-ADR040-006 (IPW). No implementar IPW sin verificar primero que el campo existe y varía en runtime."""

OLD_RULE = "- **REGLA PERMANENTE (DAY 133 — Consejo 8/8):** `cap_sys_admin` está prohibida en imágenes de producción si el kernel es ≥5.8. Usar `cap_bpf` para operaciones eBPF. Documentar fallback con DEBT-KERNEL-COMPAT-001 si necesario."
content = replace_exact(content, OLD_RULE, NEW_RULES, "reglas permanentes")
print("✓ Reglas permanentes DAY 134 añadidas")

# 2. Añadir sección CERRADO DAY 134 antes de CERRADO DAY 133
CERRADO_DAY134 = """## ✅ CERRADO DAY 134

### Pipeline E2E en hardened VM — check-prod-all PASSED
- **Status:** ✅ CERRADO DAY 134
- **Fix:** Primer pipeline end-to-end en hardened VM con 5/5 gates verdes. 15 problemas de integración resueltos (vagrant --cwd, AppArmor tunables/global, Falco offline via .deb, macros inline Falco 0.43, cmake flags, pipeline-build PROFILE=production, firewall-build faltante en pipeline-build, prod-sign PEM canónico, ownership root:argus, getcap path, check caps post-Consejo, check-prod-falco API 0.43, permisos sudo).
- **Commits:** `f256e6f0` + `2e9a5b39`

### DEBT-KERNEL-COMPAT-001
- **Status:** ✅ CERRADO DAY 134
- **Fix:** `cap_bpf` funciona correctamente con XDP en kernel 6.1.0-44-amd64 (Debian bookworm). `sniffer: cap_net_admin,cap_net_raw,cap_ipc_lock,cap_bpf=eip` — verificado en hardened VM.
- **Commit:** `2e9a5b39`

### DEBT-PAPER-FUZZING-METRICS-001
- **Status:** ✅ CERRADO DAY 134
- **Fix:** Tabla §6.8 con datos reales de tres campañas libFuzzer (DAY 130). `validate_chain_name`: 2.4M runs, 0 crashes, corpus 67, ~80K exec/s. `safe_exec`: 2.6M runs, 0 crashes, corpus 37, 42K exec/s. `validate_filepath`: 282K runs, 0 crashes, corpus 111, 4.6K exec/s. Análisis delta exec/s documentado. Paper actualizado a Draft v18.
- **Commit:** post-`2e9a5b39`

### ADR-040 + ADR-041 — Integración en BACKLOG + README
- **Status:** ✅ CERRADO DAY 134
- **Fix:** ADR-040 ML Plugin Retraining Contract v2 (8/8, 17 enmiendas) + ADR-041 Hardware Acceptance Metrics FEDER (8/8) integrados en BACKLOG.md y README.md. 25 ficheros, 4648 inserciones.
- **Commit:** `87680d83`

---

"""
OLD_CERRADO = "## ✅ CERRADO DAY 133"
content = replace_exact(content, OLD_CERRADO, CERRADO_DAY134 + "## ✅ CERRADO DAY 133", "cerrado day134")
print("✓ Sección CERRADO DAY 134 añadida")

# 3. Añadir nuevas deudas del Consejo síntesis (después de DEBT-ADR041)
NEW_DEBTS_CONSEJO = """
---

### DEBT-EMECAS-HARDENED-001 *(nueva — DAY 134, Consejo síntesis)*
**Severidad:** 🔴 Crítica | **Bloqueante:** Sí | **Target:** DAY 135
**Origen:** Consejo síntesis 8/8 DAY 134

Implementar `make hardened-full` como EMECAS sagrado de la hardened VM:
- Fail-fast obligatorio (`set -e`)
- Siempre incluye `vagrant destroy -f` al inicio
- Gates `check-prod-all` siempre completos, nunca cacheados
- Target paralelo `make hardened-redeploy` (sin destroy, para iteración en desarrollo de perfiles AppArmor/Falco)
- Documentar en `docs/EMECAS-hardened.md`: cuándo usar cada target

**Test de cierre:** `make hardened-full` desde VM destruida → check-prod-all PASSED en <45 min. Segunda ejecución de `make hardened-full` también PASSED (reproducibilidad).

---

### DEBT-VENDOR-FALCO-001 *(nueva — DAY 134, Consejo síntesis)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** DAY 135
**Origen:** Consejo síntesis 8/8 DAY 134

Formalizar gestión de artefactos binarios de terceros:
- Directorio `dist/vendor/` gitignored
- `dist/vendor/CHECKSUMS` committeado con SHA-256 de cada artefacto
- `make vendor-download` descarga y verifica hash — si no coincide → abort
- Falco .deb actual (`falco_0.43.1_amd64.deb`) mover a `dist/vendor/`
- Documentar en `docs/VENDOR-ARTIFACTS.md`

**Test de cierre:** `make vendor-download` en repo limpio descarga y verifica Falco .deb. Hash incorrecto → exit 1.

---

### DEBT-SEEDS-DEPLOY-001 *(nueva — DAY 134, Consejo síntesis)*
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** DAY 135
**Origen:** Consejo síntesis 7/8 DAY 134

Crear target `make prod-deploy-seeds` para transferencia explícita de semillas desde dev VM a hardened VM en el momento del deploy real:
- Usar `scp -F vagrant-ssh-config` (REGLA PERMANENTE DAY 129)
- Permisos `0400 argus:argus` en destino
- Convertir WARNs de `seed.bin no existe` en `check-prod-permissions` a INFO documentados
- La ausencia de seeds en EMECAS es estado correcto por diseño

**Test de cierre:** `make prod-deploy-seeds` → seeds en `/etc/ml-defender/*/seed.bin` con permisos correctos → `check-prod-permissions` sin WARNs.

---

### DEBT-CONFIDENCE-SCORE-001 *(nueva — DAY 134, Consejo síntesis)*
**Severidad:** 🔴 Crítica | **Bloqueante:** Sí (prerequisito ADR-040 Regla 4) | **Target:** DAY 135
**Origen:** Consejo síntesis 8/8 DAY 134

Verificar que ml-detector emite `confidence_score ∈ [0,1]` en salida ZeroMQ antes de implementar IPW (DEBT-ADR040-006):
- **Paso 1 — Inspección estática:** `scripts/check-confidence-score.sh` — verifica campo en `.proto` y asignación en código fuente
- **Paso 2 — Test de integración:** `tests/integration/test_confidence_score.py` — captura mensaje ZeroMQ real con golden pcap determinista, verifica presencia + rango + variabilidad (no constante entre benign/attack)
- Si el campo no existe → DEBT-ADR040-002 abierto, IPW bloqueado
- Si el campo existe pero es constante → bug de implementación, requiere fix antes de IPW

**Test de cierre:** Ambos scripts pasan. Score varía entre flows benignos y maliciosos. DEBT-ADR040-002 marcado como CERRADO solo cuando ambos pasen.

"""

# Insertar después de DEBT-ADR041 block
ADR041_ANCHOR = "**Tolerancias ML:** x86 TOLERANCE=0.0000 · ARM TOLERANCE=0.0005 (NEON vs AVX2)."
assert_anchor(content, ADR041_ANCHOR, "adr041 anchor")
pos = content.find(ADR041_ANCHOR)
row_end = content.find("\n", pos) + 1
content = content[:row_end] + NEW_DEBTS_CONSEJO + content[row_end:]
print("✓ Nuevas deudas Consejo síntesis añadidas")

# 4. Actualizar estado global
NEW_STATUS_LINES = """DEBT-PROD-APT-SOURCES-INTEGRITY-001:      0% ⏳  feature/adr030-variant-a
DEBT-PAPER-FUZZING-METRICS-001:         100% ✅  DAY 134 CERRADO
DEBT-KEY-SEPARATION-001:                  0% ⏳  post-FEDER
DEBT-KERNEL-COMPAT-001:                 100% ✅  DAY 134 CERRADO — cap_bpf ok en kernel 6.1
DEBT-PROD-APPARMOR-PORTS-001:             0% ⏳  post-JSON-estabilización
DEBT-PROD-FALCO-RULES-EXTENDED-001:       0% ⏳  DAY 135
DEBT-DEBIAN13-UPGRADE-001:                0% ⏳  post-FEDER"""

OLD_STATUS = """DEBT-PROD-APT-SOURCES-INTEGRITY-001:      0% ⏳  feature/adr030-variant-a
DEBT-PAPER-FUZZING-METRICS-001:          40% 🟡  DAY 134 (reformulación cerrada, tabla pendiente)
DEBT-KEY-SEPARATION-001:                  0% ⏳  post-FEDER
DEBT-KERNEL-COMPAT-001:                   0% ⏳  DAY 134
DEBT-PROD-APPARMOR-PORTS-001:             0% ⏳  post-JSON-estabilización
DEBT-PROD-FALCO-RULES-EXTENDED-001:       0% ⏳  DAY 135
DEBT-DEBIAN13-UPGRADE-001:                0% ⏳  post-FEDER"""
content = replace_exact(content, OLD_STATUS, NEW_STATUS_LINES, "estado global deudas")

# Añadir nuevas deudas al estado global
NEW_STATUS_ADDITIONS = """DEBT-EMECAS-HARDENED-001 (make hardened-full): 0% ⏳  DAY 135
DEBT-VENDOR-FALCO-001 (dist/vendor/CHECKSUMS): 0% ⏳  DAY 135
DEBT-SEEDS-DEPLOY-001 (prod-deploy-seeds):     0% ⏳  DAY 135
DEBT-CONFIDENCE-SCORE-001 (prerequisito IPW):  0% ⏳  DAY 135"""

OLD_ADR040_STATUS = "DEBT-ADR040-001 (golden set v1):            0% ⏳  v1.0 post-FEDER"
content = replace_exact(content, OLD_ADR040_STATUS,
                        NEW_STATUS_ADDITIONS + "\n" + OLD_ADR040_STATUS, "estado global nuevas deudas")
print("✓ Estado global actualizado")

# 5. Actualizar BACKLOG-FEDER-001 gate (check-prod-all ahora está DONE)
OLD_FEDER_GATE = "- [ ] Pipeline E2E en hardened VM verde (`make check-prod-all`) — DAY 134"
NEW_FEDER_GATE = "- [x] Pipeline E2E en hardened VM verde (`make check-prod-all`) — DAY 134 ✅"
content = replace_exact(content, OLD_FEDER_GATE, NEW_FEDER_GATE, "feder gate")
print("✓ BACKLOG-FEDER-001 gate actualizado")

# 6. Actualizar footer
content = replace_exact(
    content,
    "*DAY 134 — 28 Abril 2026 · ADR-040 + ADR-041 integrados · feature/adr030-variant-a*",
    "*DAY 134 — 28 Abril 2026 · check-prod-all PASSED · Draft v18 completo · feature/adr030-variant-a*",
    "footer"
)
print("✓ Footer actualizado")

BACKLOG.write_text(content)
print(f"✅ BACKLOG.md guardado ({len(content.splitlines())} líneas)\n")

# ─────────────────────────────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────────────────────────────
print("── README.md ───────────────────────────────────────────────────────────")
content = README.read_text()

# 1. Actualizar commit hash y paper status
OLD_HEADER = """**Tag activo:** `v0.5.2-hardened` | **Commit:** `c6e0c9f1` | **Branch activa:** `feature/adr030-variant-a`
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a7f478ee85daa`
**Paper:** arXiv:2604.04952 · Draft v18 en GitHub (pre-arXiv, pendiente tabla fuzzing §6.8)
**ADR-040:** ML Plugin Retraining Contract — PROPUESTO v2 (Consejo 8/8, 17 enmiendas)
**ADR-041:** Hardware Acceptance Metrics FEDER — PROPUESTO (Consejo 8/8)"""

# Use a softer replace since the keypair hash may vary slightly in display
content = content.replace(
    "**Paper:** arXiv:2604.04952 · Draft v18 en GitHub (pre-arXiv, pendiente tabla fuzzing §6.8)",
    "**Paper:** arXiv:2604.04952 · Draft v18 COMPLETO (tabla fuzzing §6.8 real, pre-arXiv replace pendiente)"
)
content = content.replace(
    "**Commit:** `c6e0c9f1`",
    "**Commit:** `2e9a5b39`"
)
print("✓ Commit hash y paper status actualizados")

# 2. Actualizar hitos DAY 133 → añadir DAY 134 completo
OLD_HITOS_133 = """### Hitos DAY 133
- **Paper Draft v18** — tabla BSR §6.12 con métricas reales. Reformulación §6.8 fuzzing post-Consejo (eliminada frase incorrecta). Compilado Overleaf, 42 páginas.
- **ADR-030 Variant A — infraestructura completa** — 6 perfiles AppArmor enforce, usuario `argus` no-root, `cap_bpf` en lugar de `cap_sys_admin`, Falco 10 reglas, `/tmp` noexec.
- **Makefile prod-* targets** — `prod-full-x86`, `check-prod-all` y 10+ targets de producción.
- **Consejo 8/8 DAY 133** — `cap_bpf` unánime, 3 reglas Falco nuevas, keypairs post-FEDER, reformulación fuzzing.
- **Acta del Consejo** — `docs/acta_consejo_day133.md`."""

NEW_HITOS = """### Hitos DAY 134 🎉
- **Pipeline E2E hardened VM — check-prod-all PASSED** — 5/5 gates verdes: BSR, AppArmor 6/6, cap_bpf, permisos, Falco 10 reglas. 15 problemas de integración resueltos.
- **DEBT-KERNEL-COMPAT-001 CERRADO** — `cap_bpf` verificado en kernel 6.1 con XDP.
- **Draft v18 completo** — tabla fuzzing §6.8 con datos reales: 3 targets, 0 crashes, análisis delta exec/s.
- **ADR-040 + ADR-041** — Contratos ML retraining + métricas hardware FEDER. Consejo 8/8.
- **Consejo síntesis DAY 134** — 7 decisiones vinculantes para DAY 135.

### Hitos DAY 133
- **Paper Draft v18** — tabla BSR §6.12 con métricas reales. Reformulación §6.8 fuzzing post-Consejo.
- **ADR-030 Variant A — infraestructura completa** — 6 perfiles AppArmor enforce, `cap_bpf`, Falco 10 reglas.
- **Makefile prod-* targets** — `prod-full-x86`, `check-prod-all` y 10+ targets de producción.
- **Consejo 8/8 DAY 133** — `cap_bpf` unánime, 3 reglas Falco nuevas, keypairs post-FEDER."""

content = replace_exact(content, OLD_HITOS_133, NEW_HITOS, "hitos")
print("✓ Hitos DAY 134 añadidos")

# 3. Actualizar tabla deuda técnica
OLD_DEBT_TABLE = """| DEBT-PAPER-FUZZING-METRICS-001 | 🟡 Media | DAY 134 (tabla completa) |
| DEBT-KERNEL-COMPAT-001 | 🟡 Media | DAY 134 (verificar cap_bpf+XDP) |"""
NEW_DEBT_TABLE = """| DEBT-PAPER-FUZZING-METRICS-001 | ✅ CERRADO | DAY 134 |
| DEBT-KERNEL-COMPAT-001 | ✅ CERRADO | DAY 134 |
| DEBT-EMECAS-HARDENED-001 | 🔴 Crítica | DAY 135 (make hardened-full) |
| DEBT-VENDOR-FALCO-001 | 🟡 Media | DAY 135 (dist/vendor/CHECKSUMS) |
| DEBT-SEEDS-DEPLOY-001 | 🟡 Media | DAY 135 (prod-deploy-seeds) |
| DEBT-CONFIDENCE-SCORE-001 | 🔴 Crítica | DAY 135 (prerequisito IPW) |"""
content = replace_exact(content, OLD_DEBT_TABLE, NEW_DEBT_TABLE, "tabla deuda")
print("✓ Tabla deuda técnica actualizada")

# 4. Actualizar próxima frontera
OLD_FRONTERA = """- **DAY 134** — primer pipeline end-to-end en hardened VM: `make hardened-provision-all → prod-full-x86 → check-prod-all`
- **DEBT-PENTESTER-LOOP-001** — ACRL: Caldera → eBPF capture → XGBoost retrain → Ed25519 sign → hot-swap"""
NEW_FRONTERA = """- **DAY 135** — `make hardened-full` (EMECAS sagrado desde cero) · DEBT-PROD-APT-SOURCES-INTEGRITY-001 · dist/vendor/CHECKSUMS · confidence_score verificación · arXiv replace v15→v18
- **DEBT-PENTESTER-LOOP-001** — ACRL: Caldera → eBPF capture → XGBoost retrain → Ed25519 sign → hot-swap"""
content = replace_exact(content, OLD_FRONTERA, NEW_FRONTERA, "próxima frontera")
print("✓ Próxima frontera actualizada a DAY 135")

# 5. Roadmap: NEXT DAY 134 → DONE, añadir NEXT DAY 135
OLD_NEXT = """### 🔜 NEXT — DAY 134: pipeline end-to-end en hardened VM

| Priority | Task |
|---|---|
| 🔴 P0 | `make hardened-provision-all → prod-full-x86 → check-prod-all` |
| 🔴 P0 | DEBT-KERNEL-COMPAT-001 — verificar cap_bpf + XDP en kernel 6.1 |
| 🔴 P0 | DEBT-PAPER-FUZZING-METRICS-001 — tabla completa §6.8 con datos DAY 130 |
| 🟡 P1 | DEBT-PROD-APT-SOURCES-INTEGRITY-001 — SHA-256 sources.list fail-closed |
| 🟡 P2 | DEBT-PROD-FALCO-RULES-EXTENDED-001 — ptrace, DNS tunneling, /dev/mem |"""

NEW_NEXT = """### ✅ DONE — DAY 134 (28 Apr 2026) — Pipeline E2E hardened + ADR-040/041 🎉
- [x] `make check-prod-all` PASSED — 5/5 gates verdes en hardened VM
- [x] DEBT-KERNEL-COMPAT-001 CERRADO — cap_bpf + XDP en kernel 6.1 ✅
- [x] DEBT-PAPER-FUZZING-METRICS-001 CERRADO — tabla §6.8 con datos reales ✅
- [x] Draft v18 completo — 42 páginas, listo para arXiv replace
- [x] ADR-040 ML Retraining Contract (8/8, 17 enmiendas) + ADR-041 FEDER HW Metrics (8/8)

### 🔜 NEXT — DAY 135: EMECAS hardened + apt sources + arXiv

| Priority | Task |
|---|---|
| 🔴 P0 | `make hardened-full` desde VM destruida — EMECAS sagrado (Kimi/Consejo D6) |
| 🔴 P0 | DEBT-PROD-APT-SOURCES-INTEGRITY-001 — SHA-256 sources.list fail-closed (Mistral D7) |
| 🔴 P0 | DEBT-CONFIDENCE-SCORE-001 — verificar confidence_score en ml-detector (prerequisito IPW) |
| 🟡 P1 | DEBT-VENDOR-FALCO-001 — dist/vendor/CHECKSUMS + make vendor-download |
| 🟡 P1 | DEBT-SEEDS-DEPLOY-001 — make prod-deploy-seeds + WARNs → INFO |
| 🟡 P2 | arXiv replace v15 → v18 |
| 🟢 P3 | DEBT-PROD-FALCO-RULES-EXTENDED-001 — ptrace, DNS tunneling, /dev/mem |"""

content = replace_exact(content, OLD_NEXT, NEW_NEXT, "roadmap next")
print("✓ Roadmap actualizado")

# 6. Milestone DAY 134 pipeline
OLD_MILESTONE_NEXT = "- 🔜 DAY 134: **Pipeline E2E en hardened VM · check-prod-all verde**"
content = replace_exact(content, OLD_MILESTONE_NEXT,
                        "- ✅ DAY 134: **Pipeline E2E hardened · check-prod-all PASSED · Draft v18 completo · ADR-040+041** 🎉",
                        "milestone next")
print("✓ Milestone DAY 134 actualizado a ✅")

# 7. Añadir milestone DAY 135
content = content.replace(
    "- ✅ DAY 134: **Pipeline E2E hardened · check-prod-all PASSED · Draft v18 completo · ADR-040+041** 🎉",
    "- ✅ DAY 134: **Pipeline E2E hardened · check-prod-all PASSED · Draft v18 completo · ADR-040+041** 🎉\n- 🔜 DAY 135: **make hardened-full EMECAS · apt sources integrity · arXiv replace v15→v18**"
)
print("✓ Milestone DAY 135 añadido")

# 8. Quick Start: actualizar sección Hardened VM
OLD_QS = """### Hardened VM (ADR-030 Variant A)

```bash
make hardened-up
make hardened-provision-all   # filesystem + AppArmor + Falco
make prod-full-x86            # build → sign → checksums → deploy
make check-prod-all           # 5 security gates
```"""
NEW_QS = """### Hardened VM (ADR-030 Variant A)

```bash
# EMECAS sagrado (reproducibilidad total — para demo FEDER y validación)
make hardened-full            # destroy → up → provision → build → deploy → check

# Workflow alternativo (iteración rápida durante desarrollo)
make hardened-up
make hardened-provision-all   # filesystem + AppArmor + Falco
make prod-full-x86            # build → sign → checksums → deploy
make check-prod-all           # 5 security gates
```"""
content = replace_exact(content, OLD_QS, NEW_QS, "quick start")
print("✓ Quick Start hardened VM actualizado")

README.write_text(content)
print(f"✅ README.md guardado ({len(content.splitlines())} líneas)\n")

# ─────────────────────────────────────────────────────────────────────────────
# Verificación
# ─────────────────────────────────────────────────────────────────────────────
print("── VERIFICACIÓN ────────────────────────────────────────────────────────")
bl = BACKLOG.read_text()
rm = README.read_text()

checks = [
    (bl, "CERRADO DAY 134", "BACKLOG cerrado DAY 134"),
    (bl, "DEBT-EMECAS-HARDENED-001", "BACKLOG EMECAS-HARDENED"),
    (bl, "DEBT-VENDOR-FALCO-001", "BACKLOG VENDOR-FALCO"),
    (bl, "DEBT-SEEDS-DEPLOY-001", "BACKLOG SEEDS-DEPLOY"),
    (bl, "DEBT-CONFIDENCE-SCORE-001", "BACKLOG CONFIDENCE-SCORE"),
    (bl, "check-prod-all`) — DAY 134 ✅", "BACKLOG FEDER gate"),
    (bl, "DEBT-KERNEL-COMPAT-001:                 100%", "BACKLOG kernel compat cerrado"),
    (bl, "DEBT-PAPER-FUZZING-METRICS-001:         100%", "BACKLOG fuzzing cerrado"),
    (rm, "DONE — DAY 134", "README roadmap DAY 134 done"),
    (rm, "NEXT — DAY 135", "README roadmap DAY 135 next"),
    (rm, "make hardened-full", "README quick start hardened-full"),
    (rm, "DAY 134: **Pipeline E2E hardened", "README milestone DAY 134"),
    (rm, "DAY 135:", "README milestone DAY 135"),
    (rm, "DEBT-CONFIDENCE-SCORE-001", "README deuda confidence"),
    (rm, "Draft v18 COMPLETO", "README paper v18 completo"),
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
    print("❌ Algunas verificaciones fallaron")
    sys.exit(1)
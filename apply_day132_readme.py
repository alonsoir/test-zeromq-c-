#!/usr/bin/env python3
# apply_day132_readme.py
# Aplica los cambios de DAY 132 al README.md
# macOS safe

with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Actualizar estado actual
content = content.replace(
    '## Estado actual — DAY 130 (2026-04-25)',
    '## Estado actual — DAY 132 (2026-04-26)'
)

content = content.replace(
    '**Tag activo:** `v0.5.2-hardened` | **Commit:** `aab08daa` | **Branch activa:** `main` (limpio)',
    '**Tag activo:** `v0.5.2-hardened` | **Commit:** `18d8e101` | **Branch activa:** `main` (limpio) · `feature/adr030-variant-a` (P2 en curso)'
)

content = content.replace(
    '**Keypair activo:** `1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`',
    '**Keypair activo:** `1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`\n**Paper:** arXiv:2604.04952 · Draft v17 en GitHub (pre-arXiv)'
)

# 2. Reemplazar hitos DAY 130 con DAY 132
OLD_HITOS = '''### Hitos DAY 130
- **PROTOCOLO CANÓNICO ESTABLECIDO** — `vagrant destroy -f && vagrant up && make bootstrap && make test-all` es el inicio obligatorio de toda sesión.
- **DEBT-SYSTEMD-AUTOINSTALL-001 CERRADA** — `install-systemd-units.sh` integrado en Vagrantfile.
- **DEBT-SAFE-EXEC-NULLBYTE-001 CERRADA** — `is_safe_for_exec()` en `safe_exec.hpp`. 17/17 tests GREEN.
- **DEBT-GITGUARDIAN-YAML-001 CERRADA** — `.gitguardian.yaml` paths_ignore v2.
- **DEBT-FUZZING-LIBFUZZER-001 CERRADA** — 2.4M runs, 0 crashes, corpus 67 ficheros. `make fuzz-all`.
- **DEBT-MARKDOWN-HOOK-001 CERRADA** — pre-commit hook detecta corrupción markdown en .cpp/.hpp.
- **Asciinema:** `docs/argus-day130-bootstrap-20260425-142211.cast`'''

NEW_HITOS = '''### Hitos DAY 132
- **Paper Draft v17 COMPLETADO** — 4 nuevas secciones §6: RED→GREEN gate, Fuzzing tercera capa, CWE-78 execv(), BSR axiom. +315 líneas. Compilado y verificado en Overleaf. Pendiente arXiv hasta tener métricas reales.
- **HARDWARE-REQUIREMENTS.md** — especificaciones mínimas y recomendadas para aRGus-production (ADR-030). Cierra DEBT-PROD-COMPAT-BASELINE-001.
- **vagrant/hardened-x86/Vagrantfile** — VM Debian 12 + AppArmor enforcing + BSR axiom (sin compilador). Skeleton ADR-030 Variant A.
- **README Prerequisites** — sección de instalación de Vagrant + VirtualBox + make para macOS y Linux.
- **Consejo 8/8 DAY 132** — 5 nuevas deudas de seguridad para imagen de producción: AppArmor primera línea, Falco, FS mínimo, apt sources integrity, upgrade Debian 13.
- **Arquitectura de seguridad de producción definida** — superficie de ataque mínima como principio estructural, no como lista de checks.

### Hitos DAY 130
- **PROTOCOLO CANÓNICO ESTABLECIDO** — `vagrant destroy -f && vagrant up && make bootstrap && make test-all` es el inicio obligatorio de toda sesión.
- **DEBT-SYSTEMD-AUTOINSTALL-001 CERRADA** — `install-systemd-units.sh` integrado en Vagrantfile.
- **DEBT-SAFE-EXEC-NULLBYTE-001 CERRADA** — `is_safe_for_exec()` en `safe_exec.hpp`. 17/17 tests GREEN.
- **DEBT-GITGUARDIAN-YAML-001 CERRADA** — `.gitguardian.yaml` paths_ignore v2.
- **DEBT-FUZZING-LIBFUZZER-001 CERRADA** — 2.4M runs, 0 crashes, corpus 67 ficheros. `make fuzz-all`.
- **DEBT-MARKDOWN-HOOK-001 CERRADA** — pre-commit hook detecta corrupción markdown en .cpp/.hpp.
- **Asciinema:** `docs/argus-day130-bootstrap-20260425-142211.cast`'''

content = content.replace(OLD_HITOS, NEW_HITOS)

# 3. Actualizar tabla de deuda abierta
content = content.replace(
    '''### Deuda técnica abierta
Ver [docs/BACKLOG.md](docs/BACKLOG.md) para detalle completo.

| Deuda | Prioridad | Target |
|-------|-----------|--------|
| DEBT-SEED-CAPABILITIES-001 | ⏳ Baja | v0.6+ |
| DEBT-SAFE-PATH-RESOLVE-MODEL-001 | ⏳ | feature/adr038-acrl |
| DEBT-NATIVE-LINUX-BOOTSTRAP-001 | ⏳ | post-FEDER |''',
    '''### Deuda técnica abierta
Ver [docs/BACKLOG.md](docs/BACKLOG.md) para detalle completo.

| Deuda | Prioridad | Target |
|-------|-----------|--------|
| DEBT-PROD-APPARMOR-COMPILER-BLOCK-001 | 🔴 Alta | feature/adr030-variant-a |
| DEBT-PROD-FALCO-EXOTIC-PATHS-001 | 🔴 Alta | feature/adr030-variant-a |
| DEBT-PROD-FS-MINIMIZATION-001 | 🔴 Alta | feature/adr030-variant-a |
| DEBT-PROD-APT-SOURCES-INTEGRITY-001 | 🔴 Crítica | feature/adr030-variant-a |
| DEBT-PAPER-FUZZING-METRICS-001 | 🟡 Media | DAY 133 (pre-arXiv) |
| DEBT-SEED-CAPABILITIES-001 | ⏳ Baja | v0.6+ |
| DEBT-SAFE-PATH-RESOLVE-MODEL-001 | ⏳ | feature/adr038-acrl |
| DEBT-NATIVE-LINUX-BOOTSTRAP-001 | ⏳ | post-FEDER |'''
)

# 4. Actualizar roadmap NEXT
content = content.replace(
    '### 🔜 NEXT — DAY 130: Fuzzing + Null byte + Limpieza\n\n| Priority | Task |\n|---|---|\n| 🔴 P0 BLOQUEANTE | DEBT-SAFE-EXEC-NULLBYTE-001 — null byte check en safe_exec() + test RED→GREEN |\n| 🔴 P0 | DEBT-FUZZING-LIBFUZZER-001 — libFuzzer sobre validate_chain_name + parsers ZMQ |\n| 🟡 P1 | DEBT-GITIGNORE-BUILD-001 — **/build-debug/ en .gitignore |\n| 🟡 P1 | DEBT-GITGUARDIAN-YAML-001 — limpiar deprecated keys |\n| 🟡 P1 | DEBT-MARKDOWN-HOOK-001 — pre-commit hook [word](http:// en .cpp/.hpp |\n| 🟡 P2 | Paper §5 — Draft v17 (property testing + safe_path taxonomy) |',
    '''### ✅ DONE — DAY 132 (26 Apr 2026) — Draft v17 + ADR-030 inicio 🎉
- [x] **Paper Draft v17** ✅ — §6.5 RED→GREEN gate · §6.8 Fuzzing · §6.10 CWE-78 · §6.12 BSR axiom
- [x] **HARDWARE-REQUIREMENTS.md** ✅ — DEBT-PROD-COMPAT-BASELINE-001 cerrada
- [x] **vagrant/hardened-x86/Vagrantfile** ✅ — ADR-030 Variant A skeleton
- [x] **README Prerequisites** ✅ — Vagrant + VirtualBox + make install instructions
- [x] **Consejo 8/8** ✅ — 5 nuevas deudas de seguridad producción documentadas

### 🔜 NEXT — DAY 133: Makefile prod targets + métricas paper

| Priority | Task |
|---|---|
| 🔴 P0 | Makefile targets: `prod-build-x86`, `prod-sign`, `prod-checksums`, `prod-verify`, `check-prod-no-compiler` (dpkg + command -v), `check-prod-checksec` |
| 🔴 P0 | DEBT-PAPER-FUZZING-METRICS-001 — añadir 2.4M/0 crashes/67 corpus a §6.8 + pedir al Consejo explicación de "misses nothing within CPU time" |
| 🔴 P0 | Métricas VM hardened para §6.12 BSR: `dpkg -l \| wc -l` dev vs hardened |
| 🟡 P1 | DEBT-PROD-APPARMOR-COMPILER-BLOCK-001 — perfiles AppArmor anti-compilador |
| 🟡 P1 | DEBT-PROD-APT-SOURCES-INTEGRITY-001 — SHA-256 sources.list en boot check |
| 🟡 P2 | DEBT-PROD-FALCO-EXOTIC-PATHS-001 — reglas Falco para paths exóticos |'''
)

# 5. Actualizar milestones
content = content.replace(
    '- 🔜 DAY 130: **DEBT-SAFE-EXEC-NULLBYTE-001 · libFuzzer · .gitignore · Paper §5**',
    '''- ✅ DAY 130: **DEBT-SAFE-EXEC-NULLBYTE-001 · libFuzzer 2.4M runs · .gitignore · REGLA EMECAS** 🎉
- ✅ DAY 132: **Paper Draft v17 · HARDWARE-REQUIREMENTS · vagrant/hardened-x86 · Prerequisites README · Consejo 8/8 arquitectura producción** 🎉
- 🔜 DAY 133: **Makefile prod targets · métricas fuzzing paper · AppArmor anti-compilador · apt sources integrity**'''
)

# 6. Actualizar preprint a Draft v17
content = content.replace(
    '**Published:** 3 April 2026 · **Draft v16** (updated 19 April 2026) · MIT license',
    '**Published:** 3 April 2026 · **Draft v17** (DAY 132 — pre-arXiv, en revisión) · MIT license'
)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("README.md actualizado correctamente")
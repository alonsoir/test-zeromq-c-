# Prompt de Continuidad — DAY 133
*aRGus NDR — arXiv:2604.04952 — 27 Abril 2026*

---

## REGLA EMECAS — OBLIGATORIA ANTES DE CUALQUIER ACCION

```bash
vagrant destroy -f
vagrant up
make bootstrap
make test-all
```

**Esta regla es innegociable hasta que el pipeline esté en modo solo lectura
y mantenimiento.** No se toca ningún fichero, no se ejecuta ningún comando
técnico, no se abre ningún editor hasta que `make test-all` devuelva
`ALL TESTS COMPLETE` en una VM destruida y reconstruida desde cero.

Si falla en cualquier punto: diagnosticar, corregir, repetir desde
`vagrant destroy -f`.

*Bautizada en honor de Emerson (emecas@inspiron), que intentó saltarse
Vagrant y sin querer certificó que el protocolo es sólido.*

---

## Estado del proyecto al inicio de DAY 133

**Repositorio:** `alonsoir/argus` en GitHub
**Branch activa principal:** `main` @ `18d8e101` — sagrado
**Branch de trabajo:** `feature/adr030-variant-a` @ `9b3438fb`
**Tag activo:** `v0.5.2-hardened`
**Paper:** arXiv:2604.04952 — Draft v17 en GitHub, pendiente arXiv

### Keypair activo (post-rebuild DAY 130)
`1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`

### Pipeline esperado tras REGLA EMECAS
- 6/6 RUNNING: etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall
- TEST-INTEG-SIGN: 7/7 PASSED
- make test-all: ALL TESTS COMPLETE
- Fallo pre-existente conocido (no regresión): `rag-ingester test_config_parser` 1/8

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

---

## Trabajo completado DAY 132

### Commits en `main`
- `5a22c068` — docs: ADR-025 D13 + BACKLOG DAY 131 + prompt continuidad DAY 132
- `b7d38d1f` — docs: paper Draft v17 — RED→GREEN gate, fuzzing layer, CWE-78 execv(), BSR axiom
- `18d8e101` — docs: add Prerequisites section — Vagrant + VirtualBox + make (DAY 132)

### Commits en `feature/adr030-variant-a`
- `9b3438fb` — feat: ADR-030 Variant A — HARDWARE-REQUIREMENTS.md + vagrant/hardened-x86/Vagrantfile

### Paper Draft v17 — §6 expandido
| Subsección | Contenido |
|-----------|-----------|
| §6.5 | The RED→GREEN Gate: Merge as a Non-Negotiable Contract |
| §6.8 | Fuzzing as the Third Testing Layer (harness libFuzzer concreto) |
| §6.10 | CWE-78: execv() Without a Shell as a Physical Barrier |
| §6.12 | The Build/Runtime Separation Axiom (ADR-039, Thompson 1984) |

Compilación Overleaf verificada: 42 páginas, listings 6–13 correctos, referencias [16] [23] integradas.

### Decisiones arquitectónicas DAY 132 (founder)
1. **AppArmor es la primera línea de defensa BSR**, no dpkg. check-prod-no-compiler es auditoría.
2. **Falco** entra en el stack de producción: vigilancia runtime de paths exóticos.
3. **FS de producción mínimo**: solo paths imprescindibles para el pipeline. /tmp noexec.
4. **apt sources integrity**: SHA-256 firmado. Si cambia → pipeline no arranca (fail-closed).
5. **RED→GREEN se mantiene**: formalización de Kimi rechazada por claridad operacional.
6. **Método científico puro**: medir y publicar lo que salga. Sin adornar.

### Nuevas deudas abiertas (DAY 132)
| ID | Descripción | Target |
|----|-------------|--------|
| DEBT-PROD-APPARMOR-COMPILER-BLOCK-001 | AppArmor bloquea compiladores en producción | feature/adr030-variant-a |
| DEBT-PROD-FALCO-EXOTIC-PATHS-001 | Falco vigila paths exóticos en runtime | feature/adr030-variant-a |
| DEBT-PROD-FS-MINIMIZATION-001 | FS producción: acceso mínimo + noexec en exóticos | feature/adr030-variant-a |
| DEBT-PROD-APT-SOURCES-INTEGRITY-001 | SHA-256 sources.list en boot check, fail-closed | feature/adr030-variant-a |
| DEBT-DEBIAN13-UPGRADE-001 | Upgrade path Debian 12→13 bare-metal | post-FEDER |
| DEBT-PAPER-FUZZING-METRICS-001 | Métricas reales en §6.8 + corregir frase "misses nothing" | DAY 133 |

---

## Plan DAY 133

### P0 — Métricas para el paper (pre-arXiv)
**Antes de cualquier trabajo técnico en Makefile:**

1. Arrancar la hardened VM y medir:
```bash
cd vagrant/hardened-x86
vagrant up
vagrant ssh -c 'dpkg -l | wc -l'           # paquetes en hardened
vagrant ssh -c 'du -sh /'                   # tamaño imagen
vagrant ssh -c 'dpkg -l | grep -c ^ii'      # paquetes instalados exactos
```
Comparar con dev VM: `vagrant ssh -c 'dpkg -l | wc -l'` desde raíz del proyecto.

2. Añadir tabla a §6.12 del paper (BSR axiom):
```
| Environment | Packages | Image size | Compilers |
| Dev VM      | ~XXX     | ~XX GB     | gcc, g++, clang, cmake, ... |
| Hardened VM | ~XX      | ~X GB      | NONE |
```

3. Añadir tabla a §6.8 del paper (Fuzzing):
```
| Target | Runs | Crashes | Corpus | Time |
| validate_chain_name | 2.4M | 0 | 67 files | 30s |
| validate_filepath   | ...  | 0 | ...      | ... |
| safe_exec           | ...  | 0 | ...      | ... |
```
(datos de DEBT-FUZZING-LIBFUZZER-001, ya cerrada DAY 130)

4. Pedir al Consejo (todos los modelos) explicación de la frase:
   **"Fuzzing misses nothing within CPU time"**
   — El founder no la entiende. El Consejo debe explicarla para que aprendamos
   juntos, y proponer reformulación con precisión científica. Ver Q5 en convocatoria.

### P1 — Makefile targets de producción

En `feature/adr030-variant-a`, añadir al Makefile raíz:

```makefile
# ── Producción (solo desde dev VM) ──────────────────────────────────────────

_check-dev-env:
	@which clang++ > /dev/null 2>&1 || \
	  (echo "FAIL: prod targets requieren dev VM (clang++ no encontrado)"; exit 1)

prod-build-x86: _check-dev-env
	@echo "=== Building production binaries (x86-64) ==="
	# Compilar con -O2 -DNDEBUG, sin símbolos debug, a dist/

prod-sign: _check-dev-env
	@echo "=== Signing production binaries ==="
	# Ed25519 sobre cada binario en dist/

prod-checksums: _check-dev-env
	@echo "=== Generating SHA256SUMS ==="
	# sha256sum dist/* > dist/SHA256SUMS

prod-verify:
	@echo "=== Verifying production binaries ==="
	# Verificar SHA256SUMS + firma Ed25519

check-prod-no-compiler:
	@echo "=== BSR: verifying no compiler in production ==="
	@# Capa 1: dpkg
	@if dpkg -l 2>/dev/null | grep -qE '^ii\s+(gcc|g\+\+|clang|cmake|build-essential)'; then \
	  echo "FAIL: compiler found via dpkg"; exit 1; fi
	@# Capa 2: PATH
	@for cmd in gcc g++ clang clang++ cc c++ cmake; do \
	  if command -v $$cmd > /dev/null 2>&1; then \
	    echo "FAIL: $$cmd found in PATH"; exit 1; fi; \
	done
	@echo "OK: no compiler present (dpkg + PATH verified)"

check-prod-checksec:
	@echo "=== checksec on production binaries ==="
	@which checksec > /dev/null 2>&1 || (echo "FAIL: checksec not installed"; exit 1)
	@for f in dist/*; do checksec --file=$$f; done
```

**Regla:** targets `prod-*` solo en dev VM. Targets `check-prod-*` solo en hardened VM (o CI).

### P2 — Commit y actualizar documentos (si P0 y P1 completos)

```bash
# En feature/adr030-variant-a
git add Makefile docs/latex/main.tex docs/latex/references.bib
git commit -m "feat: prod Makefile targets + paper §6.8/§6.12 métricas reales (DAY 133)"
git push origin feature/adr030-variant-a
```

### P3 — Convocatoria al Consejo DAY 133 (si tiempo)
Preguntar al Consejo:
- Q1: Revisión de los 6 Makefile targets de producción
- Q2: Revisión de la tabla BSR metrics (dev vs hardened)
- Q3: Revisión de la tabla de fuzzing en §6.8
- Q4: Estrategia de implementación de Falco en la imagen hardened (¿reglas base o custom?)
- **Q5: Explicar "Fuzzing misses nothing within CPU time" — aprender juntos, reformular con precisión**

---

## Deudas abiertas relevantes para DAY 133

| ID | Descripción | Prioridad |
|----|-------------|-----------|
| DEBT-PAPER-FUZZING-METRICS-001 | Métricas reales §6.8 + §6.12 + corregir frase | 🔴 P0 pre-arXiv |
| DEBT-PROD-APPARMOR-COMPILER-BLOCK-001 | AppArmor anti-compilador | 🔴 P1 |
| DEBT-PROD-APT-SOURCES-INTEGRITY-001 | SHA-256 sources.list en boot | 🔴 P1 |
| DEBT-PROD-FALCO-EXOTIC-PATHS-001 | Falco paths exóticos | 🟡 P2 |
| DEBT-PROD-FS-MINIMIZATION-001 | FS mínimo + noexec | 🟡 P2 |
| DEBT-DEBIAN13-UPGRADE-001 | Upgrade path Debian 13 | ⏳ post-FEDER |

---

## Backlog FEDER-001

**Deadline:** 22 septiembre 2026
**Go/no-go técnico:** 1 agosto 2026
**Contacto:** Andrés Caro Lindo (UEx/INCIBE)

**Prerequisites pendientes:**
- [ ] ADR-030 Variant A (x86 + AppArmor) estable ← en curso DAY 133
- [ ] ADR-030 Variant B (ARM64 + AppArmor + libpcap) estable
- [ ] Demo pcap reproducible en < 10 minutos (`scripts/feder-demo.sh`)
- [ ] Paper §6 con métricas reales (pre-arXiv v17)
- [ ] Clarificar con Andrés: NDR standalone vs federación (antes julio 2026)

---

## Decisiones de diseño nuevas (DAY 132) — para referencia rápida

| Decisión | Resolución |
|----------|-----------|
| **AppArmor como primera línea BSR** | AppArmor bloquea compiladores. dpkg check es auditoría, no defensa. |
| **Falco en producción** | Vigilancia runtime de paths exóticos. AppArmor previene; Falco detecta. |
| **FS mínimo en producción** | Solo paths necesarios para el pipeline. /tmp, /var/tmp noexec. |
| **apt sources integrity** | SHA-256 firmado en imagen. Si cambia → fail-closed. Sin excepciones. |
| **Makefile raíz con prefijo prod-** | Guard _check-dev-env. No Makefile.production separado. |
| **debian/bookworm64** | Reproducibilidad sobre novedad. Trixie: upgrade path documentado. |
| **Dos capas BSR check** | dpkg + command -v. La defensa real es AppArmor+Falco+FS mínimo. |
| **Método científico puro para paper** | Medir, publicar lo que salga con procedimiento. Sin adornar. |
| **RED→GREEN se mantiene** | Claridad operacional sobre formalización matemática. |

---

## Reglas permanentes del proyecto

- **REGLA EMECAS:** `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
- **macOS:** Nunca `sed -i` sin `-e ''`. Scripts con emojis → `python3 << 'PYEOF'`.
- **VM↔macOS:** Solo `scp -F /tmp/vagrant-ssh-config`. Prohibido pipe zsh.
- **vagrant ssh:** Siempre con `-c '...'`.
- **JSON es la ley:** No hardcoded values.
- **Fail-closed:** En caso de duda, rechazar.
- **dist/:** Nunca en git. SHA256SUMS obligatorio. BSR axiom (ADR-039).
- **Lógica compleja:** Siempre a `tools/script.sh`, nunca inline en Makefile.
- **Seed ChaCha20:** NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **Seguridad:** Todo fix requiere RED→GREEN + property test + test integración. Sin excepciones.
- **main:** Sagrado. Solo entra lo que pasa REGLA EMECAS en VM destruida y reconstruida.
- **AppArmor es la primera línea de defensa**, no los checks de herramientas.
- **apt sources:** Si se modifican → pipeline no arranca. Sin negociación.

---

*DAY 133 — 27 Abril 2026 · feature/adr030-variant-a @ 9b3438fb · main @ 18d8e101*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"La superficie de ataque mínima no es una aspiración. Es una decisión de diseño."*
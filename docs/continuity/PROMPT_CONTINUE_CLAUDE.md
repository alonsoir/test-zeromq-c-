# ML Defender (aRGus NDR) — DAY 128 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA SCRIPTS:** Lógica compleja → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **REGLA macOS/sed:** Nunca `sed -i` sin `-e ''`. Usar Python3 heredoc para ediciones de ficheros en macOS.
- **REGLA PERMANENTE (Consejo 7/7 DAY 124):** Ningún fix de seguridad en código de producción se mergea sin test de demostración RED→GREEN. El test debe fallar con el código antiguo y pasar con el nuevo. Sin excepciones.
- **REGLA PERMANENTE (Consejo 8/8 DAY 125):** Todo fix de seguridad incluye: (1) unit test sintético, (2) property test de invariante, (3) test de integración en componente real. Sin excepciones.
- **REGLA PERMANENTE (Consejo 8/8 DAY 127):** Toda nueva superficie de ficheros se clasifica con PathPolicy antes de implementar. Documentar en docs/SECURITY-PATH-PRIMITIVES.md.

---

## Estado al cierre de DAY 127

### Branch activa
`main` — limpio. Tag activo: `v0.5.2-hardened`.

### Último merge
DAY 127 — `fix/day127-dev-prod-symlink` mergeado a main.
Commit: `resolve_config()` + Makefile paths absolutos via `/etc/ml-defender/`.

### Hitos completados DAY 125-127
- **DAY 125:** 5 deudas cerradas. Hallazgo: property test encontró bug latente en fix F17.
- **DAY 126:** 4 deudas críticas cerradas. `lstat()` pre-`resolve()`. Prefix fijo. RED→GREEN traversal tests seed-client + firewall. Tag `v0.5.2-hardened`.
- **DAY 127:** `DEBT-DEV-PROD-SYMLINK-001` ✅ — `resolve_config()` nueva primitiva safe_path. `lexically_normal()` para verificar prefix ANTES de seguir symlinks. Paridad dev/prod via `/etc/ml-defender/*.json → /vagrant/`. 6/6 RUNNING. Consejo 8/8 feedback recibido.

### Hallazgos técnicos clave DAY 125-127
1. **`fs::is_symlink(resolved)` es inútil post-`weakly_canonical()`** — `lstat()` sobre el path original es la única defensa correcta para material criptográfico.
2. **`lexically_normal()` vs `weakly_canonical()`** — dos herramientas para dos casos de seguridad distintos. Para configs con symlinks legítimos, verificar prefix ANTES de resolver.
3. **Property test encontró bug en el propio fix** — `int64_t` overflow en fix F17. Validación empírica de la adopción sistémica de property testing.

### Taxonomía safe_path (Consejo 8/8 DAY 127 — PERMANENTE)
```
resolve()        → validación general (weakly_canonical post-resolución)
resolve_seed()   → material criptográfico (lstat pre-resolución, sin symlinks, 0400)
resolve_config() → configs con symlinks legítimos (lexically_normal pre-resolución)
resolve_model()  → [BACKLOG ADR-038] modelos firmados Ed25519
```

---

## PASO 0 — DAY 128: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE|COMPLETE"
```

Si la VM está parada: `make up && make bootstrap`

---

## PASO 1 — DEBT-SAFE-PATH-TAXONOMY-DOC-001 (30 min)

### Qué hacer
Crear `docs/SECURITY-PATH-PRIMITIVES.md` con:
- Tabla de las 4 primitivas (3 activas + 1 futura)
- Diagrama de decisión textual: "¿Material criptográfico? → resolve_seed. ¿Symlinks legítimos? → resolve_config. ¿General? → resolve. ¿Modelo firmado? → resolve_model (backlog)"
- PathPolicy enum conceptual (documentación, no implementación)
- Ejemplos de uso correcto e incorrecto

Gate: fichero existe en `docs/SECURITY-PATH-PRIMITIVES.md` y está versionado.

---

## PASO 2 — DEBT-PROPERTY-TESTING-PATTERN-001 (45 min)

### Qué hacer
Crear `docs/testing/PROPERTY-TESTING.md` con:
- Qué es un property test (invariante matemática, no caso específico)
- Cuándo usarlo (toda superficie crítica con operaciones aritméticas o de paths)
- Patrón estándar: identificar invariante → escribir loop → verificar RED con código antiguo → GREEN con código nuevo
- Ejemplos reales del proyecto: F17 (`compute_memory_mb`), `resolve_seed`, `resolve_config`
- Relación con otras técnicas: unit tests (base) → property tests (invariantes) → fuzzing libFuzzer (parsers) → mutation testing (calidad suite)

Además, añadir 3 property tests nuevos RED→GREEN:
- `safe_path::resolve_seed` — nunca escapa prefix, nunca acepta symlinks
- `safe_path::resolve_config` — nunca escapa prefix lexical, acepta symlinks dentro
- `config_parser` — prefix fijo nunca deriva del input

Gate: `docs/testing/PROPERTY-TESTING.md` existe + 3 property tests PASSED en ctest.

---

## PASO 3 — DEBT-PROVISION-PORTABILITY-001 (30 min)

### Qué hacer
Localizar todas las instancias hardcodeadas de `vagrant` como service user en `provision.sh`:

```bash
grep -n "vagrant\|chown\|chmod\|service_user" \
  /Users/aironman/CLionProjects/test-zeromq-docker/tools/provision.sh | head -30
```

Fix: añadir al inicio de `provision.sh`:
```bash
ARGUS_SERVICE_USER="${ARGUS_SERVICE_USER:-vagrant}"
```

Sustituir todas las instancias hardcodeadas de `vagrant` (en contextos de chown/permisos) por `${ARGUS_SERVICE_USER}`.

Gate: `TEST-PROVISION-1` verde tras el fix. `provision.sh` con `ARGUS_SERVICE_USER=testuser` → seeds con permisos `0400 testuser:testuser`.

---

## PASO 4 — DEBT-SNYK-WEB-VERIFICATION-001 (cuando estés en el navegador)

Ejecutar Snyk web sobre `v0.5.2-hardened`:
- URL: https://app.snyk.io
- Target: repositorio `alonsoir/argus`
- Filtro: código C++ de producción

Criterio de triage (Consejo 8/8 DAY 127):
- Código propio + path/overflow/crypto → Fix bloqueante con RED→GREEN
- Código propio + otro HIGH → Fix próximo sprint
- Third-party no alcanzable → Documentar en `docs/security/SNYK-DAY-128.md` con justificación
- Falso positivo demostrable → Cerrar con justificación documentada

Gate: `docs/security/SNYK-DAY-128.md` existe con clasificación de todos los findings.

---

## PASO 5 — Commit y push

```bash
git add -A
git commit -F - << 'EOF'
docs: DAY 128 — safe_path taxonomy + property testing pattern + provision portability

- docs/SECURITY-PATH-PRIMITIVES.md: taxonomia safe_path con PathPolicy enum
  conceptual y diagrama de decision (Consejo 8/8 DAY 127)
- docs/testing/PROPERTY-TESTING.md: patron formal de property testing
  con ejemplos reales del proyecto (F17, resolve_seed, resolve_config)
- 3 property tests nuevos RED->GREEN: resolve_seed + resolve_config + config_parser
- provision.sh: ARGUS_SERVICE_USER variable (default vagrant, portable)
- TEST-PROVISION-1: verde post-fix

Consejo 8/8 DAY 127: property testing primero, fuzzing despues.
La herramienta propone; el modelo de amenazas decide (criterio Snyk).
EOF

make test-all 2>&1 | grep -E "ALL TESTS|COMPLETE"
git push origin main
```

---


---

## Contexto estratégico — BACKLOG-BIZMODEL-001

### Decisiones tomadas DAY 127 (documentadas en docs/BACKLOG-BIZMODEL-001.md)

**Modelo de negocio:** Open-core. Core MIT siempre gratis. Servicio cloud/fleet de pago.

**Paraguas institucional:** UEx/INCIBE — preferencia clara. No es un acuerdo, es una
propuesta que hay que presentar a Andrés Caro Lindo antes de julio 2026. Ellos tienen
prioridad y autonomía total.

**Despliegue:** On-premise primero. Nube europea soberana (Hetzner/OVH/IONOS) para el
servicio cloud. Internacionalización = servidores clonados en territorio del cliente.

**Hardware pendiente (financiable FEDER):**
- Raspberry Pi 4/5 ARM64 × 4-8 → caracterización edge deployment
- Mini PCs x86 × 2-4 → perfil municipio/escuela
- Servidor de desarrollo (NVIDIA Spark DGX o equivalente MSI) → sin esto el portátil
  del 2019 es un riesgo de proyecto real
- Servidor central telemetry-server → CPD UEx o nube europea soberana

**telemetry-collector (ADR-039 pendiente):**
- Tap en el pipeline, sin tocar el flujo a rag-ingester
- Mismo crypto-transport (ChaCha20-Poly1305 + HKDF) — no se reinventa nada
- Collect-first, anonymize-later — geolocalización SIEMPRE asíncrona en servidor
- Priorización geográfica de distribución de modelos (frente de propagación)
- Bloqueante de hardware: NO para el código. SÍ para validación a escala real.

## Contexto permanente

### Secuencia canónica
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Branch activa
`main` — limpio. Tag: `v0.5.2-hardened`.

### Estado de deuda al inicio de DAY 128
```
🟡 DEBT-SNYK-WEB-VERIFICATION-001        → DAY 128 (navegador) — PASO 4
🟡 DEBT-PROPERTY-TESTING-PATTERN-001     → DAY 128 — PASO 2
🟡 DEBT-SAFE-PATH-TAXONOMY-DOC-001       → DAY 128 — PASO 1
🟢 DEBT-PROVISION-PORTABILITY-001        → DAY 128 — PASO 3
⏳ DEBT-SAFE-PATH-RESOLVE-MODEL-001      → feature/adr038-acrl
⏳ DEBT-FUZZING-LIBFUZZER-001            → post-property-testing
⏳ DEBT-PENTESTER-LOOP-001               → POST-DEUDA
```

### Modelos firmados activos
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
  wednesday_eval_report.json        (OOD finding sealed)
```

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/abs/2604.04952
Pendiente: Draft v17 con §5 actualizado:
- §5.3 "Property Testing as a Security Fix Validator" (hallazgo F17 DAY 125)
- §5.4 "Dev/Prod Parity via Symlinks, Not Conditional Logic" (DAY 127)
- §5.5 "RED→GREEN as Non-Negotiable Merge Gate"
- §5.x "Taxonomía safe_path: lexically_normal vs weakly_canonical" (distinción no documentada en literatura C++20)

Referencias a citar: QuickCheck (Claessen & Hughes), CWE-22/23, TOCTOU literature, OWASP Path Traversal.

### Pregunta crítica FEDER pendiente (Consejo DAY 127)
¿La demo FEDER requiere federación funcional (ADR-038) o es suficiente con NDR standalone?
Clarificar con Andrés Caro Lindo ANTES de julio 2026.
Deadline FEDER: 22 septiembre 2026.

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

### REGLA DE ORO DAY 128
Documentar no es burocracia — es el contrato que garantiza que el hallazgo
de hoy no se convierte en deuda de mañana.
Un patrón sin documentación es un patrón que se reinventa cada vez.

---# ML Defender (aRGus NDR) — DAY 128 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA SCRIPTS:** Lógica compleja → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **REGLA macOS/sed:** Nunca `sed -i` sin `-e ''`. Usar Python3 heredoc para ediciones de ficheros en macOS.
- **REGLA PERMANENTE (Consejo 7/7 DAY 124):** Ningún fix de seguridad en código de producción se mergea sin test de demostración RED→GREEN. El test debe fallar con el código antiguo y pasar con el nuevo. Sin excepciones.
- **REGLA PERMANENTE (Consejo 8/8 DAY 125):** Todo fix de seguridad incluye: (1) unit test sintético, (2) property test de invariante, (3) test de integración en componente real. Sin excepciones.
- **REGLA PERMANENTE (Consejo 8/8 DAY 127):** Toda nueva superficie de ficheros se clasifica con PathPolicy antes de implementar. Documentar en docs/SECURITY-PATH-PRIMITIVES.md.

---

## Estado al cierre de DAY 127

### Branch activa
`main` — limpio. Tag activo: `v0.5.2-hardened`.

### Último merge
DAY 127 — `fix/day127-dev-prod-symlink` mergeado a main.
Commit: `resolve_config()` + Makefile paths absolutos via `/etc/ml-defender/`.

### Hitos completados DAY 125-127
- **DAY 125:** 5 deudas cerradas. Hallazgo: property test encontró bug latente en fix F17.
- **DAY 126:** 4 deudas críticas cerradas. `lstat()` pre-`resolve()`. Prefix fijo. RED→GREEN traversal tests seed-client + firewall. Tag `v0.5.2-hardened`.
- **DAY 127:** `DEBT-DEV-PROD-SYMLINK-001` ✅ — `resolve_config()` nueva primitiva safe_path. `lexically_normal()` para verificar prefix ANTES de seguir symlinks. Paridad dev/prod via `/etc/ml-defender/*.json → /vagrant/`. 6/6 RUNNING. Consejo 8/8 feedback recibido.

### Hallazgos técnicos clave DAY 125-127
1. **`fs::is_symlink(resolved)` es inútil post-`weakly_canonical()`** — `lstat()` sobre el path original es la única defensa correcta para material criptográfico.
2. **`lexically_normal()` vs `weakly_canonical()`** — dos herramientas para dos casos de seguridad distintos. Para configs con symlinks legítimos, verificar prefix ANTES de resolver.
3. **Property test encontró bug en el propio fix** — `int64_t` overflow en fix F17. Validación empírica de la adopción sistémica de property testing.

### Taxonomía safe_path (Consejo 8/8 DAY 127 — PERMANENTE)
```
resolve()        → validación general (weakly_canonical post-resolución)
resolve_seed()   → material criptográfico (lstat pre-resolución, sin symlinks, 0400)
resolve_config() → configs con symlinks legítimos (lexically_normal pre-resolución)
resolve_model()  → [BACKLOG ADR-038] modelos firmados Ed25519
```

---

## PASO 0 — DAY 128: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main && git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE|COMPLETE"
```

Si la VM está parada: `make up && make bootstrap`

---

## PASO 1 — DEBT-SAFE-PATH-TAXONOMY-DOC-001 (30 min)

### Qué hacer
Crear `docs/SECURITY-PATH-PRIMITIVES.md` con:
- Tabla de las 4 primitivas (3 activas + 1 futura)
- Diagrama de decisión textual: "¿Material criptográfico? → resolve_seed. ¿Symlinks legítimos? → resolve_config. ¿General? → resolve. ¿Modelo firmado? → resolve_model (backlog)"
- PathPolicy enum conceptual (documentación, no implementación)
- Ejemplos de uso correcto e incorrecto

Gate: fichero existe en `docs/SECURITY-PATH-PRIMITIVES.md` y está versionado.

---

## PASO 2 — DEBT-PROPERTY-TESTING-PATTERN-001 (45 min)

### Qué hacer
Crear `docs/testing/PROPERTY-TESTING.md` con:
- Qué es un property test (invariante matemática, no caso específico)
- Cuándo usarlo (toda superficie crítica con operaciones aritméticas o de paths)
- Patrón estándar: identificar invariante → escribir loop → verificar RED con código antiguo → GREEN con código nuevo
- Ejemplos reales del proyecto: F17 (`compute_memory_mb`), `resolve_seed`, `resolve_config`
- Relación con otras técnicas: unit tests (base) → property tests (invariantes) → fuzzing libFuzzer (parsers) → mutation testing (calidad suite)

Además, añadir 3 property tests nuevos RED→GREEN:
- `safe_path::resolve_seed` — nunca escapa prefix, nunca acepta symlinks
- `safe_path::resolve_config` — nunca escapa prefix lexical, acepta symlinks dentro
- `config_parser` — prefix fijo nunca deriva del input

Gate: `docs/testing/PROPERTY-TESTING.md` existe + 3 property tests PASSED en ctest.

---

## PASO 3 — DEBT-PROVISION-PORTABILITY-001 (30 min)

### Qué hacer
Localizar todas las instancias hardcodeadas de `vagrant` como service user en `provision.sh`:

```bash
grep -n "vagrant\|chown\|chmod\|service_user" \
  /Users/aironman/CLionProjects/test-zeromq-docker/tools/provision.sh | head -30
```

Fix: añadir al inicio de `provision.sh`:
```bash
ARGUS_SERVICE_USER="${ARGUS_SERVICE_USER:-vagrant}"
```

Sustituir todas las instancias hardcodeadas de `vagrant` (en contextos de chown/permisos) por `${ARGUS_SERVICE_USER}`.

Gate: `TEST-PROVISION-1` verde tras el fix. `provision.sh` con `ARGUS_SERVICE_USER=testuser` → seeds con permisos `0400 testuser:testuser`.

---

## PASO 4 — DEBT-SNYK-WEB-VERIFICATION-001 (cuando estés en el navegador)

Ejecutar Snyk web sobre `v0.5.2-hardened`:
- URL: https://app.snyk.io
- Target: repositorio `alonsoir/argus`
- Filtro: código C++ de producción

Criterio de triage (Consejo 8/8 DAY 127):
- Código propio + path/overflow/crypto → Fix bloqueante con RED→GREEN
- Código propio + otro HIGH → Fix próximo sprint
- Third-party no alcanzable → Documentar en `docs/security/SNYK-DAY-128.md` con justificación
- Falso positivo demostrable → Cerrar con justificación documentada

Gate: `docs/security/SNYK-DAY-128.md` existe con clasificación de todos los findings.

---

## PASO 5 — Commit y push

```bash
git add -A
git commit -F - << 'EOF'
docs: DAY 128 — safe_path taxonomy + property testing pattern + provision portability

- docs/SECURITY-PATH-PRIMITIVES.md: taxonomia safe_path con PathPolicy enum
  conceptual y diagrama de decision (Consejo 8/8 DAY 127)
- docs/testing/PROPERTY-TESTING.md: patron formal de property testing
  con ejemplos reales del proyecto (F17, resolve_seed, resolve_config)
- 3 property tests nuevos RED->GREEN: resolve_seed + resolve_config + config_parser
- provision.sh: ARGUS_SERVICE_USER variable (default vagrant, portable)
- TEST-PROVISION-1: verde post-fix

Consejo 8/8 DAY 127: property testing primero, fuzzing despues.
La herramienta propone; el modelo de amenazas decide (criterio Snyk).
EOF

make test-all 2>&1 | grep -E "ALL TESTS|COMPLETE"
git push origin main
```

---


---

## Contexto estratégico — BACKLOG-BIZMODEL-001

### Decisiones tomadas DAY 127 (documentadas en docs/BACKLOG-BIZMODEL-001.md)

**Modelo de negocio:** Open-core. Core MIT siempre gratis. Servicio cloud/fleet de pago.

**Paraguas institucional:** UEx/INCIBE — preferencia clara. No es un acuerdo, es una
propuesta que hay que presentar a Andrés Caro Lindo antes de julio 2026. Ellos tienen
prioridad y autonomía total.

**Despliegue:** On-premise primero. Nube europea soberana (Hetzner/OVH/IONOS) para el
servicio cloud. Internacionalización = servidores clonados en territorio del cliente.

**Hardware pendiente (financiable FEDER):**
- Raspberry Pi 4/5 ARM64 × 4-8 → caracterización edge deployment
- Mini PCs x86 × 2-4 → perfil municipio/escuela
- Servidor de desarrollo (NVIDIA Spark DGX o equivalente MSI) → sin esto el portátil
  del 2019 es un riesgo de proyecto real
- Servidor central telemetry-server → CPD UEx o nube europea soberana

**telemetry-collector (ADR-039 pendiente):**
- Tap en el pipeline, sin tocar el flujo a rag-ingester
- Mismo crypto-transport (ChaCha20-Poly1305 + HKDF) — no se reinventa nada
- Collect-first, anonymize-later — geolocalización SIEMPRE asíncrona en servidor
- Priorización geográfica de distribución de modelos (frente de propagación)
- Bloqueante de hardware: NO para el código. SÍ para validación a escala real.

## Contexto permanente

### Secuencia canónica
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Branch activa
`main` — limpio. Tag: `v0.5.2-hardened`.

### Estado de deuda al inicio de DAY 128
```
🟡 DEBT-SNYK-WEB-VERIFICATION-001        → DAY 128 (navegador) — PASO 4
🟡 DEBT-PROPERTY-TESTING-PATTERN-001     → DAY 128 — PASO 2
🟡 DEBT-SAFE-PATH-TAXONOMY-DOC-001       → DAY 128 — PASO 1
🟢 DEBT-PROVISION-PORTABILITY-001        → DAY 128 — PASO 3
⏳ DEBT-SAFE-PATH-RESOLVE-MODEL-001      → feature/adr038-acrl
⏳ DEBT-FUZZING-LIBFUZZER-001            → post-property-testing
⏳ DEBT-PENTESTER-LOOP-001               → POST-DEUDA
```

### Modelos firmados activos
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017_v2.ubj + .sig  (DAY 122 — IN-DISTRIBUTION)
  wednesday_eval_report.json        (OOD finding sealed)
```

### Paper arXiv:2604.04952
Draft v16 activo. https://arxiv.org/abs/2604.04952
Pendiente: Draft v17 con §5 actualizado:
- §5.3 "Property Testing as a Security Fix Validator" (hallazgo F17 DAY 125)
- §5.4 "Dev/Prod Parity via Symlinks, Not Conditional Logic" (DAY 127)
- §5.5 "RED→GREEN as Non-Negotiable Merge Gate"
- §5.x "Taxonomía safe_path: lexically_normal vs weakly_canonical" (distinción no documentada en literatura C++20)

Referencias a citar: QuickCheck (Claessen & Hughes), CWE-22/23, TOCTOU literature, OWASP Path Traversal.

### Pregunta crítica FEDER pendiente (Consejo DAY 127)
¿La demo FEDER requiere federación funcional (ADR-038) o es suficiente con NDR standalone?
Clarificar con Andrés Caro Lindo ANTES de julio 2026.
Deadline FEDER: 22 septiembre 2026.

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

### REGLA DE ORO DAY 128
Documentar no es burocracia — es el contrato que garantiza que el hallazgo
de hoy no se convierte en deuda de mañana.
Un patrón sin documentación es un patrón que se reinventa cada vez.

---

*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"Un fix sin test de demostración es una promesa sin firma." — Qwen, Consejo DAY 124*
*"Has eliminado la confianza implícita en los fixes. Eso es lo que define un sistema de seguridad maduro." — ChatGPT, Consejo DAY 127*

*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*"Un fix sin test de demostración es una promesa sin firma." — Qwen, Consejo DAY 124*
*"Has eliminado la confianza implícita en los fixes. Eso es lo que define un sistema de seguridad maduro." — ChatGPT, Consejo DAY 127*
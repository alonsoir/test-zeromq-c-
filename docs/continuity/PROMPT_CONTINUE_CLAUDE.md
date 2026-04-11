# ML Defender (aRGus NDR) — DAY 115 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 114

### Hitos del día
**ADR-025 Plugin Integrity Verification: MERGEADO A MAIN ✅**
Tag: v0.3.0-plugin-integrity. 12/12 tests PASSED.
DEBT-SIGNAL-001/002 resueltos (async-signal-safe + atomic<bool>).
TEST-INTEG-4d implementado y PASSED (ml-detector PHASE 2d, 3/3).
Commits: 65a29034 (merge), 37c22423 (docs v15).

**arXiv Replace v15 SUBMITTED ✅**
submit/7467190 — Replacement of 2604.04952.
Párrafo Glasswing revisado. Draft v15 — DAY 114.

**Rama PHASE 3 abierta ✅**
feature/phase3-hardening desde main.

**Incidente DAY 114 (resuelto, lección aprendida):**
Pipeline no arrancaba tras rebuild — causa: libplugin_hello.so no desplegado ni firmado
tras cambio de libplugin_loader.so. ADR-025 fail-closed funcionó correctamente.
Solución: make plugin-hello-build + provision.sh sign.
Lección: la firma de plugins DEBE estar automatizada e integrada en el build system.

### Consejo DAY 114: ACTAS CERRADAS ✅
5 miembros respondieron. Veredictos definitivos del árbitro registrados.

---

## Veredictos árbitro DAY 114 (DEFINITIVOS)

**Q1 — DEBT-SIGN-AUTO:**
APROBADO CON RESTRICCIÓN DE SEGURIDAD CRÍTICA.
La firma automática SOLO ocurre en build/provision time usando artefactos recién compilados.
NUNCA firmar automáticamente en producción — solo verificar.
Mecanismo: provision.sh check-plugins verifica que todo plugin referenciado en JSON configs
existe y tiene .sig válido. Si falta → error en producción, firma en provisioning.
plugin-manifest.json (sha256+key_version): aplazado post-PHASE 3.
make diagnose: aplazado a DEBT-OPS-002.

**Q2 — DEBT-HELLO-001:**
OPCIÓN C UNÁNIME.
BUILD_DEV_PLUGINS=OFF por defecto. CI: ON explícito.
JSON producción sin referencia a libplugin_hello.
add_subdirectory(hello) envuelto en if(BUILD_DEV_PLUGINS).
make validate-prod-configs: check CI que falla si JSON producción referencia hello.

**Q3 — Orden PHASE 3 DEFINITIVO:**
1. systemd units (Restart=always, RestartSec=5s, unset LD_PRELOAD)
2. DEBT-SIGN-AUTO (firma automática build-time, idempotente, segura)
3. DEBT-HELLO-001 (BUILD_DEV_PLUGINS=OFF + JSON limpios + validate-prod-configs)
4. TEST-PROVISION-1 (gate CI — depende de 1+2+3 estables)
5. AppArmor profiles (6 componentes + denegar write /usr/bin/ml-defender-* para root)
6. DEBT-ADR025-D11 (provision.sh --reset — deadline 18 Apr, NO SE MUEVE)

**Q4 — Troubleshooting:**
docs/TROUBLESHOOTING.md en Markdown.
Resumen ejecutivo en CLAUDE.md.
make diagnose target (DEBT-OPS-002, parte de PHASE 3).
--check-config flag en componentes (candidato DAY 115+).

---

## PASO 0 — Verificación de estabilidad del pipeline (SIEMPRE PRIMERO)

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/phase3-hardening
git pull origin feature/phase3-hardening
make pipeline-stop
make pipeline-build 2>&1 | tail -5
```

```bash
# Verificar plugins firmados
vagrant ssh -c "ls -la /usr/lib/ml-defender/plugins/"
# Esperar: libplugin_hello.so + .sig + libplugin_test_message.so + .sig
```

```bash
# Firmar lo que falte (idempotente)
make sign-plugins
```

```bash
# Verificar JSON configs — plugins declarados
vagrant ssh -c "grep -h 'plugin\|libplugin' /vagrant/*/config/*.json 2>/dev/null | sort -u"
# Asegurarse de que CADA plugin referenciado en JSON tiene .so + .sig en /usr/lib/ml-defender/plugins/
```

```bash
make pipeline-start
make pipeline-status
# Esperar: 6/6 RUNNING
```

```bash
make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"
# Esperar: 12/12 PASSED
```

**Solo si los 6 componentes están RUNNING y los 12 tests PASSED se continúa con el trabajo del día.**

---

## Orden DAY 115

### PASO 1 — systemd units (PHASE 3, ítem 1)

Crear/actualizar systemd units para los 6 componentes con:
- `Restart=always`
- `RestartSec=5s`
- `Environment="LD_PRELOAD="` (unset explícito)
- `AmbientCapabilities` mínimas necesarias

Archivos objetivo:
```
/etc/systemd/system/ml-defender-etcd-server.service
/etc/systemd/system/ml-defender-sniffer.service
/etc/systemd/system/ml-defender-ml-detector.service
/etc/systemd/system/ml-defender-firewall-acl.service
/etc/systemd/system/ml-defender-rag-ingester.service
/etc/systemd/system/ml-defender-rag-security.service
```

Antes de implementar: revisar si ya existen en el repo:
```bash
vagrant ssh -c "ls /etc/systemd/system/ml-defender* 2>/dev/null || echo 'NO EXISTEN'"
find /Users/aironman/CLionProjects/test-zeromq-docker -name "*.service" 2>/dev/null
```

### PASO 2 — DEBT-SIGN-AUTO (PHASE 3, ítem 2)

Implementar en provision.sh:

```bash
# Modo provisioning (Vagrant):
provision.sh check-plugins
# → Para cada plugin en JSON configs:
#     Si .so no existe en /usr/lib/ml-defender/plugins/ → ERROR
#     Si .sig no existe → firmar (provision time)
#     Si .sig inválido para clave actual → re-firmar (provision time)
#     Si todo OK → skip
# → Idempotente, nunca firmar en producción hot

# Integración Makefile:
check-and-sign-plugins:  (depende de plugin-loader-build)
    @vagrant ssh -c "sudo bash /vagrant/tools/provision.sh check-plugins"

# pipeline-start debe depender de check-and-sign-plugins
```

### PASO 3 — DEBT-HELLO-001 (PHASE 3, ítem 3)

```cmake
# CMakeLists.txt raíz:
option(BUILD_DEV_PLUGINS "Build development plugins (hello-world)" OFF)
if(BUILD_DEV_PLUGINS)
    add_subdirectory(plugins/hello)
endif()
```

```bash
# make validate-prod-configs:
# Verificar que ningún JSON en */config/*.json referencia libplugin_hello
```

```bash
# Eliminar libplugin_hello.so de todos los JSON de producción
# Verificar: grep -r "libplugin_hello" */config/
```

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios
- **arXiv**: arXiv:2604.04952 [cs.CR] — PUBLICADO + Replace v15 submitted ✅
- **Branch activa**: feature/phase3-hardening
- **Repositorio**: https://github.com/alonsoir/argus
- **Tag estable**: v0.3.0-plugin-integrity (main)

### Regla de oro del pipeline
**El pipeline es estable cuando:** 6/6 RUNNING + 12/12 plugin-integ-test PASSED
+ ls /usr/lib/ml-defender/plugins/ muestra .so + .sig para CADA plugin referenciado en JSON configs.
  **Tras cualquier cambio en libplugin_loader.so o plugins:** make pipeline-stop → make pipeline-build
  → make sign-plugins → verificar .sig → make pipeline-start → make pipeline-status → make plugin-integ-test.

### ADR-025 keypair dev
- Private key: /etc/ml-defender/plugins/plugin_signing.sk (VM only)
- MLD_PLUGIN_PUBKEY_HEX: b824bcd7a14f6e19a0d8c9be86110828060e600723d12e118dccc95c862c8468
- Firmar plugins: make sign-plugins

### DEBT-ADR025-D11 deadline
provision.sh --reset — **deadline 18 Apr 2026 (4 días)**. No negociable.

### Deuda activa PHASE 3 (orden definitivo árbitro)
1. systemd units ← DAY 115 PASO 1
2. DEBT-SIGN-AUTO ← DAY 115 PASO 2
3. DEBT-HELLO-001 ← DAY 115 PASO 3
4. TEST-PROVISION-1
5. AppArmor profiles (incluir: denegar write /usr/bin/ml-defender-* para root)
6. DEBT-ADR025-D11 (deadline 18 Apr)

### Patrón robusto para scripts en VM (NUNCA sed -i en macOS)
cat > /tmp/script.py << 'PYEOF' → vagrant upload → vagrant ssh -c 'sudo python3 /tmp/script.py'

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.
Qwen se auto-identifica como DeepSeek — patrón consolidado.

### PHASE 2 — COMPLETA ✅
2a firewall + 2b rag-ingester + 2c sniffer + 2d ml-detector + 2e rag-security.
12/12 tests PASSED (4a+4b+4c+4d+4e+SIGN-1..7).

### Filosofía core
"Un escudo, nunca una espada."
"La verdad por delante, siempre."
Fail-closed: todo o nada. En un sistema que salva vidas, no arrancar es preferible a arrancar comprometido.
PHASE 3: ya no diseñamos seguridad — diseñamos operación segura (Consejo DAY 114).
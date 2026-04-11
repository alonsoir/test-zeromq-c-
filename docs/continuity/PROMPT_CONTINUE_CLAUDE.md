# ML Defender (aRGus NDR) — DAY 115 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 114

### Hitos del día
**ADR-025 Plugin Integrity Verification: MERGEADO A MAIN ✅**
Tag: v0.3.0-plugin-integrity. 12/12 tests PASSED.
DEBT-SIGNAL-001/002 resueltos. TEST-INTEG-4d implementado y PASSED.
Commits: 65a29034 (merge), a16a0795 (docs+backlog), commit final README+ADR-032.

**arXiv Replace v15 SUBMITTED ✅**
submit/7467190. Draft v15. Párrafo Glasswing revisado.

**ADR-032 Plugin Distribution Chain: APROBADO por Consejo ✅**
YubiKey OpenPGP Ed25519 (NO PIV — corrección técnica crítica).
Formato .sig embebido (Opción B, 4/5). Multi-key en loader. Revocación offline.
Docs: docs/adr/ADR-032-plugin-distribution-chain.md

**PHASE 3 abierta: feature/phase3-hardening ✅**

---

## Veredictos árbitro DAY 114 (DEFINITIVOS)

### Consejo DAY 114 — Q1-Q4 (PHASE 3)
Ver síntesis en docs/counsil/

**Orden PHASE 3 DEFINITIVO:**
1. systemd units (Restart=always, RestartSec=5s, unset LD_PRELOAD)
2. DEBT-SIGN-AUTO (firma build-time ÚNICAMENTE, nunca en producción hot)
3. DEBT-HELLO-001 (BUILD_DEV_PLUGINS=OFF + JSON limpios + validate-prod-configs)
4. TEST-PROVISION-1 (CI gate)
5. AppArmor profiles (6 componentes + denegar write /usr/bin/ml-defender-* para root)
6. DEBT-ADR025-D11 (provision.sh --reset — **deadline 18 Apr, NO SE MUEVE**)

**DEBT-SIGN-AUTO restricción de seguridad crítica:**
Firma automática SOLO en build/provision time con artefactos recién compilados.
NUNCA firmar automáticamente en producción — solo verificar.

### Consejo ADR-032
Ver síntesis en docs/counsil/
**Corrección técnica crítica:** YubiKey PIV NO soporta Ed25519. Usar applet OpenPGP.

---

## PASO 0 — Verificación de estabilidad del pipeline (SIEMPRE PRIMERO)

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/phase3-hardening
git pull origin feature/phase3-hardening
make pipeline-stop
make pipeline-build 2>&1 | tail -5
vagrant ssh -c "ls -la /usr/lib/ml-defender/plugins/"
# Esperar: .so + .sig para CADA plugin en JSON configs
make sign-plugins
vagrant ssh -c "grep -h 'libplugin' /vagrant/*/config/*.json 2>/dev/null | sort -u"
# Verificar que CADA plugin referenciado tiene .so + .sig
make pipeline-start && make pipeline-status
# Esperar: 6/6 RUNNING
make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"
# Esperar: 12/12 PASSED
```

**Solo si 6/6 RUNNING y 12/12 PASSED se continúa.**

---

## Orden DAY 115

### TAREA PRIORITARIA DAY 115: Sesión Consejo ADR-024

**ADR-024 (Noise_IKpsk3 P2P) tiene 4 preguntas abiertas sin resolver desde DAY 104:**
- OQ-5: revocación de clave estática
- OQ-6: continuidad de sesión durante rotación de clave
- OQ-7: replay protection
- OQ-8: rendimiento en ARMv8

**Antes de implementar ADR-024, lanzar sesión específica al Consejo con estas 4 preguntas.**
Generar el documento de consulta, lanzarlo a los 7 modelos, sintetizar veredictos, actualizar ADR-024.

```bash
# Revisar ADR-024 actual:
cat /Users/aironman/CLionProjects/test-zeromq-docker/docs/adr/ADR-024*.md
```

### PASO 1 — systemd units (PHASE 3, ítem 1)

Verificar si existen:
```bash
vagrant ssh -c "ls /etc/systemd/system/ml-defender* 2>/dev/null || echo 'NO EXISTEN'"
find /Users/aironman/CLionProjects/test-zeromq-docker -name "*.service" 2>/dev/null
```

Crear para los 6 componentes con:
- `Restart=always`
- `RestartSec=5s`
- `Environment="LD_PRELOAD="` (unset explícito)
- Capabilities mínimas necesarias

### PASO 2 — DEBT-SIGN-AUTO (PHASE 3, ítem 2)

provision.sh check-plugins:
- Solo en provisioning: firmar plugins recién compilados sin .sig
- En producción: SOLO verificar, nunca firmar
- Idempotente: si ya firmado con clave actual → skip
- Integrar en Makefile (check-and-sign-plugins target antes de pipeline-start)

### PASO 3 — DEBT-HELLO-001 (PHASE 3, ítem 3)

```cmake
option(BUILD_DEV_PLUGINS "Build development plugins" OFF)
if(BUILD_DEV_PLUGINS)
    add_subdirectory(plugins/hello)
endif()
```

```bash
# make validate-prod-configs:
# Falla si algún JSON de producción referencia libplugin_hello
grep -r "libplugin_hello" */config/
```

---

## Roadmap post-PHASE 3 (REGISTRADO DAY 114)

**Este roadmap está acordado y debe reflejarse en BACKLOG y README tras completar PHASE 3.**

### Bloque A — ADR-024 Noise_IKpsk3 (P2P cifrado sin HA)

**Contexto:** ADR-024 implementa canal cifrado directo entre componentes via
Noise Protocol Framework (Noise_IKpsk3 o Noise_KK). Modo simple, sin HA,
sin etcd en el hot path para cifrado.

**Precondición:** OQ-5/6/7/8 resueltas por Consejo (DAY 115).

**Orden:**
1. Sesión Consejo OQ-5..8 → veredictos árbitro
2. Implementación ADR-024 en rama feature/adr024-noise-p2p
3. Tests de integración E2E cifrado P2P

### Bloque B — Stress test CTU-13 Neris con pipeline real

**Precondición:** PHASE 3 completa + DEBT-TOOLS-001 resuelto.

**DEBT-TOOLS-001 CRÍTICO:** Los synthetic injectors en tools/ deben integrar
PluginLoader + plugins firmados (Ed25519) ANTES del stress test.
Sin esto, el stress test ejercita un sistema distinto al de producción.

```
tools/synthetic_sniffer_injector.cpp     → integrar PluginLoader
tools/synthetic_ml_output_injector.cpp   → integrar PluginLoader
tools/generate_synthetic_events.cpp      → integrar PluginLoader
```

**Orden:**
1. DEBT-TOOLS-001: refactorizar injectors con PluginLoader + plugins firmados
2. make sign-plugins (firmar plugins para stress test)
3. Stress test CTU-13 Neris con pipeline real (6/6 + plugins)
4. Verificar F1 = 0.9985 se mantiene con plugins activos
5. Stress test con bigFlows (throughput ceiling)

### Bloque C — Refactoring etcd legacy

**Contexto:** etcd-server y etcd-client tienen actualmente dos roles:
1. Distribución de JSON configs (MANTENER)
2. Intermediario de cifrado / semillas (DEPRECAR → reemplazar por ADR-024)

**Arquitectura objetivo:**
```
etcd-server: SOLO para
  - Distribución de JSON configs
  - Registro heartbeat de componentes
  - Seed distribution (mientras no haya ADR-032 Fase B)

Componentes entre sí: ZeroMQ + Noise_IKpsk3 (ADR-024)
  - Sin etcd en el hot path
  - Sin legacy etcd-client en la lógica de cifrado
```

**Precondición:** ADR-024 implementado y validado (Bloque A completo).

**Orden:**
1. Identificar todo el código legacy etcd-client en los 6 componentes
2. Crear rama feature/refactor-etcd-legacy
3. Migrar cifrado de canal a ADR-024
4. Mantener etcd solo para config distribution + heartbeat
5. Tests de regresión completos (F1 + stress)

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios
- **arXiv**: arXiv:2604.04952 [cs.CR] — PUBLICADO + Replace v15 submitted ✅
- **Branch activa**: feature/phase3-hardening
- **Repositorio**: https://github.com/alonsoir/argus
- **Tag estable**: v0.3.0-plugin-integrity (main)

### Regla de oro del pipeline
**Estable = 6/6 RUNNING + 12/12 plugin-integ-test PASSED**
**+ ls /usr/lib/ml-defender/plugins/ muestra .so + .sig para CADA plugin en JSON configs**
Tras cualquier cambio en libplugin_loader.so o plugins:
stop → build → sign → verify → start → status → plugin-integ-test

### ADR-025 keypair dev
- Private key: /etc/ml-defender/plugins/plugin_signing.sk (VM only)
- MLD_PLUGIN_PUBKEY_HEX: b824bcd7a14f6e19a0d8c9be86110828060e600723d12e118dccc95c862c8468
- Firmar: make sign-plugins

### DEBT-ADR025-D11 deadline
provision.sh --reset — **deadline 18 Apr 2026**. No negociable.

### Deuda activa PHASE 3 (orden definitivo)
1. systemd units ← DAY 115 PASO 1
2. DEBT-SIGN-AUTO ← DAY 115 PASO 2
3. DEBT-HELLO-001 ← DAY 115 PASO 3
4. TEST-PROVISION-1
5. AppArmor profiles
6. DEBT-ADR025-D11 (deadline 18 Apr)

### Roadmap post-PHASE 3 (acordado DAY 114)
A. ADR-024 Noise_IKpsk3 P2P (OQ-5..8 → Consejo DAY 115 mañana)
B. DEBT-TOOLS-001 + Stress test CTU-13 Neris con pipeline real
C. Refactoring etcd legacy (etcd = solo config + heartbeat)

### Patrón robusto para scripts en VM (NUNCA sed -i en macOS)
cat > /tmp/script.py << 'PYEOF' → vagrant upload → vagrant ssh -c 'sudo python3 /tmp/script.py'

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.
Qwen se auto-identifica como DeepSeek — patrón consolidado.

### PHASE 2 — COMPLETA ✅
2a+2b+2c+2d+2e. 12/12 tests PASSED.

### Filosofía core
"Un escudo, nunca una espada."
"La verdad por delante, siempre."
Fail-closed. PHASE 3: operación segura, no solo seguridad.
ADR-032: la autoridad de firma y el servidor de producción NO comparten dominio de confianza.
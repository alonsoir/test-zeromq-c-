# ML Defender (aRGus NDR) — BACKLOG

## Via Appia Quality 🏛️

---

## 📐 Criterio de compleción

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

---

## ✅ COMPLETADO

### DAY 115 (12 Apr 2026) — PHASE 3 ítems 1-4 + ADR-024 OQs cerradas

**ADR-024 OQ-5..8: CERRADAS (Consejo unanimidad) ✅**
- OQ-5: allowed_static_keys en deployment.yml + caché local + re-provision si seed_family comprometida
- OQ-6: Dual-key T=24h + versioned deployment.yml + secuencia 5 pasos cero downtime
- OQ-7: Riesgo replay aceptado v1 + nftables rate-limiting + trigger v2 si WAN
- OQ-8: Noise_IKpsk3 mantenido + benchmark ARMv8 obligatorio pre-producción
- ADR-024 actualizado con Recovery Contract + TEST-INTEG-8/9

**PHASE 3 ítem 1 — systemd units ✅**
- 6 units: Restart=always, RestartSec=5s, Environment="LD_PRELOAD="
- set-build-profile.sh: symlinks build-active → build-debug|release
- DEBT-RAG-BUILD-001 registrado; DEBT-SYSTEMD-001: checklist RPi documentado

**PHASE 3 ítem 2 — DEBT-SIGN-AUTO ✅**
- provision.sh check-plugins: sign-if-needed dev, verify-only producción. Idempotente.

**PHASE 3 ítem 3 — DEBT-HELLO-001 ✅**
- BUILD_DEV_PLUGINS=OFF guard. libplugin_hello eliminado de 5 JSONs.
- Bug resuelto: 4 componentes tenían active:true. make validate-prod-configs añadido.

**PHASE 3 ítem 4 — TEST-PROVISION-1 ✅**
- 5 checks CI gate. pipeline-start depende de test-provision-1.
- Commits: df976d90, a1b23882 (feature/phase3-hardening)

---

### DAY 114 (11 Apr 2026) — ADR-025 MERGE + Signal Safety + TEST-INTEG-4d + arXiv v15

**ADR-025 Plugin Integrity Verification: MERGEADO A MAIN ✅**
Tag: v0.3.0-plugin-integrity. 12/12 tests PASSED.
DEBT-SIGNAL-001/002 resueltos. TEST-INTEG-4d PASSED.
Commits: 65a29034 (merge), 37c22423 (docs v15)

**arXiv Replace v15 SUBMITTED ✅** — submit/7467190

**ADR-032 Plugin Distribution Chain: APROBADO ✅**
YubiKey OpenPGP Ed25519 (NO PIV). Formato .sig embebido. Multi-key loader.

---

### DAY 113 — ADR-025 IMPLEMENTADO + Paper v14 *(ver git log)*
### DAY 111 — arXiv:2604.04952 PUBLICADO 🎉 · DOI: https://doi.org/10.48550/arXiv.2604.04952
### DAY 110–62 *(ver historial completo en git log)*

---

## 📋 BACKLOG ACTIVO

### P0 — PHASE 3 restante (rama: feature/phase3-hardening)

| ID | Tarea | Estado | Deadline |
|----|-------|--------|---------|
| **DEBT-ADR025-D11** | provision.sh --reset: regenera seed_family + keypairs Ed25519 + keypair firma. SIN auto-firma. Mensaje claro post-reset indicando `make sign-plugins`. | 🔴 URGENTE | **18 Apr** |
| PHASE3-APPARMOR | AppArmor profiles 6 componentes. Flujo: complain → audit → enforce. Incluir paths de provision.sh --reset. Denegar write /usr/bin/ml-defender-* para root. | 🔴 TODO | post --reset |
| TEST-PROVISION-CHECK6 | Check #6: permisos ficheros sensibles (*.sk, deployment.yml no world-writable) | ⏳ TODO | DAY 116 |
| TEST-PROVISION-CHECK7 | Check #7: consistencia JSONs — cada plugin referenciado tiene .so + .sig presente | ⏳ TODO | DAY 116 |
| DEBT-OPS-001 | make redeploy-plugins: build+sign+deploy en un solo target | ⏳ TODO | post-PHASE3 |
| DEBT-OPS-002 | Documentación operativa + sección Troubleshooting pipeline | ⏳ TODO | post-PHASE3 |

### P1 — Deuda inmediata

| ID | Tarea | Deadline | Origen |
|----|-------|----------|--------|
| DEBT-RAG-BUILD-001 | rag/CMakeLists.txt: build-debug/release igual que resto de componentes | Pre-RPi | DAY 115 |
| REC-2 | noclobber + check 0-bytes en CI | PHASE 3 | Consejo DAY 110 |

### P2 — Antes del próximo PCAP replay

| ID | Tarea | Origen |
|----|-------|--------|
| DEBT-TOOLS-001 | Synthetic injectors: integrar PluginLoader + plugins firmados Ed25519 (3 ficheros) | Árbitro DAY 113 |
| DEBT-SNIFFER-SEED | Unificar sniffer bajo SeedClient | DAY 107 |

### P3 — Post-PHASE 3

| ID | Tarea | Origen |
|----|-------|--------|
| ADR-024 impl | Noise_IKpsk3 P2P (feature/adr024-noise-p2p). OQs 5..8 cerradas. | DAY 115 |
| ADR-030 activación | AppArmor enforcing + hardware Pi | post-PHASE 3 |
| ADR-031 spike | seL4/Genode técnico (2–3 semanas) | post-ADR-030 |
| ADR-026 | Fleet telemetry + XGBoost + BitTorrent distribution | diferido |
| ADR-032 Fase A | Plugin Distribution Chain: manifest JSON + multi-key loader + revocación | DAY 114 |
| ADR-032 Fase B | YubiKey OpenPGP (2× unidades) + firma HSM | post-ADR-032-A + hardware |
| ADR-033 | Platform Integrity: TPM 2.0 Measured Boot | propuesto DAY 114 |
| DEBT-CLI-001 | ml-defender verify-plugin --bundle CLI tool | Consejo ADR-032 |
| BARE-METAL stress | tcpreplay en NIC físico | bloqueado hardware |
| DEBT-FD-001 | Fast Detector Path A → JSON thresholds | DAY 80 |
| DEBT-CRYPTO-003a | mlock() en seed_client.cpp | ADR-022 |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | P3 |

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CryptoTransport (HKDF+nonce+AEAD):    ████████████████████ 100% ✅  DAY 97
contexts.hpp (HKDF simétricos):       ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):        ████████████████████ 100% ✅  DAY 99
plugin-loader ADR-012 PHASE 1b 6/6:   ████████████████████ 100% ✅  DAY 102
ADR-023 PHASE 2a-2e (6 componentes):  ████████████████████ 100% ✅  DAY 105-112
ADR-025 Plugin Integrity (Ed25519):   ████████████████████ 100% ✅  DAY 113-114 🎉
TEST-INTEG-4a/4b/4c/4d/4e:           ████████████████████ 100% ✅  DAY 114
TEST-INTEG-SIGN-1..7:                 ████████████████████ 100% ✅  DAY 113
DEBT-SIGNAL-001/002:                  ████████████████████ 100% ✅  DAY 114 🎉
arXiv:2604.04952 PUBLICADO:           ████████████████████ 100% ✅  DAY 111 🎉
arXiv Replace v15 SUBMITTED:          ████████████████████ 100% ✅  DAY 114 🎉
ADR-024 OQs 5..8 CERRADAS:           ████████████████████ 100% ✅  DAY 115 🎉
PHASE 3 ítem 1 (systemd units):      ████████████████████ 100% ✅  DAY 115 🎉
PHASE 3 ítem 2 (DEBT-SIGN-AUTO):     ████████████████████ 100% ✅  DAY 115 🎉
PHASE 3 ítem 3 (DEBT-HELLO-001):     ████████████████████ 100% ✅  DAY 115 🎉
PHASE 3 ítem 4 (TEST-PROVISION-1):   ████████████████████ 100% ✅  DAY 115 🎉
DEBT-ADR025-D11 (--reset):           ░░░░░░░░░░░░░░░░░░░░   0% 🔴  deadline 18 Apr
PHASE 3 ítem 5 (AppArmor):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 116
TEST-PROVISION-1 checks 6+7:         ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 116
ADR-024 Noise_IKpsk3 impl:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 3
DEBT-TOOLS-001 (injectors+plugins):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  pre-stress test
ADR-032 Fase A:                      ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 3
BARE-METAL stress test:              ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado hardware
DEBT-FD-001 (JSON thresholds):       ████░░░░░░░░░░░░░░░░  20% 🟡
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| Fail-closed plugin loading | std::terminate() si require_signature:true y plugin sin firma | 113 |
| Plugin hello en producción | PROHIBIDO — validate-prod-configs gate en Makefile | 115 |
| Signal handlers | write(STDERR_FILENO) exclusivamente | 114 |
| shutdown_called_ | std::atomic<bool> obligatorio | 114 |
| Firma automática plugins | NUNCA en producción. Dev: provision.sh check-plugins. | 115 |
| provision.sh --reset | Regenera claves SIN auto-firma. Operador firma manualmente. | 115 |
| AppArmor | complain → audit → enforce. Incluir paths de --reset en perfiles. | 115 |
| ADR-024 OQ-5 revocación | allowed_static_keys en deployment.yml + caché local | 115 |
| ADR-024 OQ-6 rotación | Dual-key T=24h + versioned deployment.yml | 115 |
| ADR-024 OQ-7 replay | Aceptado v1 + nftables. Timestamp obligatorio si WAN. | 115 |
| ADR-024 OQ-8 rendimiento | Noise_IKpsk3 v1. Benchmark ARMv8 pre-prod. | 115 |
| D8-pre bidireccional | READONLY+payload→terminate + NORMAL+nullptr→terminate | 111 |
| MAX_PLUGIN_PAYLOAD_SIZE | 64KB hard limit, std::terminate() | 111 |
| ADR-032 autoridad de firma | YubiKey OpenPGP Ed25519 (no PIV). 2 unidades. | 114 |

---

## 🔑 Procedimiento de verificación de estabilidad del pipeline

```bash
make pipeline-stop
make pipeline-build
make sign-plugins
vagrant ssh -c "ls -la /usr/lib/ml-defender/plugins/"  # .so + .sig para CADA plugin
make test-provision-1   # CI gate: 5 checks (6+7 pendientes DAY 116)
make pipeline-start && make pipeline-status  # 6/6 RUNNING
make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"  # 12/12 PASSED
```

---

### Notas del Consejo de Sabios — DAY 115

> "4 ítems de PHASE 3 cerrados + ADR-024 OQs 5..8 resueltas en una mañana de domingo.
> Hallazgo crítico: 4 componentes con active:true para hello plugin en producción — bug de
> seguridad resuelto. TEST-PROVISION-1 es ahora el root of truth del pipeline.
> Próximo: DEBT-ADR025-D11 (deadline 18 Apr) → AppArmor complain → enforce.
> Checks 6+7 a añadir en TEST-PROVISION-1: permisos ficheros sensibles + consistencia JSONs."
> — Consejo de Sabios (6 miembros) · DAY 115 · Unanimidad en Q1/Q2/Q4. Q3: 4/6 DEBT-ADR025-D11 primero.

---

*Última actualización: DAY 115 — 12 Apr 2026*
*Branch activa: feature/phase3-hardening*
*Tests: 12/12 plugin-integ-test PASSED · 6/6 RUNNING*
*arXiv: 2604.04952 · v15 submitted ✅ · Tag: v0.3.0-plugin-integrity*
*PHASE 2: ✅ COMPLETA · PHASE 3: 4/6 ✅ EN CURSO*
*"Via Appia Quality — Un escudo, nunca una espada."*
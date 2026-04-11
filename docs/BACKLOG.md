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

### DAY 114 (11 Apr 2026) — ADR-025 MERGE + Signal Safety + TEST-INTEG-4d + arXiv v15

**ADR-025 Plugin Integrity Verification: MERGEADO A MAIN ✅**
Tag: v0.3.0-plugin-integrity
- Ed25519 offline signing + TOCTOU-safe dlopen, 12/12 tests PASSED
- DEBT-SIGNAL-001: signal handlers async-signal-safe (write() en lugar de std::cout)
- DEBT-SIGNAL-002: shutdown_called_ bool → std::atomic<bool>
- TEST-INTEG-4d: ml-detector PHASE 2d, 3/3 PASSED (NUEVO)
- Commits: 65a29034 (merge), 37c22423 (docs v15)

**arXiv Replace v15 SUBMITTED ✅**
- submit/7467190 — Replacement of 2604.04952
- Párrafo Glasswing/Mythos revisado (veredicto Q5 árbitro DAY 113)
- Draft v15 — DAY 114 — April 2026

**Rama PHASE 3 abierta ✅**
- feature/phase3-hardening creada desde main

**Deuda registrada DAY 114:**
- DEBT-HELLO-001: eliminar libplugin_hello.so de configs de producción (PHASE 3)
- DEBT-OPS-001: make redeploy-plugins (build+sign+deploy en un solo target)
- DEBT-OPS-002: documentación operativa + sección Troubleshooting pipeline

---

### DAY 113 (10 Apr 2026) — ADR-025 IMPLEMENTADO + Paper v14

**ADR-025 Plugin Integrity Verification: IMPLEMENTADO ✅**
Ed25519 offline signing + TOCTOU-safe dlopen. 7/7 SIGN tests PASSED.
make test: 11/11 PASSED (4a+4b+4c+4e+SIGN-1..7).
Rama: feature/plugin-integrity-ed25519. Commits: eb2c88d9, a3819bc3, 1eb40e8b.

**Paper Draft v14: COMPILACIÓN LIMPIA ✅**
Glasswing/Mythos integrado (párrafo revisado por árbitro DAY 113).

**Consejo DAY 113: ACTAS CERRADAS ✅**

---

### DAY 112 (9 Apr 2026) — PHASE 2e + ADR-030/031
*(ver historial git)*

### DAY 111 (8 Apr 2026) — FIX-C/D + PHASE 2d + ADR-029 + arXiv PUBLICADO
**🎉 arXiv:2604.04952 [cs.CR] PUBLICADO**
DOI: https://doi.org/10.48550/arXiv.2604.04952

### DAY 110–62
*(ver historial completo en git log)*

---

## 📋 BACKLOG ACTIVO

### P0 — PHASE 3 (rama: feature/phase3-hardening)

| ID | Tarea | Origen |
|----|-------|--------|
| PHASE3-SYSTEMD | systemd units: Restart=always, RestartSec=5s, unset LD_PRELOAD | ADR-025 D10 |
| PHASE3-APPARMOR | AppArmor profiles básicos 6 componentes + denegar write /usr/bin/ml-defender-* para root | Gemini DAY 113 |
| PHASE3-CI | TEST-PROVISION-1 como gate formal CI | ChatGPT5 DAY 108 |
| DEBT-HELLO-001 | Eliminar libplugin_hello.so de JSON configs producción + CMake flag BUILD_DEV_PLUGINS=OFF | DAY 114 |
| DEBT-OPS-001 | make redeploy-plugins: build+sign+deploy en un solo target | DAY 114 |
| DEBT-OPS-002 | Documentación operativa actualizada + sección Troubleshooting (árbol: pipeline no arranca → causas → solución) | DAY 114 |

### P1 — Deuda inmediata post-merge

| ID | Tarea | Deadline | Origen |
|----|-------|----------|--------|
| DEBT-ADR025-D11 | provision.sh --reset: rotación manual keypair Ed25519 | 7 días desde merge (18 Apr) | ADR-025 D11, árbitro DAY 113 |
| DEBT-SIGN-AUTO | Firma automática de plugins: provision.sh consciente de plugins no firmados o parcialmente firmados. Vagrantfile + Makefile integrados. Sin relanzar si todo está firmado; completar lo que falte si está parcial. | PHASE 3 | DAY 114 troubleshooting |

### P2 — Antes del próximo PCAP replay

| ID | Tarea | Origen |
|----|-------|--------|
| DEBT-TOOLS-001 | Synthetic injectors: integrar PluginLoader + plugins firmados (Ed25519) | Árbitro DAY 113 |
| REC-2 | noclobber + check 0-bytes CI | Consejo DAY 110 |
| DEBT-SNIFFER-SEED | Unificar sniffer bajo SeedClient | DAY 107 |

### P3 — Post-PHASE 3

| ID | Tarea | Origen |
|----|-------|--------|
| ADR-030 activación | AppArmor enforcing + hardware Pi | post-PHASE 3 |
| ADR-031 spike | seL4/Genode técnico (2–3 semanas) | post-ADR-030 |
| ADR-026 | Fleet telemetry + XGBoost + BitTorrent distribution | diferido: construir sobre buenos andamios |
| ADR-024 impl | Noise_IKpsk3 dynamic key agreement | FASE 3 post-PHASE 3 |
| ADR-032 Fase A | Plugin Distribution Chain: formato manifest JSON + multi-key loader + revocación | DAY 114, Consejo ADR-032 |
| ADR-032 Fase B | Plugin Distribution Chain: YubiKey OpenPGP (2× unidades) + firma HSM | post-ADR-032-A + hardware |
| ADR-033 | Platform Integrity: TPM 2.0 Measured Boot | propuesto DAY 114 |
| DEBT-CLI-001 | ml-defender verify-plugin --bundle CLI tool | Qwen, Consejo ADR-032 |
| BARE-METAL-IMAGE | Imagen Debian Bookworm hardened exportable a USB | P3 |
| BARE-METAL stress | tcpreplay 100/250/500/1000 Mbps en NIC físico | P3 |
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
set_terminate() 6/6 main():           ████████████████████ 100% ✅  DAY 100
plugin-loader ADR-012 PHASE 1b 6/6:   ████████████████████ 100% ✅  DAY 101-102
ADR-023 PHASE 2a (firewall):          ████████████████████ 100% ✅  DAY 105-106
ADR-023 PHASE 2b (rag-ingester):      ████████████████████ 100% ✅  DAY 109-110
ADR-023 PHASE 2c (sniffer):           ████████████████████ 100% ✅  DAY 111
ADR-023 PHASE 2d (ml-detector):       ████████████████████ 100% ✅  DAY 111+114
ADR-023 PHASE 2e (rag-security):      ████████████████████ 100% ✅  DAY 112
ADR-025 Plugin Integrity (Ed25519):   ████████████████████ 100% ✅  DAY 113-114 🎉
TEST-INTEG-4a 3/3:                    ████████████████████ 100% ✅  DAY 105
TEST-INTEG-4b:                        ████████████████████ 100% ✅  DAY 109
TEST-INTEG-4c 3/3:                    ████████████████████ 100% ✅  DAY 111
TEST-INTEG-4d 3/3:                    ████████████████████ 100% ✅  DAY 114 🎉
TEST-INTEG-4e 3/3:                    ████████████████████ 100% ✅  DAY 112
TEST-INTEG-SIGN-1..7:                 ████████████████████ 100% ✅  DAY 113
DEBT-SIGNAL-001 (async-signal-safe):  ████████████████████ 100% ✅  DAY 114 🎉
DEBT-SIGNAL-002 (atomic<bool>):       ████████████████████ 100% ✅  DAY 114 🎉
arXiv:2604.04952 PUBLICADO:           ████████████████████ 100% ✅  DAY 111 🎉
arXiv Replace v15 SUBMITTED:          ████████████████████ 100% ✅  DAY 114 🎉
PHASE 3 (hardening):                  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/phase3-hardening
DEBT-ADR025-D11 (--reset):            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  deadline 18 Apr
DEBT-SIGN-AUTO (firma automática):    ░░░░░░░░░░░░░░░░░░░░   0% ⏳  PHASE 3
DEBT-HELLO-001 (rm hello plugin):     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  PHASE 3
DEBT-TOOLS-001 (injectors+plugins):   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  antes PCAP replay
ADR-030 AppArmor activación:          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 3
ADR-031 seL4 spike:                   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-ADR-030
ADR-026 Fleet/XGBoost/BitTorrent:     ░░░░░░░░░░░░░░░░░░░░   0% ⏳  diferido
ADR-024 Noise_IKpsk3 impl:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  FASE 3
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado hardware
DEBT-FD-001 (JSON thresholds):        ████░░░░░░░░░░░░░░░░  20% 🟡
TEST-PROVISION-1 (CI gate):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  PHASE 3
ADR-032 Fase A (manifest+multikey):   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 3
ADR-032 Fase B (YubiKey HSM):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-ADR-032-A + hardware
ADR-033 (TPM measured boot):          ░░░░░░░░░░░░░░░░░░░░   0% ⏳  propuesto
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| Fail-closed plugin loading | std::terminate() si require_signature:true y plugin sin firma | 113 |
| Plugin hello en producción | PROHIBIDO — DEBT-HELLO-001, eliminar en PHASE 3 | 114 |
| Signal handlers | write(STDERR_FILENO) exclusivamente, sin std::cout/cerr | 114 |
| shutdown_called_ | std::atomic<bool> obligatorio | 114 |
| Firma automática plugins | provision.sh debe detectar plugins sin firmar y firmarlos; Makefile/Vagrantfile conscientes | 114 |
| D8-pre bidireccional | READONLY+payload→terminate + NORMAL+nullptr→terminate | 111 |
| MAX_PLUGIN_PAYLOAD_SIZE | 64KB hard limit, std::terminate() si excedido | 111 |
| ADR-030 AppArmor-Hardened | variante producción, denegar write /usr/bin/ml-defender-* incluso para root | 112+113 |
| ADR-031 seL4/Genode | investigación pura, spike GO/NO-GO obligatorio | 112 |
| ADR-026 timing | diferido — construir sobre buenos andamios primero (PHASE 3) | 113 |
| ADR-032 autoridad de firma | Clave privada NUNCA en disco producción. YubiKey OpenPGP Ed25519 (no PIV). 2 unidades. Multi-key en loader. | 114 |
| ADR-032 formato .sig | JSON embebido: manifest + firma Ed25519 cubre sha256(so) + sha256(manifest) | 114 |
| ADR-032 customer_id | Control lógico, no barrera criptográfica fuerte. Documentado explícitamente. | 114 |
| ADR-033 TPM | ADR separado. No mezclar con distribución de plugins. | 114 |

---

## 🔑 Procedimiento de verificación de estabilidad del pipeline

**Ejecutar en este orden al inicio de cada sesión o tras cualquier cambio en libplugin_loader.so, plugins, o configuración:**

```bash
# 1. Parar pipeline limpio
make pipeline-stop

# 2. Reconstruir todo
make pipeline-build

# 3. Verificar firma de plugins (firmar los que falten)
make sign-plugins

# 4. Verificar que todos los plugins están firmados
vagrant ssh -c "ls -la /usr/lib/ml-defender/plugins/"
# Esperar: .so + .sig para CADA plugin listado en los JSON configs

# 5. Verificar JSON configs: plugins declarados vs presentes
# (ver sección Troubleshooting en DEBT-OPS-002)

# 6. Relanzar y verificar status
make pipeline-start
make pipeline-status
# Esperar: 6/6 RUNNING

# 7. Ejecutar tests de integración
make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"
# Esperar: 12/12 PASSED
```

**Causa conocida de fallo post-rebuild:** si libplugin_loader.so cambia (nueva firma, nueva dependencia), los binarios de componentes deben recompilarse Y los plugins deben estar firmados en /usr/lib/ml-defender/plugins/ con .sig válido. Sin .sig → std::terminate() fail-closed.

---

### Notas del Consejo de Sabios

> DAY 114 — ADR-025 MERGE + Signal Safety + TEST-INTEG-4d + arXiv v15:
> "ADR-025 mergeado a main. Tag v0.3.0-plugin-integrity. 12/12 tests PASSED.
> DEBT-SIGNAL-001/002 resueltos: signal handlers async-signal-safe verificados
> a nivel binario (objdump). TEST-INTEG-4d implementado y PASSED (ml-detector PHASE 2d).
> arXiv Replace v15 submitted (submit/7467190). PHASE 3 abierta.
> Incidente DAY 114: pipeline no arrancó tras rebuild — causa: libplugin_hello.so
> no desplegado ni firmado. ADR-025 fail-closed funcionó correctamente.
> DEBT-SIGN-AUTO y DEBT-OPS-001/002 registrados."
> DAY 114 — ADR-032 APROBADO (Plugin Distribution Chain):
> "YubiKey OpenPGP (no PIV) para Ed25519 — corrección técnica crítica.
> Formato .sig embebido (opción B, 4/5). Multi-key en loader desde día 1.
> customer_id como control lógico documentado. Revocación: revocation.json firmado offline.
> ADR-033 propuesto (TPM measured boot, separado). Soberanía open-source documentada.
> Dos YubiKeys obligatorios (principal + backup). DEBT-CLI-001 registrado."
> — Claude (Anthropic) · DAY 114
---

*Última actualización: DAY 114 — 11 Apr 2026*
*Branch activa: feature/phase3-hardening*
*Tests: 25/25 + TEST-INTEG-4a 3/3 + 4b + 4c 3/3 + 4d 3/3 + 4e 3/3 + SIGN-1..7*
*Paper: Draft v15 ✅ · arXiv Replace v15: submitted (submit/7467190) ✅*
*Pipeline: 6/6 RUNNING ✅*
*ADR-025: MERGEADO main ✅ · Tag: v0.3.0-plugin-integrity*
*ADR-032: APROBADO ✅ · ADR-033: PROPUESTO ⏳*
*PHASE 2: ✅ COMPLETA · PHASE 3: ⏳ EN CURSO*
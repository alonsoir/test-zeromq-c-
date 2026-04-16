# aRGus NDR — BACKLOG
*Última actualización: DAY 119 — 16 Abril 2026*

---

## 📐 Criterio de compleción

| Estado | Criterio |
|---|---|
| ✅ 100% | Implementado + probado en condiciones reales + resultado documentado |
| 🟡 80% | Implementado + compilando + smoke test pasado, sin validación E2E completa |
| 🟡 60% | Implementado parcialmente o con valores placeholder conocidos |
| ⏳ 0% | No iniciado |

---

## 📋 POLÍTICA DE DEUDA TÉCNICA

- **Bloqueante:** se cierra dentro de la feature en que se detectó. No hay merge a main sin test verde.
- **No bloqueante con feature natural:** se asigna a la feature destino. Documentada con ID de feature.
- **No bloqueante sin feature natural:** se acumula hasta abrir `feature/tech-debt-cleanup` (3+ DEBTs sin destino claro).
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad. Compilar o instalar manualmente en la VM sin actualizar estas fuentes = deuda técnica de infraestructura garantizada.

---

## ✅ COMPLETADO

### DAY 119 (16 Apr 2026) — Consolidación infraestructura + secuencia canónica reproducible

**vagrant destroy + up desde cero VALIDADO ✅** — 10 problemas detectados y resueltos
**libsodium 1.0.19 en Vagrantfile ✅** — antes de ONNX/FAISS/XGBoost · bloque build-from-source idempotente
**tmux + xxd en paquetes base Vagrantfile ✅** — línea 206 · `build-essential git wget curl vim jq make rsync locales libc-bin file tmux xxd`
**XGBoost find_lib_path() robusto ✅** — `xgboost.core.find_lib_path()[0]` reemplaza `find` con paths hardcoded
**XGBoost fallback apt + --timeout=300 ✅** — DEBT-XGBOOST-APT-001 registrado (no bloqueante)
**pipeline-build dependencias explícitas ✅** — `crypto-transport-build + etcd-client-build + plugin-loader-build` en pipeline-build
**install-systemd-units target Makefile ✅** — wrapper para install-systemd-units.sh
**set-build-profile target Makefile ✅** — wrapper para set-build-profile.sh (PROFILE=debug|production)
**plugin_xgboost API corregida ✅** — `PluginResult plugin_init(const PluginConfig*)` · `plugin_process_message` · `plugin_shutdown`
**OBS-5 contratos @requires/@ensures/@invariant ✅** — en xgboost_plugin.cpp
**/usr/lib/ml-defender/plugins/ en Vagrantfile ✅** — mkdir + build + deploy plugin_xgboost + plugin_test_message
**plugin_test_message build+deploy en Vagrantfile + Makefile ✅** — `plugin-test-message-build` target
**make sync-pubkey ✅** — lee pubkey activa VM → CMakeLists.txt → recompila plugin-loader (temporal — ver DEBT-PUBKEY-RUNTIME-001)
**Secuencia canónica 9 pasos documentada ✅** — post vagrant destroy, reproducible
**6/6 RUNNING + make test-all VERDE ✅** — incluyendo TEST-INTEG-SIGN PASSED
Commits: 8d964390 → 6055c54d · Pubkey activa: `9ac7b8c5ce2d970f77a5fcfcc3b8463b66082db50636a9e81da3cdbb7b2b8019`

---

### DAY 118 (15 Apr 2026) — PHASE 3 COMPLETADA — v0.4.0 MERGEADO A MAIN 🎉

**AppArmor enforce sniffer ✅** — 0 denials en 300s · TEST-APPARMOR-ENFORCE PASSED
**noclobber audit crítico ✅** — sin riesgos de clobber en rutas sensibles
**CHANGELOG-v0.4.0.md ✅** — docs/CHANGELOG-v0.4.0.md creado y commiteado
**MERGE feature/phase3-hardening → main ✅** — git merge --no-ff · tag v0.4.0-phase3-hardening
**feature/adr026-xgboost ABIERTA ✅** — rama creada · skeleton + Vagrantfile DAY 118
Commits: b6ee97c0 → da0296cd

---

### DAY 117 (14 Apr 2026) — PHASE 3 DEBTs bloqueantes + AppArmor enforce 5/6

**DEBT-VAGRANTFILE-001 ✅** · **DEBT-SEED-PERM-001 + TEST-PERMS-SEED ✅** · **REC-2 noclobber ✅**
**TEST-INVARIANT-SEED ✅** · **TEST-PROVISION-1 8/8 ✅** · **Backup policy .bak.* ✅**
**ADR-021 addendum ✅** · **docs/Recovery Contract ✅** · **DEBT-RAG-BUILD-001 ✅**
**AppArmor enforce 5/6 ✅** · **arXiv Draft v15 recibido de Cornell ✅**
Commits: 85197f96 → fac4cd54

---

### DAY 116–115–114–113–111–62 *(ver git log para historial completo)*

---

## 📋 BACKLOG ACTIVO

### P0 — BLOQUEANTES feature/adr026-xgboost (DAY 120 prioritarios)

| ID | Tarea | Test de cierre | Origen |
|----|-------|---------------|--------|
| **DEBT-PUBKEY-RUNTIME-001** | Mover pubkey de CMakeLists.txt a fichero runtime `/etc/ml-defender/plugins/plugin_pubkey.hex`. CMake: `file(READ ...)`. Elimina `make sync-pubkey` completamente. 100% reproducible, no mezcla host/VM. **BLOQUEANTE DAY 120.** | `vagrant destroy && vagrant up` → plugin-loader compila con pubkey correcta sin `make sync-pubkey` | ChatGPT5 DAY 119 — causa raíz de sync-pubkey |
| **DEBT-BOOTSTRAP-001** | `make bootstrap`: encadena los 9 pasos canónicos con checkpoints, verbose, idempotente. Falla ruidosamente en cada paso. Para "primer clone" y CI. **BLOQUEANTE DAY 120.** | `make bootstrap` desde VM limpia → 6/6 RUNNING + plugin-integ-test PASSED | Consejo unanimidad DAY 119 |
| **DEBT-INFRA-VERIFY-001** | `make check-system-deps`: verifica que libsodium 1.0.19, xgboost 3.2.0, tmux, xxd, etc. están presentes antes de compilar. Dependencia de `pipeline-build`. | `make check-system-deps` falla con mensaje claro si falta dependencia | Qwen + Grok DAY 119 |
| **DEBT-INFRA-VERIFY-002** | `make post-up-verify`: valida entorno post `vagrant up` antes de continuar con bootstrap. Versiones, permisos, paths. | `make post-up-verify` verde → entorno listo para bootstrap | Qwen DAY 119 |
| **OBS-1 / DEBT-XGBOOST-SIGN-001** | Firma Ed25519 del modelo (.ubj.sig). Verificación en plugin_init antes de XGBoosterLoadModel. `make sign-models`. | make sign-models → .ubj.sig válido · plugin_init verifica antes de cargar | Consejo DAY 118 |
| **OBS-2 / TEST-INTEG-XGBOOST-1** | Test unitario: modelo juguete + plugin_process_message con MessageContext sintético + salida ∈ [0,1] no NaN. | make test-all incluye TEST-INTEG-XGBOOST-1 verde | Consejo DAY 118 |

---

### P0 — BLOQUEANTES feature/adr026-xgboost (pendientes entrenamie nto)

| ID | Tarea | Test de cierre | Origen |
|----|-------|---------------|--------|
| **PASO 3** | Localizar feature set RF baseline → `docs/xgboost/features.md` | Columnas exactas, orden, normalización documentados | DAY 119 plan |
| **PASO 4** | `scripts/train_xgboost_baseline.py` — mismo feature set RF (Opción A) · random_state=42 · exportar .json + .ubj · gate: Precision≥0.99 + F1≥0.9985 | Gate verde en CTU-13 Neris · 4 runs mínimo | Consejo DAY 118 |
| **docs/xgboost/plugin-contract.md** | Contrato mínimo ctx->payload: float32[] + num_features + validación NaN/Inf + fallo explícito. Referenciar desde ADR-026. | Fichero existe · plugin valida contrato en runtime | Consejo unanimidad DAY 119 |
| **feature_schema_v1.md** | Esquema versionado de features CTU-13. Orden exacto de columnas. Requisito para reproducibilidad científica. | Documento en docs/ml/ · plugin usa versión declarada | ChatGPT5 DAY 119 |

---

### P1 — Deuda de seguridad crítica (→ feature/crypto-hardening)

| ID | Tarea | Test de cierre | Contexto |
|----|-------|---------------|---------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF en seed_client.cpp | Valgrind/ASan: seed no permanece en heap post-derivación | RAM forensics threat — DAY 116 |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient como resto de componentes | DAY 107 |
| **docs/CRYPTO-INVARIANTS.md** | Tabla invariantes criptográficos + tests de validación | Fichero existe con tabla: invariante · componentes · test | DAY 116 |
| **ADR-021 multi-familia** | Reimplementar seed_families por canal para multi-nodo | Test canal aislado: compromiso A no expone seed canal B | DAY 116 addendum |

---

### P2 — Post-enforce AppArmor (→ feature/ops-tooling)

| ID | Tarea | Test de cierre | Origen |
|----|-------|---------------|--------|
| **DEBT-OPS-001** | make redeploy-plugins: build+sign+deploy en un solo target | make redeploy-plugins → plugins firmados y desplegados | BACKLOG original |
| **DEBT-OPS-002** | docs/operations/troubleshooting.md con síntomas → solución | Fichero existe con sección pipeline + crypto | BACKLOG original |

---

### P3 — Post-PHASE 4 (features futuras, en orden de dependencia)

#### PHASE 4 — feature/adr026-xgboost (activa)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-026** | XGBoost plugins Track 1. Precision ≥ 0.99 (gate médico). DPIA pre-producción. | Plugin XGBoost + firmado + F1 ≥ 0.9985 CTU-13 | feature/adr026-xgboost |
| **OBS-3 / DEBT-XGBOOST-LATENCY** | Medir latencia por inferencia. Tabla comparativa RF vs XGBoost §4 paper. | Latencia registrada en validación CTU-13 | feature/adr026-xgboost |
| **OBS-6 / DEBT-XGBOOST-CACHE** | Cache modelo en plugin_init: static BoosterHandle. | Plugin no recarga modelo en llamadas sucesivas | feature/adr026-xgboost |
| **DEBT-XGBOOST-APT-001** | Verificar versión apt python3-xgboost en Debian bookworm. Documentar en OFFLINE-DEPLOYMENT.md. | vagrant provision red cortada → warning visible + versión documentada | DAY 119 |
| **DEBT-XGBOOST-SOFTFAIL-001** | Soft-fail: si XGBoost no carga, ml-detector continúa con RF + "Modo Protección Degradada" + alerta RAG. | ml-detector no termina si XGBoost falla, alerta CRITICAL | feature/phase5-resilience |
| **DEBT-TOOLS-001** | Synthetic injectors + PluginLoader + plugins firmados Ed25519 | Injectors generan tráfico procesado por plugin | feature/adr026-xgboost |
| **DEBT-FD-001** | Fast Detector Path A → thresholds desde JSON, no hardcoded | sniffer.json controla thresholds · tests con valores distintos | feature/adr026-xgboost |

---

#### ADR-037 — Snyk C++ Security Hardening (→ paralelo o post ADR-026, antes de ADR-036)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-037 / F-001** | Command injection `firewall-acl-agent`: `validate_chain_name()` allowlist regex. | Tests RejectsMalicious · AcceptsValid | feature/adr026-xgboost o tech-debt-cleanup |
| **ADR-037 / F-002** | Path traversal en carga config JSON: `safe_resolve_config()` centralizado. | Tests RejectsTraversal · RejectsSymlink · AcceptsValidProd | feature/adr026-xgboost o tech-debt-cleanup |
| **ADR-037 / F-003** | Integer overflows: checked arithmetic con `std::numeric_limits<>`. | Snyk re-scan C++ → 0 findings F-003 | feature/adr026-xgboost o tech-debt-cleanup |
| **ADR-037 / GATE** | Re-scan Snyk C++. Gate cierre: 0 medios. Python excluido. | Snyk report C++ → 0 medium/critical | pre-ADR-036 obligatorio |

---

#### Features de infraestructura crypto y protocolo

| ID | Tarea | Feature destino |
|----|-------|----------------|
| ADR-024 impl | Noise_IKpsk3 P2P | feature/adr024-noise-p2p |
| ADR-032 Fase A | Manifest JSON + multi-key loader + revocación | feature/adr032-hsm |
| ADR-032 Fase B | YubiKey OpenPGP (2× unidades) + firma HSM | feature/adr032-hsm (post-hardware) |
| ADR-033 TPM | TPM 2.0 Measured Boot. seed_family en hardware. | feature/crypto-hardening |
| DEBT-CLI-001 | ml-defender verify-plugin --bundle CLI | feature/adr032-hsm |

---

#### Variantes hardened y bare-metal

| ID | Tarea | Feature destino |
|----|-------|----------------|
| ADR-029 | Variantes A/B/C · x86 + ARM RPi | feature/bare-metal |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | feature/bare-metal |
| ADR-034 | deployment.yml SSOT topología hospitalaria | feature/bare-metal (tardía) |
| ADR-035 | etcd-server HA — cluster 3 nodos, Raft, mTLS | feature/bare-metal (post ADR-034) |
| ADR-036 | Verificación formal baseline (CBMC + Frama-C) | feature/formal-verification (último) |

---

### ⏸️ POSPUESTO — ADR-033 Institutional Knowledge Base via RAG

Estado: POSPUESTO indefinidamente. Condiciones de activación: operador no resuelve incidente por falta de acceso, ADRs > 40, segundo contribuidor incorporado.

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | ADR/DAY |
|---|---|---|
| seed_family single-node | UN seed compartido para 6 componentes — INVARIANTE-SEED-001 | ADR-021 addendum · DAY 116 |
| RAM protection del seed | explicit_bzero(seed) post-HKDF + mlock(subkeys). Definitivo: TPM ADR-033 | ADR-021 addendum · DAY 116 |
| Plugin integrity | Ed25519 + TOCTOU-safe dlopen + fail-closed std::terminate | ADR-025 · DAY 113 |
| D8-pre bidireccional | READONLY+payload→terminate + NORMAL+nullptr→terminate | ADR-023 FIX-C · DAY 111 |
| MAX_PLUGIN_PAYLOAD_SIZE | 64KB hard limit, std::terminate() | ADR-023 FIX-D · DAY 111 |
| ADR-032 autoridad firma | YubiKey OpenPGP Ed25519. 2 unidades. | ADR-032 · DAY 114 |
| XGBoost vs FT-Transformer | XGBoost Track 1 Year 1. vLLM Track 2 Year 2-3 | ADR-026 pre · DAY 104 |
| Precision gate médico | Precision ≥ 0.99 obligatorio pre-producción. DPIA requerida. | Consejo · DAY 104 |
| XGBoost feature set | Opción A: mismo feature set que RF baseline. | Consejo unanimidad · DAY 118 |
| XGBoost formato modelo | JSON repo (auditoría), .ubj producción (runtime). Firma Ed25519 obligatoria. | Consejo unanimidad · DAY 118 |
| plugin_invoke arquitectura | Opción B: ml-detector pre-procesa → float32[] en payload. Plugin agnóstico. | Consejo unanimidad · DAY 118 |
| std::terminate() XGBoost v0.1 | Fail-closed. Integridad > Disponibilidad. Soft-fail → PHASE 5. | Consejo unanimidad · DAY 118 |
| Vagrantfile/Makefile SSOT | Vagrantfile = sistema + deps externas. Makefile = build + tests + orquestación. NUNCA instalar manualmente en VM sin actualizar ambos. | Lección operacional · DAY 119 |
| pubkey plugin-loader | Temporal: sync-pubkey. Definitivo: fichero runtime `/etc/ml-defender/plugins/plugin_pubkey.hex`. | ChatGPT5 · DAY 119 |
| make bootstrap | 9 pasos canónicos con checkpoints + verbose + idempotente. Para primer clone. | Consejo unanimidad · DAY 119 |
| ctx->payload contrato XGBoost | float32[] + num_features + validación NaN/Inf + fallo explícito. Documentado en plugin-contract.md. | Consejo unanimidad · DAY 119 |

---

## 🔑 Secuencia canónica post `vagrant destroy + up` (DAY 119)

```bash
make up                     # vagrant up defender + client
make post-up-verify         # ⏳ DEBT-INFRA-VERIFY-002 — validar entorno
make check-system-deps      # ⏳ DEBT-INFRA-VERIFY-001 — verificar dependencias
# Temporal hasta DEBT-PUBKEY-RUNTIME-001:
make sync-pubkey            # lee pubkey activa → CMakeLists.txt → recompila plugin-loader
make set-build-profile      # activa symlinks build-active (PROFILE=debug por defecto)
make install-systemd-units  # instala 6 units en /etc/systemd/system/
make sign-plugins           # firma Ed25519 todos los plugins (ADR-025)
make test-provision-1       # CI gate PHASE 3 — 8/8 checks
make pipeline-start         # arranca 6 componentes via tmux
make pipeline-status        # verificar 6/6 RUNNING
make plugin-integ-test      # verificar 6/6 PASSED incluyendo TEST-INTEG-SIGN
```

**Una vez implementado `make bootstrap` (DEBT-BOOTSTRAP-001):**
```bash
make bootstrap   # encadena todo lo anterior
```

**Regla de oro:** 6/6 RUNNING + make test-all VERDE = pipeline estable.

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:            ████████████████████ 100% ✅
HMAC Infrastructure:                   ████████████████████ 100% ✅
F1-Score Validation (CTU-13 Neris):    ████████████████████ 100% ✅  F1=0.9985 · Recall=1.0000
CryptoTransport (HKDF+nonce+AEAD):     ████████████████████ 100% ✅  DAY 97
contexts.hpp (HKDF simétricos):        ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):         ████████████████████ 100% ✅  DAY 99
plugin-loader ADR-012 PHASE 1b:        ████████████████████ 100% ✅  DAY 102
ADR-023 PHASE 2a-2e (6 componentes):   ████████████████████ 100% ✅  DAY 105-112
ADR-025 Plugin Integrity (Ed25519):    ████████████████████ 100% ✅  DAY 113-114 🎉
TEST-INTEG-4a/4b/4c/4d/4e + SIGN:     ████████████████████ 100% ✅  DAY 119 (SIGN reparado)
arXiv:2604.04952 PUBLICADO:            ████████████████████ 100% ✅  DAY 111 🎉
arXiv Replace v15 SUBMITTED:           ████████████████████ 100% ✅  DAY 114 🎉
ADR-024 OQs 5..8 CERRADAS:            ████████████████████ 100% ✅  DAY 115 🎉
PHASE 3 ítems 1-4:                     ████████████████████ 100% ✅  DAY 115 🎉
DEBT-ADR025-D11 (--reset):             ████████████████████ 100% ✅  DAY 116 🎉
TEST-PROVISION-1 (8/8 checks):         ████████████████████ 100% ✅  DAY 117 🎉
AppArmor 6/6 enforce:                  ████████████████████ 100% ✅  DAY 118 🎉
Reproducibilidad vagrant destroy:      ████████████████████ 100% ✅  DAY 119 🎉
Vagrantfile/Makefile como SSOT:        ████████████████████ 100% ✅  DAY 119 🎉
DEBT-CRYPTO-003a (mlock+bzero):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/crypto-hardening
ADR-026 XGBoost Track 1:               ███░░░░░░░░░░░░░░░░░  15% 🟡  skeleton + API correcta + Vagrantfile DAY 119
DEBT-PUBKEY-RUNTIME-001:               ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BLOQUEANTE DAY 120
DEBT-BOOTSTRAP-001 (make bootstrap):   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BLOQUEANTE DAY 120
DEBT-INFRA-VERIFY-001/002:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BLOQUEANTE DAY 120
ADR-037 Snyk C++ Hardening:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  paralelo/post ADR-026
ADR-024 Noise_IKpsk3 impl:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/adr024-noise-p2p
ADR-032 Fase A:                        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/adr032-hsm
ADR-033 TPM Measured Boot:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/crypto-hardening
ADR-029 variantes hardened:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/bare-metal
ADR-034 Deployment Topology:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/bare-metal (post ADR-029)
ADR-035 etcd-server HA:                ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/bare-metal (post ADR-034)
ADR-036 Formal Verification:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/formal-verification (último)
BARE-METAL stress test:                ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado hardware
DEBT-FD-001 (JSON thresholds):         ████░░░░░░░░░░░░░░░░  20% 🟡
```

---

### Notas del Consejo de Sabios — DAY 119

> "DAY 119 ha sido una sesión de consolidación de infraestructura ejemplar. Diez problemas
> detectados y resueltos en una sola sesión. La lección operacional es lapidaria:
> 'El Vagrantfile y el Makefile son la única fuente de verdad.'
>
> Q1 sync-pubkey: CONDICIONAL. Mecanismo válido como parche. Solución estructural real
> (ChatGPT5): mover pubkey a fichero runtime, CMake lee con file(READ ...).
> Elimina sync-pubkey completamente. BLOQUEANTE DAY 120 — DEBT-PUBKEY-RUNTIME-001.
>
> Q2 Vagrantfile/Makefile: SEPARACIÓN CORRECTA unanimidad. Formalizar con
> make check-system-deps como contrato explícito entre ambos.
>
> Q3 make bootstrap: SÍ OBLIGATORIO unanimidad. 9 pasos con checkpoints,
> verbose, idempotente. BLOQUEANTE DAY 120 — DEBT-BOOTSTRAP-001.
>
> Q4 contrato ctx->payload: float32[] + num_features + validación NaN/Inf
> + fallo explícito. Documentar en docs/xgboost/plugin-contract.md.
    > El orden exacto de features depende del modelo — definir feature_schema_v1.md.
>
> Q5 puntos ciegos DAY 120: persistencia etcd post-destroy, permisos plugins,
> caché CMake no invalidada, reloj VM, orden firma (sync-pubkey ANTES de sign-plugins).
> Recomendación unánime: ejecutar secuencia dos veces para verificar idempotencia.
>
> Veredicto global: DAY 119 transforma el proyecto de 'funciona' a
> 'se puede desplegar y mantener'. Salto de madurez importante."
> — Consejo de Sabios (5/7) · DAY 119

---

### Notas del Consejo de Sabios — DAY 118

> "PHASE 3 COMPLETADA. AppArmor 6/6 enforce, merge --no-ff, tag v0.4.0-phase3-hardening.
> Q1 UNANIMIDAD: Opción A — mismo feature set que RF.
> Q2 UNANIMIDAD: JSON repo + .ubj producción. Firma Ed25519 BLOQUEANTE.
> Q3 UNANIMIDAD: Opción B — ml-detector pre-procesa float32[].
> OBS-4 segunda ronda (Gemini rectifica): std::terminate() v0.1 UNANIMIDAD."
> — Consejo de Sabios (5/7 + segunda ronda Gemini) · DAY 118

---

*Última actualización: DAY 119 cierre — 16 Abril 2026*
*Branch activa: feature/adr026-xgboost (main @ v0.4.0-phase3-hardening)*
*Tests: make test-all VERDE · TEST-PROVISION-1 8/8 · 6/6 RUNNING · AppArmor 6/6 enforce*
*Pubkey activa DAY 119: 9ac7b8c5ce2d970f77a5fcfcc3b8463b66082db50636a9e81da3cdbb7b2b8019*
*arXiv: 2604.04952 · v15 ✅ · Tag: v0.4.0-phase3-hardening*
*PHASE 3: COMPLETADA ✅ · PHASE 4: feature/adr026-xgboost 15% · Consejo DAY 119 incorporado*
*"Via Appia Quality — Un escudo, nunca una espada."*
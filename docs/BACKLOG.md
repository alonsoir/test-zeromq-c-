# aRGus NDR — BACKLOG
*Última actualización: DAY 120 — 17 Abril 2026*

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
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad. Compilar o instalar manualmente en la VM sin actualizar ambas fuentes = deuda técnica de infraestructura garantizada.
- **REGLA DE SCRIPTS:** Lógica compleja con quoting anidado → script en `tools/`, nunca inline en Makefile. Ejemplo: `tools/check-xgboost-version.sh`, `tools/extract-pubkey-hex.sh`.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. Nunca toca el sistema de build (CMake, logs, caché). Se lee exclusivamente en runtime con mlock() + explicit_bzero(). Diferente a la pubkey que es pública por diseño.

---

## ✅ COMPLETADO

### DAY 120 (17 Apr 2026) — DEBTs infra + ADR-026 PASO 4a-4e

**DEBT-PUBKEY-RUNTIME-001 ✅** — `tools/extract-pubkey-hex.sh` + `execute_process()` en CMakeLists.txt. `make sync-pubkey` DEPRECATED. Plugin-loader lee pubkey desde `/etc/ml-defender/plugins/plugin_signing.pk` en cmake-time.
**DEBT-BOOTSTRAP-001 ✅** — `make bootstrap` encadena 8 pasos canónicos con checkpoints. Para primer clone y CI.
**DEBT-INFRA-VERIFY-001/002 ✅** — `make check-system-deps` + `make post-up-verify`. `tools/check-xgboost-version.sh`.
**Idempotencia vagrant destroy × 2 VALIDADA ✅** — secuencia canónica verde en ambas iteraciones.
**plugin_test_message → pipeline-build dep ✅** — eliminado del Vagrantfile provisioning (requería plugin-loader headers), añadido como dep explícita de pipeline-build.
**libgomp symlink Vagrantfile ✅** — `libgomp-e985bcbb.so.1.0.0` → `/usr/local/lib/` para dlopen desde plugins C++.
**ADR-026 PASO 4a ✅** — `docs/xgboost/features.md`: 23 features LEVEL1, mapping protobuf, dataset CIC-IDS-2017.
**ADR-026 PASO 4b ✅** — `docs/xgboost/plugin-contract.md`: contrato `ctx->payload` float32[23], invariantes fail-closed, schema v1.
**ADR-026 PASO 4c ✅** — `scripts/train_xgboost_baseline.py` sobre 2.83M flows CIC-IDS-2017: F1=0.9978 (RF=0.9968, +0.001), Precision=0.9973 (RF=0.9944, +0.003), ROC-AUC=1.0000. Gates F1≥0.997 + Precision≥0.99 PASADOS.
**ADR-026 PASO 4d ✅** — `make sign-models` + `tools/sign-model.sh`: `xgboost_cicids2017.ubj.sig` 64 bytes Ed25519.
**ADR-026 PASO 4e ✅** — `TEST-INTEG-XGBOOST-1 PASSED` (contratos técnicos). **PENDIENTE mejora**: casos reales CIC-IDS-2017 (ver DEBT-XGBOOST-TEST-REAL-001).
**xgboost_plugin.cpp inferencia real ✅** — `XGBoosterPredict` implementado, score en `ctx->annotation`, `plugin_api_version()` añadido.
**make test-all VERDE ✅** — todos los tests pasan post DAY 120.
Commits: 0a2bdef3 · Pubkey activa DAY 120: `ec8c4bf0fdce51d556b99b5ca7a74aaad6f6683c6f6914784c732c4abbc8c6e1`

---

### DAY 119 (16 Apr 2026) — Consolidación infraestructura + secuencia canónica reproducible

**vagrant destroy + up desde cero VALIDADO ✅** — 10 problemas detectados y resueltos
**libsodium 1.0.19 en Vagrantfile ✅** · **tmux + xxd en paquetes base ✅** · **XGBoost find_lib_path() robusto ✅**
**pipeline-build dependencias explícitas ✅** · **install-systemd-units + set-build-profile Makefile ✅**
**plugin_xgboost API corregida ✅** · **make sync-pubkey ✅** (temporal — DEBT-PUBKEY-RUNTIME-001 cerrado DAY 120)
**6/6 RUNNING + make test-all VERDE ✅** — incluyendo TEST-INTEG-SIGN PASSED
Commits: 8d964390 → 6055c54d

---

### DAY 118 (15 Apr 2026) — PHASE 3 COMPLETADA — v0.4.0 MERGEADO A MAIN 🎉

**AppArmor enforce 6/6 ✅** · **CHANGELOG-v0.4.0.md ✅** · **MERGE feature/phase3-hardening → main ✅**
**feature/adr026-xgboost ABIERTA ✅**
Commits: b6ee97c0 → da0296cd

---

### DAY 117–111–62 *(ver git log para historial completo)*

---

## 📋 BACKLOG ACTIVO

### P0 — BLOQUEANTES feature/adr026-xgboost (DAY 121 prioritarios)

| ID | Tarea | Test de cierre | Origen |
|----|-------|---------------|--------|
| **DEBT-XGBOOST-TEST-REAL-001** | TEST-INTEG-XGBOOST-1 con casos reales CIC-IDS-2017. Extraer fixtures del CSV: 3 flows ATTACK, 3 BENIGN. Verificar score ATTACK > 0.5, score BENIGN < 0.1. Scores actuales (0.0007/0.0034) son out-of-distribution — test técnicamente correcto pero científicamente insuficiente. **BLOQUEANTE MERGE.** | TEST-INTEG-XGBOOST-1 PASSED con scores discriminantes en datos reales | Consejo UNÁNIME 7/7 DAY 120 |
| **DEBT-SEED-AUDIT-001** | Auditar todos los CMakeLists.txt y fuentes C++. La seed ChaCha20 es material secreto — NUNCA en CMake (expone en CMakeCache.txt y logs CI). Leer exclusivamente en runtime con mlock() + explicit_bzero(). Usar SecureBuffer C++20. Verificar con `grep -r "seed" CMakeLists.txt`. | grep -r "seed" CMakeLists.txt → 0 resultados con hex literal | Qwen (argumento más fuerte) + Consejo DAY 120 |
| **DEBT-XGBOOST-DDOS-001** | `scripts/ddos_detection/train_xgboost_ddos.py` con `ddos_detection_dataset.json` (27MB sintético DeepSeek). Mismas 10 features DDOS_FEATURES. Gate: superar RF baseline. Exportar `.json` + `.ubj`. | F1 + Precision superiores al RF baseline ddos_detection_model.pkl | DAY 120 reflexión |
| **DEBT-XGBOOST-RANSOMWARE-001** | `scripts/ransomware/train_xgboost_ransomware.py` con `data/*_guaranteed.csv` (network + files + processes). Mismas features. Gate: superar RF baseline. Exportar `.json` + `.ubj`. | F1 + Precision superiores al RF baseline simple_effective_model.pkl | DAY 120 reflexión |
| **DEBT-SIGN-MODELS-EXTEND-001** | Extender `make sign-models` para firmar los 3 modelos: xgboost_cicids2017.ubj + xgboost_ddos.ubj + xgboost_ransomware.ubj. | make sign-models → 3 modelos firmados con .sig 64 bytes | DAY 120 plan |

---

### P0 — BLOQUEANTES feature/adr026-xgboost (paper §4)

| ID | Tarea | Test de cierre | Origen |
|----|-------|---------------|--------|
| **OBS-3 / DEBT-XGBOOST-LATENCY** | Tabla comparativa RF vs XGBoost: latencia (μs/flow), F1, Precision, ROC-AUC para los 3 detectores. §4 del paper arXiv:2604.04952. | Tabla en docs/xgboost/ con datos medidos | Consejo DAY 118 + Gemini DAY 120 |
| **PAPER-SECTION-4** | Separar §4 en dos subsecciones explícitas: §4.1 CIC-IDS-2017 real (RF vs XGBoost) + §4.2 DeepSeek sintético (ransomware/DDoS, proof-of-concept con limitaciones explícitas). Riesgo de rechazo si se mezclan. | Sección §4 actualizada en arXiv draft | Consejo UNÁNIME 7/7 DAY 120 |

---

### P1 — Deuda de seguridad crítica (→ feature/crypto-hardening)

| ID | Tarea | Test de cierre | Contexto |
|----|-------|---------------|---------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF en seed_client.cpp. SecureBuffer C++20 (mlock en constructor, explicit_bzero en destructor). | Valgrind/ASan: seed no permanece en heap post-derivación | RAM forensics threat — DAY 116 |
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
| **ADR-026** | XGBoost plugins Track 1. Precision ≥ 0.99 (gate médico). DPIA pre-producción. | 3 plugins XGBoost firmados + F1 superiores RF | feature/adr026-xgboost |
| **OBS-6 / DEBT-XGBOOST-CACHE** | Cache modelo en plugin_init: static BoosterHandle. | Plugin no recarga modelo en llamadas sucesivas | feature/adr026-xgboost |
| **DEBT-XGBOOST-APT-001** | Verificar versión apt python3-xgboost en Debian bookworm. Documentar en OFFLINE-DEPLOYMENT.md. | vagrant provision red cortada → warning visible | DAY 119 |
| **DEBT-XGBOOST-SOFTFAIL-001** | Soft-fail: si XGBoost no carga, ml-detector continúa con RF + "Modo Protección Degradada" + alerta RAG. | ml-detector no termina si XGBoost falla | feature/phase5-resilience |
| **DEBT-FD-001** | Fast Detector Path A → thresholds desde JSON, no hardcoded | sniffer.json controla thresholds · tests con valores distintos | feature/adr026-xgboost |

#### Entrenamiento in-situ (→ investigación futura, Q3 2026)

| ID | Tarea | Gates mínimos | Feature destino |
|----|-------|--------------|----------------|
| **RESEARCH-INSITU-001** | XGBoost warm-start in-situ con tráfico real local del hospital. Distribución via mirror oficial firmado (no BitTorrent puro — riesgo adversarial poisoning). | G1: firma Ed25519 obligatoria + G2: golden test suite en nodo central + G3: KL-divergence < umbral + G4: sandbox 24h + G5: rollback automático si F1 < 0.95 + Human-in-the-loop | feature/phase5-resilience |

#### ADR-037 — Snyk C++ Security Hardening

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-037 / F-001** | Command injection `firewall-acl-agent`: `validate_chain_name()` allowlist regex. | Tests RejectsMalicious · AcceptsValid | feature/adr026-xgboost o tech-debt-cleanup |
| **ADR-037 / F-002** | Path traversal en carga config JSON: `safe_resolve_config()` centralizado. | Tests RejectsTraversal · RejectsSymlink | feature/adr026-xgboost o tech-debt-cleanup |
| **ADR-037 / F-003** | Integer overflows: checked arithmetic con `std::numeric_limits<>`. | Snyk re-scan C++ → 0 findings F-003 | feature/adr026-xgboost o tech-debt-cleanup |
| **ADR-037 / GATE** | Re-scan Snyk C++. Gate cierre: 0 medios. Python excluido. | Snyk report C++ → 0 medium/critical | pre-ADR-036 obligatorio |

#### Features de infraestructura crypto y protocolo

| ID | Tarea | Feature destino |
|----|-------|----------------|
| ADR-024 impl | Noise_IKpsk3 P2P | feature/adr024-noise-p2p |
| ADR-032 Fase A | Manifest JSON + multi-key loader + revocación | feature/adr032-hsm |
| ADR-032 Fase B | YubiKey OpenPGP (2× unidades) + firma HSM | feature/adr032-hsm (post-hardware) |
| ADR-033 TPM | TPM 2.0 Measured Boot. seed_family en hardware. | feature/crypto-hardening |
| DEBT-CLI-001 | ml-defender verify-plugin --bundle CLI | feature/adr032-hsm |

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
| Vagrantfile/Makefile SSOT | Vagrantfile = sistema + deps externas. Makefile = build + tests + orquestación. | Lección operacional · DAY 119 |
| Scripts tools/ para quoting | Lógica compleja → tools/script.sh. Nunca inline en Makefile. | Lección operacional · DAY 120 |
| pubkey plugin-loader | RESUELTO: fichero runtime `/etc/ml-defender/plugins/plugin_signing.pk`. CMake: execute_process(). Sin sync-pubkey. | DEBT-PUBKEY-RUNTIME-001 · DAY 120 |
| seed ChaCha20 | NUNCA en CMake ni build. Solo runtime: mlock() + explicit_bzero(). SecureBuffer C++20. | Qwen + Consejo DAY 120 |
| make bootstrap | 8 pasos canónicos con checkpoints + verbose + idempotente. Para primer clone. | Consejo unanimidad · DAY 120 |
| ctx->payload contrato XGBoost | float32[23] + validación NaN/Inf + fallo explícito. Schema v1. | Consejo unanimidad · DAY 119 |
| Datasets: real vs sintético | CIC-IDS-2017 real (level1). DeepSeek sintético (ransomware/DDoS). Paper §4 separado explícitamente. | Consejo UNÁNIME 7/7 · DAY 120 |
| XGBoost in-situ distribución | Mirror oficial firmado (no BitTorrent puro). Gates G1-G5. Human-in-the-loop. Q3 2026+. | Consejo DAY 120 |

---

## 🔑 Secuencia canónica post `vagrant destroy + vagrant up` (DAY 120)

```bash
make up                     # vagrant up defender + client (~20-30 min provisioning)
make post-up-verify         # valida entorno post-up (plugins, seeds, signing key)
make check-system-deps      # verifica libsodium 1.0.19, xgboost 3.2.0, tmux, xxd...
make set-build-profile      # activa symlinks build-active (PROFILE=debug por defecto)
make install-systemd-units  # instala 6 units en /etc/systemd/system/
make pipeline-build         # compila pipeline completo (pubkey leída en cmake-time automáticamente)
make sign-plugins           # firma Ed25519 todos los plugins (ADR-025)
make sign-models            # firma Ed25519 modelos XGBoost (.ubj → .sig)
make test-provision-1       # CI gate PHASE 3 — 8/8 checks
make pipeline-start         # arranca 6 componentes via tmux
make pipeline-status        # verificar 6/6 RUNNING
make plugin-integ-test      # verificar 6/6 PASSED incluyendo TEST-INTEG-SIGN
```

**Equivalente en un solo comando (DEBT-BOOTSTRAP-001 cerrado DAY 120):**
```bash
make bootstrap   # encadena todo lo anterior con checkpoints
```

**Regla de oro:** 6/6 RUNNING + make test-all VERDE = pipeline estable.

---

## 📊 Estado global del proyecto

Foundation + Thread-Safety:            ████████████████████ 100% ✅
HMAC Infrastructure:                   ████████████████████ 100% ✅
F1-Score Validation (CTU-13 Neris):    ████████████████████ 100% ✅  F1=0.9985 · Recall=1.0000
CryptoTransport (HKDF+nonce+AEAD):     ████████████████████ 100% ✅  DAY 97
contexts.hpp (HKDF simétricos):        ████████████████████ 100% ✅  DAY 99
TEST-INTEG-1/2/3 (gate arXiv):         ████████████████████ 100% ✅  DAY 99
plugin-loader ADR-012 PHASE 1b:        ████████████████████ 100% ✅  DAY 102
ADR-023 PHASE 2a-2e (6 componentes):   ████████████████████ 100% ✅  DAY 105-112
ADR-025 Plugin Integrity (Ed25519):    ████████████████████ 100% ✅  DAY 113-114 🎉
TEST-INTEG-4a/4b/4c/4d/4e + SIGN:     ████████████████████ 100% ✅  DAY 120
arXiv:2604.04952 PUBLICADO:            ████████████████████ 100% ✅  DAY 111 🎉
arXiv Replace v15 SUBMITTED:           ████████████████████ 100% ✅  DAY 114 🎉
PHASE 3 ítems 1-4:                     ████████████████████ 100% ✅  DAY 115 🎉
TEST-PROVISION-1 (8/8 checks):         ████████████████████ 100% ✅  DAY 117 🎉
AppArmor 6/6 enforce:                  ████████████████████ 100% ✅  DAY 118 🎉
Reproducibilidad vagrant destroy × 2:  ████████████████████ 100% ✅  DAY 120 🎉
DEBT-PUBKEY-RUNTIME-001:               ████████████████████ 100% ✅  DAY 120 🎉
DEBT-BOOTSTRAP-001 (make bootstrap):   ████████████████████ 100% ✅  DAY 120 🎉
DEBT-INFRA-VERIFY-001/002:             ████████████████████ 100% ✅  DAY 120 🎉
ADR-026 PASO 4a-4e (XGBoost level1):   ████████████████████ 100% ✅  DAY 120 🎉  F1=0.9978
ADR-026 PASO 4e TEST-INTEG-XGBOOST-1: ████████████████░░░░  80% 🟡  contratos OK, scores reales pendientes
DEBT-XGBOOST-TEST-REAL-001:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BLOQUEANTE MERGE DAY 121
DEBT-SEED-AUDIT-001:                   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BLOQUEANTE DAY 121
DEBT-XGBOOST-DDOS-001:                 ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 121
DEBT-XGBOOST-RANSOMWARE-001:           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 121
PAPER-SECTION-4 (real vs sintético):   ░░░░░░░░░░░░░░░░░░░░   0% ⏳  DAY 121
DEBT-CRYPTO-003a (mlock+bzero):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/crypto-hardening
ADR-037 Snyk C++ Hardening:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  paralelo/post ADR-026
ADR-024 Noise_IKpsk3 impl:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/adr024-noise-p2p
ADR-032 Fase A:                        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/adr032-hsm
ADR-033 TPM Measured Boot:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/crypto-hardening
ADR-029 variantes hardened:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/bare-metal
ADR-034/035/036:                       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/bare-metal (post ADR-029)
BARE-METAL stress test:                ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado hardware

---

### Notas del Consejo de Sabios — DAY 120

> "DAY 120 ha cerrado los 3 DEBTs bloqueantes de infraestructura y ha completado los
> 5 pasos del PASO 4 de ADR-026. El primer plugin ML real de aRGus está operativo.
>
> UNANIMIDAD 7/7 en las 4 preguntas:
>
> P1 — TEST-INTEG-XGBOOST-1: RECHAZADO como gate de merge. Los scores 0.0007/0.0034
> son out-of-distribution. El test valida contratos técnicos pero no utilidad del modelo.
> EXIGIDO: casos reales CIC-IDS-2017 con score ATTACK > 0.5, BENIGN < 0.1.
> DEBT-XGBOOST-TEST-REAL-001 BLOQUEANTE MERGE.
>
> P2 — Paper §4: SEPARACIÓN OBLIGATORIA. §4.1 CIC-IDS-2017 real + §4.2 DeepSeek
> sintético con limitaciones explícitas. Mezclarlos = rechazo seguro por revisores.
>
> P3 — In-situ + BitTorrent: VIABLE TÉCNICAMENTE pero INACEPTABLE sin gates G1-G5.
> Qwen/Gemini alertan adversarial poisoning. Mirror oficial firmado, no P2P puro.
> Human-in-the-loop obligatorio. Investigación Q3 2026+.
>
> P4 — DEBT-SEED: Qwen tiene el argumento más fuerte: la seed NO toca el sistema de
> build. execute_process() expone en CMakeCache.txt y logs CI. Solo runtime:
> mlock() + explicit_bzero(). SecureBuffer C++20. AUDITAR MAÑANA.
>
> Nota Gemini: comparar tamaño modelo XGBoost vs RF para Raspberry Pi (caché L2/L3).
> Nota Mistral/Kimi: documentar README.md en data/ sobre origen real vs sintético.
>
> Nuevos miembros del Consejo: Kimi y Mistral incorporados DAY 120. El Consejo
> opera ahora con 7 modelos + Claude como sintetizador."
> — Consejo de Sabios (7/7) · DAY 120

---

### Notas del Consejo de Sabios — DAY 119

> "DAY 119 transforma el proyecto de 'funciona' a 'se puede desplegar y mantener'.
> Salto de madurez importante. Q3 make bootstrap: SÍ OBLIGATORIO unanimidad.
> Q1 sync-pubkey: CONDICIONAL (parche). Solución estructural: fichero runtime.
> BLOQUEANTE DAY 120 — resuelto."
> — Consejo de Sabios (5/7) · DAY 119

---

*Última actualización: DAY 120 cierre — 17 Abril 2026*
*Branch activa: feature/adr026-xgboost (main @ v0.4.0-phase3-hardening)*
*Tests: make test-all VERDE · TEST-PROVISION-1 8/8 · 6/6 RUNNING · AppArmor 6/6 enforce*
*Pubkey activa DAY 120: ec8c4bf0fdce51d556b99b5ca7a74aaad6f6683c6f6914784c732c4abbc8c6e1*
*arXiv: 2604.04952 · v15 ✅ · Tag: v0.4.0-phase3-hardening*
*PHASE 3: COMPLETADA ✅ · PHASE 4: feature/adr026-xgboost 60% · Consejo DAY 120 incorporado*
*"Via Appia Quality — Un escudo, nunca una espada."*
MDEOF
cp /tmp/BACKLOG.md /Users/aironman/CLionProjects/test-zeromq-docker/docs/BACKLOG.md
echo "✅ BACKLOG.md actualizado"
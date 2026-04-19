# aRGus NDR — BACKLOG
*Última actualización: DAY 122 — 19 Abril 2026*

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
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA DE SCRIPTS:** Lógica compleja → `tools/script.sh`, nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().

---

## ✅ COMPLETADO

### DAY 122 (19 Apr 2026) — PHASE 4 COMPLETADA — v0.5.0-preproduction MERGEADO A MAIN 🎉

**DEBT-PRECISION-GATE-001 ✅ CERRADO CON HALLAZGO CIENTÍFICO** — Protocolo Wednesday held-out ejecutado con rigor máximo (md5 sellado, threshold calibrado en validation, Wednesday abierto una sola vez). Resultado: Precision=0.9945/Recall=0.9818 in-distribution (gates satisfechos). OOD Wednesday: impossibility result documentado — covariate shift estructural por diseño de CIC-IDS-2017 (DoS Hulk/GoldenEye/Slowloris ausentes del train). Consejo 7/7 unánime: MERGE AUTORIZADO con caveat PRE-PRODUCTION. Contribución publicable: evidencia cuantitativa de que datasets académicos son insuficientes como fuente única para NDR producción. Cita Sommer & Paxson 2010.

**train_xgboost_level1_v2.py ✅** — Split temporal Tue+Thu+Fri / validation 20% estratificado / Wednesday BLIND. scale_pos_weight=4.273. XGBoost 3.2.0 early stopping ronda 724. Threshold calibrado en validation: 0.8211.

**xgboost_cicids2017_v2.ubj + .sig ✅** — Modelo exportado .json + .ubj. Firmado Ed25519 via `tools/sign-model.sh` (openssl pkeyutl). 2,262,134 bytes. 64 bytes firma.

**wednesday_eval_report.json ✅** — Reporte OOD sellado permanentemente. md5 Wednesday: bf0dd7e9d991987df4e13ea58a1b409c. Artefacto científico permanente en repositorio.

**Paper Draft v16 ✅** — arXiv:2604.04952 actualizado. §8 XGBoost in-distribution + §8 Wednesday OOD impossibility result. §10.13 structural bias academic datasets. §11.18 ACRL (Adversarial Capture-Retrain Loop). Citas: sommer2010 + caldera2024.

**DEBT-PENTESTER-LOOP-001 ABIERTA** → BACKLOG P3. Primera aproximación: MITRE Caldera. Ver §11.18 paper.

**make sign-models actualizado ✅** — Ahora firma 4 modelos (añadido xgboost_cicids2017_v2.ubj).

**Vagrantfile: pandas + scikit-learn ✅** — Añadir a paso pip de Python en provisioning (deuda no bloqueante de DAY 122).

Commits: hasta HEAD · Branch mergeada: feature/adr026-xgboost → main · Tag: v0.5.0-preproduction

---

### DAY 121 (18 Apr 2026) — DEBTs bloqueantes cerrados

**fix(provision)**: circular dependency plugin_signing.pk → plugin-loader cmake. Idempotencia × 3 certificada.
**DEBT-SEED-AUDIT-001 ✅** · **DEBT-XGBOOST-TEST-REAL-001 ✅** — gate médico PASADO (FTP-Patator real).
**DEBT-XGBOOST-DDOS-001 ✅** — F1=1.0 (DeepSeek sintético 50k, 20× más rápido que RF).
**DEBT-XGBOOST-RANSOMWARE-001 ✅** — F1=0.9932 (DeepSeek sintético 3k, 6× más rápido que RF).
**DEBT-SIGN-MODELS-EXTEND-001 ✅** · **docs/xgboost/comparison-table.md ✅** · **PAPER-SECTION-4 ✅**
Commits: 5b1a3021 → 55880c7c

---

### DAY 120 (17 Apr 2026) — DEBTs infra + ADR-026 PASO 4a-4e

**DEBT-PUBKEY-RUNTIME-001 ✅** · **DEBT-BOOTSTRAP-001 ✅** · **DEBT-INFRA-VERIFY-001/002 ✅**
**Idempotencia vagrant destroy × 2 ✅** · **ADR-026 PASO 4a-4e ✅** — F1=0.9978, Precision=0.9973, ROC-AUC=1.0
Commits: 0a2bdef3

---

### DAY 119–118–117–111 *(ver git log)*

---

## 📋 BACKLOG ACTIVO

### P0 — BLOQUEANTES próxima feature

*Sin bloqueantes activos. PHASE 4 completada. Main @ v0.5.0-preproduction.*

---

### P1 — Deuda de infraestructura inmediata (→ hotfix o próxima feature)

| ID | Tarea | Test de cierre | Contexto |
|----|-------|---------------|---------|
| **DEBT-PROVISION-PANDAS** | Añadir `pandas scikit-learn` al paso pip del Vagrantfile (provisioning) | vagrant destroy + bootstrap → train_xgboost_level1_v2.py sin ModuleNotFoundError | DAY 122 |

---

### P2 — Deuda de seguridad crítica (→ feature/crypto-hardening)

| ID | Tarea | Test de cierre | Contexto |
|----|-------|---------------|---------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF en seed_client.cpp. SecureBuffer C++20. | Valgrind/ASan: seed no permanece en heap | RAM forensics threat — DAY 116 |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient como resto de componentes | DAY 107 |
| **docs/CRYPTO-INVARIANTS.md** | Tabla invariantes criptográficos + tests validación | Fichero existe con tabla completa | DAY 116 |

---

### P3 — Post-PHASE 4 (features futuras)

#### PHASE 5 — Loop Adversarial (→ feature/adr038-acrl)

| ID | Tarea | Gates mínimos | Feature destino |
|----|-------|--------------|----------------|
| **DEBT-PENTESTER-LOOP-001** | Adversarial Capture-Retrain Loop (ACRL). Fase 1: MITRE Caldera como red team determinista (ATT&CK-mapped). Captura con eBPF/XDP → flows etiquetados → reentrenamiento XGBoost warm-start → firma Ed25519 → hot-swap. | G1: reproducibilidad (seed fijo + versioned scenario) · G2: ground-truth a nivel de flow (logs Caldera) · G3: cobertura ≥3 familias ATT&CK · G4: tráfico RFC-válido (tshark validation) · G5: red sandbox aislada | feature/adr038-acrl |
| **ADR-038** | Adversarial Capture-Retrain Loop — ADR formal. Incluye: arquitectura del loop, especificaciones mínimas pentester, criterios de validación modelo resultante, protocolo de distribución (Ed25519 + hot-swap). | ADR-038 aprobado por Consejo | feature/adr038-acrl |

#### ADR-037 — Snyk C++ Security Hardening

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-037 / F-001** | Command injection `firewall-acl-agent`: `validate_chain_name()` allowlist regex. | Tests RejectsMalicious · AcceptsValid | feature/tech-debt-cleanup |
| **ADR-037 / F-002** | Path traversal en carga config JSON: `safe_resolve_config()` centralizado. | Tests RejectsTraversal · RejectsSymlink | feature/tech-debt-cleanup |
| **ADR-037 / F-003** | Integer overflows: checked arithmetic con `std::numeric_limits<>`. | Snyk re-scan C++ → 0 findings F-003 | feature/tech-debt-cleanup |
| **ADR-037 / GATE** | Re-scan Snyk C++. Gate cierre: 0 medios. Python excluido. | Snyk report C++ → 0 medium/critical | pre-ADR-036 obligatorio |

#### Features de infraestructura crypto y protocolo

| ID | Tarea | Feature destino |
|----|-------|----------------|
| ADR-024 impl | Noise_IKpsk3 P2P | feature/adr024-noise-p2p |
| ADR-032 Fase A | Manifest JSON + multi-key loader + revocación | feature/adr032-hsm |
| ADR-032 Fase B | YubiKey OpenPGP (2× unidades) + firma HSM | feature/adr032-hsm (post-hardware) |
| ADR-033 TPM | TPM 2.0 Measured Boot. seed_family en hardware. | feature/crypto-hardening |

#### Variantes hardened y bare-metal

| ID | Tarea | Feature destino |
|----|-------|----------------|
| ADR-029 | Variantes A/B/C · x86 + ARM RPi | feature/bare-metal |
| ADR-034 | deployment.yml SSOT topología hospitalaria | feature/bare-metal |
| ADR-035 | etcd-server HA — cluster 3 nodos, Raft, mTLS | feature/bare-metal |
| ADR-036 | Verificación formal baseline (CBMC + Frama-C) | feature/formal-verification |

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | ADR/DAY |
|---|---|---|
| Datasets académicos para NDR | INSUFICIENTES como fuente única. Covariate shift estructural demostrado empiricamente (Wednesday OOD). Modelo de arranque válido; modelo fundacional requiere ACRL. | Consejo 7/7 · DAY 122 |
| DEBT-PRECISION-GATE-001 | CERRADO con hallazgo. Gate certifica in-distribution (Prec=0.9945/Rec=0.9818). Caveat PRE-PRODUCTION. No desplegar en hospitales hasta ACRL completado. | Consejo 7/7 · DAY 122 |
| Loop adversarial (ACRL) | Arquitectura: IA pentester → captura real → reentrenamiento → hot-swap firmado. Fase 1: MITRE Caldera determinista. Fase 2: generative AI para variantes. | Consejo 7/7 · DAY 122 |
| Nomenclatura loop | "Adversarial Capture-Retrain Loop (ACRL)". Relacionado con "Adversarial Data Flywheel". Citado en paper §11.18. | Consejo 7/7 · DAY 122 |
| seed_family single-node | UN seed compartido para 6 componentes — INVARIANTE-SEED-001 | ADR-021 · DAY 116 |
| Plugin integrity | Ed25519 + TOCTOU-safe dlopen + fail-closed std::terminate | ADR-025 · DAY 113 |
| XGBoost vs FT-Transformer | XGBoost Track 1 Year 1. vLLM Track 2 Year 2-3 | ADR-026 pre · DAY 104 |
| Precision gate médico | Precision ≥ 0.99 obligatorio pre-producción. DPIA requerida. | Consejo · DAY 104 |
| Vagrantfile/Makefile SSOT | Vagrantfile = sistema + deps externas. Makefile = build + tests + orquestación. | Lección operacional · DAY 119 |
| pubkey plugin-loader | Fichero runtime `/etc/ml-defender/plugins/plugin_signing.pk`. Sin hardcoding. | DEBT-PUBKEY-RUNTIME-001 · DAY 120 |
| make bootstrap | 8 pasos canónicos con checkpoints + verbose + idempotente. | Consejo unanimidad · DAY 120 |

---

## 🔑 Secuencia canónica (DAY 122+)

```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

**Regla de oro:** 6/6 RUNNING + make test-all VERDE = pipeline estable.

---

## 📊 Estado global del proyecto

```
Foundation + Thread-Safety:            ████████████████████ 100% ✅
HMAC Infrastructure:                   ████████████████████ 100% ✅
F1-Score Validation (CTU-13 Neris):    ████████████████████ 100% ✅  F1=0.9985
CryptoTransport (HKDF+nonce+AEAD):     ████████████████████ 100% ✅
ADR-025 Plugin Integrity (Ed25519):    ████████████████████ 100% ✅
TEST-INTEG-4a/4b/4c/4d/4e + SIGN:     ████████████████████ 100% ✅
AppArmor 6/6 enforce:                  ████████████████████ 100% ✅
arXiv:2604.04952 PUBLICADO:            ████████████████████ 100% ✅
arXiv Replace v16 SUBMITTED:           ████████████████████ 100% ✅  DAY 122 🎉
PHASE 3 completada v0.4.0:             ████████████████████ 100% ✅
PHASE 4 completada v0.5.0-preprod:     ████████████████████ 100% ✅  DAY 122 🎉
ADR-026 XGBoost level1 in-dist:        ████████████████████ 100% ✅  Prec=0.9945/Rec=0.9818
Wednesday OOD finding documentado:     ████████████████████ 100% ✅  DAY 122 🎉
DEBT-PRECISION-GATE-001:               ████████████████████ 100% ✅  CERRADO CON HALLAZGO
make bootstrap idempotente:            ████████████████████ 100% ✅
DEBT-CRYPTO-003a (mlock+bzero):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-037 Snyk C++ Hardening:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳
DEBT-PENTESTER-LOOP-001 (ACRL):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  PRÓXIMA FRONTERA
ADR-038 ACRL formal:                   ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-024 Noise_IKpsk3 impl:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-032 Fase A:                        ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-033 TPM Measured Boot:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳
ADR-029 variantes hardened:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳
BARE-METAL stress test:                ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado hardware
```

---

### Notas del Consejo de Sabios — DAY 122

> "DAY 122 transforma un intento de cerrar un gate en una contribución científica
> de primer orden. El protocolo experimental fue ejecutado con disciplina de acero:
> threshold calibrado solo en validation, Wednesday abierto una sola vez, md5 sellado.
>
> UNANIMIDAD 7/7 en las 6 preguntas:
>
> P1 — Hallazgo publicable: SÍ. Evidencia cuantitativa (threshold sweep completo)
> de imposibilidad operativa. Novedoso respecto a literatura existente.
>
> P2 — DEBT-PRECISION-GATE-001: CERRADO con Opción A + etiqueta PRE-PRODUCTION.
> Merge autorizado. No desplegar en hospitales hasta ACRL completado.
>
> P3 — Paper §4/§5: framing correcto. No es 'el modelo falla', es 'cualquier modelo
> supervisado estático entrenado en estas condiciones fallará'. Citar Sommer & Paxson 2010.
>
> P4 — Nomenclatura: 'Adversarial Capture-Retrain Loop (ACRL)'. Relacionado con
> 'Adversarial Data Flywheel'. Citar literatura existente de red teaming y MITRE ATT&CK.
>
> P5 — DEBT-PENTESTER-LOOP-001: MITRE Caldera como primera aproximación.
> Especificaciones mínimas: determinismo, ground-truth a nivel flow, cobertura ATT&CK,
> tráfico RFC-válido, sandbox aislado. IA generativa → Fase 2.
>
> P6 — Protocolo experimental: riguroso y publicable sin reservas metodológicas.
>
> La arquitectura de aRGus queda REFORZADA, no debilitada. El plugin XGBoost fue
> diseñado desde el principio para ser reemplazable. Hoy tenemos la evidencia empírica
> de por qué eso era necesario.
>
> Frase del día (Kimi): 'No entrenamos con Wednesday porque Wednesday no existe en el
> entrenamiento. Entrenamos con Tuesday, y aprendemos a detectar Wednesday en producción.'
>
> Próxima frontera: DEBT-PENTESTER-LOOP-001 → MITRE Caldera → flows reales → ACRL."
> — Consejo de Sabios (7/7) · DAY 122

---

*Última actualización: DAY 122 cierre — 19 Abril 2026*
*Branch mergeada: feature/adr026-xgboost → main @ v0.5.0-preproduction*
*Tests: make test-all VERDE · TEST-PROVISION-1 8/8 · 6/6 RUNNING · AppArmor 6/6 enforce*
*arXiv: 2604.04952 · Draft v16 ✅ · Tag: v0.5.0-preproduction*
*PHASE 4: COMPLETADA ✅ · PHASE 5: DEBT-PENTESTER-LOOP-001 (ACRL) · PRÓXIMA FRONTERA*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
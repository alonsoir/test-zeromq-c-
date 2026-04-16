# aRGus NDR — BACKLOG
*Última actualización: DAY 118 — 15 Abril 2026*

---

## 📐 Criterio de compleción

### DEBT-XGBOOST-APT-001 — Verificar versión apt python3-xgboost en Debian bookworm
- **Feature destino:** feature/adr026-xgboost
- **Bloqueante:** NO (fallback solo para air-gapped)
- **Contexto:** el fallback apt del Vagrantfile instala python3-xgboost sin pin de versión.
  Debian bookworm puede proveer una versión != 3.2.0, rompiendo reproducibilidad científica.
- **Tarea:** `apt show python3-xgboost` en VM limpia → documentar versión en OFFLINE-DEPLOYMENT.md
- **Test de cierre:** `vagrant provision` en red cortada → warning visible + versión documentada

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

---

## ✅ COMPLETADO

### DAY 118 (15 Apr 2026) — PHASE 3 COMPLETADA — v0.4.0 MERGEADO A MAIN 🎉

**AppArmor enforce sniffer ✅** — 0 denials en 300s · TEST-APPARMOR-ENFORCE PASSED
**noclobber audit crítico ✅** — sin riesgos de clobber en rutas sensibles
**CHANGELOG-v0.4.0.md ✅** — docs/CHANGELOG-v0.4.0.md creado y commiteado
**MERGE feature/phase3-hardening → main ✅** — git merge --no-ff · tag v0.4.0-phase3-hardening
**feature/adr026-xgboost ABIERTA ✅** — rama creada · XGBOOST-VALIDATION.md pendiente
Commits: b6ee97c0 → da0296cd

---

### DAY 117 (14 Apr 2026) — PHASE 3 DEBTs bloqueantes + AppArmor enforce 5/6

**DEBT-VAGRANTFILE-001 ✅** — apparmor-utils + apparmor-profiles en Vagrantfile
**DEBT-SEED-PERM-001 + TEST-PERMS-SEED ✅** — seed_client.cpp 0600→0640 · ctest 2/2
**REC-2 noclobber + hook 0-bytes ✅** — set -o noclobber en 3 scripts · pre-commit hook
**TEST-INVARIANT-SEED ✅** — make test-invariant-seed integrado en make test-all
**TEST-PROVISION-1 8/8 ✅** — check #8 apparmor-utils · echoes 7/7→8/8 · test-all gate
**Backup policy .bak.* ✅** — cleanup_old_backups() · 3 resets → 14 backups (2×7) ✅
**ADR-021 addendum ✅** — INVARIANTE-SEED-001 validado producción + backup policy
**docs/Recovery Contract (OQ-6 ADR-024) ✅** — 5 pasos + lección pubkey hardcoded
**DEBT-RAG-BUILD-001 ✅** — rag/build→build-debug · build-active symlink · 6/6 en set-build-profile.sh
**apparmor-utils check #8 ✅** — TEST-PROVISION-1 check #8 verde
**tools/apparmor-promote.sh ✅** — complain→enforce · monitor 5min · rollback automático
**AppArmor enforce 5/6 ✅** — etcd-server · rag-security · rag-ingester · ml-detector · firewall (0 denials cada uno)
**AppArmor enforce sniffer ✅** — DAY 118 · 0 denials 300s · TEST-APPARMOR-ENFORCE PASSED
**Pubkey dev rotada ✅** — e51a91e91d72f74fe97e8a4eb883c9c6eb41dd2fc994feaf59d5ba2177720f3d
**arXiv Draft v15 recibido de Cornell ✅** — https://arxiv.org/abs/2604.04952
Commits: 85197f96 → fac4cd54 (7 commits)

---

### DAY 116 (13 Apr 2026) — PHASE 3 CORE + Bug crítico seed_family

**DEBT-ADR025-D11: provision.sh --reset ✅**
- reset_all_keys(): UN seed_family compartido → 6 componentes (INVARIANTE-SEED-001)
- reset_plugin_signing_keypair(): backup + regeneración + mensaje operacional
- TEST-RESET-1/2/3: PASSED
- Nueva pubkey dev: c44a4fe2bfe4ee8ad86f840277625e10ca1c97e85671f366c38a38e6bf02d575
- Bug arquitectural resuelto: seeds independientes → HKDF MAC fail → ver ADR-021 addendum
- Commits: 3c0a214f

**TEST-PROVISION-1 checks 6+7 ✅**
- Check #6: permisos .sk no world-writable, seed.bin = 640
- Check #7: plugins activos en JSONs tienen .so + .sig
- 7/7 checks PASSED — Commit: e01b5919

**AppArmor complain mode (6/6) ✅**
- 6 perfiles en tools/apparmor/, instalados en /etc/apparmor.d/
- 0 denials con pipeline 6/6 + 12/12 PASSED
- Commit: efe203bf

**Documentación DAY 116 ✅**
- README.md + BACKLOG.md actualizados
- ADR-021 addendum: INVARIANTE-SEED-001 + threat model RAM
- Prompt de continuidad DAY 117
- Commit: 9bf0209d

---

### DAY 115 (12 Apr 2026) — PHASE 3 ítems 1-4 + ADR-024 OQs

**ADR-024 OQ-5..8: CERRADAS (Consejo unanimidad) ✅**
**PHASE 3 ítem 1 — systemd units ✅**
**PHASE 3 ítem 2 — DEBT-SIGN-AUTO ✅**
**PHASE 3 ítem 3 — DEBT-HELLO-001 ✅**
**PHASE 3 ítem 4 — TEST-PROVISION-1 (5/5) ✅**
Commits: df976d90, a1b23882

---

### DAY 114 (11 Apr 2026) — ADR-025 MERGE + Signal Safety + arXiv v15

**ADR-025 Plugin Integrity: MERGEADO A MAIN ✅** — Tag: v0.3.0-plugin-integrity. 12/12 tests PASSED.
**arXiv Replace v15 SUBMITTED ✅** — submit/7467190
**ADR-032 Plugin Distribution Chain: APROBADO ✅** — YubiKey OpenPGP Ed25519. Formato .sig embebido.
Commits: 65a29034, 37c22423

---

### DAY 113 — ADR-025 IMPLEMENTADO + Paper v14 *(ver git log)*
### DAY 111 — arXiv:2604.04952 PUBLICADO 🎉 · DOI: https://doi.org/10.48550/arXiv.2604.04952
### DAY 110–62 *(ver historial completo en git log)*

---

## 📋 BACKLOG ACTIVO

### P0 — BLOQUEANTES feature/phase3-hardening
**✅ TODOS CERRADOS DAY 118 — PHASE 3 COMPLETADA · main @ v0.4.0-phase3-hardening**

---

### P1 — Deuda de seguridad crítica (→ feature/crypto-hardening)

| ID | Tarea | Test de cierre | Contexto |
|----|-------|---------------|---------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF en seed_client.cpp. Post-bzero solo viven los subkeys en RAM con mlock(). | Verificar con valgrind/ASan que seed no permanece en heap post-derivación | RAM forensics threat — DAY 116. Ver ADR-021 addendum. |
| **DEBT-SNIFFER-SEED** | Unificar sniffer bajo SeedClient | sniffer arranca con SeedClient como resto de componentes | DAY 107 |
| **docs/CRYPTO-INVARIANTS.md** | Tabla invariantes criptográficos + tests de validación asociados | Fichero existe con tabla: invariante · componentes · test | DAY 116 — ver ADR-021 addendum |
| **ADR-021 multi-familia** | Reimplementar seed_families por canal para topología multi-nodo | Test canal aislado: compromiso componente A no expone seed canal B | DAY 116 addendum — en single-node seed compartido es aceptable |

---

### P2 — Post-enforce AppArmor (→ feature/ops-tooling)

| ID | Tarea | Test de cierre | Origen |
|----|-------|---------------|--------|
| **DEBT-OPS-001** | make redeploy-plugins: build+sign+deploy en un solo target | make redeploy-plugins → plugins firmados y desplegados sin pasos manuales | BACKLOG original |
| **DEBT-OPS-002** | Documentación operativa + sección Troubleshooting pipeline | docs/operations/troubleshooting.md con síntomas → solución | BACKLOG original |

---

### P3 — Post-PHASE 3 (features futuras, en orden recomendado)

> **Nota de ordenación:** los ítems se listan en orden de dependencia lógica.
> ADR-037 puede ejecutarse en paralelo con ADR-026 o inmediatamente después.
> ADR-034 y ADR-035 son los últimos antes de ADR-036, ya que requieren
> topología multi-nodo y bare-metal completamente validados.

#### PHASE 4 — feature/adr026-xgboost (activa)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-026** | XGBoost plugins Track 1. Precision ≥ 0.99 (gate médico). DPIA requerida pre-producción. Pre-req: AppArmor enforce completo + todos los DEBTs bloqueantes cerrados. | Plugin XGBoost cargado + firmado + F1 ≥ 0.9985 en replay CTU-13 | feature/adr026-xgboost |
| **OBS-1 / DEBT-XGBOOST-SIGN-001** | Firma Ed25519 del modelo (.ubj.sig). Mismo esquema ADR-025. Verificación antes de XGBoosterLoadModel. **BLOQUEANTE merge.** | make sign-models → .ubj.sig válido · plugin_init verifica antes de cargar | feature/adr026-xgboost |
| **OBS-2 / TEST-INTEG-XGBOOST-1** | Test unitario: cargar modelo juguete + plugin_invoke con MessageContext sintético + verificar salida ∈ [0,1] y no NaN. **BLOQUEANTE merge.** | make test-all incluye TEST-INTEG-XGBOOST-1 verde | feature/adr026-xgboost |
| **OBS-3 / DEBT-XGBOOST-LATENCY** | Medir latencia por inferencia desde Fase 3. Para tabla comparativa RF vs XGBoost en §4 paper. | Latencia registrada en cada run de validación CTU-13 | feature/adr026-xgboost |
| **OBS-5 / DEBT-XGBOOST-CONTRACTS** | Contratos informales ADR-036 en xgboost_plugin.cpp: @requires @ensures @invariant. | Comentarios presentes antes de merge | feature/adr026-xgboost |
| **OBS-6 / DEBT-XGBOOST-CACHE** | Cache modelo en plugin_init: static BoosterHandle. Evitar reload en cada invocación. | Plugin no recarga modelo en llamadas sucesivas | feature/adr026-xgboost |
| **DEBT-XGBOOST-SOFTFAIL-001** | Soft-fail: si XGBoost no carga, ml-detector continúa con RF + "Modo Protección Degradada" + alerta RAG. | ml-detector no termina si XGBoost falla, pero alerta CRITICAL | feature/phase5-resilience |
| **DEBT-XGBOOST-PROVISION-001** | ✅ DAY 118 — Vagrantfile bloque XGBoost 3.2.0 (líneas 327-348). Fallback apt pendiente DAY 119. | vagrant destroy && vagrant up → XGBoost 3.2.0 disponible | feature/adr026-xgboost |
| **DEBT-TOOLS-001** | Synthetic injectors + PluginLoader + plugins firmados Ed25519 | Injectors generan tráfico procesado por plugin correctamente | feature/adr026-xgboost |
| **DEBT-FD-001** | Fast Detector Path A → thresholds desde JSON, no hardcoded | sniffer.json controla thresholds · tests con valores distintos pasan | feature/adr026-xgboost |

---

#### ADR-037 — Snyk C++ Security Hardening (→ paralelo o post ADR-026, antes de ADR-036)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-037 / F-001** | Command injection `firewall-acl-agent`: `validate_chain_name()` en `IPTablesWrapper`. Allowlist regex `[A-Z0-9_\-]{1,28}`. Aplicar también en deserializador JSON. **Fix más urgente.** | `RejectsMaliciousChainName` · `AcceptsValidChainName` · `RejectsLowerCaseChainName` | feature/adr026-xgboost o feature/tech-debt-cleanup |
| **ADR-037 / F-002** | Path traversal en carga de config JSON: `safe_resolve_config()` centralizado en todos los componentes. Prefix whitelist: `../config` (dev) + `/etc/argus` (prod). | `RejectsTraversalPath` · `RejectsSymlink` · `AcceptsValidProdPath` · `AcceptsValidDevPath` | feature/adr026-xgboost o feature/tech-debt-cleanup |
| **ADR-037 / F-003** | Integer overflows en operaciones numéricas C++. Checked arithmetic con `std::numeric_limits<>` en buffer sizes + índices. Tipos explícitos en contadores. | Snyk re-scan C++ → 0 findings F-003 | feature/adr026-xgboost o feature/tech-debt-cleanup |
| **ADR-037 / GATE** | Re-scan Snyk sobre sources C++ una vez backlog completo (pre-ADR-036). **Gate de cierre ADR-037: 0 medios en C++.** Python excluido (fuera de superficie AppArmor/Falco). | Snyk report C++ → 0 medium/critical findings | pre-ADR-036 obligatorio |

---

#### Features de infraestructura crypto y protocolo

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| ADR-024 impl | Noise_IKpsk3 P2P. OQs 5..8 cerradas DAY 115. Listo para implementar. | TEST-INTEG-8/9 PASSED (definidos en ADR-024) | feature/adr024-noise-p2p |
| ADR-032 Fase A | Manifest JSON + multi-key loader + revocación. Ver ADR-032 DAY 114. | Plugin cargado desde manifest firmado + revocación funciona | feature/adr032-hsm |
| ADR-032 Fase B | YubiKey OpenPGP (2× unidades) + firma HSM. Pre-req: hardware. | Plugin firmado con YubiKey verificado por plugin-loader | feature/adr032-hsm (post-hardware) |
| **ADR-033 TPM** | TPM 2.0 Measured Boot. seed_family nunca en userspace. Solución definitiva RAM forensics. Ver ADR-021 addendum DAY 116. | seed no presente en /proc/PID/mem post-arranque | feature/crypto-hardening |
| DEBT-CLI-001 | ml-defender verify-plugin --bundle CLI. Ver ADR-032 DAY 114. | CLI verifica bundle sin pipeline activo | feature/adr032-hsm |

---

#### Variantes hardened y bare-metal

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| ADR-029 | Variantes hardened A/B/C. x86 + ARM RPi. Delta A vs C publicable. Variante A: Debian+AppArmor+eBPF/XDP. Variante B: Debian+AppArmor+libpcap. Variante C: seL4+libpcap. | F1 ≥ 0.9985 + 0 paquetes perdidos bajo carga X Mbps en cada variante | feature/bare-metal |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | vagrant up desde Vagrantfile nuevo → 6/6 RUNNING | feature/bare-metal |
| BARE-METAL stress | tcpreplay en NIC físico. 0 drops a 100 Mbps. | 0 drops · latencia < 2× baseline VM | bloqueado hardware |
| ADR-021 multi-familia | Reimplementar seed_families por canal para multi-nodo. | Test: compromiso componente A no expone seed canal B | feature/crypto-hardening |

---

#### ADR-034 — Deployment Topology Declarativa (→ post ADR-029 + bare-metal)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-034** | `deployment.yml` como SSOT de topología hospitalaria. Ansible + Jinja2 como motor. Jenkins CI/CD. seed_families por planta (ADR-021 multi-familia). | `make validate-topology` verde · despliegue reproducible desde `deployment.yml` | feature/bare-metal (fase tardía) |
| ADR-034 / OQ-1 | Fanout N rag-ingesters → 1 rag-security: benchmark ZeroMQ PUSH/PULL a >50 nodos. ¿Requiere rag-ingester-coordinator? | Pendiente Consejo | feature/bare-metal |
| ADR-034 / OQ-2 | CI/CD: Jenkins vs GitHub Actions vs Gitea Actions (air-gap). Implicaciones supply chain. | Pendiente Consejo | feature/bare-metal |
| ADR-034 / firma | `deployment.yml` firmado con Ed25519 (mismo esquema ADR-025/032). | `deployment.yml.sig` verificado antes de cualquier despliegue | feature/bare-metal |

---

#### ADR-035 — etcd-server Alta Disponibilidad (→ post ADR-034)

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-035** | Cluster etcd 3 nodos mínimo. Raft consensus. mTLS peer-to-peer. Integración con `deployment.yml`. Failover automático para los 6 componentes del pipeline. | Cluster 3 nodos · quorum con 1 nodo caído · componentes reconectan automáticamente | feature/bare-metal (fase tardía) |
| ADR-035 / OQ-1 | CA para mTLS etcd: ¿Ed25519/libsodium (coherente ADR-025) o X.509 ECDSA P-256? | Pendiente Consejo | feature/bare-metal |
| ADR-035 / OQ-2 | Despliegues muy pequeños (1-2 nodos): ¿single-node etcd con SPOF documentado, o modo embedded? | Pendiente Consejo | feature/bare-metal |
| ADR-035 / backup | `make etcd-snapshot` + systemd timer diario + retención 7 días. Integrado en Recovery Contract. | Snapshot generado · restauración documentada en docs/operations/etcd-recovery.md | feature/bare-metal |

---

#### ADR-036 — Formal Verification Baseline (→ último, requiere todo lo anterior)

> **Pre-requisitos hard:** ADR-037 cerrado (0 medios Snyk C++) + ADR-029 + ADR-034 + ADR-035 + merge de todas las features anteriores.

| ID | Tarea | Test de cierre | Feature destino |
|----|-------|---------------|----------------|
| **ADR-036** | Verificación formal baseline. CBMC primero (propiedades de seguridad acotadas). Frama-C/WP para componentes P0: seed_client + crypto-transport. IEC 62443-4-2 SL2 como objetivo. Variante C → rama `research/sel4-verification`. | `make verify-P0` verde · P1+P3 demostradas o criterio de parada activado (3 meses) | feature/formal-verification |
| ADR-036 / OQ-1 | Reducir Fase A a 2 componentes P0: seed_client + crypto-transport únicamente. | Definition of Done explícita redactada en ADR-036 final | feature/formal-verification |
| ADR-036 / OQ-2 | P5 (terminación pipeline) reformular como "ausencia de deadlocks bajo carga". | Propiedad formalizada en ADR-036 final | feature/formal-verification |

---

### ⏸️ POSPUESTO — ADR-033 Institutional Knowledge Base via RAG

**Estado:** POSPUESTO indefinidamente. No es una feature activa.

**Decisión del Consejo (DAY 116, unanimidad):**
La idea es correcta pero prematura. El source of truth actual (ADRs + BACKLOG + scripts)
es suficiente. Cualquier mecanismo de curación manual muere en DAY 150 en modo solopreneur.
El RAG con docs obsoletos es peor que no tener RAG.

**Condiciones de activación** (cualquiera desbloquea):
1. Un operador no resuelve un incidente porque el conocimiento no está accesible.
2. El número de ADRs supera 40 y la búsqueda manual se vuelve lenta.
3. Un segundo contribuidor se incorpora y necesita onboarding estructurado.

**Alternativa más simple cuando llegue el momento:**
`ONBOARDING.md` con estructura "si te encuentras X, mira Y". Sin infraestructura nueva.

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | ADR/DAY |
|---|---|---|
| seed_family single-node | UN seed compartido para 6 componentes — INVARIANTE-SEED-001 | ADR-021 addendum · DAY 116 |
| seed_family multi-nodo | Un seed por familia de canal — blast radius limitado a canal comprometido | ADR-021 · DAY 100/116 |
| RAM protection del seed | explicit_bzero(seed) post-HKDF + mlock(subkeys). Definitivo: TPM ADR-033 | ADR-021 addendum · DAY 116 |
| Firma automática plugins | NUNCA en producción. Dev: provision.sh check-plugins. DEBT-SIGN-AUTO | ADR-025 · DAY 115 |
| provision.sh --reset | Regenera claves SIN auto-firma. Operador firma manualmente post-reset | ADR-025 D11 · DAY 116 |
| AppArmor estrategia | complain → audit → enforce. Sniffer: 48h mínimo en complain | Consejo Q1 · DAY 116 |
| AppArmor enforce orden | etcd-server → rag-* → ml-detector → firewall → sniffer (último) | Consejo Q1 unanimidad · DAY 116 |
| Plugin integrity | Ed25519 + TOCTOU-safe dlopen + fail-closed std::terminate | ADR-025 · DAY 113 |
| D8-pre bidireccional | READONLY+payload→terminate + NORMAL+nullptr→terminate | ADR-023 FIX-C · DAY 111 |
| MAX_PLUGIN_PAYLOAD_SIZE | 64KB hard limit, std::terminate() | ADR-023 FIX-D · DAY 111 |
| ADR-032 autoridad firma | YubiKey OpenPGP Ed25519 (NO PIV). 2 unidades. Firma y prod no comparten dominio | ADR-032 · DAY 114 |
| XGBoost vs FT-Transformer | XGBoost Track 1 Year 1. vLLM explainability Track 2 Year 2-3 | ADR-026 pre · DAY 104 |
| Precision gate médico | Precision ≥ 0.99 obligatorio antes de producción en hospitales | Consejo · DAY 104 · DPIA requerida |
| HKDF telemetría | HTTPS:443. LZ4 antes de ChaCha20 siempre | ADR-013/020 · DAY 94 |
| Deuda bloqueante | Cierra en su feature. Sin merge a main sin test verde | Política · DAY 116 |
| Deuda no bloqueante | Asignada a feature destino o tech-debt-cleanup | Política · DAY 116 |
| ADR-033 KB RAG | POSPUESTO. Condiciones de activación definidas. Alternativa: ONBOARDING.md | Consejo · DAY 116 |
| XGBoost feature set | Opción A: mismo feature set que RF baseline. Ablation study como experimento secundario. | Consejo unanimidad · DAY 118 |
| XGBoost formato modelo | JSON en repo (auditoría), .ubj en producción (runtime). Firma Ed25519 obligatoria (.ubj.sig). | Consejo unanimidad · DAY 118 |
| plugin_invoke arquitectura | Opción B: ml-detector pre-procesa features → float32[] en payload. Plugin agnóstico al formato ZeroMQ. | Consejo unanimidad · DAY 118 |
| std::terminate() XGBoost v0.1 | Fail-closed unanimidad. Integridad > Disponibilidad en v0.1. Soft-fail → DEBT-XGBOOST-SOFTFAIL-001 PHASE 5. | Consejo unanimidad (incl. Gemini 2ª ronda) · DAY 118 |
| ADR-037 Snyk scope | Solo C++. Python excluido (fuera superficie AppArmor/Falco). Re-scan post-backlog pre-ADR-036. | ADR-037 · DAY 118 |
| ADR-034 SSOT topología | deployment.yml como SSOT. Ansible+Jinja2 motor. Jenkins CI/CD. seed_families por planta. | ADR-034 DRAFT · DAY 118 |
| ADR-035 etcd HA | Cluster 3 nodos mínimo. Raft. mTLS peer-to-peer. Failover automático pipeline. | ADR-035 DRAFT · DAY 118 |

---

## 🔑 Procedimiento de verificación de estabilidad del pipeline

```bash
make pipeline-stop
make pipeline-build 2>&1 | tail -5
vagrant ssh -c "sudo bash /vagrant/etcd-server/config/set-build-profile.sh debug"
make sign-plugins
make test-provision-1      # CI gate: 8/8 checks
make pipeline-start && make pipeline-status  # 6/6 RUNNING
make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"  # 12/12 PASSED
```

**Regla de oro:** 6/6 RUNNING + 12/12 PASSED = pipeline estable.
Tras cualquier cambio: stop → build → sign → test-provision-1 → start → status → plugin-integ-test.

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
TEST-INTEG-4a/4b/4c/4d/4e:            ████████████████████ 100% ✅  DAY 114
TEST-INTEG-SIGN-1..7:                  ████████████████████ 100% ✅  DAY 113
DEBT-SIGNAL-001/002:                   ████████████████████ 100% ✅  DAY 114 🎉
arXiv:2604.04952 PUBLICADO:            ████████████████████ 100% ✅  DAY 111 🎉
arXiv Replace v15 SUBMITTED:           ████████████████████ 100% ✅  DAY 114 🎉
ADR-024 OQs 5..8 CERRADAS:            ████████████████████ 100% ✅  DAY 115 🎉
PHASE 3 ítems 1-4:                     ████████████████████ 100% ✅  DAY 115 🎉
DEBT-ADR025-D11 (--reset):             ████████████████████ 100% ✅  DAY 116 🎉
TEST-PROVISION-1 (8/8 checks):         ████████████████████ 100% ✅  DAY 117 🎉
AppArmor complain (6/6 perfiles):      ████████████████████ 100% ✅  DAY 116 🎉
AppArmor enforce (6/6):                ████████████████████ 100% ✅  DAY 118 🎉
DEBT-VAGRANTFILE-001:                  ████████████████████ 100% ✅  DAY 117 🎉
DEBT-SEED-PERM-001 + TEST-PERMS-SEED:  ████████████████████ 100% ✅  DAY 117 🎉
REC-2 (noclobber + 0-bytes):           ████████████████████ 100% ✅  DAY 117 🎉
TEST-INVARIANT-SEED:                   ████████████████████ 100% ✅  DAY 117 🎉
Backup policy .bak.*:                  ████████████████████ 100% ✅  DAY 117 🎉
ADR-021 addendum (repo):               ████████████████████ 100% ✅  DAY 117 🎉
docs/Recovery Contract:                ████████████████████ 100% ✅  DAY 117 🎉
DEBT-RAG-BUILD-001:                    ████████████████████ 100% ✅  DAY 117 🎉
DEBT-CRYPTO-003a (mlock+bzero):        ░░░░░░░░░░░░░░░░░░░░   0% ⏳  feature/crypto-hardening
ADR-026 XGBoost Track 1:               ██░░░░░░░░░░░░░░░░░░  10% 🟡  feature/adr026-xgboost (skeleton + Vagrantfile DAY 118)
ADR-037 Snyk C++ Hardening:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳  paralelo/post ADR-026 · pre-ADR-036
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

### Notas del Consejo de Sabios — DAY 118 (segunda ronda incluida)

> "PHASE 3 COMPLETADA. AppArmor 6/6 enforce, 13/13 DEBTs cerrados, merge --no-ff,
> tag v0.4.0-phase3-hardening. PHASE 4 abierta: feature/adr026-xgboost.
>
> Q1 UNANIMIDAD: Opción A — mismo feature set que RF. La única variable que cambia
> es el algoritmo. Opción B como ablation study secundario.
>
> Q2 UNANIMIDAD: JSON en repo (auditoría científica), .ubj en producción
> (3× más rápido de cargar, menor superficie). Firma Ed25519 del modelo BLOQUEANTE.
>
> Q3 UNANIMIDAD: Opción B — ml-detector pre-procesa features y serializa como
> float32[] en payload. Plugin XGBoost agnóstico al formato ZeroMQ.
>
> Q4 MAYORÍA: pip 3.2.0 primero + fallback apt + docs/OFFLINE-DEPLOYMENT.md.
>
> OBS-4 segunda ronda (Gemini rectifica): std::terminate() en v0.1 UNANIMIDAD.
> Integridad sobre disponibilidad. Soft-fail → DEBT-XGBOOST-SOFTFAIL-001 PHASE 5.
> Condición: Garantía del Provisioning (provision.sh verifica modelo antes de arranque).
>
> Items bloqueantes nuevos para merge a main feature/adr026-xgboost:
> OBS-1: firma Ed25519 del modelo (.ubj.sig) igual que plugins.
> OBS-2: TEST-INTEG-XGBOOST-1 en make test-all.
> Items no bloqueantes: OBS-3 latencia, OBS-5 contratos ADR-036, OBS-6 cache modelo."
> — Consejo de Sabios (5/7 + segunda ronda Gemini) · DAY 118

---

### Notas del Consejo de Sabios — DAY 116

> "PHASE 3 CORE completada. Bug arquitectural crítico resuelto: seeds independientes
> rompían HKDF/MAC — INVARIANTE-SEED-001 ahora explícita en ADR-021 addendum.
> AppArmor 6/6 en complain, 0 denials. TEST-PROVISION-1 a 7/7.
> 13 DEBTs bloqueantes identificados para DAY 117-118.
>
> Consejo unánime: AppArmor enforce DAY 117 (etcd-server → rag-* → ml-detector →
> firewall → sniffer 48h). DEBTs VAGRANTFILE-001 + SEED-PERM-001 DAY 117.
> ADR-026 XGBoost solo cuando enforce completo y DEBTs cerrados.
>
> Amenaza RAM forensics sobre seed_family: mitigación DEBT-CRYPTO-003a
> (explicit_bzero + mlock). Solución definitiva: ADR-033 TPM post-PHASE 4.
>
> ADR-033 Knowledge Base RAG: POSPUESTO por unanimidad. Prematuro.
> Alternativa cuando llegue: ONBOARDING.md + Golden Sources.
> Condiciones de activación definidas.
>
> Política de deuda técnica formalizada: bloqueante cierra en su feature,
> no bloqueante asignada a feature destino. Toda deuda tiene test de cierre."
> — Consejo de Sabios (6 miembros) · DAY 116

---

*Última actualización: DAY 118 cierre — 15 Abril 2026*
*Branch activa: feature/adr026-xgboost (main @ v0.4.0-phase3-hardening)*
*Tests: make test-all VERDE · TEST-PROVISION-1 8/8 · 6/6 RUNNING · AppArmor 6/6 enforce*
*arXiv: 2604.04952 · v15 ✅ · Tag: v0.4.0-phase3-hardening*
*PHASE 3: COMPLETADA ✅ · PHASE 4: feature/adr026-xgboost skeleton DAY 118 · Consejo veredictos incorporados*
*"Via Appia Quality — Un escudo, nunca una espada."*
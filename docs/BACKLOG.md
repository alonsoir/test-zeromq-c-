# aRGus NDR — BACKLOG
*Última actualización: DAY 116 — 13 Abril 2026*

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

### DAY 116 (13 Apr 2026) — PHASE 3 CORE + Bug crítico seed_family

**DEBT-ADR025-D11: provision.sh --reset ✅**
- reset_all_keys(): UN seed_family compartido → 6 componentes (INVARIANTE-SEED-001)
- reset_plugin_signing_keypair(): backup + regeneración + mensaje operacional
- TEST-RESET-1/2/3: PASSED
- Nueva pubkey dev: c44a4fe2bfe4ee8ad86f840277625e10ca1c97e85671f366c38a38e6bf02d575
- Bug arquitectural resuelto: seeds independientes → HKDF MAC fail
- Commits: 3c0a214f

**TEST-PROVISION-1 checks 6+7 ✅**
- Check #6: permisos .sk no world-writable, seed.bin = 640
- Check #7: plugins activos en JSONs tienen .so + .sig
- 7/7 checks PASSED — Commit: e01b5919

**AppArmor complain mode (6/6) ✅**
- 6 perfiles en tools/apparmor/, instalados en /etc/apparmor.d/
- 0 denials con pipeline 6/6 + 12/12 PASSED
- Commit: efe203bf

---

### DAY 115 (12 Apr 2026) — PHASE 3 ítems 1-4 + ADR-024 OQs

*(ver BACKLOG anterior)*

### DAY 114 y anteriores

*(ver git log)*

---

## 📋 BACKLOG ACTIVO

### P0 — DAY 117 (inmediato)

| ID | Tarea | Deadline |
|----|-------|---------|
| **DEBT-VAGRANTFILE-001** | Añadir apparmor-utils al bloque apt del Vagrantfile | DAY 117 |
| **DEBT-SEED-PERM-001** | Corregir mensaje SeedClient: chmod 600 → chmod 640. Añadir TEST-PERMS-SEED | DAY 117 |
| **ADR-021-ADDENDUM** | Documentar INVARIANTE-SEED-001 + threat model RAM + regresión vs multi-familia | DAY 117 |
| **TEST-INVARIANT-SEED** | Verifica post-reset: todos los seed.bin son byte-a-byte idénticos | DAY 117 |
| **APPARMOR-ENFORCE** | Enforce secuencial: etcd-server → rag-* → ml-detector → firewall → sniffer (48h) | DAY 117-118 |

### P1 — Deuda de seguridad crítica

| ID | Tarea | Contexto |
|----|-------|---------|
| **DEBT-CRYPTO-003a** | mlock() + explicit_bzero(seed) post-derivación HKDF en seed_client.cpp. El seed solo se necesita durante la derivación — post-bzero solo viven los subkeys. | RAM forensics threat — DAY 116 |
| DEBT-RAG-BUILD-001 | rag/CMakeLists.txt: build-debug/release como resto de componentes | DAY 115 |

### P2 — Post-enforce AppArmor

| ID | Tarea | Origen |
|----|-------|--------|
| **APPARMOR-PROMOTE-SH** | tools/apparmor-promote.sh: enforce → monitor 5min → rollback si denials | Consejo DAY 116 (Qwen) |
| DOCS-RECOVERY-CONTRACT | Documento operacional rotación claves zero downtime (OQ-6 ADR-024) | Consejo DAY 115 |
| DOCS-CRYPTO-INVARIANTS | docs/CRYPTO-INVARIANTS.md con tabla invariantes + tests de validación | Consejo DAY 116 |
| REC-2 | noclobber + check 0-bytes en CI | Consejo DAY 110 |

### P3 — Post-PHASE 3

| ID | Tarea | Origen |
|----|-------|--------|
| **ADR-026** | XGBoost plugins Track 1. Precision ≥ 0.99 (gate médico). Pre-req: AppArmor enforce completo + DEBTs cerrados | DAY 104 |
| ADR-024 impl | Noise_IKpsk3 P2P. OQs 5..8 cerradas, listo para implementar | DAY 115 |
| ADR-032 Fase A | Plugin Distribution Chain: manifest JSON + multi-key loader + revocación | DAY 114 |
| ADR-032 Fase B | YubiKey OpenPGP (2× unidades) + firma HSM | post-ADR-032-A |
| **ADR-033** | TPM 2.0 Measured Boot. Objetivo: seed_family nunca en userspace RAM. Derivación HKDF en hardware. Solución definitiva a RAM forensics threat. | DAY 116 |
| ADR-029 | Variantes hardened: A=AppArmor+eBPF/XDP · B=AppArmor+libpcap · C=seL4+libpcap. x86 + ARM RPi. Delta A vs C = coste medible de seguridad formal. | DAY 109 |
| ADR-021 multi-familia | Reimplementar seed_families por canal para topología multi-nodo. En single-node el seed compartido es aceptable; en multi-nodo limita blast radius. | DAY 116 addendum |
| DEBT-TOOLS-001 | Synthetic injectors: PluginLoader + plugins firmados | DAY 113 |
| BARE-METAL stress | tcpreplay en NIC físico | bloqueado hardware |
| DEBT-FD-001 | Fast Detector Path A → JSON thresholds | DAY 80 |

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| seed_family single-node | UN seed compartido para 6 componentes — INVARIANTE-SEED-001 | 116 |
| seed_family multi-nodo | Un seed por familia de canal (ADR-021) — pendiente implementación | 100/116 |
| RAM protection del seed | explicit_bzero post-HKDF + mlock subkeys. Solución definitiva: TPM (ADR-033) | 116 |
| Firma automática plugins | NUNCA en producción. Dev: provision.sh check-plugins | 115 |
| provision.sh --reset | Regenera claves SIN auto-firma. Operador firma manualmente post-reset | 116 |
| AppArmor | complain → audit → enforce. Sniffer: 48h mínimo en complain | 116 |
| AppArmor enforce orden | etcd-server → rag-* → ml-detector → firewall → sniffer (último) | 116 |
| Plugin integrity | Ed25519 + TOCTOU-safe dlopen. Fail-closed std::terminate | 113 |
| D8-pre bidireccional | READONLY+payload→terminate + NORMAL+nullptr→terminate | 111 |
| ADR-032 autoridad firma | YubiKey OpenPGP Ed25519 (no PIV). 2 unidades. | 114 |

---

## 📊 Estado global del proyecto

Foundation + Thread-Safety:           ████████████████████ 100% ✅
HMAC Infrastructure:                  ████████████████████ 100% ✅
F1-Score Validation (CTU-13):         ████████████████████ 100% ✅
CryptoTransport (HKDF+nonce+AEAD):    ████████████████████ 100% ✅
Plugin-loader ADR-023 PHASE 2 (6/6):  ████████████████████ 100% ✅
ADR-025 Plugin Integrity (Ed25519):   ████████████████████ 100% ✅ DAY 114 🎉
arXiv:2604.04952 PUBLICADO:           ████████████████████ 100% ✅ DAY 111 🎉
arXiv Replace v15:                    ████████████████████ 100% ✅ DAY 114 🎉
ADR-024 OQs 5..8:                     ████████████████████ 100% ✅ DAY 115 🎉
PHASE 3 ítems 1-4:                    ████████████████████ 100% ✅ DAY 115 🎉
DEBT-ADR025-D11 (--reset):            ████████████████████ 100% ✅ DAY 116 🎉
TEST-PROVISION-1 (7/7):               ████████████████████ 100% ✅ DAY 116 🎉
AppArmor complain (6/6):              ████████████████████ 100% ✅ DAY 116 🎉
AppArmor enforce (5/6):               ░░░░░░░░░░░░░░░░░░░░   0% 🔄 DAY 117
AppArmor enforce sniffer:             ░░░░░░░░░░░░░░░░░░░░   0% ⏳ DAY 118+
DEBT-VAGRANTFILE-001:                 ░░░░░░░░░░░░░░░░░░░░   0% 🔄 DAY 117
DEBT-SEED-PERM-001:                   ░░░░░░░░░░░░░░░░░░░░   0% 🔄 DAY 117
DEBT-CRYPTO-003a (mlock+bzero):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳ P1
ADR-026 XGBoost Track 1:              ░░░░░░░░░░░░░░░░░░░░   0% ⏳ DAY 118+
ADR-033 TPM Measured Boot:            ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-PHASE 4
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴 bloqueado hardware
DEBT-FD-001:                          ████░░░░░░░░░░░░░░░░  20% 🟡

---

### Notas del Consejo de Sabios — DAY 116

> "PHASE 3 CORE completada. Bug arquitectural crítico resuelto: seeds independientes
> rompían HKDF/MAC — INVARIANTE-SEED-001 ahora explícito. AppArmor 6/6 en complain,
> 0 denials. TEST-PROVISION-1 a 7/7 checks.
>
> Consejo unánime: AppArmor enforce DAY 117 (orden: etcd-server → rag-* → ml-detector
> → firewall → sniffer 48h). DEBTs VAGRANTFILE-001 + SEED-PERM-001 DAY 117.
> ADR-026 XGBoost solo cuando enforce completo. Addendum ADR-021 obligatorio.
>
> Amenaza identificada: RAM forensics sobre seed_family compartido. Mitigación:
> explicit_bzero(seed) post-HKDF + mlock(subkeys). Solución definitiva: ADR-033 TPM.
> Blast radius en multi-nodo: reimplementar seed_families por canal (ADR-021)."
> — Consejo de Sabios (5 miembros) · DAY 116

---

*Última actualización: DAY 116 — 13 Abril 2026*
*Branch activa: feature/phase3-hardening*
*Tests: 12/12 plugin-integ-test PASSED · 7/7 TEST-PROVISION-1 · 6/6 RUNNING*
*arXiv: 2604.04952 · v15 submitted ✅ · Tag: v0.3.0-plugin-integrity*
*PHASE 3: CORE COMPLETADO ✅ | DEBTs PENDIENTES: 3*
*"Via Appia Quality — Un escudo, nunca una espada."*
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

### Day 111 (8 Apr 2026) — FIX-C/D + TEST-INTEG-4c + PHASE 2d + ADR-029

**🎉 arXiv:2604.04952 [cs.CR] PUBLICADO**
"ML Defender (aRGus NDR): An Open-Source Embedded ML NIDS for Botnet and
Anomalous Traffic Detection in Resource-Constrained Organizations"
DOI: https://doi.org/10.48550/arXiv.2604.04952 — 3 Apr 2026, 28 páginas.

**FIX-C — D8-pre inverso (ChatGPT5 DAY 110, obligatorio) ✅**
- `plugin_loader.cpp`: PLUGIN_MODE_NORMAL + payload==nullptr → std::terminate()
- Contrato D8-pre bidireccional completo

**FIX-D — MAX_PLUGIN_PAYLOAD_SIZE 64KB (ChatGPT5 DAY 110, obligatorio) ✅**
- `plugin_loader.hpp`: constexpr MAX_PLUGIN_PAYLOAD_SIZE = 65536
- `plugin_loader.cpp`: payload_len > MAX → std::terminate()

**TEST-INTEG-4c — 3/3 PASSED ✅**
- Caso A: NORMAL + payload real → PLUGIN_OK, errors==0
- Caso B: D8 VIOLATION campo read-only detectada → PASS
- Caso C: result_code=-1 → error registrado, no crash

**PHASE 2d — ml-detector ✅**
- `zmq_handler.hpp`: set_plugin_loader() + plugin_loader_ member + include
- `zmq_handler.cpp`: invoke_all(ctx) post-inferencia, PLUGIN_MODE_NORMAL
- `main.cpp`: set_plugin_loader(&plugin_loader_) tras start()
- Pipeline: 6/6 RUNNING

**ADR-029 — g_plugin_loader + async-signal-safe ✅**
- Documenta patrón global exclusivo de rag-security
- D1-D5 formalizados. TEST-INTEG-4e definido.

**Commits DAY 111:**
- b23eca66: FIX-C + FIX-D + TEST-INTEG-4c
- 58d73c04: PHASE 2d ml-detector
- Commit 3: ADR-029
- Branch: feature/plugin-crypto

---

### Day 110 (7 Apr 2026) — PluginMode + PHASE 2c + Paper v13
*(ver historial git)*

### Day 109–62
*(ver historial completo en git log)*

---

## 🔜 PRÓXIMO — PHASE 2e (rag-security)

### PHASE 2e — rag-security (ADR-029 D1-D5)

Archivos: `rag-security/src/main.cpp`
Patrón: g_plugin_loader global + signal handler async-signal-safe + invoke_all READONLY
Gate: TEST-INTEG-4e (3 casos).

### TEST-INTEG-4e

- Caso A: READONLY + evento real → PLUGIN_OK, result_code ignorado
- Caso B: g_plugin_loader=nullptr → invoke_all no llamado, no crash
- Caso C: simulación lógica signal handler → shutdown limpio

---

## 📋 BACKLOG ACTIVO

### P1 — Deuda inmediata

| ID | Tarea | Origen |
|----|-------|--------|
| TEST-INTEG-4e | Gate PHASE 2e: rag-security | ADR-029 |
| arXiv Replace v13 | Subir main_v13.tex — v1 anunciada 2604.04952 | DAY 111 |
| TEST-PROVISION-1 | Gate CI: vagrant destroy → up → 6/6 RUNNING | ChatGPT5 DAY 108 |
| DEBT-SNIFFER-SEED | Unificar sniffer bajo SeedClient | DAY 107 |
| REC-2 | noclobber + check 0-bytes CI | Consejo DAY 110 |

### P1 — feature/plugin-crypto (branch activa)

| ID | Tarea |
|----|-------|
| ADR-025 | Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen) — post PHASE 2 |
| ADR-024 impl | Noise_IKpsk3 dynamic key agreement — FASE 3 post-arXiv |

### P2 — Post-PHASE 2

| ID | Tarea | Origen |
|----|-------|--------|
| BARE-METAL-IMAGE | Imagen Debian Bookworm hardened — exportable a USB | P2 |
| BARE-METAL stress | tcpreplay 100/250/500/1000 Mbps en NIC físico | P2 |
| DEBT-FD-001 | Fast Detector Path A → JSON thresholds | DAY 80 |
| DEBT-CRYPTO-003a | mlock() en seed_client.cpp | ADR-022 |
| DEBT-INFRA-001 | Migrar box Vagrant a Debian Trixie | P2 |
| FEAT-ROTATION-1 | provision.sh rotate-all + SEED_ROTATION_DAYS | P2 |
| DOCS-APPARMOR | 6 perfiles AppArmor por componente | P2 |

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
FIX-C D8-pre inverso:                 ████████████████████ 100% ✅  DAY 111
FIX-D MAX_PLUGIN_PAYLOAD_SIZE:        ████████████████████ 100% ✅  DAY 111
TEST-INTEG-4c (gate PHASE 2c):        ████████████████████ 100% ✅  DAY 111
ADR-023 PHASE 2d (ml-detector):       ████████████████████ 100% ✅  DAY 111
ADR-029 (g_plugin_loader async-safe): ████████████████████ 100% ✅  DAY 111
PluginMode + mode field:              ████████████████████ 100% ✅  DAY 110
D8-pre coherence check:               ████████████████████ 100% ✅  DAY 110
Paper v13:                            ████████████████████ 100% ✅  DAY 110
ADR-028 (RAG Ingestion Trust Model):  ████████████████████ 100% ✅  DAY 109
arXiv PUBLICADO (2604.04952):         ████████████████████ 100% ✅  DAY 111 🎉
ADR-023 PHASE 2e (rag-security):      ████████████████████ 100% ✅  DAY 112
TEST-INTEG-4e (gate PHASE 2e):        ████████████████████ 100% ✅  DAY 112
ADR-030 (AppArmor-Hardened variant):  ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BACKLOG post-PHASE 3
ADR-031 (seL4/Genode research):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  BACKLOG post-ADR-030
arXiv Replace v13:                    ░░░░░░░░░░░░░░░░░░░░   0% ⏳  pendiente Consejo Q3-112
ADR-025 impl. (plugin signing):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post-PHASE 2
ADR-024 impl. (Noise IK):             ░░░░░░░░░░░░░░░░░░░░   0% ⏳  FASE 3
BARE-METAL stress test:               ░░░░░░░░░░░░░░░░░░░░   0% 🔴  bloqueado por hardware
DEBT-FD-001 (JSON thresholds):        ████░░░░░░░░░░░░░░░░  20% 🟡
TEST-PROVISION-1 (CI gate):           ░░░░░░░░░░░░░░░░░░░░   0% ⏳  post PHASE 2
```

---

## 🔑 Decisiones de diseño consolidadas

| Decisión | Resolución | DAY |
|---|---|---|
| D8-pre bidireccional | READONLY+payload→terminate + NORMAL+nullptr→terminate | 111 |
| MAX_PLUGIN_PAYLOAD_SIZE | 64KB hard limit, std::terminate() si excedido | 111 |
| PHASE 2d contrato | payload=evento serializado, NORMAL, post-inferencia | 111 |
| ADR-029 g_plugin_loader | global exclusivo rag-security, async-signal-safe | 111 |
| ADR-030 AppArmor-Hardened | variante producción Linux+AppArmor, Vagrant-compatible, ARM64/x86 | 112 |
| ADR-031 seL4/Genode | investigación pura, spike GO/NO-GO obligatorio, XDP inviable en guest | 112 |
| PHASE 2e invoke_all modo | READONLY — guardián semántico, result_code ignorado | 111 |
| PluginMode field | flag explícito uint8_t, D8-pre enforce | 110 |
| Integration Philosophy | 4 argumentos: latencia, superficie, SPOF, footprint | 110 |
| ADR-028 timing | diferir hasta primer write-capable plugin | 109 |

---

### Notas del Consejo de Sabios

> DAY 112 — PHASE 2e completa + ADR-030/031 incorporados:
> "PHASE 2e: rag-security integrado con ADR-029 D1-D5. TEST-INTEG-4e 3/3 PASSED.
> PHASE 2 Multi-Layer Plugin Architecture COMPLETA (5/5 componentes, 4a+4b+4c+4e).
> ADR-030 (AppArmor-Hardened) y ADR-031 (seL4/Genode) aprobados por Consejo 5/5
> unanimidad DAY 109. Incorporados a BACKLOG como deuda post-PHASE 3.
> Commit: 10d678ed. Branch: feature/plugin-crypto."
> — Claude (Anthropic) · DAY 112

> DAY 111 — FIX-C/D + PHASE 2d + ADR-029:
> "Consejo no convocado. Implementación directa de mandatos DAY 110.
> FIX-C y FIX-D (ChatGPT5 obligatorios) implementados y testados.
> TEST-INTEG-4c: 3/3 PASSED. PHASE 2d ml-detector compilación limpia 6/6 RUNNING.
> ADR-029 escrito: formaliza g_plugin_loader + async-signal-safe antes de PHASE 2e.
> Hito del día: arXiv:2604.04952 [cs.CR] anunciado públicamente."
> — Claude (Anthropic) · Implementación DAY 111

---

*Última actualización: DAY 112 — 9 Apr 2026*
*Branch: feature/plugin-crypto*
*Tests: 25/25 + TEST-INTEG-4a 3/3 + TEST-INTEG-4b PASSED + TEST-INTEG-4c 3/3 PASSED + TEST-INTEG-4e 3/3 PASSED*
*Paper: Draft v13 ✅ · arXiv: 2604.04952 PUBLICADO ✅*
*Pipeline: 6/6 RUNNING ✅*
*PHASE 2: ✅ COMPLETA (2a+2b+2c+2d+2e) · ADR-030/031: BACKLOG ⏳*
## DEBT-TOOLS-001 — Synthetic injectors: integrar plugin-loader (ADR-025)

**Prioridad:** P3 (no bloqueante)
**Origen:** DAY 113 — observacion durante implementacion ADR-025
**Ficheros afectados:**
- `tools/synthetic_sniffer_injector.cpp`
- `tools/synthetic_ml_output_injector.cpp`
- `tools/generate_synthetic_events.cpp`

**Descripcion:**
Los scripts de stress actuan como sustitutos de los componentes reales para pruebas
de carga. Actualmente usan ZeroMQ + crypto-transport pero NO integran plugin-loader.
Los componentes reales ahora cargan plugins con verificacion Ed25519 (ADR-025).
Para que los synthetic injectors sean representativos del comportamiento real,
deben tambien instanciar PluginLoader y cargar plugins firmados.

**Condicion de activacion:** Post-ADR-025 estable en main. Antes de stress tests
formales de PHASE 3.

**Dependencias:** ADR-025 (Ed25519), make sign-plugins (firma previa al test)

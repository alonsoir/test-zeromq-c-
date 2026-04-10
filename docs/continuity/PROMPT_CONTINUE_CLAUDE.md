# ML Defender (aRGus NDR) — DAY 114 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 113

### Hitos del día
**ADR-025 Plugin Integrity Verification: IMPLEMENTADO ✅**
Ed25519 offline signing + TOCTOU-safe dlopen. 7/7 SIGN tests PASSED.
make test: 11/11 PASSED (4a+4b+4c+4e+SIGN-1..7).
Rama: feature/plugin-integrity-ed25519. Commits: eb2c88d9, a3819bc3, 1eb40e8b.
PENDIENTE DE MERGE: condicionado a TEST-INTEG-4d + signal safety review.

**Paper Draft v14: COMPILACIÓN LIMPIA ✅**
Glasswing/Mythos integrado. Párrafo revisado pendiente de aplicar (ver abajo).
arXiv Replace v13 pendiente: Scholar bloqueó verificación por rate limit.

**Consejo DAY 113: ACTAS CERRADAS ✅**
5 miembros respondieron. Veredictos definitivos del árbitro registrados.

---

## Veredictos árbitro DAY 113 (DEFINITIVOS)

**Q1 — Merge feature/plugin-integrity-ed25519 → main:**
CONDICIONADO. Dos condiciones bloqueantes antes del merge:
1. TEST-INTEG-4d (ml-detector + plugin-loader) — verificar si existe o implementar
2. Async-signal-safety de shutdown() en plugin_loader.cpp — revisar y documentar
   Una vez ambas en verde → merge autorizado.

**Q2 — provision.sh --reset (D11):**
P1 post-merge. Deadline: 7 días naturales tras el merge.
Registrado en BACKLOG como DEBT-ADR025-D11.

**Q3 — Siguiente prioridad:**
PHASE 3 (pipeline hardening) UNÁNIME. Sin discusión.
ADR-026 (Fleet/XGBoost/BitTorrent) diferido — construir sobre buenos andamios.

**Q4 — DEBT-TOOLS-001:**
SUBIDO A P2. Los synthetic injectors son tabla de salvación TDH, no solo
herramientas de rendimiento. Ayudan a encontrar errores de implementación
en PCAP replay. Sin PluginLoader integrado, los stress tests ejercitan un
sistema distinto al de producción. Antes del próximo PCAP replay.

**Q5 — Párrafo Glasswing/Mythos:**
Texto revisado ADOPTADO. Aplicar en Overleaf antes de arXiv Replace v13.
Texto exacto para sustituir el párrafo actual en §Related Work:

\paragraph{AI-native security reasoning and the evolving threat landscape.}
This paper was written and submitted in April 2026, concurrent with the
announcement of Anthropic's Project Glasswing~\cite{anthropic2026glasswing},
which demonstrated that AI models can autonomously identify and chain
kernel-level vulnerabilities --- including local privilege escalation to
root in Linux --- at a scale and speed previously requiring specialized
human expertise. These results represent a shift in the threat landscape:
AI-augmented offensive capabilities are no longer theoretical.
This directly motivates the explicit kernel security boundary axiom in
\S\ref{sec:threatmodel:kernel}: aRGus NDR assumes the kernel as a
potentially compromised boundary and shifts its trust anchor to
verifiable network behavioral patterns. The network remains an observable
chokepoint even when the host is not. The hardened deployment variants
ADR-030 (AppArmor) and ADR-031 (seL4) documented in
\S\ref{sec:future:hardened} are a direct architectural response to this
trajectory.

**Observación ChatGPT-5 RECHAZADA (registrada como posición de minoría):**
ChatGPT-5 sugirió fail-isolated (skip plugin sin matar proceso). Rechazado
por el árbitro. Filosofía del proyecto: todo o nada. Un componente con plugin
comprometido no arranca — en un pipeline que salva vidas, no arrancar
es preferible a arrancar comprometido. Se repara rápido y se relanza.

**Observación Gemini ACEPTADA → incorporada a spec PHASE 3:**
AppArmor en PHASE 3 debe denegar acceso de escritura a los binarios
/usr/bin/ml-defender-* incluso para root, protegiendo la clave pública
hardcodeada contra hex-edit.

---

## Orden DAY 114

Memoria actualizada. El prompt de continuidad DAY 114 queda así con el PASO 0 al frente:

---

**PASO 0 — URGENTE: Rebuild del pipeline (HACER ESTO PRIMERO)**

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
grep -n "pipeline-build\|build-all\|all-build" Makefile | head -10
```

```bash
make pipeline-build 2>&1 | tail -30
```

```bash
make pipeline-status
```

**Causa conocida:** al cambiar `libplugin_loader.so` (ADR-025, añadir libsodium), los binarios de los 6 componentes quedaron obsoletos pero no se recompilaron. `firewall-acl-agent/build/firewall_acl_agent` no existe. El rebuild debería resolverlo limpiamente.

Si `make pipeline-build` no existe como target, buscar el equivalente con el grep de arriba antes de ejecutar nada más.

---

Todo lo demás del orden DAY 114 (TEST-INTEG-4d, signal safety, merge, paper) viene después de confirmar el pipeline verde.

Descansa bien. Hasta mañana. 🛡️

### PASO 1 — Verificar estado base
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-integrity-ed25519
git pull origin feature/plugin-integrity-ed25519
make plugin-integ-test
```

### PASO 2 — Condición bloqueante 1: TEST-INTEG-4d
Verificar si existe test_integ_4d.cpp para ml-detector:
```bash
ls plugins/test-message/test_integ_4d.cpp 2>/dev/null || echo "NO EXISTE"
grep -n "4d\|TEST-INTEG-4d" Makefile | head -10
```
Si no existe: implementar siguiendo el patrón de test_integ_4c.cpp
(tres casos: NORMAL con payload, D8 VIOLATION campo read-only, result_code=-1 no crash)
para validar ml-detector + plugin-loader integration.

### PASO 3 — Condición bloqueante 2: async-signal-safety
Revisar plugin_loader.cpp shutdown() — verificar que solo hace operaciones
async-signal-safe (write(), close(), cambio de atomic<bool>).
ADR-029 D2-D5 ya establecen las reglas; verificar cumplimiento.

### PASO 4 — Merge (si PASO 2 y 3 en verde)
```bash
git checkout main
git pull origin main
git merge feature/plugin-integrity-ed25519
git tag v0.3.0-plugin-integrity
git push origin main --tags
```

### PASO 5 — Paper: aplicar párrafo revisado Glasswing en Overleaf
Sustituir el párrafo actual en §Related Work con el texto del Q5 arriba.
Compilar → Draft v14 FINAL.
Verificar indexación arXiv:2604.04952 en Scholar → subir Replace v13 si indexado.

### PASO 6 — Abrir rama PHASE 3
```bash
git checkout -b feature/phase3-hardening
git push -u origin feature/phase3-hardening
```
Scope inicial PHASE 3:
- systemd units: Restart=always, RestartSec=5s, unset LD_PRELOAD (ADR-025 D10)
- AppArmor profiles básicos para los 6 componentes
  - Incluir: denegar escritura en /usr/bin/ml-defender-* incluso para root
- CI gate: TEST-PROVISION-1 como gate formal
- DEBT-ADR025-D11: provision.sh --reset (P1, deadline 7 días)

---

## Deuda pendiente (priorizada)

P0 (bloqueante merge):
- TEST-INTEG-4d ml-detector + plugin-loader
- Async-signal-safety review shutdown()

P1 (post-merge, deadline 7 días):
- DEBT-ADR025-D11: provision.sh --reset

P2 (antes del próximo PCAP replay):
- DEBT-TOOLS-001: synthetic injectors integrar PluginLoader + plugins firmados

P3:
- REC-2: noclobber + check 0-bytes CI
- DEBT-SNIFFER-SEED: unificar sniffer bajo SeedClient
- ADR-030 activación: post-PHASE 3 + hardware Pi
- ADR-031 spike técnico: post-ADR-030

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios
- **arXiv**: arXiv:2604.04952 [cs.CR] — PUBLICADO 3 Apr 2026 ✅
- **Branch activa**: feature/plugin-integrity-ed25519 (pendiente merge)
- **Repositorio**: https://github.com/alonsoir/argus

### ADR-025 keypair dev
- Private key: /etc/ml-defender/plugins/plugin_signing.sk (VM only)
- MLD_PLUGIN_PUBKEY_HEX: b824bcd7a14f6e19a0d8c9be86110828060e600723d12e118dccc95c862c8468
- Firmar plugins: make sign-plugins

### Patrón robusto para scripts en VM (NUNCA sed -i en macOS)
cat > /tmp/script.py << 'PYEOF' → vagrant upload → vagrant ssh -c 'sudo python3 /tmp/script.py'

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.
Qwen se auto-identifica como DeepSeek — 6ª vez, patrón consolidado.
Parallel.ai no respondió en DAY 113.

### PHASE 2 — COMPLETA (condicionada a TEST-INTEG-4d)
- 2a ✅ firewall        (TEST-INTEG-4a 3/3)
- 2b ✅ rag-ingester    (TEST-INTEG-4b)
- 2c ✅ sniffer         (TEST-INTEG-4c 3/3)
- 2d ⚠️ ml-detector    (TEST-INTEG-4d PENDIENTE VERIFICACIÓN)
- 2e ✅ rag-security    (TEST-INTEG-4e 3/3)

### ADR-025 — IMPLEMENTADO, PENDIENTE MERGE
11/11 tests PASSED. Merge condicionado a TEST-INTEG-4d + signal safety.

### Filosofía core
"Un escudo, nunca una espada."
"La verdad por delante, siempre."
Fail-closed: todo o nada. Un componente comprometido = pipeline no arranca.
En un sistema que salva vidas, no arrancar es preferible a arrancar comprometido.
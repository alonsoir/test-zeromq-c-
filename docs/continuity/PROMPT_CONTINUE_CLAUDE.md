# ML Defender (aRGus NDR) — DAY 113 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 112

### Hitos del día
**PHASE 2 Multi-Layer Plugin Architecture: COMPLETA ✅**
5/5 componentes integrados. make plugin-integ-test: 4a+4b+4c+4e PASSED.

**ADR-030 + ADR-031 incorporados al repositorio y BACKLOG.**

### Completado DAY 112

**PHASE 2e — rag-security ✅**
- `rag/src/main.cpp` reescrito con ADR-029 D1-D5 completos:
  - D1: `static ml_defender::PluginLoader* g_plugin_loader = nullptr`
  - D2: `signalHandler` async-signal-safe (write()+shutdown()+raise())
  - D3: orden inicialización: loader → asignación → signal handlers
  - D4: invoke_all READONLY post-processCommand, result_code ignorado
  - D5: invoke_all NUNCA desde signal handler
  - double-shutdown guard: `g_plugin_loader = nullptr` tras shutdown()

**TEST-INTEG-4e 3/3 PASSED ✅**
- Caso A: READONLY + evento real → errors=0, result_code ignorado
- Caso B: g_plugin_loader=nullptr → invoke_all no llamado, no crash
- Caso C: simulación signal handler → shutdown limpio, g_plugin_loader=nullptr

**ADR-030 — aRGus-AppArmor-Hardened ✅ (BACKLOG)**
- Variante producción Linux 6.12 LTS + Debian 13 + AppArmor enforcing
- Vagrant-compatible. ARM64 (Raspberry Pi 4/5) + x86-64
- Mitiga confused deputy AppArmor (Hugo Vázquez Caramés)
- Activar post-PHASE 3. Bloqueado: hardware Pi pendiente adquisición
- Aprobado Consejo 5/5 unanimidad DAY 109

**ADR-031 — aRGus-seL4-Genode ✅ (BACKLOG/RESEARCH)**
- Investigación pura: ¿cuánto cuesta seguridad formal en rendimiento real?
- seL4 ~12.000 líneas C verificadas Isabelle/HOL. Guest Linux no privilegiado
- XDP inviable en guest (H1) → fallback libpcap obligatorio
- Overhead estimado 40-60%. Spike técnico 2-3 semanas obligatorio
- QEMU directo (Vagrant incompatible). Raspberry Pi 5 preferida (EL2)
- Activa post-ADR-030 + spike GO
- Aprobado Consejo 5/5 unanimidad DAY 109

**README.md + BACKLOG.md actualizados ✅**

**Commits DAY 112:**
- Commit 1 (10d678ed): PHASE 2e + TEST-INTEG-4e
- Commit 2 (1691db06): BACKLOG + ADR-030/031 + README
- Branch: feature/plugin-crypto

---

## Consejo DAY 112 — Preguntas abiertas para DAY 113

**Q1-112 — PHASE 2e: ¿invoke_all READONLY o NORMAL en rag-security?**
Respondido: READONLY por ADR-029 D4. rag-security es guardián semántico,
no actúa sobre eventos — no hay caso válido para NORMAL.

**Q2-112 — TEST-INTEG-4e Caso C: ¿cómo testear SIGTERM sin fork()?**
Respondido: simulación de lógica del handler sin señal real. Caso C
verifica las postcondiciones (shutdown ejecutado, g_plugin_loader=nullptr)
sin necesidad de fork()/kill(). Resultado: limpio y suficiente.

**Q3-112 — arXiv Replace v13: ¿subir ahora o esperar indexación?**
Decisión: ESPERAR. v1 publicada el 3 de abril. Google Scholar / Semantic
Scholar tardan 1-2 semanas en indexar. No hay urgencia. Subir v13 cuando
v1 esté indexada en Scholar.

**Pendiente DAY 113:**
- Incorporar implicaciones Mythos Preview en el paper (DAY 108 pendiente):
  (1) axioma kernel inseguro como limitación declarada del scope
  (2) aRGus válido contra threat model real hospitales/municipios
  (3) detección de red como capa defensiva incluso con kernel comprometido
  (4) referencia a Glasswing/Mythos como contexto temporal del paper
  Esto desbloquea arXiv Replace v13.

---

## Orden DAY 113

### PASO 1 — Verificar estado
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/plugin-crypto
git pull origin feature/plugin-crypto
make pipeline-status
make plugin-integ-test
```

### PASO 2 — Paper: incorporar implicaciones Mythos Preview
Editar `main.tex` (LaTeX en Overleaf o local):
- §Threat Model: añadir axioma kernel inseguro como limitación declarada
- §Conclusions o §Discussion: aRGus válido dentro de su capa aunque kernel comprometido
- §Related Work o footnote: referencia a Mythos Preview + ADR-030/031 como trabajo futuro
- Gate: Draft v14 producido, compilación limpia en Overleaf

### PASO 3 — arXiv Replace v13 (si Scholar ya indexó v1)
Verificar: https://scholar.google.com/scholar?q=arXiv:2604.04952
Si indexado → subir v13 al panel arXiv.
Si no → documentar como pendiente y continuar.

### PASO 4 — Decisión: ¿abrir PR feature/plugin-crypto → main?
PHASE 2 completa. Candidato natural para merge.
Consejo DAY 113 debe pronunciarse sobre timing del merge.

### PASO 5 (opcional) — Iniciar ADR-025 implementación
Plugin Integrity Verification (Ed25519). Post-PHASE 2, desbloqueado.
Solo si PASO 2 y 3 completos.

---

## Deuda pendiente (no bloqueante)

- Paper Mythos Preview integration (DAY 108) — PASO 2 DAY 113
- arXiv Replace v13 — esperar indexación v1 en Scholar
- PR feature/plugin-crypto → main — candidato post-DAY 113
- ADR-025 impl. (Plugin Integrity Ed25519) — post-PHASE 2 ✅ desbloqueado
- REC-2: noclobber + check 0-bytes CI (P2)
- TEST-PROVISION-1 como gate CI formal
- DEBT-SNIFFER-SEED — unificar sniffer bajo SeedClient
- ADR-030 activación — post-PHASE 3 + hardware Pi
- ADR-031 spike técnico — post-ADR-030

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios
- **arXiv**: arXiv:2604.04952 [cs.CR] — PUBLICADO 3 Apr 2026 ✅
- **Branch activa**: feature/plugin-crypto
- **Repositorio**: https://github.com/alonsoir/argus

### PHASE 2 estado — COMPLETA
- 2a ✅ firewall        (TEST-INTEG-4a 3/3)
- 2b ✅ rag-ingester    (TEST-INTEG-4b)
- 2c ✅ sniffer         (TEST-INTEG-4c 3/3)
- 2d ✅ ml-detector     (post-inferencia)
- 2e ✅ rag-security    (TEST-INTEG-4e 3/3, ADR-029 D1-D5)

### Comandos VM críticos
- Editar ficheros en VM: python3 << 'PYEOF' (nunca sed -i sin -e '' en macOS)
- vagrant ssh -c '...' con comillas simples para CMAKE_FLAGS
- NUNCA > fichero para escribir código — usar python3 heredoc
- CMake: NO_DEFAULT_PATH para libsodium
- CI: .github/workflows/ci.yml

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.
Qwen se auto-identifica como DeepSeek — registrar como Qwen en actas.

### Notas arquitectónicas seL4 (DAY 112)
- XDP requiere acceso directo a descriptores DMA de la NIC física
- Bajo Genode, el datapath pasa por virtio-net → XDP inviable en guest
- eBPF kernel programs: mismo problema
- Sniffer es el único componente que toca kernel space hoy
- Fallback libpcap ya previsto en ADR-031
- Todo el resto del pipeline (ZeroMQ, ONNX, FAISS, plugins): userspace, sin cambios
- El spike técnico es una pregunta acotada: solo el mecanismo de captura
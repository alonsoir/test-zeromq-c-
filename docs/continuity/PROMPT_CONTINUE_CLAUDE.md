# Prompt de Continuidad — DAY 132
*aRGus NDR — arXiv:2604.04952 — 26 Abril 2026*

---

## REGLA EMECAS — OBLIGATORIA ANTES DE CUALQUIER ACCION

```bash
vagrant destroy -f
vagrant up
make bootstrap
make test-all
```

**Esta regla es innegociable hasta que el pipeline este en modo solo lectura
y mantenimiento.** No se toca ningun fichero, no se ejecuta ningun comando
tecnico, no se abre ningun editor hasta que `make test-all` devuelva
`ALL TESTS COMPLETE` en una VM destruida y reconstruida desde cero.

Si falla en cualquier punto: diagnosticar, corregir, repetir desde
`vagrant destroy -f`.

*Bautizada en honor de Emerson (emecas@inspiron), que intento saltarse
Vagrant y sin querer certifico que el protocolo es solido.*

---

## Estado del proyecto al inicio de DAY 132

**Repositorio:** `alonsoir/argus` en GitHub
**Branch activa:** `main`
**Commit:** `84dc3af9` (docs: corregir referencias ADR-029 en BACKLOG)
**Tag activo:** `v0.5.2-hardened`
**Paper:** arXiv:2604.04952 — Draft v16 activo
**Keypair activo (post-rebuild DAY 130):**
`1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`

### Pipeline esperado tras REGLA EMECAS
- 6/6 RUNNING: etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall
- TEST-INTEG-SIGN: 7/7 PASSED (SIGN-8/9/10 son post-FEDER, no implementados aun)
- make test-all: ALL TESTS COMPLETE
- Fallo pre-existente conocido (no regresion): `rag-ingester test_config_parser`
  1/8 — safe_path rechaza `/vagrant/...`, prefijo `/etc/ml-defender/` requerido

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

---

## Trabajo completado DAY 131

### Documentacion actualizada (sin commits tecnicos)

**ADR-025 — Extension D13 documentada:**
- Emergency Patch Protocol: Plugin Unload via Signed Message [POST-FEDER]
- Payload `action="unload"` + `target_plugin` firmado con Ed25519
- Reutiliza canal ZeroMQ + verificacion Ed25519 + `dlclose()` existentes
- Zero new attack surface — semantica extendida sobre infraestructura existente
- Tests SIGN-8/9/10 anadidos a la tabla de tests
- D13 correctamente posicionado dentro de "Decisiones detalladas" (despues D12)
- Origen: sugerencia de founder externo via LinkedIn (DAY 131)

**BACKLOG actualizado:**
- ADR-025-EXT-001 anadido a tabla PHASE 5 (post-FEDER)
- "Plugin unload via mensaje firmado" anadido a Decisiones de diseno consolidadas

**Analisis EventSentinel.ai:**
- No es competidor directo de aRGus
- EventSentinel = observabilidad de hardware (predice fallos de disco/CPU/NIC)
- aRGus = seguridad NDR (detecta ataques en red)
- Precio EventSentinel: $384/nodo/ano → ~19.000€/ano para 50 nodos
- Argumento FEDER: ese coste es solo monitoring, aRGus da seguridad completa gratis
- Enterprise ofrece on-premise pero custom pricing — soberania de datos cuestionable

**LinkedIn — respuesta a founder sobre compiler removal:**
- Founder identifico blind spot: runtime-only updates tienen sus propios riesgos
- Respuesta enviada: reconoce critica + describe D13 como solucion elegante
- D13 responde exactamente al blind spot: rollback sin compilador, sin protocolo nuevo

### Ficheros pendientes de commit (P0 DAY 131 no completado)
- `docs/adr/ADR-025.md` — D13 + tests SIGN-8/9/10 + entrada en Registro
- `docs/adr/ADR-039-build-runtime-separation.md`
- `docs/consejo/CONSEJO-ADR039-DAY130.md`
- `docs/continuity/PROMPT_CONTINUE_CLAUDE.md`
- `.gitignore` (anadir `dist/`)
- `docs/BACKLOG.md` (deudas DAY 130 + ADR-025-EXT-001 + decision plugin unload)

---

## Plan DAY 132

### P0 — Commits pendientes (arrastre de DAY 131)
Commitear todo lo pendiente en un commit atomico de documentacion:
```bash
git add docs/adr/ADR-025.md docs/adr/ADR-039-build-runtime-separation.md \
        docs/consejo/CONSEJO-ADR039-DAY130.md \
        docs/continuity/PROMPT_CONTINUE_CLAUDE.md \
        .gitignore docs/BACKLOG.md
git commit -m "docs: ADR-025 D13 + ADR-039 + BACKLOG DAY 131"
```

### P1 — Paper §5 Draft v17
Actualizar arXiv:2604.04952 con contribuciones metodologicas DAY 124-131:

| Seccion | Contenido |
|---------|-----------|
| §5.1 | Test-Driven Hardening (TDH) — metodologia completa |
| §5.2 | RED->GREEN como gate de merge no negociable |
| §5.3 | Property Testing como validador de fixes de seguridad (hallazgo F17) |
| §5.4 | Taxonomia safe_path: lexically_normal vs weakly_canonical |
| §5.5 | CWE-78: execv() sin shell como barrera fisica |
| §5.6 | Axioma BSR: Build/Runtime Separation como principio estructural |
| §5.7 | Fuzzing como tercera capa de testing (unit -> property -> fuzzing) |
| §5.8 | Dev/Prod parity via symlinks, not conditional logic |

### P2 — ADR-030 Implementacion Variant A (prerequisito FEDER)
Implementar lo aprobado en ADR-039 para Variant A (x86):

1. Anadir `dist/` a `.gitignore`
2. Targets Makefile: `build-production-x86`, `sign-production`, `checksums-production`
3. `check-prod-no-compiler` (via dpkg)
4. `check-prod-checksec` (via checksec)
5. `vagrant/hardened-x86/Vagrantfile`
6. `docs/HARDWARE-REQUIREMENTS.md` (DEBT-PROD-COMPAT-BASELINE-001)

### P3 — ADR-030 Variant B (ARM64, prerequisito FEDER)
- `vagrant/hardened-arm64/Vagrantfile`
- Cross-compilation toolchain: `aarch64-linux-gnu-g++` en VM de dev

---

## BACKLOG-FEDER-001

**Deadline:** 22 septiembre 2026
**Go/no-go tecnico:** 1 agosto 2026
**Contacto:** Andres Caro Lindo (UEx/INCIBE)
**Argumento central:** 19.000€/ano solo para monitoring de hardware (EventSentinel)
vs aRGus NDR completo open-source. Brecha de mercado demostrable con numeros reales.

**Prerequisites pendientes:**
- [ ] ADR-030 Variant A (x86 + AppArmor + eBPF/XDP) estable
- [ ] ADR-030 Variant B (ARM64 + AppArmor + libpcap) estable
- [ ] Demo pcap reproducible en < 10 minutos (`scripts/feder-demo.sh`)
- [ ] Paper §5 Draft v17 con axioma BSR
- [ ] Clarificar con Andres: NDR standalone vs federacion funcional (antes julio 2026)

---

## Deudas abiertas relevantes

| ID | Descripcion | Target |
|----|-------------|--------|
| DEBT-PROD-METRICS-001 | Completar tabla metricas §5 paper | DAY 132-135 |
| DEBT-PROD-COMPAT-BASELINE-001 | HARDWARE-REQUIREMENTS.md | DAY 132 |
| DEBT-BUILD-PIPELINE-001 | Builder VM separada (Opcion A ADR-039) | post-FEDER |
| DEBT-PROD-DEBUG-SYMBOLS-001 | Simbolos debug separados para forense | v1.1 |
| DEBT-NATIVE-LINUX-BOOTSTRAP-001 | Flujo nativo Linux sin Vagrant | post-FEDER |
| DEBT-SEED-CAPABILITIES-001 | CAP_DAC_READ_SEARCH en lugar de sudo | v0.6+ |
| DEBT-CRYPTO-003a | mlock() + explicit_bzero(seed) post-HKDF | feature/crypto-hardening |

## Backlog post-FEDER relevante

| ID | Descripcion |
|----|-------------|
| ADR-025-EXT-001 | Emergency Patch Protocol — implementar D13 (action="unload") |
| DEBT-PENTESTER-LOOP-001 | ACRL: Caldera + ATT&CK + XGBoost warm-start |
| ADR-030 aRGus-seL4 | Branch independiente, nunca a main |

---

## Correcciones de nomenclatura (DAY 130)
- ADR-029 = rag-security plugin integration (existente, no tocar)
- ADR-030 = aRGus-AppArmor-Hardened (variantes produccion x86 + ARM64)
- ADR-031 = aRGus-seL4-Genode (investigacion)
- ADR-039 = Build/Runtime Separation (APROBADO DAY 130)
- Proximos numeros libres: ADR-040+

---

## Reglas permanentes del proyecto

- **REGLA EMECAS:** vagrant destroy -f && vagrant up && make bootstrap && make test-all
- **macOS:** Nunca `sed -i` sin `-e ''`. Scripts con emojis -> `/tmp/script.py`.
- **VM<->macOS:** Solo `scp -F /tmp/vagrant-ssh-config`. Prohibido pipe zsh.
- **vagrant ssh:** Siempre con `-c '...'`.
- **JSON es la ley:** No hardcoded values.
- **Fail-closed:** En caso de duda, rechazar.
- **dist/:** Nunca en git. SHA256SUMS obligatorio.
- **Lógica compleja:** Siempre a `tools/script.sh`, nunca inline en Makefile.
- **Seed ChaCha20:** NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().
- **Seguridad:** Todo fix requiere RED->GREEN + property test + test integracion. Sin excepciones.

---

*DAY 132 — 26 Abril 2026 · main @ 84dc3af9 · v0.5.2-hardened*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
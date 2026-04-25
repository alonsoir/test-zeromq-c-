# Prompt de Continuidad — DAY 131
*aRGus NDR — arXiv:2604.04952 — 25 Abril 2026*

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

## Estado del proyecto al inicio de DAY 131

**Repositorio:** `alonsoir/argus` en GitHub
**Branch activa:** `main`
**Commit:** `84dc3af9` (docs: corregir referencias ADR-029 en BACKLOG)
**Tag activo:** `v0.5.2-hardened`
**Paper:** arXiv:2604.04952 — Draft v16 activo
**Keypair activo (post-rebuild DAY 130):**
`1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`

### Pipeline esperado tras REGLA EMECAS
- 6/6 RUNNING: etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall
- TEST-INTEG-SIGN: 7/7 PASSED
- make test-all: ALL TESTS COMPLETE
- Fallo pre-existente conocido (no regresion): `rag-ingester test_config_parser`
  1/8 — safe_path rechaza `/vagrant/...`, prefijo `/etc/ml-defender/` requerido

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

---

## Trabajo completado DAY 130

### 5 deudas cerradas
| Deuda | Commit |
|-------|--------|
| DEBT-SYSTEMD-AUTOINSTALL-001 | `8e57aad2` |
| DEBT-SAFE-EXEC-NULLBYTE-001 — is_safe_for_exec() 17/17 tests | `c8e293a8` |
| DEBT-GITGUARDIAN-YAML-001 — paths_ignore v2 | `06228a67` |
| DEBT-FUZZING-LIBFUZZER-001 — 2.4M runs, 0 crashes, corpus 67 | `f5994c4a` |
| DEBT-MARKDOWN-HOOK-001 — pre-commit hook | `aab08daa` |

### ADR-039 aprobado (Consejo 8/8)
Build/Runtime Separation for Production Variants
- Opcion B aprobada para FEDER
- Axioma BSR publicable (con "trusted build environment assumption")
- Flags produccion con enmiendas: -fstack-clash-protection, -fno-strict-overflow,
  -Werror=format-security, -fasynchronous-unwind-tables, -Wl,-z,noexecstack
- -march=x86-64 baseline como default (5/8), x86-64-v2 como opt-in (8/8)
- SHA256SUMS obligatorio en dist/
- CHECK-PROD-NO-COMPILER via dpkg (no solo which)
- CHECK-PROD-CHECKSEC gate BLOQUEANTE

### Nuevas deudas abiertas (DAY 130)
| ID | Descripcion | Target |
|----|-------------|--------|
| DEBT-BUILD-PIPELINE-001 | Builder VM separada (Opcion A) | post-FEDER |
| DEBT-PROD-METRICS-001 | Completar tabla metricas §5 paper | DAY 131-135 |
| DEBT-PROD-COMPAT-BASELINE-001 | HARDWARE-REQUIREMENTS.md | DAY 131 |
| DEBT-PROD-DEBUG-SYMBOLS-001 | Simbolos debug separados para forense | v1.1 |
| DEBT-NATIVE-LINUX-BOOTSTRAP-001 | Flujo nativo Linux sin Vagrant | post-FEDER |

---

## Plan DAY 131

### P0 — Commitear ADR-039 + acta + LinkedIn + prompt
Ficheros pendientes de commit:
- `docs/adr/ADR-039-build-runtime-separation.md`
- `docs/consejo/CONSEJO-ADR039-DAY130.md`
- `docs/continuity/PROMPT_CONTINUE_CLAUDE.md`
- `.gitignore` (anadir `dist/`)
- `docs/BACKLOG.md` (anadir nuevas deudas DAY 130)

### P1 — Paper §5 Draft v17
Actualizar arXiv:2604.04952 con contribuciones metodologicas DAY 124-130:

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

### P2 — ADR-030 Implementacion (prerequisito FEDER)
Implementar lo aprobado en ADR-039 para Variant A (x86):

1. Anadir `dist/` a `.gitignore`
2. Targets Makefile: `build-production-x86`, `sign-production`, `checksums-production`
3. `check-prod-no-compiler` (via dpkg)
4. `check-prod-checksec` (via checksec)
5. `vagrant/hardened-x86/Vagrantfile`
6. `docs/HARDWARE-REQUIREMENTS.md`

### P3 — ADR-030 Variant B (ARM64, prerequisito FEDER)
- `vagrant/hardened-arm64/Vagrantfile`
- Cross-compilation toolchain: `aarch64-linux-gnu-g++` en VM de dev

---

## BACKLOG-FEDER-001

**Deadline:** 22 septiembre 2026
**Go/no-go tecnico:** 1 agosto 2026
**Contacto:** Andres Caro Lindo (UEx/INCIBE)

**Prerequisites pendientes:**
- [ ] ADR-030 Variant A (x86 + AppArmor + eBPF/XDP) estable
- [ ] ADR-030 Variant B (ARM64 + AppArmor + libpcap) estable
- [ ] Demo pcap reproducible en < 10 minutos (`scripts/feder-demo.sh`)
- [ ] Paper §5 Draft v17 con axioma BSR

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

---

*DAY 130 — 25 Abril 2026 · main @ 84dc3af9 · v0.5.2-hardened*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
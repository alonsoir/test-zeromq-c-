# Prompt de Continuidad — DAY 131
*aRGus NDR — arXiv:2604.04952 — 25 Abril 2026*

---

## 🔴 REGLA EMECAS o REGLA EMECAS— OBLIGATORIA ANTES DE CUALQUIER ACCIÓN

```bash
vagrant destroy -f
vagrant up
make bootstrap
make test-all
```

**Esta regla es innegociable hasta que el pipeline esté en modo solo lectura y mantenimiento.**
No se toca ningún fichero, no se ejecuta ningún comando técnico, no se abre ningún editor
hasta que `make test-all` devuelva `ALL TESTS COMPLETE` en una VM destruida y reconstruida desde cero.
Si falla en cualquier punto → diagnosticar, corregir, repetir desde `vagrant destroy -f`.

---

## Estado del proyecto al inicio de DAY 131

**Repositorio:** `alonsoir/argus` en GitHub
**Branch activa:** `main`
**Commit:** `aab08daa` (docs: DAY 130 — README + BACKLOG actualizados)
**Tag activo:** `v0.5.2-hardened`
**Paper:** arXiv:2604.04952 — Draft v16 activo
**Keypair activo (post-rebuild DAY 130):** `1f48b75054fe98e8371653607caaf028b3f688bc055782c9c9c6d0e3494dad54`

### Pipeline esperado tras REGLA EMECAS
- 6/6 RUNNING: etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall
- TEST-INTEG-SIGN: 7/7 PASSED
- make test-all: ALL TESTS COMPLETE
- Fallo pre-existente conocido (no regresión): `rag-ingester test_config_parser` 1/8 — safe_path rechaza `/vagrant/...`, prefijo `/etc/ml-defender/` requerido

### Consejo de Sabios (8 modelos)
Claude · Grok · ChatGPT · DeepSeek · Qwen · Gemini · Kimi · Mistral

---

## Trabajo completado DAY 130

### 5 deudas cerradas
| Deuda | Commit |
|-------|--------|
| DEBT-SYSTEMD-AUTOINSTALL-001 — install-systemd-units.sh en Vagrantfile | `8e57aad2` |
| DEBT-SAFE-EXEC-NULLBYTE-001 — is_safe_for_exec() + 17/17 tests GREEN | `c8e293a8` |
| DEBT-GITGUARDIAN-YAML-001 — .gitguardian.yaml paths_ignore v2 | `06228a67` |
| DEBT-FUZZING-LIBFUZZER-001 — 2.4M runs, 0 crashes, corpus 67 ficheros | `f5994c4a` |
| DEBT-MARKDOWN-HOOK-001 — pre-commit hook [word](http://) en .cpp/.hpp | `aab08daa` |

### Nuevas reglas permanentes
- **REGLA EMECAS (DAY 130):** `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
- **is_safe_for_exec() es un contrato de seguridad**, no una optimización
- **libFuzzer baseline certificado:** 2.4M runs sin crash = superficie validada

### Hallazgos técnicos
- Colaborador externo (emecas@inspiron) intentó `make bootstrap` en Linux nativo → fallo `llama.h`. DEBT-NATIVE-LINUX-BOOTSTRAP-001 abierta (backlog post-FEDER).
- `.gitguardian.yaml` estaba corrupto con dos entradas fusionadas en una línea.
- `fuzz_validate_chain` binario se coló en un commit — añadido a `.gitignore`.

---

## Plan DAY 131

### Prioridad P0 — Paper §5 Draft v17
El paper arXiv:2604.04952 necesita §5 actualizado con las contribuciones metodológicas DAY 124-130:

| Sección | Contenido |
|---------|-----------|
| §5.1 | Test-Driven Hardening (TDH) — metodología completa |
| §5.2 | RED→GREEN como gate de merge no negociable |
| §5.3 | Property Testing como validador de fixes de seguridad (hallazgo F17) |
| §5.4 | Taxonomía safe_path: lexically_normal vs weakly_canonical — distinción no documentada en literatura C++20 |
| §5.5 | CWE-78: execv() sin shell como barrera física, no promesa |
| §5.6 | Fuzzing como tercera capa de testing (unit → property → fuzzing) |
| §5.7 | Dev/Prod parity via symlinks, not conditional logic |

### Prioridad P1 — ADR-029 Variant A (prerequisito FEDER)
- x86 + AppArmor + eBPF/XDP
- Vagrantfile separado: `vagrant/hardened-x86/`
- Prerequisito para BACKLOG-FEDER-001 (deadline 22 septiembre 2026, go/no-go 1 agosto 2026)

### Prioridad P1 — ADR-029 Variant B
- ARM64 + AppArmor + libpcap (Raspberry Pi 4/5)
- Vagrantfile separado: `vagrant/hardened-arm64/`

---

## Deuda técnica abierta relevante

| ID | Prioridad | Target |
|----|-----------|--------|
| DEBT-SEED-CAPABILITIES-001 | ⏳ Baja | v0.6+ |
| DEBT-SAFE-PATH-RESOLVE-MODEL-001 | ⏳ | feature/adr038-acrl |
| DEBT-NATIVE-LINUX-BOOTSTRAP-001 | ⏳ | post-FEDER |
| DEBT-CRYPTO-003a (mlock+bzero) | ⏳ | feature/crypto-hardening |

---

## BACKLOG-FEDER-001

**Deadline:** 22 septiembre 2026
**Go/no-go técnico:** 1 agosto 2026
**Contacto:** Andrés Caro Lindo (UEx/INCIBE)

**Prerequisites pendientes:**
- [ ] ADR-029 Variant A (x86 + AppArmor + eBPF/XDP) estable
- [ ] ADR-029 Variant B (ARM64 + AppArmor + libpcap) estable
- [ ] Demo pcap reproducible en < 10 minutos (`scripts/feder-demo.sh`)

---

## Reglas permanentes del proyecto

- **macOS:** Nunca `sed -i` sin `-e ''`. Usar `python3 << 'PYEOF'` o scripts en `/tmp/`.
- **Scripts largos con emojis:** Escribir a `/tmp/script.py` y ejecutar con `python3 /tmp/script.py`. Los emojis se corrompen en heredocs del terminal.
- **VM↔macOS:** Solo `scp -F /tmp/vagrant-ssh-config` o `vagrant scp`. Prohibido `vagrant ssh -c "cat ..." > fichero`.
- **vagrant ssh:** Siempre con `-c '...'`.
- **JSON es la ley:** No hardcoded values.
- **Fail-closed:** En caso de duda, rechazar.

---

*DAY 130 — 25 Abril 2026 · main @ aab08daa · v0.5.2-hardened*
*"Via Appia Quality — Un escudo que aprende de su propia sombra."*
*REGLA EMECAS en honor de mi amigo Emerson que me recordó que hay que ser más cuidadoso"
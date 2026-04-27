# Prompt de Continuidad — DAY 134
*aRGus NDR · arXiv:2604.04952 · 28 Abril 2026*

---

## Estado al cierre de DAY 133

**Branch activa:** `feature/adr030-variant-a`
**Último commit:** post-Consejo DAY 133 (cap_bpf + Falco 10 reglas + §6.8 fuzzing reformulado)
**REGLA EMECAS:** 6/6 RUNNING · TEST-INTEG-SIGN 7/7 PASSED · ALL TESTS COMPLETE
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
*(Regenera en cada `vagrant destroy && vagrant up`)*

---

## Lo que se completó en DAY 133

### REGLA EMECAS — verde desde el inicio
`vagrant destroy -f && vagrant up && make bootstrap && make test-all` — 6/6 RUNNING.
Fallo pre-existente conocido (no regresión): `rag-ingester test_config_parser` 1/8.

### Paper Draft v18 (commit `c6e0c9f1`)
- §6.12: Tabla BSR con métricas reales medidas en VMs recién provisionadas:
   - Dev VM: 719 pkgs, 5.9 GB, compiladores presentes
   - Hardened VM: 304 pkgs, 1.3 GB, NONE (check-prod-no-compiler: OK)
   - Reducción: 58% paquetes, 78% disco
   - Minbase target (~100 pkgs / ~0.4 GB) documentado como DEBT-PROD-FS-MINIMIZATION-001

### ADR-030 Variant A — infraestructura completa (commit `c6e0c9f1`)
- 6 perfiles AppArmor enforce en `security/apparmor/`
- Usuario `argus` (system, nologin, no home) + /tmp noexec,nosuid,nodev
- setcap mínimo por componente + tools/prod/ completo
- Makefile: `prod-full-x86`, `check-prod-all` y 10+ targets
- Falco 10 reglas en `vagrant/hardened-x86/scripts/setup-falco.sh`

### Post-Consejo DAY 133 (8/8 unánime)
- `argus.sniffer`: `cap_sys_admin` → `cap_bpf` (Linux ≥5.8)
- `argus.etcd-server`: `cap_net_bind_service` ELIMINADA (2379 > 1024)
- `deploy-hardened.sh`: detección automática kernel + fallback documentado
- `setup-falco.sh`: 10 reglas (añadidas config tamper, model/plugin replace, AA tamper)
- `main.tex`: §6.8 reformulado — eliminada frase "misses nothing within CPU time"
- `acta_consejo_day133.md` generada

### Nuevas deudas abiertas DAY 133
| ID | Descripción | Target |
|----|-------------|--------|
| DEBT-KEY-SEPARATION-001 | Keypair separado pipeline vs plugins | post-FEDER |
| DEBT-KERNEL-COMPAT-001 | Verificar cap_bpf + XDP en kernel 6.1 | DAY 134 (P0) |
| DEBT-PROD-APPARMOR-PORTS-001 | Restringir network a puertos ZeroMQ | post-JSON-estabilización |
| DEBT-PROD-FALCO-RULES-EXTENDED-001 | ptrace, DNS tunneling, /dev/mem | DAY 135 |

---

## P0 para DAY 134 — Pipeline end-to-end en hardened VM

### Paso 1: Provisionar la hardened VM
```bash
make hardened-up
make hardened-provision-all
# → setup-filesystem.sh (usuario argus, /tmp noexec)
# → setup-apparmor.sh (6 perfiles en complain mode primero)
# → setup-falco.sh (10 reglas, prioridad WARNING durante tuning)
```

### Paso 2: Compilar y desplegar
```bash
make prod-full-x86
# → prod-build-x86 (compila en dev VM con -O3 -march=native -DNDEBUG -flto)
# → prod-collect-libs (runtime-only libs)
# → prod-sign (Ed25519 sobre binarios + plugins)
# → prod-checksums (SHA256SUMS)
# → prod-deploy-x86 (instala en /opt/argus/, setcap, sin SUID)
```

### Paso 3: Verificar DEBT-KERNEL-COMPAT-001
```bash
# En la hardened VM:
vagrant --cwd vagrant/hardened-x86 ssh -c 'uname -r'
vagrant --cwd vagrant/hardened-x86 ssh -c 'getcap /opt/argus/bin/sniffer'
# Esperado: /opt/argus/bin/sniffer cap_net_admin,cap_net_raw,cap_bpf,cap_ipc_lock+eip
# Si falla XDP con cap_bpf → documentar DEBT-KERNEL-COMPAT-001 y revertir a cap_sys_admin
```

### Paso 4: Estrategia de maduración AppArmor+Falco
```bash
# Fase 1 — AppArmor complain mode (observar logs 30 min)
vagrant --cwd vagrant/hardened-x86 ssh -c 'sudo aa-status'
vagrant --cwd vagrant/hardened-x86 ssh -c 'sudo journalctl -u falco -f'
# Ajustar perfiles según denegaciones reales con: sudo aa-logprof
# Especial atención: ZeroMQ puede abrir sockets temporales o /dev/shm (Gemini)

# Fase 2 — Pasar a enforce cuando 30 min sin FP
vagrant --cwd vagrant/hardened-x86 ssh -c 'sudo aa-enforce /etc/apparmor.d/argus.*'
```

### Paso 5: Gates de seguridad
```bash
make check-prod-all
# → check-prod-no-compiler  (dpkg + PATH, dos capas)
# → check-prod-apparmor     (6 perfiles en enforce mode)
# → check-prod-capabilities (setcap correcto en sniffer y firewall)
# → check-prod-permissions  (ownership y modos de /opt/argus/, etc.)
# → check-prod-falco        (servicio activo + reglas cargadas)
```

---

## P1 para DAY 134 — Tabla fuzzing §6.8

**DEBT-PAPER-FUZZING-METRICS-001** (pre-arXiv)

La reformulación de la frase está cerrada. Pendiente: tabla con datos reales.

```bash
# Recuperar de DAY 130:
make fuzz-all  # en dev VM
# Objetivo: completar tabla con 3 targets:
# | Target              | Runs  | Crashes | Corpus | Time |
# | validate_chain_name | 2.4M  | 0       | 67     | 30s  |
# | validate_filepath   | ?     | 0       | ?      | ?    |
# | safe_exec           | ?     | 0       | ?      | ?    |
```

Una vez completa la tabla → actualizar `main.tex` → push → arXiv replace v15 → v18.

---

## P2 para DAY 134 — DEBT-PROD-APT-SOURCES-INTEGRITY-001

Si queda tiempo después de P0 y P1:
- SHA-256 de `sources.list` en imagen
- Check en boot: si cambia → fail-closed
- AppArmor deny `/etc/apt/**` w en todos los perfiles
- Falco regla `argus_apt_sources_modified`

---

## Reglas permanentes a recordar

- **REGLA EMECAS:** `vagrant destroy -f && vagrant up && make bootstrap && make test-all` al inicio
- **macOS sed:** nunca `sed -i` sin `-e ''`; usar `python3 << 'PYEOF'` para ediciones en VM
- **Vagrant ssh:** `vagrant ssh -c '...'` (con -c y comillas simples)
- **Makefile:** espacios no tabs; nunca heredoc con `vagrant ssh -c` (quoting issues)
- **cap_bpf:** Linux ≥5.8. Debian bookworm kernel 6.1 lo soporta. DEBT-KERNEL-COMPAT-001 si falla.
- **AppArmor maduración:** complain → 30 min sin FP → enforce. Nunca enforce sin baseline.
- **Falco + AppArmor en paralelo:** ajustar ambos a la vez, se retroalimentan (Consejo).
- **ZeroMQ y AppArmor:** puede abrir sockets temporales o /dev/shm — observar en complain mode.
- **"JSON es la ley":** puertos ZeroMQ no hardcodeados en perfiles AA (DEBT-PROD-APPARMOR-PORTS-001).

---

## Ficheros modificados en DAY 133 (todos en `feature/adr030-variant-a`)

```
security/apparmor/argus.sniffer          # cap_bpf, deny explícitos
security/apparmor/argus.etcd-server      # cap_net_bind_service eliminada
security/apparmor/argus.ml-detector      # sin caps, no-root
security/apparmor/argus.firewall-acl-agent # cap_net_admin, execv
security/apparmor/argus.rag-ingester     # sin caps, no-root
security/apparmor/argus.rag-security     # sin caps, TinyLlama local
tools/prod/build-x86.sh                  # -O3 -march=native -DNDEBUG -flto
tools/prod/collect-libs.sh               # runtime-only libs
tools/prod/sign-binaries.sh              # Ed25519 (reutiliza ADR-025)
tools/prod/deploy-hardened.sh            # setcap cap_bpf + detección kernel
tools/prod/check-permissions.sh          # audit filesystem
vagrant/hardened-x86/scripts/setup-filesystem.sh   # usuario argus, /tmp noexec
vagrant/hardened-x86/scripts/setup-apparmor.sh     # 6 perfiles enforce
vagrant/hardened-x86/scripts/setup-falco.sh        # 10 reglas modern_ebpf
Makefile                                 # prod-* + check-prod-* + hardened-*
docs/latex/main.tex                      # Draft v18 (§6.12 BSR + §6.8 fuzzing)
docs/argus_ndr_v18.pdf                   # compilado Overleaf, 42 páginas
docs/acta_consejo_day133.md              # acta completa DAY 133
docs/BACKLOG.md                          # DAY 133 cerrados + nuevas deudas
README.md                                # badges Falco+BSR, sección hardened
```

---

## Prompt para iniciar DAY 134

Pegar en Claude al inicio de la sesión:

```
Continuamos aRGus NDR en DAY 134 (28 Abril 2026).

Branch activa: feature/adr030-variant-a
Keypair: b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa (regenera en cada vagrant destroy)

DAY 133 cerrado: ADR-030 Variant A infraestructura completa. 6 perfiles AppArmor enforce
(cap_bpf en sniffer post-Consejo), Falco 10 reglas (modern_ebpf), usuario argus no-root,
/tmp noexec, Makefile prod-full-x86 + check-prod-all. Paper Draft v18 con métricas BSR
reales (58% reducción paquetes, 78% disco) y §6.8 fuzzing reformulado (post-Consejo 8/8).

P0 DAY 134:
1. make hardened-up && make hardened-provision-all
2. make prod-full-x86
3. DEBT-KERNEL-COMPAT-001: verificar cap_bpf funciona con XDP en kernel 6.1
4. Estrategia maduración: AppArmor complain 30 min → enforce. Falco WARNING → CRITICAL.
5. make check-prod-all (5 gates)

P1 DAY 134: DEBT-PAPER-FUZZING-METRICS-001 — tabla §6.8 con datos reales DAY 130
P2 DAY 134: DEBT-PROD-APT-SOURCES-INTEGRITY-001 si queda tiempo

Regla macOS: nunca sed -i sin -e ''; usar python3 << 'PYEOF' para ediciones.
Regla Vagrant: vagrant ssh -c '...' con -c y comillas simples.
```

---

*DAY 133 cerrado — 27 Abril 2026 · c6e0c9f1 + post-Consejo*
*"Piano piano. Via Appia Quality."*
# Prompt de Continuidad — DAY 135
*aRGus NDR · arXiv:2604.04952 · 29 Abril 2026*

---

## Estado al cierre de DAY 134

**Branch activa:** `feature/adr030-variant-a`
**Último commit:** `2e9a5b39` (prod-full-x86 + check-prod-all PASSED)
**REGLA EMECAS dev VM:** `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
*(Regenera en cada `vagrant destroy && vagrant up`)*

---

## Lo que se completó en DAY 134

### Madrugada (05:45–08:00) — ADR-040 + ADR-041
- **ADR-040 ML Plugin Retraining Contract v2** — 7 reglas, 12 deudas, Consejo 8/8 (17 enmiendas)
- **ADR-041 Hardware Acceptance Metrics FEDER** — 3 niveles despliegue, 10 métricas, Consejo 8/8
- BACKLOG.md + README.md actualizados — commit `87680d83`

### Mañana (08:30–14:00) — Pipeline E2E hardened VM
- `hardened-provision-all` verde: filesystem + AppArmor 6/6 enforce + Falco 10 reglas (0.43.1)
- `prod-full-x86` verde: pipeline-build PROFILE=production → sign Ed25519 → deploy
- **`check-prod-all` PASSED — 5/5 gates:**
  - BSR: no compiler (gcc-12-base excluido del regex)
  - AppArmor: 6/6 enforce
  - Capabilities: `cap_bpf` + `cap_net_admin` (post-Consejo DAY 133)
  - Permissions: `root:argus` ownership correcto
  - Falco: 10 reglas aRGus cargadas (grep en argus.yaml, API 0.43)
- **DEBT-KERNEL-COMPAT-001 CERRADO** — `cap_bpf` funciona en kernel 6.1

### Tarde — Paper Draft v18 completo
- **DEBT-PAPER-FUZZING-METRICS-001 CERRADO** — tabla §6.8 con datos reales:
  - `validate_chain_name`: 2.4M runs, 0 crashes, corpus 67, ~80K exec/s
  - `safe_exec`: 2.6M runs, 0 crashes, corpus 37, 42K exec/s
  - `validate_filepath`: 282K runs, 0 crashes, corpus 111, 4.6K exec/s
- Draft v18 versión actualizada en main.tex — 42 páginas

### Síntesis Consejo de Sabios (8/8 — post-sesión)
7 decisiones vinculantes para DAY 135 (ver sección completa en BACKLOG.md):
- D1: `hardened-full` fail-fast + destroy / `hardened-redeploy` sin destroy
- D2: Seeds NO en EMECAS — WARNs → INFO — `prod-deploy-seeds` explícito
- D3: Gates `check-prod-all` siempre completos, nunca cacheados
- D4: `dist/vendor/CHECKSUMS` committeado, .deb gitignored, `make vendor-download` + SHA-256
- D5: `confidence_score`: inspección estática + test ZeroMQ + variabilidad
- D6 (Kimi, adoptado): DAY 135 arranca con `make hardened-full` desde VM destruida
- D7 (Mistral, adoptado): DEBT-PROD-APT-SOURCES-INTEGRITY-001 antes de nuevos avances

---

## Fixes técnicos permanentes de DAY 134 (aplicar si se revierten)

```
Makefile:
  - vagrant --cwd → cd $(HARDENED_X86_DIR) && vagrant  (11 targets)
  - pipeline-build añade firewall-build
  - check-prod-capabilities: cap_bpf en lugar de cap_sys_admin
  - check-prod-no-compiler: [[:space:]] tras nombre (excluye gcc-12-base)
  - check-prod-falco: grep en argus.yaml (falco --list deprecated 0.43)
  - prod-sign: sudo bash
  - check-prod-permissions: sudo bash
  - getcap → /usr/sbin/getcap

security/apparmor/*.6: #include <tunables/global> al inicio

vagrant/hardened-x86/scripts/setup-falco.sh:
  - Instalación offline via .deb en /vagrant
  - Macros inline: open_write, open_read, spawned_process
  - evt.dir eliminado de spawned_process (redundante en execve, Falco 0.43)

tools/prod/build-x86.sh:
  - Usa make PROFILE=production pipeline-build (no reinventa el build)
  - Recolecta de build-production/ no build/

tools/prod/sign-binaries.sh:
  - openssl pkeyutl -sign -rawin (PEM canónico)
  - find con ! -name "*.sig" (no [[ != *.sig ]] que no funciona en bash)

tools/prod/deploy-hardened.sh:
  - /opt/argus/ root:argus, /opt/argus/lib root:argus, /opt/argus/plugins root:argus

tools/prod/collect-libs.sh:
  - ldconfig pipe añade || true

Vagrantfile:
  - Descarga automática falco .deb en all-dependencies (antes del DEPENDENCIES_EOF)
```

---

## P0 para DAY 135 — EMECAS sagrado hardened VM

**Decisión Consejo D6 (Kimi, adoptado):** El primer acto es `make hardened-full` desde VM destruida. Cualquier fallo es bloqueante para el merge de `feature/adr030-variant-a` a `main`.

```bash
# Paso 0: Implementar make hardened-full (si no existe)
# Ver arquitectura: fail-fast + destroy + provision + build + deploy + check

# Paso 1: EMECAS sagrado
make hardened-full
# Debe completar en <45 min con check-prod-all PASSED

# Paso 2: Segunda ejecución (sin destroy — hardened-redeploy)
make hardened-redeploy
# Debe ser más rápida (~5-10 min) y también PASSED
```

Target `hardened-full` a implementar:
```makefile
hardened-full:
	@echo "=== EMECAS HARDENED — destroy → provision → build → deploy → check ==="
	$(MAKE) hardened-destroy
	$(MAKE) hardened-up
	$(MAKE) vendor-download        # Falco .deb verificado
	$(MAKE) hardened-provision-all
	$(MAKE) PROFILE=production pipeline-build
	$(MAKE) prod-sign
	$(MAKE) prod-checksums
	$(MAKE) prod-deploy-x86
	$(MAKE) check-prod-all
	@echo "✅ EMECAS HARDENED PASSED"
```

---

## P1 para DAY 135 — DEBT-PROD-APT-SOURCES-INTEGRITY-001

**Severidad:** 🔴 Crítica | Decisión Mistral D7

```bash
# En hardened VM:
# 1. SHA-256 de sources.list al momento del provisioning
# 2. Check en boot: si cambia → fail-closed
# 3. AppArmor deny /etc/apt/** w en todos los perfiles
# 4. Falco regla argus_apt_sources_modified
```

---

## P2 para DAY 135 — DEBT-VENDOR-FALCO-001

```bash
mkdir -p dist/vendor
mv falco_0.43.1_amd64.deb dist/vendor/
sha256sum dist/vendor/falco_0.43.1_amd64.deb > dist/vendor/CHECKSUMS
echo "dist/vendor/*.deb" >> .gitignore
# dist/vendor/CHECKSUMS sí se committea
```

Target `vendor-download`:
```makefile
vendor-download:
	@EXPECTED=$$(grep falco dist/vendor/CHECKSUMS | cut -d' ' -f1); \
	if [ -f dist/vendor/falco_*.deb ]; then \
		ACTUAL=$$(sha256sum dist/vendor/falco_*.deb | cut -d' ' -f1); \
		[ "$$ACTUAL" = "$$EXPECTED" ] && echo "✅ Falco .deb verificado" && exit 0; \
		echo "⚠️ Hash no coincide — re-descargando"; \
	fi; \
	# Descargar desde dev VM si está levantada
	vagrant ssh -c 'cd /vagrant/dist/vendor && apt-get download falco 2>/dev/null || true'
```

---

## P3 para DAY 135 — DEBT-SEEDS-DEPLOY-001

```bash
# Convertir WARNs de seed.bin en INFO en check-prod-permissions
# Crear target prod-deploy-seeds (scp -F vagrant-ssh-config)
```

---

## P4 para DAY 135 — DEBT-CONFIDENCE-SCORE-001

```bash
# Paso 1: Inspección estática
grep -r "confidence" ml-detector/src/ --include="*.cpp" --include="*.hpp" --include="*.proto"

# Paso 2: Si el campo existe, test de integración ZeroMQ
# python3 scripts/check-confidence-score.py --golden-pcap data/golden_flow_001.pcap
# assert confidence_score ∈ [0,1] y varía entre benign/attack
```

---

## P5 para DAY 135 — arXiv replace v15 → v18

```bash
# Paper Draft v18 completo:
# - §6.12 BSR métricas reales (DAY 133)
# - §6.8 tabla fuzzing datos reales (DAY 134)
# - Version v17 → v18 en main.tex (DAY 134)
# Upload en arXiv como "replace" de arXiv:2604.04952
```

---

## Reglas permanentes

- **REGLA EMECAS dev:** `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
- **REGLA EMECAS hardened:** `make hardened-full` (destroy incluido)
- **REGLA EMECAS hardened (dev):** `make hardened-redeploy` (sin destroy, iteración)
- **macOS sed:** nunca `sed -i` sin `-e ''`; usar `python3` para ediciones
- **Vagrant ssh:** `vagrant ssh -c '...'` con -c y comillas simples
- **Hardened VM ssh:** `cd vagrant/hardened-x86 && vagrant ssh -c '...'`
- **Semillas:** NO en EMECAS. `prod-deploy-seeds` para deploy real.
- **Falco .deb:** en `dist/vendor/` (gitignored). Hash en `dist/vendor/CHECKSUMS` (committeado).
- **cap_bpf:** Linux ≥5.8. Debian bookworm kernel 6.1 lo soporta. ✅ Verificado DAY 134.
- **Falco 0.43:** macros open_write/open_read/spawned_process inline. evt.dir eliminado.
- **"JSON es la ley":** puertos ZeroMQ no hardcodeados en perfiles AA.

---

## Ficheros críticos modificados en DAY 134

```
Makefile                                 # 11 fixes vagrant --cwd + checks post-Consejo
Vagrantfile                              # descarga automática falco .deb
security/apparmor/argus.*               # #include <tunables/global> en 6 perfiles
vagrant/hardened-x86/scripts/setup-falco.sh  # offline .deb + macros Falco 0.43
tools/prod/build-x86.sh                 # pipeline-build PROFILE=production
tools/prod/collect-libs.sh              # ldconfig || true
tools/prod/deploy-hardened.sh           # ownership root:argus
tools/prod/sign-binaries.sh             # openssl PEM + find correcto
docs/latex/main.tex                     # Draft v18 — tabla §6.8 + versión
docs/BACKLOG.md                         # DAY 134 completo + Consejo síntesis
README.md                               # DAY 134 completo
docs/counsil/                           # 8 respuestas Consejo ADR-040 + ADR-041 + síntesis
```

---

## Prompt para iniciar DAY 135

Pegar en Claude al inicio de la sesión:

```
Continuamos aRGus NDR en DAY 135 (29 Abril 2026).

Branch activa: feature/adr030-variant-a
Keypair: b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa (regenera en cada vagrant destroy)

DAY 134 cerrado: check-prod-all PASSED (5/5 gates en hardened VM). ADR-040 ML Retraining
Contract (8/8, 17 enmiendas) + ADR-041 FEDER HW Metrics (8/8). Draft v18 completo con
tabla fuzzing §6.8 real. DEBT-KERNEL-COMPAT-001 CERRADO (cap_bpf ok en kernel 6.1).

Consejo síntesis DAY 134 (8/8) — 7 decisiones vinculantes:
D1: hardened-full fail-fast+destroy / hardened-redeploy sin destroy
D2: Seeds NO en EMECAS — WARNs→INFO — prod-deploy-seeds explícito
D3: check-prod-all siempre completo
D4: dist/vendor/CHECKSUMS committeado, .deb gitignored, make vendor-download
D5: confidence_score: inspección estática + test ZeroMQ + variabilidad
D6: DAY 135 arranca con make hardened-full desde VM destruida (gate pre-merge)
D7: DEBT-PROD-APT-SOURCES-INTEGRITY-001 antes de nuevos avances

P0 DAY 135: make hardened-full (implementar + ejecutar desde VM destruida — gate pre-merge)
P1 DAY 135: DEBT-PROD-APT-SOURCES-INTEGRITY-001 (SHA-256 sources.list fail-closed)
P2 DAY 135: DEBT-VENDOR-FALCO-001 (dist/vendor/CHECKSUMS + make vendor-download)
P3 DAY 135: DEBT-SEEDS-DEPLOY-001 (prod-deploy-seeds + WARNs→INFO)
P4 DAY 135: DEBT-CONFIDENCE-SCORE-001 (inspección + test ZeroMQ variabilidad)
P5 DAY 135: arXiv replace v15 → v18

Regla macOS: nunca sed -i sin -e ''; usar python3 para ediciones.
Regla hardened ssh: cd vagrant/hardened-x86 && vagrant ssh -c '...'
Regla vendor: dist/vendor/CHECKSUMS committeado, *.deb gitignored.
Regla seeds: NO en EMECAS. prod-deploy-seeds para deploy real.
```

---

*DAY 134 cerrado — 28 Abril 2026 · Commits 87680d83..2e9a5b39 · check-prod-all PASSED*
*"Piano piano. Via Appia Quality. Feliz cumpleaños."* 🏛️
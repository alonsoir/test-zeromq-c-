# aRGus NDR — DAY 137
# Continuación de DAY 136 — 30 Abril 2026

## Estado del repo
- Branch activa: main (post-merge v0.6.0-hardened-variant-a)
- Último commit: 737ba0d5 (hardened-full-with-seeds target)
- arXiv: 2604.04952 — v18 enviado (Cornell procesando)
- Keypair activo: b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa

## REGLA EMECAS DEV (obligatoria — primera acción del día)
vagrant destroy -f && vagrant up && make bootstrap && make test-all

## REGLA EMECAS HARDENED (gate pre-merge)
make hardened-full

## Lo que se completó en DAY 136

### Tareas técnicas ✅
- EMECAS dev + hardened PASSED
- PASO 0: make hardened-full + prod-deploy-seeds + check-prod-all → PASSED sin WARNs
- PASO 1: argus-apt-integrity.service verificado (ExecStartPre ✅, FailureAction=poweroff ✅)
- PASO 2: docs/KNOWN-DEBTS-v0.6.md creado (6 deudas)
- PASO 3: Merge --no-ff feature/adr030-variant-a → main + tag v0.6.0-hardened-variant-a
- PASO 4: hardened-full-with-seeds añadido al Makefile (FEDER ONLY)
- Consejo 8/8: 3 nuevas deudas identificadas (Jenkins, Vault, compiler warnings)
- BACKLOG.md + README.md actualizados

## Pendientes para DAY 137 — en orden

### PASO 0 — EMECAS completo (siempre primero)
vagrant destroy -f && vagrant up && make bootstrap && make test-all
make hardened-full
make prod-deploy-seeds
make check-prod-all

### PASO 1 — feature/variant-b-libpcap (ADR-029 Variant B)
git checkout -b feature/variant-b-libpcap
cp -r vagrant/hardened-x86 vagrant/hardened-arm64
# Modificar Vagrantfile: sin eBPF headers, con libpcap-dev
# Objetivo: delta XDP/libpcap publicable para paper + FEDER

### PASO 2 — fix/compiler-warnings-cleanup-001
git checkout -b fix/compiler-warnings-cleanup-001
# Prioridad 1: ODR violations internal_trees_inline vs traffic_trees_inline
# Prioridad 2: Protobuf ODR (build-production/proto vs src/protobuf)
# Prioridad 3: signed/unsigned, OpenSSL deprecated, Wreorder

### PASO 3 — DEBT-IRP-NFTABLES-001 (P0 pre-FEDER)
# Implementar /usr/local/bin/argus-network-isolate con nftables drop-all
# Referenciado en argus-apt-integrity.service ExecStartPre

### PASO 4 — DEBT-CRYPTO-MATERIAL-STORAGE-001 (propuesta)
# Prototipo HashiCorp Vault en Vagrantfile
# make vault-init && make vault-deploy-seeds

## Nuevas deudas DAY 136 (Consejo 8/8)
- DEBT-JENKINS-SEED-DISTRIBUTION-001 🔴 pre-FEDER
- DEBT-CRYPTO-MATERIAL-STORAGE-001 🔴 pre-FEDER
- DEBT-COMPILER-WARNINGS-CLEANUP-001 🔴 DAY 137+ rama dedicada

## Decisiones vinculantes Consejo DAY 136

D1: DEBT-IRP-NFTABLES-001 es P0 pre-FEDER
D2: Jenkins para distribución de seeds (mecanismo mínimo viable)
D3: HashiCorp Vault para demo FEDER + TPM 2.0 objetivo final
D4: fix/compiler-warnings-cleanup-001 — rama dedicada
D5: feature/variant-b-libpcap — DAY 137 PASO 1

## Reglas permanentes

- REGLA EMECAS dev: vagrant destroy -f && vagrant up && make bootstrap && make test-all
- REGLA EMECAS hardened: make hardened-full (gate pre-merge, destroy incluido)
- REGLA EMECAS hardened iter: make hardened-redeploy (sin destroy, iteración rápida)
- REGLA apt-integrity: FailureAction=poweroff. Inmediato. Sin gracia. Sin excepciones.
- REGLA vendor: dist/vendor/CHECKSUMS committeado. vendor-download solo verifica.
- REGLA seeds: NO en EMECAS. prod-deploy-seeds explícito.
- REGLA macOS sed: nunca sed -i sin -e ''; usar python3 para ediciones.
- Hardened VM ssh: cd vagrant/hardened-x86 && vagrant ssh -c '...'
- REGLA IRP systemd: nunca ExecStopPre — usar ExecStartPre.

## BACKLOG-FEDER-001
Deadline: 22 septiembre 2026
Go/no-go técnico: 1 agosto 2026
Prerequisites activos:
✅ ADR-026 mergeado (XGBoost)
✅ ADR-030 Variant A — v0.6.0-hardened-variant-a en main
✅ Pipeline E2E hardened verde
⏳ ADR-029 Variant B (libpcap) — DAY 137
⏳ DEBT-IRP-NFTABLES-001 (P0)
⏳ DEBT-CRYPTO-MATERIAL-STORAGE-001 (Vault prototipo)
⏳ scripts/feder-demo.sh reproducible
⏳ ADR-041 métricas validadas en x86 + ARM

## Commits DAY 136 (referencia)
b2a46836 → 23d99b60 → (merge commit) → 737ba0d5
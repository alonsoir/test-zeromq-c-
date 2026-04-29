# aRGus NDR — DAY 136
# Continuación de DAY 135 — 30 Abril 2026

## Estado del repo
- Branch: feature/adr030-variant-a
- Último commit: 7c8e750b
- arXiv: 2604.04952 — v18 enviado DAY 135 (Cornell procesando)
- Keypair activo: b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa

## REGLA EMECAS DEV (obligatoria — primera acción del día)
vagrant destroy -f && vagrant up && make bootstrap && make test-all

## REGLA EMECAS HARDENED (gate pre-merge)
make hardened-full

## Lo que se completó en DAY 135

### Tareas técnicas ✅
- EMECAS dev PASSED
- DEBT-VENDOR-FALCO-001 CERRADO (commit 531b9792)
- DEBT-PROD-APT-SOURCES-INTEGRITY-001 CERRADO (commits 57bbe236, e8709788)
  - FailureAction=poweroff (Voto de Oro Alonso — material radiactivo)
  - ExecStopPre → ExecStartPre corregido (DEBT-IRP-SYSTEMD-FIX-001, Kimi)
  - StartLimitIntervalSec=300 + StartLimitBurst=2
- DEBT-SEEDS-DEPLOY-001 CERRADO (commit bf4c21bf)
  - 6 seeds (400 argus:argus) + plugin_signing.pk (444 root:argus)
  - check-prod-permissions PASSED sin WARNs
- DEBT-CONFIDENCE-SCORE-001 CERRADO (commit 7691f867)
  - BENIGN=0.854557, RANSOMWARE=0.700000 — variabilidad confirmada
- arXiv replace v15→v18 ENVIADO
- BACKLOG.md + README.md actualizados
- ADR-042 IRP PRE-DISEÑO DRAFT v2 creado
  - Consejo 8/8 x2 rondas adversariales completadas
  - APROBADO como arquitectura, NO implementable directamente
  - 5 enmiendas obligatorias identificadas

## Pendientes para DAY 136 — en orden

### PASO 0 — EMECAS completo (siempre primero)
vagrant destroy -f && vagrant up && make bootstrap && make test-all
make hardened-full
make prod-deploy-seeds
make check-prod-all

Si todo verde: procedemos al merge.

### PASO 1 — argus-apt-integrity.service: verificar correcciones
Verificar que setup-apt-integrity.sh tiene la unit correcta:
- ExecStartPre (NO ExecStopPre — bug corregido DAY 135)
- FailureAction=poweroff
- StartLimitIntervalSec=300 + StartLimitBurst=2
- ExecStartPre=/usr/local/bin/argus-network-isolate (pendiente implementar)

### PASO 2 — docs/KNOWN-DEBTS-v0.6.md
Crear antes del merge:
- DEBT-COMPILER-WARNINGS-001
- DEBT-SEEDS-SECURE-TRANSFER-001
- DEBT-SEEDS-LOCAL-GEN-001
- DEBT-SEEDS-BACKUP-001
- DEBT-IRP-NFTABLES-001
- DEBT-IRP-QUEUE-PROCESSOR-001

### PASO 3 — Merge feature/adr030-variant-a → main
git checkout main
git merge --no-ff feature/adr030-variant-a
make test-all   # verificar en main
git tag v0.6.0-hardened-variant-a
git push origin main --tags

### PASO 4 — hardened-full-with-seeds (TEST/FEDER ONLY)
Añadir en Makefile después del merge:
hardened-full-with-seeds:
	@echo "⚠️  TESTING/FEDER ONLY — NO usar en producción"
	$(MAKE) hardened-full
	$(MAKE) prod-deploy-seeds
	$(MAKE) check-prod-all

### PASO 5 — Iniciar feature/variant-b-libpcap (ADR-029 Variant B)
git checkout -b feature/variant-b-libpcap
cp -r vagrant/hardened-x86 vagrant/hardened-arm64
Modificar Vagrantfile: sin eBPF headers, con libpcap-dev
Objetivo: delta XDP/libpcap publicable para paper + FEDER

## Decisiones vinculantes Consejo DAY 135 (pendientes)

D1: Merge --no-ff ← PASO 3
D2: Tag v0.6.0-hardened-variant-a ← PASO 3
D3: argus-apt-integrity.service corregido ← PASO 1
D6: hardened-full-with-seeds TEST/FEDER ONLY ← PASO 4
D7: feature/variant-b-libpcap ← PASO 5

## ADR-042 IRP — estado y próximos pasos

Estado: DRAFT v2 — aprobado como arquitectura, NO como implementación
Enmiendas obligatorias pre-implementación:
  E1: nftables drop-all en argus-network-isolate (ip link insuficiente)
  E2: ExecStopPre → ExecStartPre — CORREGIDO DAY 135 ✅
  E3: Cola irp-queue con límites + procesador systemd
  E4: post-recovery-check bloquea boot si falla
  E5: Advertencia Secure Boot para initramfs

Deudas IRP pendientes:
  DEBT-IRP-NFTABLES-001 (🔴 Alta — post-merge)
  DEBT-IRP-QUEUE-PROCESSOR-001 (🔴 Alta — post-merge)
  DEBT-IRP-A-002 forensic-collect + initramfs (🟡 post-FEDER)
  DEBT-IRP-STANDBY-ATTEST-001 (🟡 post-FEDER)
  DEBT-IRP-RF-FALLBACK-GATE-001 (🟡 post-merge)
  DEBT-IRP-SECUREBOOT-001 (🟡 post-FEDER)
  DEBT-IRP-GDPR-001 (🟡 post-FEDER)

## Reglas permanentes

- REGLA EMECAS dev: vagrant destroy -f && vagrant up && make bootstrap && make test-all
- REGLA EMECAS hardened: make hardened-full (gate pre-merge, destroy incluido)
- REGLA EMECAS hardened iter: make hardened-redeploy (sin destroy, iteración rápida)
- REGLA apt-integrity: FailureAction=poweroff. Inmediato. Sin gracia. Sin excepciones.
  "Un nodo comprometido es material radiactivo. No se reanima."
- REGLA vendor: dist/vendor/CHECKSUMS committeado. vendor-download solo verifica.
- REGLA seeds: NO en EMECAS. prod-deploy-seeds explícito (D2 Consejo DAY 134).
- REGLA macOS sed: nunca sed -i sin -e ''; usar python3 para ediciones.
- Hardened VM ssh: cd vagrant/hardened-x86 && vagrant ssh -c '...'
- REGLA IRP systemd: nunca ExecStopPre — usar ExecStartPre.

## BACKLOG-FEDER-001
Deadline: 22 septiembre 2026
Go/no-go técnico: 1 agosto 2026
Prerequisites activos:
  ✅ ADR-026 mergeado (XGBoost)
  ✅ ADR-030 Variant A infraestructura
  ✅ Pipeline E2E hardened verde
  ⏳ ADR-029 Variant B (libpcap ARM64) — DAY 136
  ⏳ scripts/feder-demo.sh reproducible
  ⏳ ADR-041 métricas validadas en x86 + ARM

## Commits DAY 135 (referencia)
531b9792 → ecce0334 → 4ac23ad6 → 57bbe236 → e8709788
bf4c21bf → 7691f867 → 43bef427 → d4afb03f → 7c8e750b

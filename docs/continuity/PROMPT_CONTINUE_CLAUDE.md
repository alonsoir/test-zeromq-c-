# ML Defender (aRGus NDR) — DAY 117 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.

---

## Estado al cierre de DAY 116

### Hitos completados
- DEBT-ADR025-D11: provision.sh --reset con seed_family compartido ✅
  - TEST-RESET-1/2/3: PASSED
  - Bug crítico: seeds independientes → HKDF MAC fail → resuelto
  - Nueva pubkey dev: c44a4fe2bfe4ee8ad86f840277625e10ca1c97e85671f366c38a38e6bf02d575
  - seed_family compartido: 75deaca96768b5d973a4339faf2325c058969bf93c00c0d21eef703a2ab91360
- TEST-PROVISION-1 7/7 checks ✅ (check #6 permisos + #7 consistencia JSONs)
- AppArmor 6 perfiles complain mode, 0 denials ✅
- Commits: 3c0a214f, e01b5919, efe203bf (feature/phase3-hardening)

### Consejo DAY 116 — Veredictos
- Q1 AppArmor: etcd-server → rag-* → ml-detector → firewall → sniffer (48h mínimo)
- Q2 DEBT-SEED-PERM-001: opción (a) corregir mensaje + test (unanimidad)
- Q3 Próxima fase: AppArmor enforce + DEBTs antes de ADR-026 (unanimidad)
- Q4 seed_family: addendum ADR-021 (árbitro)

---

## PASO 0 — Verificación de estabilidad (SIEMPRE PRIMERO)

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/phase3-hardening
git pull origin feature/phase3-hardening
make pipeline-stop
make pipeline-build 2>&1 | tail -5
vagrant ssh -c "sudo bash /vagrant/etcd-server/config/set-build-profile.sh debug"
make sign-plugins
make test-provision-1
make pipeline-start && make pipeline-status
# Esperar: 6/6 RUNNING
make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"
# Esperar: 12/12 PASSED
```

**Solo si 6/6 RUNNING y 12/12 PASSED se continúa.**

---

## CRITERIO DE CIERRE DE FEATURE

La feature/phase3-hardening NO puede mergearse a main hasta que
todos los items siguientes tengan test verde:

□ DEBT-VAGRANTFILE-001    + vagrant provision OK con apparmor-utils
□ DEBT-SEED-PERM-001      + TEST-PERMS-SEED verde
□ REC-2                   + noclobber funciona + hook rechaza 0-bytes
□ TEST-INVARIANT-SEED     + 6 seeds idénticos post-reset (gate en suite)
□ TEST-PROVISION-1 (7/7)  + todos los echoes muestran 7/7
□ ADR-021 addendum        + commiteado al repo docs/adr/
□ AppArmor enforce 5/6    + TEST-APPARMOR-ENFORCE: 6/6 + 12/12 con enforce
□ AppArmor sniffer        + 48h complain → enforce DAY 118+
□ apparmor-promote.sh     + tools/apparmor-promote.sh funcional
□ apparmor-utils check    + TEST-PROVISION-1 check #8
□ docs/Recovery Contract  + OQ-6 ADR-024 documentado
□ Backup policy .bak.*    + 3 resets → max 2 backups por componente
□ DEBT-RAG-BUILD-001      + rag-security usa build-active symlink

---

## Orden DAY 117

### PASO 1 — DEBT-VAGRANTFILE-001 (5 min)
```bash
grep -n 'apt-get\|apt install\|apparmor' Vagrantfile | head -10
```
Añadir `apparmor-utils` a la línea de instalación de paquetes.
**Test de cierre:** `vagrant provision 2>&1 | grep apparmor-utils`
O en su defecto: `vagrant ssh -c "which aa-complain"` sin instalación manual previa.

### PASO 2 — DEBT-SEED-PERM-001 + TEST-PERMS-SEED
```bash
vagrant ssh -c "grep -n 'chmod 600\|chmod 640\|0600\|0640\|perm' \
  /vagrant/libs/seed-client/src/seed_client.cpp | head -15"
```
Cambiar condición check: `!= 0600` → `!= 0640`.
Cambiar mensaje warning: `chmod 600` → `chmod 640`.
**Test de cierre TEST-PERMS-SEED:**
- seed.bin con permisos 640 → NO warning
- seed.bin con permisos 600 → SÍ warning (mensaje correcto)
- seed.bin con permisos 644 → SÍ warning
  Añadir al suite de tests o a make test-provision-1 como check #8 (apparmor-utils)
  y TEST-PERMS-SEED separado.

### PASO 3 — REC-2: noclobber + 0-bytes
```bash
grep -n 'noclobber\|set -o\|0-byte\|zero.*byte' /vagrant/tools/provision.sh | head -10
grep -rn 'noclobber' /vagrant/Makefile /vagrant/tools/ | head -10
```
Añadir `set -o noclobber` al inicio de provision.sh y scripts que usen redirección.
Añadir check pre-commit o en CI: rechazar ficheros de 0 bytes.
**Test de cierre:** script con `>` no trunca fichero existente → falla con error.

### PASO 4 — TEST-INVARIANT-SEED
Implementar test que verifica post-reset todos los seed.bin son idénticos:
```bash
# El test debe ejecutar --reset y luego comparar:
vagrant ssh -c 'sudo bash -c "
  hashes=""
  for c in etcd-server ml-detector sniffer firewall-acl-agent rag-ingester rag-security; do
    h=\$(od -A n -t x1 /etc/ml-defender/\$c/seed.bin | tr -d \" \\n\")
    hashes=\"\$hashes \$h\"
  done
  unique=\$(echo \$hashes | tr \" \" \"\\n\" | sort -u | wc -l)
  [ \"\$unique\" -eq 1 ] && echo PASSED || echo FAILED
"'
```
Integrar en `make test-reset` o en TEST-PROVISION-1 como check.

### PASO 5 — TEST-PROVISION-1 echoes 5/5 → 7/7
```bash
vagrant ssh -c "grep -n '5/5\|5\/5' /vagrant/Makefile"
```
Actualizar todos los echoes "Check X/5" y "5/5 OK" a "7/7".
**Test de cierre:** `make test-provision-1 2>&1 | grep '5/5'` → vacío.

### PASO 6 — Backup policy .bak.*
En `reset_all_keys()` y `reset_plugin_signing_keypair()`, añadir limpieza:
keep last 2 backups por componente, eliminar el más antiguo si hay 3+.
**Test de cierre:** ejecutar --reset 3 veces → `ls /etc/ml-defender/*.bak.* | wc -l`
≤ 12 (2 backups × 6 componentes).

### PASO 7 — ADR-021 addendum
Escribir el addendum en `docs/adr/ADR-021-*.md` (o inline en el ADR existente).
Contenido: INVARIANTE-SEED-001, regresión vs multi-familia, threat model RAM,
DEBT-CRYPTO-003a como mitigación, ADR-033 como solución definitiva.
**Test de cierre:** `git log docs/adr/` muestra commit del addendum.

### PASO 8 — docs/Recovery Contract (OQ-6 ADR-024)
Documento operacional: rotación de claves con zero downtime.
5 pasos: dual-key T=24h + versioned deployment.yml + secuencia.
Guardar en `docs/operations/key-rotation-contract.md`.
**Test de cierre:** fichero existe y tiene los 5 pasos documentados.

### PASO 9 — DEBT-RAG-BUILD-001
```bash
vagrant ssh -c "grep -n 'build\|RAG_BUILD\|rag.*build' /vagrant/rag/CMakeLists.txt | head -10"
vagrant ssh -c "ls -la /vagrant/rag/build-active 2>/dev/null || echo 'NO SYMLINK'"
```
Crear build-active symlink para rag-security igual que los demás componentes.
Actualizar set-build-profile.sh para incluir rag-security.
**Test de cierre:** `set-build-profile.sh debug` → rag-security: build-active → build-debug ✅

### PASO 10 — apparmor-utils en TEST-PROVISION-1 (check #8)
Añadir al Makefile check #8:
```bash
vagrant ssh -c "which aa-complain aa-enforce aa-logprof > /dev/null 2>&1 \
  && echo OK || echo FAIL"
```
Fallo → mensaje claro: "Ejecuta: sudo apt-get install -y apparmor-utils"

### PASO 11 — tools/apparmor-promote.sh
Script que:
1. Recibe componente como parámetro
2. Ejecuta aa-enforce
3. Reinicia pipeline
4. Monitorea journalctl 5 minutos
5. Si hay denials AppArmor → rollback automático a aa-complain + log
6. Si 0 denials → confirma enforce + log auditado
   **Test de cierre:** promote.sh etcd-server → enforce + 0 denials → estado enforce confirmado.

### PASO 12 — AppArmor enforce secuencial
Usando apparmor-promote.sh:
```bash
bash tools/apparmor-promote.sh etcd-server
# Verificar: 6/6 RUNNING + 0 denials → siguiente
bash tools/apparmor-promote.sh rag-security
bash tools/apparmor-promote.sh rag-ingester
bash tools/apparmor-promote.sh ml-detector
# Verificar: 6/6 RUNNING + 0 denials → siguiente
bash tools/apparmor-promote.sh firewall-acl-agent
# sniffer: NO enforce hasta 48h mínimo en complain (DAY 118+)
```
**Test de cierre TEST-APPARMOR-ENFORCE:**
`make plugin-integ-test` → 12/12 PASSED con 5/6 perfiles en enforce.

---

## Contexto permanente

### ADR-025 keypair dev (post-reset DAY 116)
MLD_PLUGIN_PUBKEY_HEX: c44a4fe2bfe4ee8ad86f840277625e10ca1c97e85671f366c38a38e6bf02d575

### seed_family post-reset DAY 116
Seed compartido (6 componentes): 75deaca96768b5d973a4339faf2325c058969bf93c00c0d21eef703a2ab91360
INVARIANTE-SEED-001: todos los seed.bin DEBEN ser idénticos.

### Regla de oro del pipeline
6/6 RUNNING + 12/12 plugin-integ-test PASSED

### Patrón robusto para scripts en VM

cat > /tmp/script.py << 'PYEOF'
...
PYEOF
vagrant upload /tmp/script.py /tmp/script.py
vagrant ssh -c "sudo python3 /tmp/script.py"

NUNCA sed -i en macOS. NUNCA Python inline con paréntesis en zsh.

### DEBTs no bloqueantes (NO tocar en esta feature)
- DEBT-CRYPTO-003a → feature/crypto-hardening
- DEBT-OPS-001/002 → feature/ops-tooling
- DEBT-TOOLS-001 → feature/adr026-xgboost
- DEBT-SNIFFER-SEED → feature/crypto-hardening
- DEBT-FD-001 → feature/adr026-xgboost
- DEBT-INFRA-001 → feature/bare-metal
- DEBT-CLI-001 → feature/adr032-hsm
- docs/CRYPTO-INVARIANTS.md → feature/crypto-hardening
- ADR-033 TPM → post-PHASE 4

### PHASE 3 estado

systemd units              ✅ DAY 115
DEBT-SIGN-AUTO             ✅ DAY 115
DEBT-HELLO-001             ✅ DAY 115
TEST-PROVISION-1 (5/5)     ✅ DAY 115
DEBT-ADR025-D11 --reset    ✅ DAY 116
TEST-PROVISION-1 (7/7)     ✅ DAY 116
AppArmor complain (6/6)    ✅ DAY 116
DEBT-VAGRANTFILE-001       🔄 DAY 117
DEBT-SEED-PERM-001         🔄 DAY 117
REC-2 noclobber            🔄 DAY 117
TEST-INVARIANT-SEED        🔄 DAY 117
TEST-PROVISION-1 (7/7 echoes) 🔄 DAY 117
Backup policy .bak.*       🔄 DAY 117
ADR-021 addendum           🔄 DAY 117
Recovery Contract          🔄 DAY 117
DEBT-RAG-BUILD-001         🔄 DAY 117
apparmor-utils check #8    🔄 DAY 117
apparmor-promote.sh        🔄 DAY 117
AppArmor enforce (5/6)     🔄 DAY 117
AppArmor enforce sniffer   ⏳ DAY 118+ (48h complain)
────────────────────────────────────────
MERGE A MAIN: cuando 1-19 cerrados + sniffer enforce DAY 118+
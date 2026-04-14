Leídas las 5 respuestas. Resumen de veredictos consolidados:

**Q1 — ALLOWED sniffer:** 4/5 dicen revisar el log completo antes de enforce. Solo Qwen dice proceder directamente confiando en el rollback. **Veredicto: revisar primero.** Acción DAY 118: `sudo journalctl -k | grep -E 'apparmor.*sniffer|ALLOWED.*sniffer' -A3 -B3` antes de ejecutar promote.

**Q2 — noclobber audit:** Unanimidad en hacer audit, pero división entre completo (ChatGPT, Grok) vs limitado a rutas críticas (DeepSeek, Qwen). **Veredicto práctico: audit limitado a ficheros críticos** — `/etc/`, `.sk`, `seed.bin`, `*.sig`. Tiempo estimado: 30 minutos.

**Q3 — Merge strategy:** **Unanimidad total: `git merge --no-ff`.** Trazabilidad científica sobre limpieza visual. Tag anotado `v0.4.0-phase3-hardening` con changelog estructurado.

**Q4 — XGBoost:** **Unanimidad: rama separada** `feature/adr026-xgboost`. Feature flag en JSON solo dentro de la rama para testing A/B interno, nunca en main hasta que Precision ≥ 0.99 + F1 ≥ 0.9985.

**¿Segunda ronda?** No es necesaria. Los 4 veredictos son claros y accionables. Nota: Parallel.ai no respondió — lo dejamos para DAY 118 si quieres intentarlo de nuevo.

Escribo el prompt de continuidad ahora.

---

# ML Defender (aRGus NDR) — DAY 118 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.

---

## Estado al cierre de DAY 117

### Hitos completados
- 12/13 DEBTs bloqueantes `feature/phase3-hardening` cerrados ✅
- AppArmor enforce 5/6: etcd-server, rag-security, rag-ingester, ml-detector, firewall-acl-agent (0 denials cada uno) ✅
- tools/apparmor-promote.sh: complain→enforce con rollback automático ✅
- TEST-PROVISION-1 8/8 · make test-all CI gate completo ✅
- DEBT-VAGRANTFILE-001, DEBT-SEED-PERM-001, REC-2, TEST-INVARIANT-SEED, Backup policy, ADR-021 addendum, Recovery Contract, DEBT-RAG-BUILD-001 ✅
- Pubkey dev rotada (3 resets): `e51a91e91d72f74fe97e8a4eb883c9c6eb41dd2fc994feaf59d5ba2177720f3d`
- arXiv Draft v15 recibido de Cornell ✅
- Commits: 85197f96 → fac4cd54 (7 commits)

### Consejo DAY 117 — Veredictos
- Q1 ALLOWED sniffer: revisar log completo antes de enforce (4/5) — ver detalle antes de promote
- Q2 noclobber: audit limitado a ficheros críticos (`/etc/`, `.sk`, `seed.bin`, `*.sig`)
- Q3 merge: `git merge --no-ff` unanimidad — trazabilidad científica
- Q4 XGBoost: rama separada `feature/adr026-xgboost` hasta Precision ≥ 0.99 + F1 ≥ 0.9985

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
# Esperar: 6/6 PASSED
```

**Solo si 6/6 RUNNING y 6/6 PASSED se continúa.**

---

## CRITERIO DE CIERRE DE FEATURE

La feature/phase3-hardening se mergea a main cuando:

✅ 12/13 DEBTs cerrados (DAY 117)
□ AppArmor enforce sniffer + TEST-APPARMOR-ENFORCE final
□ noclobber audit ficheros críticos
□ CHANGELOG-v0.4.0.md
□ git merge --no-ff → main + tag v0.4.0-phase3-hardening

---

## Orden DAY 118

### PASO 1 — Revisar ALLOWED sniffer (Consejo Q1)
```bash
vagrant ssh -c "sudo journalctl -k | grep -E 'apparmor.*sniffer|ALLOWED.*sniffer' -A3 -B3"
```
Clasificar el evento. Si es legítimo y cubierto por el perfil → proceder.
Si revela gap → actualizar perfil con `aa-logprof` antes de enforce.

### PASO 2 — AppArmor enforce sniffer
```bash
vagrant ssh -c "sudo bash /vagrant/tools/apparmor-promote.sh sniffer 2>&1"
```
**Test de cierre TEST-APPARMOR-ENFORCE:**
```bash
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS"
vagrant ssh -c "sudo aa-status 2>&1 | awk '/enforce mode/{found=1;next} /complain mode/{found=0} found && /vagrant/{print}'"
# Esperado: 6/6 en enforce
```

### PASO 3 — noclobber audit ficheros críticos (Consejo Q2)
```bash
grep -n '>[^|]' tools/provision.sh | grep -E '/etc/|\.sk|seed\.bin|\.sig|\.pem|\.pk|\.env'
```
Para cada ocurrencia: decidir si es intencional (`>|` + comentario) o protegida (`>` está bien).
**Test de cierre:** `make test-all` verde tras cambios.

### PASO 4 — CHANGELOG-v0.4.0.md
Crear `docs/CHANGELOG-v0.4.0.md` con estructura:
```markdown
## Security
## Operations  
## Tests
## Bug fixes
```
Referenciando commits y ADRs del DAY 115-118.

### PASO 5 — Merge a main
```bash
git checkout main
git pull origin main
git merge --no-ff feature/phase3-hardening -m "Merge feature/phase3-hardening

PHASE 3 hardening completion — v0.4.0:
- AppArmor enforce 6/6 componentes (0 denials)
- TEST-PROVISION-1 8/8 + make test-all CI gate
- INVARIANTE-SEED-001 + backup policy
- Recovery Contract (OQ-6 ADR-024)
- tools/apparmor-promote.sh rollback automático
- Pubkey dev rotada post-reset documentada

Closes: DEBT-VAGRANTFILE-001, DEBT-SEED-PERM-001, REC-2,
TEST-INVARIANT-SEED, Backup policy, ADR-021 addendum,
docs/Recovery Contract, DEBT-RAG-BUILD-001, apparmor-utils #8,
apparmor-promote.sh, AppArmor enforce 6/6"

git tag -a v0.4.0-phase3-hardening -m "PHASE 3 hardening complete"
git push origin main --tags
```

### PASO 6 — Actualizar BACKLOG.md + README.md (merge completado)
- Mover feature/phase3-hardening → COMPLETADO
- Badge AppArmor → 6/6 enforce
- Abrir `feature/adr026-xgboost` en roadmap

### PASO 7 — Abrir feature/adr026-xgboost
```bash
git checkout main
git checkout -b feature/adr026-xgboost
git push origin feature/adr026-xgboost
```
Crear `docs/XGBOOST-VALIDATION.md` con checklist de métricas obligatorias (Precision ≥ 0.99, F1 ≥ 0.9985, gate médico, revisión Consejo).

---

## Contexto permanente

### ADR-025 keypair dev (post-reset DAY 117)
MLD_PLUGIN_PUBKEY_HEX: `e51a91e91d72f74fe97e8a4eb883c9c6eb41dd2fc994feaf59d5ba2177720f3d`

### seed_family post-reset DAY 117
Seed compartido (6 componentes): 75deaca96768b5d973a4339faf2325c058969bf93c00c0d21eef703a2ab91360
INVARIANTE-SEED-001: todos los seed.bin DEBEN ser idénticos.

### Lección operacional crítica (DAY 117)
`provision.sh --reset` rota la keypair Ed25519. Pubkey hardcoded en CMakeLists.
**Siempre después de --reset:** `make pipeline-build` + `make sign-plugins` + `make test-all`

### Regla de oro del pipeline
6/6 RUNNING + make test-all VERDE

### Patrón robusto para scripts en VM
```bash
cat > /tmp/script.py << 'PYEOF'
...
PYEOF
vagrant upload /tmp/script.py /tmp/script.py
vagrant ssh -c "sudo python3 /tmp/script.py"
```
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

systemd units                  ✅ DAY 115
DEBT-SIGN-AUTO                 ✅ DAY 115
DEBT-HELLO-001                 ✅ DAY 115
TEST-PROVISION-1 (5/5)         ✅ DAY 115
DEBT-ADR025-D11 --reset        ✅ DAY 116
AppArmor complain (6/6)        ✅ DAY 116
DEBT-VAGRANTFILE-001           ✅ DAY 117
DEBT-SEED-PERM-001             ✅ DAY 117
REC-2 noclobber                ✅ DAY 117
TEST-INVARIANT-SEED            ✅ DAY 117
TEST-PROVISION-1 (8/8)         ✅ DAY 117
Backup policy .bak.*           ✅ DAY 117
ADR-021 addendum               ✅ DAY 117
Recovery Contract              ✅ DAY 117
DEBT-RAG-BUILD-001             ✅ DAY 117
apparmor-utils check #8        ✅ DAY 117
apparmor-promote.sh            ✅ DAY 117
AppArmor enforce (5/6)         ✅ DAY 117
noclobber audit crítico        🔄 DAY 118
AppArmor enforce sniffer       🔄 DAY 118
CHANGELOG-v0.4.0.md            🔄 DAY 118
MERGE A MAIN                   🔄 DAY 118
────────────────────────────────────────────
Tag: v0.4.0-phase3-hardening · post merge DAY 118

---


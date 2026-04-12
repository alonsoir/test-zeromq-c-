# ML Defender (aRGus NDR) — DAY 116 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## Estado al cierre de DAY 115

### Hitos del día
**PHASE 3 ítems 1-4: COMPLETADOS ✅**
- systemd units (6 componentes): Restart=always, LD_PRELOAD=unset, build-active symlinks
- DEBT-SIGN-AUTO: provision.sh check-plugins (dev sign-if-needed, prod verify-only)
- DEBT-HELLO-001: BUILD_DEV_PLUGINS=OFF + 5 JSONs limpios (bug: 4 tenían active:true)
- TEST-PROVISION-1: CI gate 5 checks, pipeline-start depende de él
- Commits: df976d90, a1b23882 (feature/phase3-hardening)

**ADR-024 OQ-5..8: CERRADAS (Consejo DAY 115 unanimidad) ✅**
ADR-024 actualizado con Recovery Contract + TEST-INTEG-8/9.

**Consejo DAY 115 cierre — Veredictos:**
- Q1 AppArmor: complain → audit → enforce (6/6 unanimidad)
- Q2 --reset: regenera claves SIN auto-firma; operador firma manualmente post-reset
- Q3 Orden: DEBT-ADR025-D11 primero (4/6, incluido árbitro), AppArmor después
- Q4 TEST-PROVISION-1: añadir check #6 (permisos ficheros sensibles) + check #7 (consistencia JSONs con plugins reales)

---

## PASO 0 — Verificación de estabilidad del pipeline (SIEMPRE PRIMERO)

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

## Orden DAY 116

### PASO 1 — DEBT-ADR025-D11: provision.sh --reset (URGENTE, deadline 18 Apr)

**Diseño aprobado por Consejo:**
- `--reset` regenera: seed_family + keypairs Ed25519 de 6 componentes + keypair de firma de plugins
- NO auto-firma plugins tras reset
- Post-reset: mensaje claro "Claves rotadas. Ejecuta: make sign-plugins"
- En dev: flag opcional `--dev` puede llamar a check-plugins automáticamente
- En producción: solo verificar, nunca firmar

```bash
# Ver provision.sh actual para diseñar --reset
vagrant ssh -c "grep -n 'reset\|RESET\|reprovision' /vagrant/tools/provision.sh"
```

**Tests requeridos:**
- TEST-RESET-1: `--reset` regenera todos los keypairs
- TEST-RESET-2: post-reset, pipeline en fail-closed hasta re-firma
- TEST-RESET-3: post-reset + sign, pipeline arranca 6/6

### PASO 2 — TEST-PROVISION-1 checks 6+7

**Check #6 (permisos):**
```bash
find /etc/ml-defender /usr/lib/ml-defender -type f \
  \( -name "*.sk" -o -name "deployment.yml" \) -perm /022
# Debe retornar vacío
```

**Check #7 (consistencia JSONs):**
Cada plugin referenciado en JSONs de producción tiene `.so` + `.sig` en `/usr/lib/ml-defender/plugins/`

### PASO 3 — AppArmor profiles (complain first)

6 perfiles para:
- etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall-acl-agent

Requisitos del Consejo:
- Incluir paths de `provision.sh --reset` en perfiles (evitar bloqueo futuro)
- Denegar `write /usr/bin/ml-defender-*` para root
- Sniffer: CAP_NET_RAW, CAP_NET_ADMIN, CAP_SYS_ADMIN, CAP_BPF
- Firewall: CAP_NET_ADMIN, CAP_NET_RAW
- Iniciar en modo **complain**, verificar pipeline 6/6, luego enforce

---

## Contexto permanente

### Proyecto
- **aRGus NDR (ML Defender)**: C++20 NDR para hospitales, escuelas, municipios
- **arXiv**: arXiv:2604.04952 [cs.CR] — v15 submitted ✅
- **Branch activa**: feature/phase3-hardening
- **Repositorio**: https://github.com/alonsoir/argus
- **Tag estable**: v0.3.0-plugin-integrity (main)

### Regla de oro del pipeline
**Estable = 6/6 RUNNING + 12/12 plugin-integ-test PASSED**
Tras cualquier cambio: stop → build → sign → test-provision-1 → start → status → plugin-integ-test

### ADR-025 keypair dev
- MLD_PLUGIN_PUBKEY_HEX: b824bcd7a14f6e19a0d8c9be86110828060e600723d12e118dccc95c862c8468
- Private key: /etc/ml-defender/plugins/plugin_signing.sk (VM only)
- Firmar: make sign-plugins

### DEBT-ADR025-D11 deadline
provision.sh --reset — **deadline 18 Apr 2026**. No negociable.

### PHASE 3 estado
```
1. systemd units         ✅ COMPLETADO DAY 115
2. DEBT-SIGN-AUTO        ✅ COMPLETADO DAY 115
3. DEBT-HELLO-001        ✅ COMPLETADO DAY 115
4. TEST-PROVISION-1      ✅ COMPLETADO DAY 115
5. AppArmor profiles     ← DAY 116 PASO 3
6. DEBT-ADR025-D11       ← DAY 116 PASO 1 (deadline 18 Apr)
+ checks 6+7 TEST-PROVISION-1 ← DAY 116 PASO 2
```

### Veredictos Consejo DAY 115 (definitivos)
- AppArmor: complain → audit → enforce (6/6)
- --reset: SIN auto-firma producción (6/6)
- Orden: DEBT-ADR025-D11 antes de AppArmor (4/6 + árbitro)
- TEST-PROVISION-1: añadir checks 6+7 (6/6)

### Patrón robusto para scripts en VM
```
cat > /tmp/script.py << 'PYEOF'
...
PYEOF
vagrant upload /tmp/script.py /tmp/script.py
vagrant ssh -c "sudo python3 /tmp/script.py"
```
NUNCA sed -i en macOS. Para scripts Python en VM: heredoc con 'PYEOF'.

### Consejo de Sabios (7 miembros)
Claude, Grok, ChatGPT, DeepSeek, Qwen (Alibaba), Gemini, Parallel.ai.

### PHASE 2 — COMPLETA ✅
2a+2b+2c+2d+2e. 12/12 tests PASSED.

### Filosofía core
"Un escudo, nunca una espada."
"La verdad por delante, siempre."
Fail-closed. PHASE 3: operación segura, no solo seguridad.
ADR-032: la autoridad de firma y el servidor de producción NO comparten dominio de confianza.
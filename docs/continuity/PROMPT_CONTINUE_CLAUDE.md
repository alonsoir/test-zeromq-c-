# ML Defender (aRGus NDR) — DAY 120 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad. Nunca compilar o instalar manualmente en la VM sin actualizar ambos.

---

## Estado al cierre de DAY 119

### Hitos completados
- **Reproducibilidad `vagrant destroy` VALIDADA** ✅ — 10 problemas detectados y resueltos
- **Vagrantfile consolidado** ✅ — libsodium 1.0.19 · tmux · xxd · plugin_xgboost · plugin_test_message · /usr/lib/ml-defender/plugins/
- **Makefile consolidado** ✅ — pipeline-build con deps explícitas · install-systemd-units · set-build-profile · sync-pubkey · plugin-test-message-build
- **plugin_xgboost API corregida** ✅ — `PluginResult plugin_init(const PluginConfig*)` · `plugin_process_message` · `plugin_shutdown` · contratos @requires/@ensures
- **make sync-pubkey** ✅ — temporal, pendiente DEBT-PUBKEY-RUNTIME-001
- **6/6 RUNNING + make test-all VERDE** ✅ — incluyendo TEST-INTEG-SIGN PASSED
- **Consejo DAY 119 sintetizado** ✅ — BACKLOG.md + README.md actualizados
- **Commits:** 8d964390 → 6055c54d · Branch: `feature/adr026-xgboost`

### Pubkey activa DAY 119 (post vagrant destroy)
`MLD_PLUGIN_PUBKEY_HEX: 9ac7b8c5ce2d970f77a5fcfcc3b8463b66082db50636a9e81da3cdbb7b2b8019`

### Seed activo DAY 119
`75deaca96768b5d973a4339faf2325c058969bf93c00c0d21eef703a2ab91360`
INVARIANTE-SEED-001: todos los seed.bin DEBEN ser idénticos.

---

## PASO 0 — DAY 120: vagrant destroy OBLIGATORIO (validar idempotencia)

**⚠️ DAY 120 arranca con vagrant destroy. El Consejo recomienda ejecutar la secuencia DOS VECES para verificar idempotencia. Es el objetivo del día antes de avanzar con ADR-026.**

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/adr026-xgboost
git pull origin feature/adr026-xgboost

vagrant destroy -f
vagrant up
# Esperar ~20-30 minutos
```

### Verificaciones post-up
```bash
vagrant ssh -c "sudo ldconfig -p | grep xgboost"
# Esperado: libxgboost.so => /usr/local/lib/libxgboost.so

vagrant ssh -c "python3 -c 'import xgboost; print(xgboost.__version__)'"
# Esperado: 3.2.0

vagrant ssh -c "pkg-config --modversion libsodium"
# Esperado: 1.0.19

vagrant ssh -c "which tmux && xxd --version 2>&1 | head -1"
# Esperado: /usr/bin/tmux · xxd ...

vagrant ssh -c "ls /usr/lib/ml-defender/plugins/"
# Esperado: libplugin_test_message.so libplugin_xgboost.so
```

### Secuencia canónica completa (ejecutar en orden estricto)
```bash
make sync-pubkey
# Esperado: ✅ pubkey actualizada + plugin-loader recompilado

make set-build-profile
# Esperado: 6/6 build-active symlinks

make install-systemd-units
# Esperado: 6/6 units instalados

make pipeline-build 2>&1 | tail -5
# Esperado: ✅ Sniffer built (debug)

make sign-plugins
# Esperado: ✅ 2 plugin(s) firmados

make test-provision-1
# Esperado: ✅ TEST-PROVISION-1 PASSED 8/8

make pipeline-start && make pipeline-status
# Esperado: 6/6 RUNNING

make plugin-integ-test 2>&1 | grep -E "PASSED|FAILED"
# Esperado: 6/6 PASSED incluyendo TEST-INTEG-SIGN

make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS"
# Esperado: ✅ ALL TESTS COMPLETE
```

**Solo si primera ejecución verde → repetir secuencia sin vagrant destroy para verificar idempotencia.**

---

## PASO 1 — DEBT-PUBKEY-RUNTIME-001 (BLOQUEANTE DAY 120)

**Objetivo:** eliminar `make sync-pubkey` moviendo la pubkey de CMakeLists.txt a fichero runtime en la VM.

### Diseño (ChatGPT5 DAY 119)
```cmake
# plugin-loader/CMakeLists.txt — reemplazar pubkey hardcodeada por:
file(READ "/etc/ml-defender/plugins/plugin_signing.pk" PUBKEY_PEM)
# Luego extraer hex desde PEM via script o cmake custom command
```

### Implementación
1. Crear script `tools/extract-pubkey-hex.sh` que lea `/etc/ml-defender/plugins/plugin_signing.pk` y devuelva hex
2. Modificar `plugin-loader/CMakeLists.txt` para ejecutar script en tiempo de cmake y inyectar resultado
3. Eliminar `make sync-pubkey` del Makefile (o dejarlo como deprecated con warning)
4. Verificar que `vagrant destroy + vagrant up + make plugin-loader-build` funciona sin sync-pubkey

**Test de cierre:** `vagrant destroy && vagrant up && make plugin-loader-build && make sign-plugins && make plugin-integ-test` → TEST-INTEG-SIGN PASSED sin ejecutar `make sync-pubkey`.

---

## PASO 2 — DEBT-BOOTSTRAP-001 (BLOQUEANTE DAY 120)

**Objetivo:** `make bootstrap` que encadene los 9 pasos canónicos con checkpoints, verbose e idempotencia.

### Diseño (Consejo unanimidad)
```makefile
.PHONY: bootstrap
bootstrap:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  🚀 aRGus NDR — Bootstrap from scratch                    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo "[1/9] make up..."
	@$(MAKE) up
	@echo "[2/9] make post-up-verify..."
	@$(MAKE) post-up-verify
	@echo "[3/9] make check-system-deps..."
	@$(MAKE) check-system-deps
	@echo "[4/9] make pipeline-build..."
	@$(MAKE) pipeline-build
	@echo "[5/9] make set-build-profile..."
	@$(MAKE) set-build-profile
	@echo "[6/9] make install-systemd-units..."
	@$(MAKE) install-systemd-units
	@echo "[7/9] make sign-plugins..."
	@$(MAKE) sign-plugins
	@echo "[8/9] make test-provision-1..."
	@$(MAKE) test-provision-1
	@echo "[9/9] make pipeline-start..."
	@$(MAKE) pipeline-start
	@$(MAKE) pipeline-status
	@$(MAKE) plugin-integ-test
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║  ✅ Bootstrap completado — 6/6 RUNNING                    ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
```

**Nota:** una vez implementado DEBT-PUBKEY-RUNTIME-001, el paso [2/9] elimina sync-pubkey y pipeline-build gestiona la pubkey automáticamente vía CMake.

**Test de cierre:** `vagrant destroy && vagrant up && make bootstrap` → 6/6 RUNNING + plugin-integ-test PASSED.

---

## PASO 3 — DEBT-INFRA-VERIFY-001/002 (BLOQUEANTE DAY 120)

### make check-system-deps
```makefile
check-system-deps:
	@echo "🔍 Verifying system dependencies..."
	@vagrant ssh -c "command -v xxd >/dev/null || { echo '❌ xxd missing'; exit 1; }"
	@vagrant ssh -c "command -v tmux >/dev/null || { echo '❌ tmux missing'; exit 1; }"
	@vagrant ssh -c "pkg-config --modversion libsodium 2>/dev/null | grep -q '1.0.19' || { echo '❌ libsodium 1.0.19 missing'; exit 1; }"
	@vagrant ssh -c "python3 -c 'import xgboost; assert xgboost.__version__ == \"3.2.0\"' || { echo '❌ xgboost 3.2.0 missing'; exit 1; }"
	@vagrant ssh -c "test -f /usr/local/lib/libxgboost.so || { echo '❌ libxgboost.so missing'; exit 1; }"
	@vagrant ssh -c "test -f /usr/local/lib/libcrypto_transport.so || { echo '❌ libcrypto_transport.so missing'; exit 1; }"
	@echo "✅ All system dependencies present"
```

### make post-up-verify
```makefile
post-up-verify:
	@echo "🔍 Verifying post-up environment..."
	@$(MAKE) check-system-deps
	@vagrant ssh -c "test -d /usr/lib/ml-defender/plugins || { echo '❌ plugins dir missing'; exit 1; }"
	@vagrant ssh -c "test -f /usr/lib/ml-defender/plugins/libplugin_xgboost.so || { echo '❌ plugin_xgboost missing'; exit 1; }"
	@vagrant ssh -c "test -f /usr/lib/ml-defender/plugins/libplugin_test_message.so || { echo '❌ plugin_test_message missing'; exit 1; }"
	@vagrant ssh -c "sudo find /etc/ml-defender -name 'seed.bin' | wc -l | grep -q '6' || { echo '❌ seeds missing'; exit 1; }"
	@echo "✅ Post-up environment verified"
```

**Test de cierre:** `make post-up-verify` verde tras `vagrant up` limpio.

---

## PASO 4 — Avanzar con ADR-026 (solo si PASOS 0-3 verdes)

### PASO 4a — Localizar feature set RF baseline
```bash
find /vagrant/ml-detector -name "*.h" -o -name "*.cpp" 2>/dev/null | \
  xargs grep -l -i "feature\|extract" 2>/dev/null
```
Documentar en `docs/xgboost/features.md` (Opción A — mismo feature set que RF, unanimidad Consejo).

### PASO 4b — docs/xgboost/plugin-contract.md
Crear documento con contrato mínimo `ctx->payload`:
- `float32[]` contiguo en row-major
- `payload_size % sizeof(float) == 0`
- `num_features` = columnas modelo
- Sin NaN ni Inf
- Versión del esquema declarada

### PASO 4c — scripts/train_xgboost_baseline.py
- Mismo feature set que RF (Opción A)
- `random_state=42`
- Exportar `xgboost_ctu13.json` + `xgboost_ctu13.ubj`
- Gate: Precision ≥ 0.99 + F1 ≥ 0.9985

### PASO 4d — make sign-models (OBS-1 BLOQUEANTE)
Extender Makefile:
- Firma `xgboost_ctu13.ubj` con keypair Ed25519 (mismo esquema ADR-025)
- Genera `xgboost_ctu13.ubj.sig`
- Verificación en `plugin_init` antes de `XGBoosterLoadModel`

### PASO 4e — TEST-INTEG-XGBOOST-1 (OBS-2 BLOQUEANTE)
Test unitario: modelo juguete + plugin_process_message con MessageContext sintético → salida ∈ [0,1] no NaN.

---

## Contexto permanente

### ADR-025 keypair dev (post-reset DAY 119 vagrant destroy)
`MLD_PLUGIN_PUBKEY_HEX: 9ac7b8c5ce2d970f77a5fcfcc3b8463b66082db50636a9e81da3cdbb7b2b8019`

### Seed activo
`75deaca96768b5d973a4339faf2325c058969bf93c00c0d21eef703a2ab91360`
INVARIANTE-SEED-001: todos los seed.bin DEBEN ser idénticos.

### Lección operacional crítica DAY 119
El Vagrantfile y el Makefile son la única fuente de verdad. Compilar o instalar manualmente en la VM sin actualizar estas fuentes = deuda técnica de infraestructura garantizada.

### Regla de oro
6/6 RUNNING + make test-all VERDE

### Secuencia canónica post vagrant destroy (DAY 119)
```
make sync-pubkey           ← temporal hasta DEBT-PUBKEY-RUNTIME-001
make set-build-profile
make install-systemd-units
make pipeline-build
make sign-plugins
make test-provision-1
make pipeline-start && make pipeline-status
make plugin-integ-test
```

**Una vez DEBT-PUBKEY-RUNTIME-001 + DEBT-BOOTSTRAP-001 cerrados:**
```
make bootstrap
```

### Patrón robusto para scripts en VM
```bash
cat > /tmp/script.py << 'PYEOF'
...
PYEOF
vagrant upload /tmp/script.py /tmp/script.py
vagrant ssh -c "sudo python3 /tmp/script.py"
```
NUNCA `sed -i` sin `-e ''` en macOS. NUNCA Python inline con paréntesis en zsh.

### feature/adr026-xgboost — estado DAY 119 cierre
```
plugins/xgboost/CMakeLists.txt              ✅ commit 9500300a
plugins/xgboost/xgboost_plugin.cpp          ✅ API corregida DAY 119 · contratos OBS-5
docs/XGBOOST-VALIDATION.md                  ✅ gate médico
docs/consejo/CONSEJO-DAY119-preguntas.md    ✅ veredictos Consejo
Vagrantfile libsodium 1.0.19                ✅ antes de ONNX
Vagrantfile tmux + xxd                      ✅ paquetes base
Vagrantfile plugin_xgboost + plugin_test    ✅ build + deploy
Makefile pipeline-build deps explícitas     ✅
Makefile install-systemd-units              ✅
Makefile set-build-profile                  ✅
Makefile sync-pubkey                        ✅ temporal
Makefile plugin-test-message-build          ✅
Feature set                                 Opción A (mismo RF) — UNANIMIDAD
Formato modelo                              JSON repo + .ubj producción — UNANIMIDAD
plugin_invoke                               Opción B (ml-detector pre-procesa) — UNANIMIDAD
std::terminate() v0.1                       Fail-closed — UNANIMIDAD
DEBT-PUBKEY-RUNTIME-001                     ⏳ BLOQUEANTE DAY 120
DEBT-BOOTSTRAP-001                          ⏳ BLOQUEANTE DAY 120
DEBT-INFRA-VERIFY-001/002                   ⏳ BLOQUEANTE DAY 120
DEBT-XGBOOST-SIGN-001                       ⏳ BLOQUEANTE merge
TEST-INTEG-XGBOOST-1                        ⏳ BLOQUEANTE merge
```

### DEBTs no bloqueantes (NO tocar en esta feature)
- DEBT-CRYPTO-003a → feature/crypto-hardening
- DEBT-OPS-001/002 → feature/ops-tooling
- DEBT-SNIFFER-SEED → feature/crypto-hardening
- DEBT-INFRA-001 → feature/bare-metal
- DEBT-CLI-001 → feature/adr032-hsm
- DEBT-XGBOOST-SOFTFAIL-001 → feature/phase5-resilience
- DEBT-XGBOOST-APT-001 → verificar versión apt bookworm (no bloqueante)
- ADR-033 TPM → post-PHASE 4

### Paper arXiv
arXiv:2604.04952 — Draft v15.
Tabla RF vs XGBoost (latencia + F1 + Precision) irá en §4.
Contribución científica: delta RF vs XGBoost en CTU-13 Neris.

---

*"Via Appia Quality — un escudo, nunca una espada."*
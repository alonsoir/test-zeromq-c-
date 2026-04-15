# ML Defender (aRGus NDR) — DAY 119 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.

---

## Estado al cierre de DAY 118

### Hitos completados
- **PHASE 3 CERRADA** ✅ — `git merge --no-ff` + tag `v0.4.0-phase3-hardening` (da0296cd)
- AppArmor 6/6 enforce (0 denials) ✅ · make test-all VERDE ✅
- CHANGELOG-v0.4.0.md ✅ · README + BACKLOG actualizados ✅
- **`feature/adr026-xgboost` abierta** ✅ — commit `9500300a`
    - `plugins/xgboost/CMakeLists.txt` ✅
    - `plugins/xgboost/xgboost_plugin.cpp` (skeleton fail-closed) ✅
    - `docs/XGBOOST-VALIDATION.md` ✅
    - Vagrantfile bloque XGBoost 3.2.0 (líneas 327-348) ✅
    - `docs/consejo/CONSEJO-DAY118-sintesis.md` ✅

### XGBoost en VM (instalación manual DAY 118 — pendiente verificar desde cero)
- `/usr/local/lib/libxgboost.so` + `/usr/local/include/xgboost/` ✅ (manual)
- Compilación C con `-lxgboost` OK ✅
- **DAY 119 PASO 0: verificar que Vagrantfile reproduce esto desde cero**

### Consejo DAY 118 — Veredictos DEFINITIVOS (5/7 + segunda ronda Gemini)

| Q | Veredicto | Unanimidad |
|---|-----------|------------|
| Q1 feature set | Opción A — mismo que RF | 5/5 ✅ |
| Q2 formato modelo | JSON repo + .ubj producción + .ubj.sig | 5/5 ✅ |
| Q3 plugin_invoke | Opción B — ml-detector pre-procesa float32[] | 5/5 ✅ |
| Q4 Vagrantfile | pip 3.2.0 + fallback apt + docs offline | 4/5 ✅ |
| OBS-4 std::terminate | Fail-closed v0.1 — Integridad > Disponibilidad | 5/5 ✅ |

### Items bloqueantes nuevos (merge feature/adr026-xgboost → main)
- **OBS-1 / DEBT-XGBOOST-SIGN-001**: firma Ed25519 del modelo (.ubj.sig)
- **OBS-2 / TEST-INTEG-XGBOOST-1**: test en make test-all

### Items no bloqueantes (backlog)
- OBS-3: latencia desde Fase 3 (para paper §4)
- OBS-5: contratos informales ADR-036 en código
- OBS-6: cache modelo en plugin_init
- DEBT-XGBOOST-SOFTFAIL-001 → feature/phase5-resilience

---

## PASO 0 — ESPECIAL DAY 119: Vagrantfile desde cero (SIEMPRE PRIMERO)

**⚠️ DAY 119 arranca con vagrant destroy. No saltar este paso.**

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/adr026-xgboost
git pull origin feature/adr026-xgboost

# Destruir y recrear VM desde cero
vagrant destroy -f
vagrant up
# Esperar ~20-30 minutos — provisioning completo

# Verificar XGBoost instalado automáticamente
vagrant ssh -c "sudo ldconfig -p | grep xgboost"
# Esperado: libxgboost.so (libc6,x86-64) => /usr/local/lib/libxgboost.so

vagrant ssh -c "python3 -c 'import xgboost; print(xgboost.__version__)'"
# Esperado: 3.2.0

vagrant ssh -c "ls /usr/local/include/xgboost/"
# Esperado: base.h  c_api.h

# Test compilación C API
vagrant ssh -c "echo '#include <xgboost/c_api.h>\n#include <stdio.h>\nint main(){printf(\"OK\\n\");return 0;}' > /tmp/t.c && gcc /tmp/t.c -lxgboost -o /tmp/t && /tmp/t"
# Esperado: OK
```

**Si PASO 0 falla:** diagnosticar bloque Vagrantfile (líneas 327-348) antes de continuar.

### Luego: verificación pipeline estándar

```bash
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

## Orden DAY 119

### PASO 1 — Fallback apt en Vagrantfile (Q4 Consejo)
Actualizar bloque XGBoost con fallback:
```bash
pip3 install xgboost==3.2.0 --break-system-packages || {
    echo "⚠️  PyPI inaccesible — fallback apt (versión no garantizada)"
    apt-get install -y python3-xgboost
    echo "❗ WARNING: xgboost $(python3 -c 'import xgboost; print(xgboost.__version__)')"
    echo "❗ Para reproducibilidad científica, usar xgboost==3.2.0"
}
```
Crear `docs/OFFLINE-DEPLOYMENT.md` con instrucciones para entornos air-gapped.

### PASO 2 — Compilar plugin_xgboost
```bash
make pipeline-build 2>&1 | grep -E "plugin_xgboost|error:|warning:"
make sign-plugins
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS"
```
Esperado: `libplugin_xgboost.so` compilado sin errores + `libplugin_xgboost.so.sig` ✅

### PASO 3 — Localizar feature set RF baseline
```bash
find /vagrant/ml-detector -name "*.h" -o -name "*.cpp" 2>/dev/null | \
  xargs grep -l -i "feature\|extract" 2>/dev/null
```
Identificar: columnas exactas, orden, normalización.
Documentar en `docs/xgboost/features.md` (Opción A — mismo que RF).

### PASO 4 — Script entrenamiento XGBoost (Fase 2)
Crear `scripts/train_xgboost_baseline.py`:
- Mismo feature set que RF (Opción A — UNANIMIDAD Consejo)
- `random_state=42` para reproducibilidad
- Exportar `xgboost_ctu13.json` + `xgboost_ctu13.ubj`
- Validar equivalencia JSON/ubj (mismo output)
- Registrar: F1, Precision, Recall, FPR, latencia
- Gate: Precision ≥ 0.99 + F1 ≥ 0.9985

### PASO 5 — Firma del modelo (OBS-1 — BLOQUEANTE)
Extender Makefile con `make sign-models`:
- Firma `xgboost_ctu13.ubj` con keypair Ed25519 (mismo esquema ADR-025)
- Genera `xgboost_ctu13.ubj.sig`
- Verificación en `plugin_init` antes de `XGBoosterLoadModel`

### PASO 6 — Contratos informales ADR-036 (OBS-5)
Añadir en `xgboost_plugin.cpp`:
```cpp
// @requires: ctx != nullptr && ctx->payload != nullptr && ctx->payload_size == N*sizeof(float)
// @ensures: return_value == 0 (OK) || std::terminate() (fail-closed)
// @invariant: no side effects on MessageContext si falla
```

---

## Contexto permanente

### ADR-025 keypair dev (post-reset DAY 117)
`MLD_PLUGIN_PUBKEY_HEX: e51a91e91d72f74fe97e8a4eb883c9c6eb41dd2fc994feaf59d5ba2177720f3d`

### seed_family post-reset DAY 117
Seed: `75deaca96768b5d973a4339faf2325c058969bf93c00c0d21eef703a2ab91360`
INVARIANTE-SEED-001: todos los seed.bin DEBEN ser idénticos.

### Lección operacional crítica
`provision.sh --reset` rota keypair. Siempre post-reset:
`make pipeline-build` → `make sign-plugins` → `make test-all`

### Regla de oro
6/6 RUNNING + make test-all VERDE

### Patrón robusto para scripts en VM
```bash
cat > /tmp/script.py << 'PYEOF'
...
PYEOF
vagrant upload /tmp/script.py /tmp/script.py
vagrant ssh -c "sudo python3 /tmp/script.py"
```
NUNCA `sed -i` sin `-e ''` en macOS. NUNCA Python inline con paréntesis en zsh.

### feature/adr026-xgboost — estado DAY 118 cierre
```
plugins/xgboost/CMakeLists.txt          ✅ commit 9500300a
plugins/xgboost/xgboost_plugin.cpp      ✅ skeleton fail-closed
docs/XGBOOST-VALIDATION.md              ✅ gate médico
docs/consejo/CONSEJO-DAY118-sintesis.md ✅ veredictos + segunda ronda
Vagrantfile bloque XGBoost (327-348)    ✅ pip 3.2.0 (fallback apt → DAY 119)
Feature set                              Opción A (mismo RF) — UNANIMIDAD
Formato modelo                           JSON repo + .ubj producción — UNANIMIDAD
plugin_invoke                            Opción B (ml-detector pre-procesa) — UNANIMIDAD
std::terminate() v0.1                    Fail-closed — UNANIMIDAD (incl. Gemini 2ª ronda)
DEBT-XGBOOST-SIGN-001                   ⏳ BLOQUEANTE — pendiente DAY 119
TEST-INTEG-XGBOOST-1                    ⏳ BLOQUEANTE — pendiente DAY 119
```

### DEBTs no bloqueantes (NO tocar en esta feature)
- DEBT-CRYPTO-003a → feature/crypto-hardening
- DEBT-OPS-001/002 → feature/ops-tooling
- DEBT-SNIFFER-SEED → feature/crypto-hardening
- DEBT-INFRA-001 → feature/bare-metal
- DEBT-CLI-001 → feature/adr032-hsm
- DEBT-XGBOOST-SOFTFAIL-001 → feature/phase5-resilience
- ADR-033 TPM → post-PHASE 4

### Paper arXiv
arXiv:2604.04952 — Draft v15.
Tabla RF vs XGBoost (latencia + F1 + Precision) irá en §4.
Contribución científica: delta RF vs XGBoost en CTU-13 Neris.

---

*"Via Appia Quality — un escudo, nunca una espada."*
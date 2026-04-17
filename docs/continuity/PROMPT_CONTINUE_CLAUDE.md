cat > /tmp/DAY121_prompt.md << 'MDEOF'
# ML Defender (aRGus NDR) — DAY 121 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad. Nunca compilar o instalar manualmente en la VM sin actualizar ambos.
- **REGLA SCRIPTS:** Lógica compleja con quoting anidado → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs de build. Solo runtime: mlock() + explicit_bzero(). SecureBuffer C++20.

---

## Estado al cierre de DAY 120

### Hitos completados
- **DEBT-PUBKEY-RUNTIME-001** ✅ — `tools/extract-pubkey-hex.sh` + `execute_process()`. Sin `make sync-pubkey`.
- **DEBT-BOOTSTRAP-001** ✅ — `make bootstrap` 8 pasos canónicos, idempotente.
- **DEBT-INFRA-VERIFY-001/002** ✅ — `make check-system-deps` + `make post-up-verify`.
- **Idempotencia × 2** ✅ — vagrant destroy verde ambas iteraciones.
- **ADR-026 PASO 4a** ✅ — `docs/xgboost/features.md` — 23 features LEVEL1, dataset CIC-IDS-2017.
- **ADR-026 PASO 4b** ✅ — `docs/xgboost/plugin-contract.md` — contrato float32[23], schema v1.
- **ADR-026 PASO 4c** ✅ — XGBoost entrenado: F1=0.9978, Precision=0.9973, ROC-AUC=1.0 (2.83M flows CIC-IDS-2017).
- **ADR-026 PASO 4d** ✅ — `make sign-models` + `tools/sign-model.sh` — Ed25519 firma `.ubj`.
- **ADR-026 PASO 4e** ✅ — `TEST-INTEG-XGBOOST-1 PASSED` — inferencia real, contratos técnicos OK.
- **make test-all VERDE** ✅
- **Consejo 7/7 incorporados** ✅ — Kimi y Mistral nuevos miembros.
- **Commits:** hasta `0a2bdef3` · Branch: `feature/adr026-xgboost`

### Pubkey activa DAY 120
`ec8c4bf0fdce51d556b99b5ca7a74aaad6f6683c6f6914784c732c4abbc8c6e1`

### Datasets localizados DAY 120
- **CIC-IDS-2017** (real): `ml-training/datasets/CIC-IDS-2017/MachineLearningCVE/` — 8 CSVs, 2.83M flows, 23 features
- **DDoS sintético DeepSeek**: `ml-training/scripts/ddos_detection/ddos_detection_dataset.json` — 27MB, 10 features DDOS_FEATURES
- **Ransomware sintético DeepSeek**: `ml-training/scripts/ransomware/data/network_guaranteed.csv` + `files_guaranteed.csv` + `processes_guaranteed.csv`

### Nota crítica Consejo DAY 120
TEST-INTEG-XGBOOST-1 scores actuales: BENIGN=0.000706, ATTACK=0.003414 — ambos out-of-distribution (features sintéticas extremas). El Consejo (7/7 unánime) rechaza esto como gate de merge. Necesita casos reales del CSV con score ATTACK>0.5 y BENIGN<0.1.

---

## PASO 0 — DAY 121: vagrant destroy OBLIGATORIO (tercera validación idempotencia)

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/adr026-xgboost
git pull origin feature/adr026-xgboost

vagrant destroy -f && vagrant up
# ~20-30 minutos

# Tras provisioning:
make bootstrap
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS"
```

**Si verde → PASO 0 certificado definitivamente. Secuencia canónica = `make bootstrap`.**

---

## PASO 1 — DEBT-SEED-AUDIT-001 (BLOQUEANTE DAY 121)

**Objetivo:** Verificar que la seed ChaCha20 NO está hardcodeada en ningún CMakeLists.txt ni fuente C++.

```bash
grep -r "seed" /Users/aironman/CLionProjects/test-zeromq-docker/plugin-loader/CMakeLists.txt
grep -r "seed" /Users/aironman/CLionProjects/test-zeromq-docker/crypto-transport/CMakeLists.txt
grep -rn "[0-9a-f]\{32,64\}" /Users/aironman/CLionProjects/test-zeromq-docker/plugin-loader/CMakeLists.txt
```

**Si se encuentra seed hardcodeada:** eliminar del CMake. La seed se lee exclusivamente en runtime:
```cpp
// SecureBuffer C++20 — mlock + explicit_bzero en destructor
class SecureBuffer : public std::vector<uint8_t> {
public:
    explicit SecureBuffer(size_t n) : std::vector<uint8_t>(n) { mlock(data(), size()); }
    ~SecureBuffer() { explicit_bzero(data(), size()); munlock(data(), size()); }
};
// Uso: SecureBuffer seed(32); leer de /etc/ml-defender/<comp>/seed.bin
```

**Test de cierre:** `grep -r "seed" CMakeLists.txt` → 0 resultados con hex literal.

---

## PASO 2 — DEBT-XGBOOST-TEST-REAL-001 (BLOQUEANTE MERGE)

**Objetivo:** Añadir a TEST-INTEG-XGBOOST-1 casos reales de CIC-IDS-2017.

**Estrategia:** Extraer 3 flows ATTACK + 3 flows BENIGN del CSV de test, convertir features a float32[23] en el orden exacto de `LEVEL1_FEATURE_NAMES`, incrustar como arrays estáticos en el test C++.

```bash
# Extraer muestras reales del CSV
python3 << 'PYEOF'
import pandas as pd
import numpy as np

FEATURES = [
    " Packet Length Std", " Subflow Fwd Bytes", " Fwd Packet Length Max",
    " Avg Fwd Segment Size", " ACK Flag Count", " Packet Length Variance",
    " PSH Flag Count", "Bwd Packet Length Max", " act_data_pkt_fwd",
    "Total Length of Fwd Packets", " Fwd Packet Length Std", "Fwd Packets/s",
    " Subflow Bwd Bytes", " Destination Port", "Init_Win_bytes_forward",
    "Subflow Fwd Packets", " Fwd IAT Min", " Packet Length Mean",
    " Total Length of Bwd Packets", " Bwd Packet Length Mean",
    " Bwd Packet Length Min", " Flow Duration", " Flow Packets/s"
]

df = pd.read_csv("ml-training/datasets/CIC-IDS-2017/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=False)
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

benign = df[df[" Label"].str.strip() == "BENIGN"][FEATURES].head(3)
attack = df[df[" Label"].str.strip() != "BENIGN"][FEATURES].head(3)

for i, row in benign.iterrows():
    vals = ", ".join(f"{v:.6f}f" for v in row.values)
    print(f"// BENIGN sample\n{{{vals}}},")

for i, row in attack.iterrows():
    vals = ", ".join(f"{v:.6f}f" for v in row.values)
    print(f"// ATTACK sample\n{{{vals}}},")
PYEOF
```

**Test de cierre:** `make test-integ-xgboost-1` con scores ATTACK>0.5 + BENIGN<0.1 + gate explícito.

---

## PASO 3 — DEBT-XGBOOST-DDOS-001

**Dataset:** `ml-training/scripts/ddos_detection/ddos_detection_dataset.json` (27MB)
**Features:** 10 features `DDOS_FEATURES` (ver `DDOSFeatures.py`)
**Script a crear:** `ml-training/scripts/ddos_detection/train_xgboost_ddos.py`
**Patrón:** mismo que `train_xgboost_baseline.py` adaptado al JSON format + normalización MinMaxScaler
**Gate:** F1 + Precision superiores al RF `ddos_detection_model.pkl`
**Exports:** `xgboost_ddos.json` + `xgboost_ddos.ubj`

---

## PASO 4 — DEBT-XGBOOST-RANSOMWARE-001

**Dataset:** `ml-training/scripts/ransomware/data/network_guaranteed.csv` + `files_guaranteed.csv` + `processes_guaranteed.csv`
**Script a crear:** `ml-training/scripts/ransomware/train_xgboost_ransomware.py`
**Patrón:** mismo que `train_simple_effective.py` sustituyendo RF por XGBoost
**Gate:** F1 + Precision superiores al RF `simple_effective_model.pkl`
**Exports:** `xgboost_ransomware.json` + `xgboost_ransomware.ubj`

---

## PASO 5 — Extender make sign-models + tabla comparativa paper

- Extender `make sign-models` para firmar los 3 modelos
- `docs/xgboost/comparison-table.md`: latencia (μs), F1, Precision, ROC-AUC para RF vs XGBoost en los 3 detectores
- Separar §4 del paper: §4.1 CIC-IDS-2017 real + §4.2 DeepSeek sintético con limitaciones explícitas

---

## Contexto permanente

### Secuencia canónica DAY 120+
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Datasets origen (integridad científica)
- **level1_attack_detector** (RF + XGBoost): CIC-IDS-2017 REAL
- **ransomware_detector**: DeepSeek sintético — `data/*_guaranteed.csv`
- **ddos_detector**: DeepSeek sintético — `ddos_detection_dataset.json`
- **Paper §4**: SEPARAR explícitamente §4.1 real + §4.2 sintético con limitaciones

### Consejo DAY 120 — decisiones clave
- TEST-INTEG-XGBOOST-1 con scores reales BLOQUEANTE MERGE (7/7 unánime)
- Seed: NUNCA en CMake. Solo runtime mlock()+explicit_bzero()
- In-situ+BitTorrent: investigación Q3 2026, gates G1-G5 obligatorios
- Paper §4: separar real vs sintético o riesgo de rechazo

### NO mergear a main hasta
1. DEBT-XGBOOST-TEST-REAL-001 verde
2. DEBT-SEED-AUDIT-001 verde

### Regla de oro
6/6 RUNNING + make test-all VERDE

*"Via Appia Quality — un escudo, nunca una espada."*
MDEOF
cp /tmp/DAY121_prompt.md /Users/aironman/CLionProjects/test-zeromq-docker/docs/DAY121_prompt.md
echo "✅ Prompt DAY 121 creado"
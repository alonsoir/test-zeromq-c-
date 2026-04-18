Los tres en orden.

**Prompt DAY 122:**

```bash
cat > /Users/aironman/CLionProjects/test-zeromq-docker/docs/DAY122_prompt.md << 'MDEOF'
# ML Defender (aRGus NDR) — DAY 122 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA SCRIPTS:** Lógica compleja → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().

---

## Estado al cierre de DAY 121

### Hitos completados DAY 121
- **fix(provision)** ✅ — circular dependency plugin_signing.pk → plugin-loader cmake. Idempotencia × 3 certificada.
- **DEBT-SEED-AUDIT-001** ✅ — seed ChaCha20 no está en ningún CMakeLists.txt ni fuente C++.
- **DEBT-XGBOOST-TEST-REAL-001** ✅ — TEST-INTEG-XGBOOST-1 con flows reales FTP-Patator. BENIGN: 0.000111/0.000120/0.000228. ATTACK: 0.999894/0.999258/0.999904.
- **DEBT-XGBOOST-DDOS-001** ✅ — XGBoost DDoS F1=1.0 (sintético DeepSeek 50k). 20× más rápido que RF.
- **DEBT-XGBOOST-RANSOMWARE-001** ✅ — XGBoost Ransomware F1=0.9932 (sintético 3k). 6× más rápido que RF.
- **DEBT-SIGN-MODELS-EXTEND-001** ✅ — make sign-models firma 3 modelos Ed25519.
- **docs/xgboost/comparison-table.md** ✅ — tabla RF vs XGBoost × 3 detectores con latencias.
- **PAPER-SECTION-4** ✅ — §4.1 CIC-IDS-2017 real + §4.2 DeepSeek sintético con limitaciones.
- **deploy-models en make bootstrap paso 6/8** ✅ — idempotente.
- **make test-all VERDE** ✅
- **Commits:** hasta `55880c7c` · Branch: `feature/adr026-xgboost`

### Pubkey activa DAY 121
`fc895faac3e8c533d0cf4463637bbb1d2a3fb09dc6e84f7282dc427dd876f238`

### BLOQUEANTE ÚNICO DAY 122
**DEBT-PRECISION-GATE-001** 🔴 — XGBoost level1 Precision=0.9875 < 0.99 (gate médico ADR-026).

---

## PASO 0 — DAY 122: vagrant destroy OBLIGATORIO

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout feature/adr026-xgboost
git pull origin feature/adr026-xgboost
vagrant destroy -f && vagrant up
make bootstrap
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS"
```

---

## PASO 1 — DEBT-PRECISION-GATE-001 (BLOQUEANTE MERGE)

### Protocolo Consejo 7/7 unánime

**Regla de oro:** El test set Wednesday es BLIND. Se abre UNA SOLA VEZ para el reporte final.
**NUNCA** calibrar threshold sobre el test set (data snooping → paper inválido).

### Split temporal obligatorio
```
Train:      Tuesday + Thursday + Friday (CSVs CIC-IDS-2017)
Validation: 20% del train (stratificado) — para calibrar threshold y early stopping
Test:       Wednesday-WorkingHours.pcap_ISCX.csv — BLIND hasta evaluación final
```

### Secuencia de trabajo

**1. Preparar datasets**
```bash
python3 << 'PYEOF'
import pandas as pd
import numpy as np
import hashlib

BASE = "ml-training/datasets/CIC-IDS-2017/MachineLearningCVE"
files = {
    "Tuesday":   f"{BASE}/Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday": f"{BASE}/Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday":  f"{BASE}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday":    f"{BASE}/Friday-WorkingHours-Morning.pcap_ISCX.csv",
}
# Registrar md5 de Wednesday ANTES de abrirlo para entrenamiento
with open(files["Wednesday"], "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"Wednesday md5 (BLIND seal): {md5}")
for day, path in files.items():
    df = pd.read_csv(path, low_memory=False)
    print(f"{day}: {len(df)} rows, attacks={sum(df[' Label'].str.strip()!='BENIGN')}")
PYEOF
```

**2. Script de re-entrenamiento**
Crear `ml-training/scripts/train_xgboost_level1_v2.py` con:
- Features: LEVEL1_FEATURE_NAMES (23 features de `docs/xgboost/features.md`)
- Train: Tuesday + Thursday + Friday
- Validation: 20% del train (para early stopping + calibración threshold)
- `scale_pos_weight` = n_benign / n_attack
- `precision_recall_curve` sobre validation para encontrar threshold con Precision ≥ 0.99
- Threshold documentado y exportado junto al modelo

**3. Evaluación final en Wednesday (UNA SOLA VEZ)**
- Cargar modelo + threshold calibrado
- Evaluar sobre Wednesday completo
- Si Precision ≥ 0.99 → MERGE AUTORIZADO
- Si no → iterar hiperparámetros sin tocar Wednesday

**4. Exportar modelo**
- `xgboost_cicids2017_v2.json` + `xgboost_cicids2017_v2.ubj`
- `make sign-models` actualizado
- `TEST-INTEG-XGBOOST-1` actualizado con flows de Wednesday

### Gate de cierre
```
Precision ≥ 0.99 en Wednesday held-out
Recall ≥ 0.95 en Wednesday held-out
Latencia < 2 µs/sample
make test-all VERDE
```

---

## Contexto permanente

### Secuencia canónica DAY 121+
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Datasets CIC-IDS-2017 disponibles
```
ml-training/datasets/CIC-IDS-2017/MachineLearningCVE/
  Monday-WorkingHours.pcap_ISCX.csv       (solo BENIGN)
  Tuesday-WorkingHours.pcap_ISCX.csv      (FTP-Patator, SSH-Patator) ← train
  Wednesday-workingHours.pcap_ISCX.csv    (DoS, Heartbleed) ← TEST BLIND
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv ← train
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv ← train
  Friday-WorkingHours-Morning.pcap_ISCX.csv ← train
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv ← train
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv ← train
```

### Consejo DAY 121 — decisiones clave (7/7 unánime)
- Wednesday como held-out test set OBLIGATORIO — data snooping invalida el paper
- Threshold calibración SOLO en validation set, nunca en test
- RF level1 pkl no recuperar — RF sin split = overfitting, no es baseline válido
- 125 falsas alarmas/hora = alert fatigue clínico = sistema inutilizable
- Gate ≥0.99 inamovible — "La seguridad de los hospitales no admite atajos estadísticos"

### NO mergear a main hasta
1. **DEBT-PRECISION-GATE-001** verde — Precision ≥ 0.99 en Wednesday held-out

### Regla de oro
6/6 RUNNING + make test-all VERDE + Precision ≥ 0.99 en Wednesday

*"Via Appia Quality — un escudo, nunca una espada."*
MDEOF
echo "✅ Prompt DAY 122 creado"
```
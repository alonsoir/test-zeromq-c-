cat > /Users/aironman/CLionProjects/test-zeromq-docker/docs/DAY123_prompt.md << 'MDEOF'
# ML Defender (aRGus NDR) — DAY 123 Continuity Prompt

Buenos días Claude. Soy Alonso (aRGus NDR, ML Defender).

## POLÍTICA DE DEUDA TÉCNICA (leer antes de empezar)

- **Bloqueante:** debe cerrarse en esta feature. No hay merge a main sin test verde.
- **No bloqueante:** asignada a feature destino en BACKLOG. No toca esta feature.
- **Toda deuda tiene test de cierre.** Implementado sin test = no cerrado.
- **REGLA CRÍTICA:** El Vagrantfile y el Makefile son la única fuente de verdad.
- **REGLA SCRIPTS:** Lógica compleja → `tools/script.sh`. Nunca inline en Makefile.
- **REGLA SEED:** La seed ChaCha20 es material criptográfico secreto. NUNCA en CMake ni logs. Solo runtime: mlock() + explicit_bzero().

---

## Estado al cierre de DAY 122

### Hitos completados DAY 122
- **PHASE 4 COMPLETADA** ✅ — feature/adr026-xgboost mergeada a main como `v0.5.0-preproduction`
- **DEBT-PRECISION-GATE-001** ✅ — CERRADO CON HALLAZGO CIENTÍFICO (no con gate pasado)
- **train_xgboost_level1_v2.py** ✅ — split temporal Tue+Thu+Fri / val 20% / Wed BLIND
- **XGBoost in-distribution** ✅ — Precision=0.9945 / Recall=0.9818 / threshold=0.8211 / latencia=1.986µs
- **Wednesday OOD finding** ✅ — impossibility result sellado (md5=bf0dd7e9...). Covariate shift estructural CIC-IDS-2017 documentado. Consejo 7/7 unánime.
- **xgboost_cicids2017_v2.ubj.sig** ✅ — firmado Ed25519 via tools/sign-model.sh
- **Paper Draft v16** ✅ — arXiv:2604.04952 actualizado. §8 XGBoost eval + §10.13 + §11.18 ACRL + sommer2010 + caldera2024
- **BACKLOG.md + README.md** ✅ — actualizados con estado DAY 122
- **DEBT-PENTESTER-LOOP-001** 🔵 ABIERTA — próxima frontera

### Hallazgo científico DAY 122 (resumen ejecutivo)
Los datasets académicos (CIC-IDS-2017) son insuficientes como fuente única para
entrenar clasificadores NDR de producción. La separación temporal de attack types
(DoS Hulk/GoldenEye/Slowloris exclusivos de Wednesday, ausentes del train) crea
un covariate shift estructural que hace imposible generalizar independientemente
del algoritmo o hiperparámetros. Corroborado y cuantificado: threshold sweep
completo, ningún punto satisface Precision≥0.99 ∩ Recall≥0.95. Consejo 7/7:
hallazgo publicable, Sommer & Paxson 2010 citado.

### Tag activo
`v0.5.0-preproduction` — PRE-PRODUCTION. No desplegar en hospitales hasta ACRL.

### Pubkey activa DAY 122
Nueva keypair generada en vagrant destroy de DAY 122 (no hardcodeada — runtime).

---

## PASO 0 — DAY 123: verificar entorno

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git checkout main
git status
make pipeline-status
make test-all 2>&1 | grep -E "PASSED|FAILED|ALL TESTS|VERDE"
```

Si la VM está parada: `make up && make bootstrap`

---

## PASO 1 — Decisión de dirección DAY 123

### Opción A — Hotfix inmediato (30 min)
Añadir `pandas scikit-learn` al Vagrantfile provisioning step de Python.
Sin esto, un `vagrant destroy + bootstrap` falla en train_xgboost_level1_v2.py.

```bash
# Verificar que está en el Vagrantfile:
grep "pandas\|scikit" /Users/aironman/CLionProjects/test-zeromq-docker/Vagrantfile
# Si no está, añadirlo al paso de pip3 install
```

### Opción B — ADR-037 Snyk Hardening (P1)
23 vulnerabilidades C++ pendientes. Bloqueante para ADR-036 (verificación formal).
- F-001: command injection `validate_chain_name()` — regex allowlist
- F-002: path traversal config loading — `safe_resolve_config()`
- F-003: integer overflows — `std::numeric_limits<>`

### Opción C — ADR-038 ACRL Diseño (P3, próxima frontera)
Escribir el ADR formal del Adversarial Capture-Retrain Loop.
Especificar: arquitectura Caldera → captura → reentrenamiento → hot-swap.
Establecer gates mínimos de validación del dataset generado.

### Opción D — Actualizar LinkedIn/arXiv con el hallazgo
Redactar post técnico público sobre el Wednesday OOD finding.
Contacto con Sebastian Garcia (CTU Prague) sobre colaboración dataset.

---

## Contexto permanente

### Secuencia canónica DAY 122+
```bash
make up           # vagrant up
make bootstrap    # 8 pasos, todo automático
make test-all     # verificación completa
```

### Estado de los modelos firmados
```
/vagrant/ml-detector/models/production/level1/
  xgboost_cicids2017.ubj       + .sig  (v1, DAY 120)
  xgboost_cicids2017_v2.ubj    + .sig  (v2, DAY 122 — IN-DISTRIBUTION VALIDATED)
  wednesday_eval_report.json          (OOD finding sealed)
  xgboost_cicids2017_v2_threshold.json

/vagrant/ml-detector/models/production/level2/ddos/
  xgboost_ddos.ubj             + .sig  (DAY 121)

/vagrant/ml-detector/models/production/level3/ransomware_xgboost_v2/
  xgboost_ransomware.ubj       + .sig  (DAY 121)
```

### Paper arXiv:2604.04952
- Draft v16 submitted DAY 122: https://arxiv.org/submit/7495855/view
- Sección §8: XGBoost in-distribution + Wednesday OOD
- Sección §10.13: structural bias academic datasets
- Sección §11.18: ACRL (Adversarial Capture-Retrain Loop)

### DEBT-PENTESTER-LOOP-001 (próxima frontera)
Loop adversarial para generar datos fundacionales reales:
1. MITRE Caldera (ATT&CK-mapped, determinista) — Fase 1
2. Captura eBPF/XDP en entorno aislado → flows etiquetados ground-truth
3. Reentrenamiento XGBoost warm-start
4. Firma Ed25519 + hot-swap (ADR-025/026)
5. Validación en held-out del mismo entorno

Gates mínimos Caldera Fase 1:
- Seed fijo + versioned scenario (reproducibilidad)
- Ground-truth a nivel flow (logs Caldera → etiquetas)
- Cobertura ≥3 familias ATT&CK
- Tráfico RFC-válido (tshark validation, 0 malformed flows)
- Red sandbox aislada (no contaminar producción)

### Decisiones de diseño clave (inamovibles)
- Datasets académicos: solo útiles para bootstrap. Modelo fundacional requiere ACRL.
- Plugin XGBoost: PRE-PRODUCTION hasta ACRL completado. No desplegar en hospitales.
- ACRL Fase 1: MITRE Caldera (no IA generativa directa — demasiado no determinista en v1)
- Paper framing: hallazgo DAY 122 = validación de la arquitectura de reentrenamiento, no limitación del modelo

### NO mergear a main nuevas features hasta
1. `make test-all VERDE` en entorno limpio

### Regla de oro
6/6 RUNNING + make test-all VERDE + Precision ≥ 0.99 in-distribution (ya certificado)

*"Via Appia Quality — un escudo que aprende de su propia sombra."*
MDEOF
echo "✅ Prompt DAY 123 creado"
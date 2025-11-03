# 🔧 Troubleshooting - Level 2 DDoS Training

## ❌ Problema: "Terminado (killed)" - Out of Memory

### Causa
El dataset CIC-DDoS-2019 es enorme (~50M flows, varios GB). Si tu VM tiene poca RAM (4-6GB), se queda sin memoria al intentar cargar todo.

### ✅ Solución: Usar Script Optimizado

**Descarga el nuevo script:**
[train_level2_ddos_binary_optimized.py](computer:///mnt/user-data/outputs/train_level2_ddos_binary_optimized.py)

**Características:**
- ✅ Carga datos en chunks (no todo de golpe)
- ✅ Sampling inteligente estratificado
- ✅ Libera memoria agresivamente
- ✅ Configurable según tu RAM

---

## 🚀 QUICK START (Para VMs con poca RAM)

### 1. Verificar RAM disponible

```bash
# Ver RAM total de la VM
free -h

# Output esperado:
#               total        used        free
# Mem:           3.8Gi       1.2Gi       2.1Gi
```

### 2. Configurar sample size según tu RAM

Edita `train_level2_ddos_binary_optimized.py` línea ~30:

```python
# Para VM con 4GB RAM:
SAMPLE_SIZE = 200000  # 200k flows (~300MB)

# Para VM con 6GB RAM:
SAMPLE_SIZE = 400000  # 400k flows (~600MB)

# Para VM con 8GB+ RAM:
SAMPLE_SIZE = 800000  # 800k flows (~1.2GB)

# Si quieres TODO el dataset (requiere 16GB+ RAM):
SAMPLE_SIZE = None
```

### 3. Entrenar con script optimizado

```bash
cd /vagrant/ml-training
source .venv/bin/activate

# Instalar dependencia extra si no está
pip install psutil

# Ejecutar script optimizado
python scripts/train_level2_ddos_binary_optimized.py
```

**Tiempo estimado:**
- 200k samples: ~3-5 minutos
- 400k samples: ~8-12 minutos
- 800k samples: ~15-20 minutos

---

## 📊 Comparación de Configuraciones

| RAM VM | SAMPLE_SIZE | Tiempo | Accuracy Esperada |
|--------|-------------|--------|-------------------|
| 4GB    | 200,000     | 3-5min | 96-97% |
| 6GB    | 400,000     | 8-12min | 97-98% |
| 8GB    | 800,000     | 15-20min | 98%+ |
| 16GB+  | None (todo) | 30-40min | 98-99% |

**Nota:** Con 200k-400k samples ya obtienes un modelo muy bueno (>96% accuracy).

---

## 🔍 Monitoreo Durante Entrenamiento

El script optimizado muestra el uso de memoria en cada paso:

```
📊 CARGANDO CIC-DDoS-2019 DATASET (OPTIMIZADO)
✅ Archivos encontrados: 18
🎯 Target sample size: 300,000 flows
📦 Max rows per file: 50,000

[1/18] DrDoS_DNS.csv                   ✅  16,667 flows (total: 16,667)
[2/18] DrDoS_LDAP.csv                  ✅  16,667 flows (total: 33,334)
...
💾 Memoria usada: 456.2 MB

🧹 PREPROCESAMIENTO
...
💾 Memoria usada: 523.8 MB

🌲 ENTRENAMIENTO RANDOM FOREST
...
💾 Memoria usada: 1,234.5 MB
```

Si ves que la memoria crece mucho (>90% de tu RAM total), **reduce SAMPLE_SIZE**.

---

## ⚙️ Otras Optimizaciones

### Opción 1: Aumentar RAM de la VM

Edita `Vagrantfile`:

```ruby
config.vm.provider "virtualbox" do |vb|
  vb.memory = "8192"  # Aumentar de 6GB a 8GB
  vb.cpus = 4         # Más CPUs también ayuda
end
```

Luego:
```bash
vagrant reload
```

### Opción 2: Usar Swap (Último Recurso)

Si no puedes aumentar RAM:

```bash
# Crear swap de 4GB
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verificar
free -h
```

**Advertencia:** Swap es MUY lento. Solo úsalo como último recurso.

### Opción 3: Entrenar en Host (macOS)

Si tu Mac tiene más RAM:

```bash
# Desde macOS (no la VM)
cd ~/Code/test-zeromq-docker/ml-training
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn joblib imbalanced-learn psutil

# Ejecutar
python scripts/train_level2_ddos_binary_optimized.py
```

---

## 🐛 Otros Errores Comunes

### Error: `ModuleNotFoundError: No module named 'psutil'`

```bash
pip install psutil
```

### Error: `ModuleNotFoundError: No module named 'imblearn'`

```bash
pip install imbalanced-learn
```

### Error: Dataset no encontrado

```bash
# Verificar que el dataset está descargado
ls -lh /vagrant/ml-training/datasets/CIC-DDoS-2019/

# Si no existe, descargar:
# https://www.unb.ca/cic/datasets/ddos-2019.html
```

### Error: "sklearn version mismatch"

```bash
pip install --upgrade scikit-learn
```

### Warning: "Using n_jobs=-1 can exhaust memory"

Es normal. El script optimizado usa `n_jobs=2` para evitar esto.

---

## 📈 ¿Cómo Saber si el Modelo es Bueno?

Métricas mínimas aceptables:

- ✅ **Accuracy > 95%**: Bueno para producción
- ✅ **Precision > 94%**: Pocos falsos positivos
- ✅ **Recall > 95%**: Detecta la mayoría de ataques
- ✅ **F1-Score > 94%**: Balance precision/recall
- ✅ **ROC AUC > 0.98**: Excelente separación de clases

Si tus métricas son inferiores:
1. Aumenta `SAMPLE_SIZE`
2. Verifica que no haya errores en los logs
3. Revisa que el dataset esté bien descargado

---

## 🎯 Próximos Pasos Después de Entrenar

1. **Convertir a ONNX:**
   ```bash
   python scripts/convert_level2_ddos_to_onnx.py
   ```

2. **Copiar a ml-detector:**
   ```bash
   cp outputs/onnx/level2_ddos_binary_detector.onnx \
      ../ml-detector/models/production/level2/
   ```

3. **Actualizar config JSON**

4. **Integrar en C++**

---

## 💡 Tips de Performance

### Para Entrenar Más Rápido

1. **Reducir árboles del RandomForest:**
   ```python
   # Línea ~280 del script optimizado
   n_estimators=50,  # En vez de 100
   ```

2. **Usar menos features:**
   ```python
   # Línea ~38 del script optimizado
   # Comentar algunas features menos importantes
   ```

3. **Desactivar SMOTE:**
   ```python
   # Línea ~380 del script optimizado
   use_smote=False
   ```

### Para Mejor Accuracy

1. **Aumentar sample size**
2. **Más árboles:** `n_estimators=150`
3. **Mayor profundidad:** `max_depth=25`

---

## 📞 ¿Más Problemas?

1. **Ver logs completos:**
   ```bash
   python scripts/train_level2_ddos_binary_optimized.py 2>&1 | tee training.log
   ```

2. **Monitorear memoria en tiempo real:**
   ```bash
   # Terminal 1:
   python scripts/train_level2_ddos_binary_optimized.py
   
   # Terminal 2:
   watch -n 1 free -h
   ```

3. **Verificar que no hay otros procesos consumiendo RAM:**
   ```bash
   htop  # o: top
   ```

---

## ✅ Checklist Pre-Training

- [ ] VM tiene al menos 4GB RAM (`free -h`)
- [ ] Dataset descargado (`ls datasets/CIC-DDoS-2019/`)
- [ ] Virtualenv activado (`which python` → debe mostrar .venv)
- [ ] Dependencias instaladas (`pip list | grep -E "sklearn|pandas|imblearn"`)
- [ ] `SAMPLE_SIZE` configurado según tu RAM
- [ ] Directorio `outputs/` existe y tiene permisos de escritura

---

**¿Todo configurado? ¡A entrenar! 🚀**

```bash
python scripts/train_level2_ddos_binary_optimized.py
```

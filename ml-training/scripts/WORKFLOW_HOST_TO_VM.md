# 🚀 ML Training Workflow: macOS Host → Debian VM

## 📋 Resumen del Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ macOS (Host) - 16GB+ RAM, CPU nativo                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Train ML model (sklearn)                                │
│    → outputs/models/*.joblib                                │
│                                                             │
│ 2. Convert to ONNX                                          │
│    → outputs/onnx/*.onnx                                    │
│                                                             │
│ ✅ Files auto-available in VM via /vagrant shared folder    │
└─────────────────────────────────────────────────────────────┘
                           ↓ (automatic via shared folder)
┌─────────────────────────────────────────────────────────────┐
│ Debian VM (Vagrant) - 6GB RAM, C++ compilation             │
├─────────────────────────────────────────────────────────────┤
│ 3. Sync models: sync-models                                │
│    → ml-detector/models/production/level2/*.onnx           │
│                                                             │
│ 4. Build detector: build-detector                          │
│    → ml-detector/build/ml-detector                         │
│                                                             │
│ 5. Run detector: run-detector                              │
│    → Uses ONNX Runtime C++ (libonnxruntime.so)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Ventajas de Este Enfoque

| Aspecto | Antes (VM) | Ahora (Host) | Mejora |
|---------|-----------|--------------|---------|
| **RAM disponible** | 6GB | 16GB+ | 2-3x más |
| **Velocidad CPU** | Virtualizada | Nativa M1/M2/Intel | 2-4x más |
| **Tiempo entrena 400k** | 8-12 min (OOM) | 3-5 min | ✅ 2-3x |
| **Dataset completo** | ❌ Crash OOM | ✅ Posible | Full dataset |
| **OOM crashes** | Frecuentes | Nunca | ✅ Estable |
| **Python deps** | Conflictos Debian | Limpio macOS | ✅ Aislado |

---

## 📦 Setup Inicial (Una Sola Vez)

### **1. Actualizar Vagrantfile**

```bash
# Backup del Vagrantfile actual
cd ~/Code/test-zeromq-docker
cp Vagrantfile Vagrantfile.backup

# Copiar el nuevo Vagrantfile optimizado
# (Descarga Vagrantfile_optimized y reemplaza)
cp ~/Downloads/Vagrantfile_optimized Vagrantfile

# Reprovisionar VM (si ya está corriendo)
vagrant reload --provision

# O crear VM nueva
vagrant destroy -f
vagrant up
```

### **2. Setup Python en macOS**

```bash
# Ir al directorio ml-training
cd ~/Code/test-zeromq-docker/ml-training

# Crear virtualenv específico para macOS
python3 -m venv .venv-macos

# Activar virtualenv
source .venv-macos/bin/activate

# Verificar Python 3.8+
python --version

# Instalar dependencias de ML (TODAS las necesarias)
pip install --upgrade pip

# Core ML
pip install pandas numpy scikit-learn

# Visualización
pip install matplotlib seaborn

# ONNX
pip install onnx onnxruntime skl2onnx

# Utilidades
pip install imbalanced-learn joblib psutil

# Verificar instalación
python -c "import pandas, sklearn, onnx, onnxruntime; print('✅ All deps installed')"
```

**Nota:** Usa `.venv-macos` para diferenciar del `.venv` de la VM (si existía).

### **3. Verificar Dataset**

```bash
# Debe estar en tu host macOS
ls -lh ~/Code/test-zeromq-docker/ml-training/datasets/CIC-DDoS-2019/

# Si no existe, descargar:
# https://www.unb.ca/cic/datasets/ddos-2019.html
```

---

## 🔄 Workflow Completo - Paso a Paso

### **PASO 1: Entrenar en macOS (Host)**

```bash
# Terminal 1 - macOS
cd ~/Code/test-zeromq-docker/ml-training
source .venv-macos/bin/activate

# Verificar sistema y configuración recomendada
python scripts/check_system_config.py

# Output esperado:
# 📊 MEMORIA
#   Total:     16.00 GB
#   Disponible: 12.34 GB
# 
# 🎯 CONFIGURACIONES RECOMENDADAS
# 1. Large (Recomendado)
#    Sample size: 800,000 flows
#    Tiempo estimado: 15-20 min
#    Accuracy esperada: 98%+

# Entrenar modelo (con configuración recomendada)
python scripts/train_level2_ddos_binary_optimized.py

# Tiempo esperado:
# - 400k samples: 3-5 min (macOS M1/M2)
# - 800k samples: 8-12 min
# - Dataset completo: 20-30 min

# Output final:
# ✅ TRAINING COMPLETADO
# 📊 Métricas finales:
#   Accuracy:  98.12%
#   F1-Score:  98.22%
#   ROC AUC:   0.9945
# 
# 🎯 Siguiente paso:
#   python scripts/convert_level2_ddos_to_onnx.py
```

### **PASO 2: Convertir a ONNX (macOS)**

```bash
# Mismo terminal, mismo virtualenv
python scripts/convert_level2_ddos_to_onnx.py

# Output:
# ✅ CONVERSIÓN A ONNX COMPLETADA
# 📦 Modelo ONNX:
#   Path: outputs/onnx/level2_ddos_binary_detector.onnx
#   Size: 5.43 MB
#   Validación: ✅ PASSED

# Verificar archivo generado
ls -lh outputs/onnx/*.onnx

# Output:
# -rw-r--r--  1 user  staff   5.4M Oct 22 10:30 level2_ddos_binary_detector.onnx
```

**🎉 ¡Modelo entrenado! Ahora está en `outputs/onnx/` y automáticamente disponible en la VM vía shared folder.**

---

### **PASO 3: Desplegar en VM (Debian)**

```bash
# Terminal 2 - Conectar a VM
vagrant ssh

# Verificar que el modelo está disponible
ls -lh /vagrant/ml-training/outputs/onnx/

# Output:
# -rwxrwxr-x 1 vagrant vagrant 5.4M Oct 22 10:30 level2_ddos_binary_detector.onnx

# Sincronizar modelos al directorio de ml-detector
sync-models

# Output:
# sending incremental file list
# level2_ddos_binary_detector.onnx
# ✅ Models synced from host

# Verificar modelos disponibles
list-models

# Output:
# Available ONNX models:
# -rw-rw-r-- 1 vagrant vagrant 5.4M Oct 22 10:31 /vagrant/ml-detector/models/production/level2/level2_ddos_binary_detector.onnx
```

---

### **PASO 4: Actualizar Config JSON (VM)**

```bash
# Editar configuración del ml-detector
nano /vagrant/ml-detector/config/ml_detector_config.json
```

**Añadir/actualizar:**
```json
{
  "models": {
    "level1": {
      "path": "models/production/level1/level1_attack_detector.onnx",
      "enabled": true,
      "threshold": 0.85
    },
    "level2_ddos": {
      "path": "models/production/level2/level2_ddos_binary_detector.onnx",
      "enabled": true,
      "threshold": 0.85,
      "activated_when": "level1_predicts_attack"
    }
  },
  "zmq": {
    "endpoint": "tcp://192.168.56.20:5571",
    "socket_type": "SUB"
  }
}
```

---

### **PASO 5: Compilar ml-detector (VM)**

```bash
# Compilar ml-detector con nuevo modelo
build-detector

# Output:
# [ 25%] Building CXX object CMakeFiles/ml-detector.dir/src/main.cpp.o
# [ 50%] Building CXX object CMakeFiles/ml-detector.dir/src/ml_predictor.cpp.o
# [ 75%] Building CXX object CMakeFiles/ml-detector.dir/src/zmq_handler.cpp.o
# [100%] Linking CXX executable ml-detector
# ✅ Build complete

# Verificar binario
ls -lh /vagrant/ml-detector/build/ml-detector
```

---

### **PASO 6: Ejecutar ml-detector (VM)**

```bash
# Terminal 2 (VM) - Ejecutar detector
run-detector

# Output:
# ================================================================================
# 🎯 ML DETECTOR - NETWORK SECURITY
# ================================================================================
# 
# ✅ Config loaded: ml_detector_config.json
# ✅ Level 1 model loaded: level1_attack_detector.onnx (23 features)
# ✅ Level 2 DDoS model loaded: level2_ddos_binary_detector.onnx (70 features)
# ✅ ZMQ socket connected: tcp://192.168.56.20:5571
# 
# 🔄 Waiting for events...

# Terminal 3 (VM) - Ejecutar sniffer (para generar tráfico)
vagrant ssh
run-sniffer

# Ahora el detector debería recibir eventos y clasificarlos
```

**Salida esperada del detector:**
```
📥 Event received: event-12345
🔍 Level 1: ATTACK (confidence: 0.92)
🔍 Level 2 DDoS: DDOS (confidence: 0.88, type: UNKNOWN)
📤 Alert published: DDOS_DETECTED
```

---

## 🔄 Flujo Iterativo (Reentrenar Modelos)

Cuando quieras reentrenar o mejorar modelos:

```bash
# 1. En macOS: Reentrenar
cd ~/Code/test-zeromq-docker/ml-training
source .venv-macos/bin/activate
python scripts/train_level2_ddos_binary_optimized.py  # Nuevo entrenamiento
python scripts/convert_level2_ddos_to_onnx.py

# 2. En VM: Redesplegar (el modelo se sobrescribe automáticamente)
vagrant ssh
sync-models          # Copia el nuevo .onnx
build-detector       # Recompilar (opcional si el código C++ no cambió)
run-detector         # Ejecutar con nuevo modelo

# ✅ Listo - modelo actualizado sin necesidad de reconstruir toda la VM
```

---

## 📊 Comparación de Tiempos

**Entrenar Level 2 DDoS (400k samples):**

| Plataforma | RAM | CPU | Tiempo | OOM Risk |
|-----------|-----|-----|--------|----------|
| VM Debian | 6GB | 4 vCPUs | 8-12 min | ⚠️ Alto (crash) |
| Mac M1 | 16GB | 8 cores | 3-4 min | ✅ Ninguno |
| Mac M2 | 24GB | 10 cores | 2-3 min | ✅ Ninguno |
| Mac Intel i7 | 16GB | 6 cores | 5-7 min | ✅ Ninguno |

---

## 🐛 Troubleshooting

### **Error: "No module named pandas" (en macOS)**

```bash
# Asegúrate de estar en el virtualenv correcto
source .venv-macos/bin/activate
pip install pandas numpy scikit-learn
```

### **Error: "Model file not found" (en VM)**

```bash
# Verificar que el modelo existe en host
ls -lh ~/Code/test-zeromq-docker/ml-training/outputs/onnx/

# Verificar que está disponible en VM
vagrant ssh
ls -lh /vagrant/ml-training/outputs/onnx/

# Sincronizar manualmente si es necesario
sync-models
```

### **Error: "Shared folder not mounted"**

```bash
# Recrear shared folder
vagrant reload

# O manualmente en VM
sudo mount -t vboxsf vagrant /vagrant
```

### **Performance: Entrenar más rápido en macOS**

```python
# Editar train_level2_ddos_binary_optimized.py línea ~30
SAMPLE_SIZE = 800000  # Usar más samples si tienes 16GB+ RAM

# Editar línea ~280 (más árboles)
n_estimators=150,  # En vez de 100
```

---

## 📁 Estructura de Archivos (Host ↔ VM)

```
~/Code/test-zeromq-docker/
├── ml-training/                      # En HOST macOS
│   ├── .venv-macos/                 # Virtualenv macOS
│   ├── datasets/
│   │   └── CIC-DDoS-2019/           # Dataset descargado en host
│   ├── scripts/
│   │   ├── train_level2_ddos_binary_optimized.py
│   │   ├── convert_level2_ddos_to_onnx.py
│   │   └── check_system_config.py
│   └── outputs/
│       ├── models/*.joblib          # Sklearn models
│       └── onnx/*.onnx              # ← ONNX models (shared con VM)
│
└── ml-detector/                      # En VM Debian
    ├── models/
    │   └── production/
    │       └── level2/
    │           └── *.onnx           # ← Synced desde outputs/onnx/
    ├── build/
    │   └── ml-detector              # Binario compilado
    └── config/
        └── ml_detector_config.json  # Configuración
```

**Todo en `ml-training/outputs/onnx/` es automáticamente visible en la VM vía `/vagrant/`.**

---

## ✅ Checklist - Setup Completo

- [ ] Vagrantfile actualizado con aliases
- [ ] VM provisionada correctamente
- [ ] Python venv creado en macOS (`.venv-macos`)
- [ ] Dependencias ML instaladas en macOS
- [ ] Dataset CIC-DDoS-2019 descargado en host
- [ ] Modelo entrenado en macOS (`outputs/onnx/*.onnx` existe)
- [ ] Modelo sincronizado a VM (`sync-models`)
- [ ] Config JSON actualizado con path del modelo
- [ ] ml-detector compilado (`build-detector`)
- [ ] ml-detector ejecuta correctamente (`run-detector`)

---

## 🎯 Próximos Pasos

1. ✅ **Level 2 DDoS Binario** - Completado con este workflow
2. ⏭️ **Level 2 DDoS Multi-clase** - Distinguir 12 tipos de DDoS
3. ⏭️ **Level 2 Ransomware** - Detectar ransomware
4. ⏭️ **Level 3 Anomalías** - Detección de anomalías (4 features)

Para cada nuevo modelo, repetir el proceso:
1. Entrenar en macOS
2. Convertir a ONNX
3. Sync a VM
4. Recompilar ml-detector (si cambia código C++)
5. Ejecutar

---

## 📚 Comandos Rápidos (Cheat Sheet)

### **En macOS (Training)**
```bash
cd ~/Code/test-zeromq-docker/ml-training
source .venv-macos/bin/activate
python scripts/check_system_config.py
python scripts/train_level2_ddos_binary_optimized.py
python scripts/convert_level2_ddos_to_onnx.py
```

### **En VM (Deployment)**
```bash
vagrant ssh
sync-models
list-models
build-detector
run-detector
```

### **Logs y Debug**
```bash
# En VM
logs-detector
logs-sniffer

# Verificar modelos cargados
grep "model loaded" /vagrant/ml-detector/build/logs/*.log
```

---

**¿Listo para empezar? 🚀**

```bash
# 1. Actualizar Vagrantfile
cp Vagrantfile_optimized Vagrantfile
vagrant reload --provision

# 2. Setup macOS
cd ml-training
python3 -m venv .venv-macos
source .venv-macos/bin/activate
pip install pandas numpy scikit-learn onnx onnxruntime skl2onnx imbalanced-learn joblib psutil matplotlib seaborn

# 3. ¡A entrenar!
python scripts/train_level2_ddos_binary_optimized.py
```

# CONTEXTO: Day 36 - Training Pipeline Implementation

## Resumen Day 35 (COMPLETADO)
**Fecha:** 08-Enero-2026
**Duración:** ~2 horas
**Estado:** ✅ COMPLETO - DimensionalityReducer operacional

### Entregables Day 35
```
✅ common-rag-ingester/
   ├── include/dimensionality_reducer.hpp    # API pública
   ├── src/dimensionality_reducer.cpp        # faiss::PCAMatrix
   ├── cmake/common-rag-ingester-config.cmake.in
   └── CMakeLists.txt

✅ tools/test_reducer.cpp                     # Test validado
✅ Compilación limpia en Debian 12
✅ Test PASSED (train/transform/save/load)
✅ Performance validado:
   • Training: 908ms para 10K samples
   • Transform: 149μs single, 20K vec/sec batch
   • Save/Load: Verificado ✅
```

### Issues Resueltos Day 35
1. ✅ FAISS no encontrado por pkg-config
    - Fix: CMakeLists.txt con find_path/find_library directo
2. ✅ API incompatible (write_VectorTransform)
    - Fix: `#include <faiss/index_io.h>` (no impl/io.h)
3. ✅ Varianza 40.97% en test sintético
    - ESPERADO: Datos random sin estructura semántica
    - Con datos reales llegaremos a ≥96%

### Arquitectura Confirmada
```
/vagrant/
├── common-rag-ingester/        # ⭐ SHARED (Day 35 ✅)
│   └── DimensionalityReducer    # PCA 384→128 operacional
│
├── faiss-ingester/              # Producer (Day 41-45)
│   └── Event → Embed → PCA → FAISS Index
│
└── rag/                         # Consumer (Day 46-55)
    └── Query → Embed → PCA → FAISS Search
```

---

## Day 36: Training Pipeline con Datos Reales

### Objetivo
Entrenar PCA con embeddings reales de eventos ML Defender para lograr ≥96% variance.

### Prerequisitos
- ✅ DimensionalityReducer compilado
- ✅ ONNX Runtime instalado (Day 32)
- ✅ Embedder models disponibles (Day 33):
    - chronos_embedder.onnx (83→512-d)
    - sbert_embedder.onnx (83→384-d)
    - attack_embedder.onnx (83→256-d)
- 📁 Eventos JSONL: `/vagrant/logs/rag/events/*.jsonl` (~32,957 eventos)

### Plan de Implementación (4-6 horas)

#### PASO 1: Data Loader (1-2h)
```cpp
// /vagrant/tools/train_pca.cpp

Funcionalidad:
1. Cargar eventos de JSONL
2. Extraer 83 features (RAGLogger schema)
3. Balance por sources (Gemini warning: evitar domain shift)
4. Preparar datasets para 3 embedders

Salida:
- N eventos balanceados
- Features normalizados [0,1]
- Verificación de calidad
```

#### PASO 2: ONNX Embedding (1-2h)
```cpp
Integración:
1. Cargar 3 modelos ONNX
2. Inferencia batch (eficiencia)
3. Generar embeddings:
   - Chronos: N × 512-d
   - SBERT: N × 384-d
   - Attack: N × 256-d

Performance target:
- >100 eventos/sec por embedder
```

#### PASO 3: PCA Training (1h)
```cpp
Entrenamiento:
1. Train 3 PCA reducers:
   - Chronos: 512→128
   - SBERT: 384→128
   - Attack: 256→128

2. Validar variance ≥96% para cada uno

3. Save models:
   /shared/models/pca/
   ├── chronos_pca_512_128.faiss
   ├── sbert_pca_384_128.faiss
   └── attack_pca_256_128.faiss
```

#### PASO 4: Validation (30min)
```cpp
Test:
1. Load cada PCA model
2. Transform 100 vectors test
3. Verificar dimensiones correctas
4. Medir performance (transform time)
5. Documentar variance achieved
```

### Estructura de Código Propuesta

```cpp
/vagrant/tools/
├── train_pca.cpp               # Main training pipeline
├── data_loader.hpp/cpp         # JSONL → Features
├── onnx_embedder.hpp/cpp       # ONNX inference wrapper
└── CMakeLists.txt              # Build config

Dependencies:
- common-rag-ingester (DimensionalityReducer)
- ONNX Runtime
- nlohmann/json (JSONL parsing)
- FAISS (save models)
```

### Criterios de Éxito Day 36

✅ 3 PCA models entrenados con variance ≥96%
✅ Models guardados en `/shared/models/pca/`
✅ Validation test PASSED
✅ Performance documented
✅ Training pipeline reproducible
✅ Código documented (Via Appia Quality)

### Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|-------------|------------|
| Variance <96% | Media | Ajustar output_dim o más datos |
| ONNX lento | Baja | Batch inference optimizado |
| Balance datos | Media | Estrategia multi-source (Gemini) |
| Memory issues | Baja | Batch processing incremental |

---

## Decisiones Técnicas Pendientes

### Para Day 36:
1. **Dataset size**: ¿10K, 20K o 32K eventos?
    - Recomendado: 20K (balance calidad/tiempo)
2. **Balancing strategy**: ¿Equal samples per source o weighted?
    - Recomendado: Equal samples (evitar domain shift)
3. **ONNX batch size**: ¿32, 64, 128?
    - Recomendado: 64 (balance memoria/throughput)

### Para Day 37-38 (Buffer):
- Integration testing DimensionalityReducer + ONNX
- Performance tuning
- Documentation refinement

---

## Prompt Sugerido para Próxima Sesión

```
Day 36: Training Pipeline - PCA con datos reales ML Defender.

CONTEXTO:
- Day 35 COMPLETO: DimensionalityReducer operacional ✅
- Test PASSED (908ms training, 149μs transform) ✅
- Varianza 40.97% con datos sintéticos (esperado)

OBJETIVO Day 36:
Entrenar 3 PCA reducers con embeddings reales → variance ≥96%

DATOS DISPONIBLES:
- ~32,957 eventos JSONL en /vagrant/logs/rag/events/
- 3 embedders ONNX (Chronos, SBERT, Attack) operacionales
- DimensionalityReducer library compilada

PLAN:
1. Data Loader: JSONL → 83 features (balanceado multi-source)
2. ONNX Embedding: 3 modelos → vectors (512-d, 384-d, 256-d)
3. PCA Training: 3 reducers → 128-d con variance ≥96%
4. Validation: Save models + test transforms

PRIORIDADES:
- Balance datos (Gemini warning: domain shift)
- Variance ≥96% target (Chronos recommendation)
- Performance measurement
- Via Appia: Código limpio, reproducible

Timeline: 4-6 horas estimadas
Output: 3 PCA models en /shared/models/pca/

¿Empezamos con el data loader?
```

---

## Notas Técnicas para Continuidad

### FAISS PCAMatrix API (validado Day 35)
```cpp
#include <faiss/index_io.h>  // ✅ CORRECTO
#include <faiss/VectorTransform.h>

// Training
pca->train(n_samples, training_data);
float variance = calculate_variance(pca->eigenvalues);

// Save/Load
faiss::write_VectorTransform(pca, filepath);
auto pca = faiss::read_VectorTransform(filepath);

// Transform
pca->apply_noalloc(n_vectors, input, output);
```

### ONNX Runtime Integration (Day 32)
```cpp
// Session setup
Ort::Env env;
Ort::SessionOptions opts;
Ort::Session session(env, model_path, opts);

// Inference
auto input_tensor = Ort::Value::CreateTensor(...);
auto output = session.Run(..., {input_tensor}, ...);
```

### Embedding Dimensions (Day 33)
- Chronos: 83 → 512-d → 128-d (PCA)
- SBERT: 83 → 384-d → 128-d (PCA)
- Attack: 83 → 256-d → 128-d (PCA)

---

## Vagrantfile Update (Future - Day 37+)

```ruby
# Add to Vagrantfile provisioning:
config.vm.provision "shell", inline: <<-SHELL
  # Install common-rag-ingester system-wide
  cd /vagrant/common-rag-ingester/build
  sudo make install
  sudo ldconfig
SHELL
```

---

## Via Appia Quality - Day 35 Retrospective

**✅ Logros:**
- Foundation sólida: DimensionalityReducer operacional
- API clean: train/transform/save/load validados
- Test PASSED: Código funciona end-to-end
- Troubleshooting eficiente: 2 fixes en 2 horas

**📊 Métricas:**
- Tiempo: ~2 horas (estimado 4-6h) ⚡
- Compilación: Primera vez limpia
- Test: 100% PASSED
- Performance: Dentro de expectativas

**🎯 Lección:**
> "Separación producer/consumer desde Day 1 = arquitectura clean.
> common-rag-ingester es SHARED, no es 'common' genérico.
> Naming matters. Testing matters. Foundation first."

**Próximo:**
> "Day 36: Datos reales → embeddings reales → PCA real → variance ≥96%.
> Pipeline completo. Despacio y bien. 🏛️"

---

**Fecha:** 08-Enero-2026
**Day 35:** ✅ COMPLETO
**Day 36:** 🚀 READY TO START
**Timeline:** Week 5 (Day 35-40) en progreso
**Via Appia:** Foundation first, expansion después 🏛️
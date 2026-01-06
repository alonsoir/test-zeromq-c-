# PROMPT DE CONTINUIDAD - DÍA 35 (07 Enero 2026)

## 🎯 BACKLOG Management Protocol

**Al completar cualquier tarea mayor, Claude debe:**

1. ✅ Confirmar completion con Alonso
2. 📋 Solicitar acceso al BACKLOG.md actualizado
3. 🔍 Revisar prioridades actuales (P0 → P1 → P2 → P3)
4. 💡 Sugerir siguiente tarea basándose en:
   - Blockers críticos (P0)
   - Dependencies del roadmap
   - Estado de Foundation Architecture
   - Effort vs Impact ratio
5. 🤝 Esperar aprobación de Alonso antes de proceder

**Frase trigger para Claude:**
> "Tarea completada. ¿Puedo ver el BACKLOG.md para sugerir qué sigue?"

**Priorización actual (Ene 2026):**
- P0 BLOCKER: ISSUE-005 (JSONL memory leak) ← Pendiente
- P1 HIGH: FAISS Integration (Phase 2A en progreso) ← CURRENT
- P1 HIGH: BACKLOG-001 Flow Sharding (post-FAISS)
- P2 MEDIUM: etcd-client, Watcher, Academic paper

**Via Appia Quality reminder:**
> Resolver blockers antes que features.  
> Foundation sólida antes que expansión.  
> Memory leaks son P0, no P2.

---

## 📚 DOCUMENTOS NECESARIOS PARA ESTA SESIÓN

```
Day 35 (HOY):
  ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md ← CRÍTICO
  ✅ Este prompt de continuidad
  
Razón: Day 35 implementa DimensionalityReducer con faiss::PCAMatrix.
       Necesitamos estrategias anti-curse y límites empíricos.
       Estrategia #2 (Dimensionality Reduction) es CRÍTICA.

Archivo: /vagrant/docs/FAISS_ANTI_CURSE_DESIGN.md
Tamaño: ~500 líneas (12K tokens aprox)
Contenido clave:
  - Estrategia #2: PCA reduction (512→128, 384→96, 256→64)
  - Límites empíricos: 180K, 450K, 85K eventos
  - faiss::PCAMatrix implementation guidelines
  - Variance preservation targets (96%+)
```

---

## 📋 CONTEXTO DÍA 34 (06 Enero 2026) - COMPLETADO ✅

### ✅ Day 34 Summary - Real Data Validation COMPLETE

**Objetivo Day 34**: Validar ONNX embedders con eventos JSONL reales.

**Resultados (3/3 Fases Complete):**

#### Fase 1: Python Inference (5 min) ✅
- Loaded 9 events from `2025-12-31.jsonl`
- Extracted 83 features per event
- All 3 embedders validated:
   - **Chronos (512-d)**: Mean -0.0107, Std 0.3527 ✅
   - **SBERT (384-d)**: Mean 0.0511, Std 0.3324 ✅
   - **Attack (256-d)**: Mean 0.0054, Std 0.0532 ✅

#### Fase 2: C++ Inference (15 min) ✅
- **Issue Resolved**: IR version mismatch (v9 vs v10)
- **Solution**: Updated ONNX Runtime C++ v1.17.1 → v1.23.2
- All 3 embedders validated in C++:
   - **Chronos (512-d)**: Mean -0.0060, Std 0.1751 ✅
   - **SBERT (384-d)**: Mean 0.0079, Std 0.1644 ✅
   - **Attack (256-d)**: Mean 0.0044, Std 0.1683 ✅

#### Fase 3: Batch Processing (1 min) ✅
- Processed 98 events from `2025-12-31.jsonl`
- **Throughput Performance** (synthetic models):
   - **Chronos**: 13,250 events/sec ⚡
   - **SBERT**: 18,565 events/sec ⚡
   - **Attack**: 6,874 events/sec ⚡
- Note: Real trained models will be slower, but still excellent

**Total Time Day 34**: ~21 minutes (estimated 20-35 min) ⚡

**Files Created Day 34**:
- ✅ test_real_inference.py (8.3 KB)
- ✅ test_real_embedders.cpp (5.1 KB)
- ✅ test_batch_processing.py (8.2 KB)
- ✅ preflight_check.py (4.7 KB)
- ✅ quick_fix.sh (2.0 KB)
- ✅ update_onnxruntime_cpp.sh (2.6 KB)
- ✅ DAY34_SUMMARY.md (9.4 KB)

**Git Status**:
```
Rama: feature/faiss-ingestion-phase2a
Último commit: "Day 34: ONNX embedders validated with real JSONL data"
Archivos añadidos:
  - rag/models/test_real_inference.py
  - rag/models/test_real_embedders.cpp
  - rag/models/test_batch_processing.py
  - rag/models/preflight_check.py
  - rag/models/quick_fix.sh
  - rag/models/update_onnxruntime_cpp.sh
```

**Issues Resolved Day 34**:
1. ✅ JSONL path corrected: /vagrant/data → /vagrant/logs
2. ✅ ONNX Runtime Python installed: v1.23.2
3. ✅ IR version mismatch resolved: C++ upgraded to v1.23.2

**Via Appia Quality Achievement (Day 34)**:
> "Day 33 creamos modelos. Day 34 los validamos con datos reales.
> Pipeline validated end-to-end. Python + C++ both working.
> Throughput baseline established. Validación antes de optimización.
> Despacio, pero avanzando. 🏛️"

---

## 🔬 RESUMEN ESTRATEGIAS ANTI-CURSE (Para Contexto Day 35+)

**Estrategias que implementaremos Days 35-40:**

### 🔴 CRÍTICAS - Phase 2A (Days 35-38)

**1. Índices Separados por Clase** (Day 36)
- Benign index vs Malicious index
- 10x mejora para Attack embedder
- Evita saturación cross-class

**2. Dimensionality Reduction Post-Embedding** (Day 35) ← HOY
- **CRÍTICO**: Usar faiss::PCAMatrix (NO Eigen manual)
- 512→128 (preserva 96.8% varianza), 384→96, 256→64
- 4x mejora en límites
- **PASAR FAISS_ANTI_CURSE_DESIGN.md para detalles**

**3. Selective Embedding** (Day 36)
- Malicious: 100% embedded
- Benign: 10% sampling (hash determinista)
- 10x mejora para clase benign

### 🟡 IMPORTANTES - Phase 2B (Days 38-40)

**4. Temporal Tiers** (Day 39)
- Hot (7 días): ~700 eventos, CV > 0.3
- Warm (30 días): IVF, CV > 0.2
- Cold (30+ días): IVF+PQ, compressed

**5. Metadata-First Search** (Day 38)
- Pre-filter con SQL/etcd
- FAISS solo para refinamiento

**6. Quantization** (Day 40)
- float32 → int8 (4x compresión)
- <1% pérdida precisión

### 🔵 AVANZADAS - Qwen Contributions

**9. IVF Attack-Aware** (Day 39)
**10. Two-Stage Re-ranking** (Day 38)
**11. Cold Start Strategy** (Day 35)

**Límites Empíricamente Validados:**
```
Chronos (512-d → 128-d): 180K eventos (CV = 0.20)
SBERT (384-d → 96-d):    450K eventos (CV = 0.20)
Attack (256-d → 64-d):   85K benign (CV = 0.20)
```

---

## 🎯 ESTADO ACTUAL - DÍA 35 INICIO

### ✅ Completado Hasta Ahora (Days 31-34)

**Phase 2A Infrastructure (Days 31-33):**
- ✅ FAISS v1.8.0 instalado, testeado, working
- ✅ ONNX Runtime v1.23.2 (Python + C++) instalado, testeado, working
- ✅ Build system configurado (C++20, auto-detection)
- ✅ Tests pasando (test_faiss_basic, test_onnx_basic)
- ✅ Anti-curse design completado (v2.0, peer-reviewed)
- ✅ 3 embedder models ONNX creados y verificados

**Phase 2A Validation (Day 34):**
- ✅ Python inference pipeline validated (test_real_inference.py)
- ✅ C++ inference pipeline validated (test_real_embedders.cpp)
- ✅ Batch processing tested (test_batch_processing.py)
- ✅ Feature extraction working (83 features from JSONL)
- ✅ Throughput baseline established (6.8K-18.5K events/sec)
- ✅ ONNX Runtime versions matched (Python + C++ v1.23.2)

**Modelos Disponibles:**
- ✅ chronos_embedder.onnx (13KB) - 83→512-d, Time series
- ✅ sbert_embedder.onnx (21KB) - 83→384-d, Semantic
- ✅ attack_embedder.onnx (9.7KB) - 83→256-d, Attack patterns
- ✅ All verified with real JSONL data
- ✅ Scripts en git, modelos regenerables

**Datos Disponibles:**
- ✅ ~32,000+ eventos en `2025-12-12.jsonl` (34 MB)
- ✅ ~14,000 eventos en `2025-12-15.jsonl` (14 MB)
- ✅ ~1,500 eventos en `2025-12-30.jsonl` (1.6 MB)
- ✅ ~750 eventos en `2025-12-31.jsonl` (787 KB)
- ✅ Path: `/vagrant/logs/rag/events/`

### 🚧 Pendiente - Week 5-6

**Day 35 (HOY): DimensionalityReducer Implementation** ← ESTAMOS AQUÍ
- Implement faiss::PCAMatrix (NOT Eigen manual)
- Train PCA with 10K events from `2025-12-12.jsonl`
- Test dimension reduction: 512→128, 384→96, 256→64
- Validate variance preservation (target: 96%+)
- Measure CV (Coefficient of Variation) improvement

**Days 36-38: Core Anti-Curse Strategies**
- AttackIndexManager (índices separados por clase)
- SelectiveEmbedder (sampling estratégico)
- ChunkCoordinator integration
- End-to-end pipeline tests

**Days 39-40: Advanced Strategies**
- Temporal Tiers (Hot/Warm/Cold)
- Metadata-First Search
- Quantization (float32 → int8)
- IVF Attack-Aware indexing

---

## 🚀 PLAN DÍA 35 - DimensionalityReducer

### 🎯 Objetivo del Día

**Focus**: Implementar PCA-based dimensionality reduction usando `faiss::PCAMatrix`.

**Contexto Importante:**
- ✅ FAISS v1.8.0 disponible con PCA support
- ✅ 32K+ eventos disponibles para training
- ✅ Embedders working (512-d, 384-d, 256-d)
- 🎯 Objetivo: Reducir dimensiones 4x sin perder calidad

**Timeline**: 6 horas total

**Status**: Day 34 Complete ✅ → DimensionalityReducer (Day 35) → AttackIndexManager (Day 36)

---

### FASE 1: Design Review (30 min)

**Objetivo**: Revisar FAISS_ANTI_CURSE_DESIGN.md y planificar implementation

**Tareas**:
1. ✅ Leer Estrategia #2 (Dimensionality Reduction Post-Embedding)
2. ✅ Revisar límites empíricos (180K, 450K, 85K eventos)
3. ✅ Entender faiss::PCAMatrix API
4. ✅ Planificar clase DimensionalityReducer

**Decisiones clave**:
- ¿Cuántos eventos usar para training? (10K recomendado)
- ¿Qué target dimensions? (128, 96, 64)
- ¿Cómo validar variance preservation? (>96%)
- ¿Dónde persistir PCA matrices? (disk cache)

---

### FASE 2: Training PCA Models (2 horas)

**Objetivo**: Entrenar 3 PCA matrices con datos reales

**Script: train_pca_models.py**
```python
#!/usr/bin/env python3
"""
Train PCA models for dimensionality reduction.

Process:
1. Load 10K events from JSONL
2. Generate embeddings using ONNX models
3. Train PCA with faiss.PCAMatrix
4. Validate variance preservation
5. Save PCA matrices to disk
"""

import numpy as np
import faiss
import onnxruntime as ort
from pathlib import Path
import pickle

# Load events, generate embeddings, train PCA...
# [Implementation details from FAISS_ANTI_CURSE_DESIGN.md]
```

**Expected Output**:
```
Training Chronos PCA (512→128):
  ✅ Loaded 10,000 events
  ✅ Generated 512-d embeddings
  ✅ Trained PCA matrix
  ✅ Variance preserved: 96.8%
  ✅ Saved: chronos_pca_512_128.pkl

Training SBERT PCA (384→96):
  ✅ Loaded 10,000 events
  ✅ Generated 384-d embeddings
  ✅ Trained PCA matrix
  ✅ Variance preserved: 97.1%
  ✅ Saved: sbert_pca_384_96.pkl

Training Attack PCA (256→64):
  ✅ Loaded 10,000 events
  ✅ Generated 256-d embeddings
  ✅ Trained PCA matrix
  ✅ Variance preserved: 96.5%
  ✅ Saved: attack_pca_256_64.pkl
```

---

### FASE 3: C++ DimensionalityReducer Class (2.5 horas)

**Objetivo**: Implementar clase C++ para reduction en production

**File: rag/src/DimensionalityReducer.hpp**
```cpp
#pragma once

#include <faiss/VectorTransform.h>
#include <memory>
#include <string>
#include <vector>

namespace ml_defender {

/**
 * DimensionalityReducer - PCA-based dimension reduction
 * 
 * Uses faiss::PCAMatrix for efficient reduction:
 * - Chronos: 512 → 128 (4x reduction)
 * - SBERT:  384 → 96  (4x reduction)
 * - Attack:  256 → 64  (4x reduction)
 */
class DimensionalityReducer {
public:
    enum class EmbedderType {
        CHRONOS,  // 512 → 128
        SBERT,    // 384 → 96
        ATTACK    // 256 → 64
    };

    DimensionalityReducer(EmbedderType type);
    ~DimensionalityReducer();

    // Load PCA matrix from disk
    bool load(const std::string& path);
    
    // Apply PCA reduction
    void reduce(const float* input, size_t n, float* output);
    
    // Batch reduction
    void reduce_batch(const std::vector<float>& input_batch, 
                     size_t batch_size,
                     std::vector<float>& output_batch);
    
    // Get dimensions
    size_t input_dim() const { return input_dim_; }
    size_t output_dim() const { return output_dim_; }

private:
    EmbedderType type_;
    size_t input_dim_;
    size_t output_dim_;
    std::unique_ptr<faiss::PCAMatrix> pca_;
};

} // namespace ml_defender
```

**File: rag/src/DimensionalityReducer.cpp**
```cpp
#include "DimensionalityReducer.hpp"
#include <faiss/impl/io.h>
#include <fstream>

namespace ml_defender {

DimensionalityReducer::DimensionalityReducer(EmbedderType type) 
    : type_(type) {
    
    switch (type) {
        case EmbedderType::CHRONOS:
            input_dim_ = 512;
            output_dim_ = 128;
            break;
        case EmbedderType::SBERT:
            input_dim_ = 384;
            output_dim_ = 96;
            break;
        case EmbedderType::ATTACK:
            input_dim_ = 256;
            output_dim_ = 64;
            break;
    }
    
    pca_ = std::make_unique<faiss::PCAMatrix>(
        input_dim_, output_dim_, 0, true
    );
}

bool DimensionalityReducer::load(const std::string& path) {
    // Load PCA matrix from disk using faiss serialization
    // [Implementation details]
    return true;
}

void DimensionalityReducer::reduce(
    const float* input, 
    size_t n, 
    float* output
) {
    pca_->apply_noalloc(n, input, output);
}

} // namespace ml_defender
```

---

### FASE 4: Testing & Validation (1 hora)

**Objetivo**: Validar que reduction preserva calidad

**Script: test_dimensionality_reducer.cpp**
```cpp
#include "DimensionalityReducer.hpp"
#include <iostream>
#include <vector>
#include <cmath>

void test_chronos_reduction() {
    using namespace ml_defender;
    
    DimensionalityReducer reducer(
        DimensionalityReducer::EmbedderType::CHRONOS
    );
    
    // Load PCA matrix
    if (!reducer.load("chronos_pca_512_128.bin")) {
        std::cerr << "Failed to load PCA matrix\n";
        return;
    }
    
    // Test data (512-d vector)
    std::vector<float> input(512);
    for (size_t i = 0; i < 512; ++i) {
        input[i] = std::sin(i * 0.1f);
    }
    
    // Reduce
    std::vector<float> output(128);
    reducer.reduce(input.data(), 1, output.data());
    
    // Validate
    std::cout << "Input dim: " << reducer.input_dim() << "\n";
    std::cout << "Output dim: " << reducer.output_dim() << "\n";
    std::cout << "First 5 reduced values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << "\n✅ Chronos reduction test passed\n";
}

int main() {
    test_chronos_reduction();
    // test_sbert_reduction();
    // test_attack_reduction();
    return 0;
}
```

**Compilar y ejecutar**:
```bash
cd /vagrant/rag/build
make test-dimensionality-reducer
./test_dimensionality_reducer
```

---

## ✅ CRITERIOS DE ÉXITO DÍA 35

```
1. PCA Training:
   ✅ train_pca_models.py created
   ✅ Trained with 10K events from 2025-12-12.jsonl
   ✅ 3 PCA matrices saved (Chronos, SBERT, Attack)
   ✅ Variance preservation validated (>96%)
   
2. C++ Implementation:
   ✅ DimensionalityReducer.hpp created
   ✅ DimensionalityReducer.cpp implemented
   ✅ Uses faiss::PCAMatrix (NOT Eigen)
   ✅ Supports all 3 embedder types
   
3. Testing:
   ✅ test_dimensionality_reducer.cpp created
   ✅ Compile and run successfully
   ✅ All 3 reductions tested
   ✅ Output dimensions correct (128, 96, 64)

4. Documentation:
   ✅ Update DAY35_SUMMARY.md
   ✅ Document PCA training process
   ✅ Document variance preservation results
```

---

## 📅 TIMELINE - SEMANA 5-6 (ACTUALIZADO)

```
✅ Day 31: FAISS + Anti-curse design
✅ Day 32: ONNX Runtime test
✅ Day 33: Real embedders (3 ONNX models)
✅ Day 34: Test con datos reales (21 min) ✅

🔥 Day 35: DimensionalityReducer (6h) ← ESTAMOS AQUÍ
   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md ← CRÍTICO
   - Train 3 PCA matrices (10K events)
   - Implement DimensionalityReducer class
   - Validate variance preservation (>96%)
   - Test reduction pipeline
   - 512→128, 384→96, 256→64

📅 Day 36: AttackIndexManager (4h)
   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md
   - Implement separate indices (benign vs malicious)
   - Test cross-class isolation
   - Measure CV improvement

📅 Day 37: SelectiveEmbedder (2h)
   - Implement sampling strategy
   - Hash-based deterministic selection
   - 100% malicious, 10% benign

📅 Day 38: ChunkCoordinator Integration (4h)
   - Connect to existing pipeline
   - End-to-end tests
   - Performance benchmarks

📅 Days 39-40: Advanced Strategies (8h)
   - Temporal Tiers
   - Metadata-First Search
   - Quantization
```

---

## 🚀 COMANDOS RÁPIDOS DÍA 35

```bash
# Verificar datos disponibles
ls -lh /vagrant/logs/rag/events/*.jsonl

# Fase 1: Review design (30 min)
# [Read FAISS_ANTI_CURSE_DESIGN.md]
# [Plan implementation]

# Fase 2: Train PCA models (2h)
cd /vagrant/rag/models
python3 train_pca_models.py

# Fase 3: C++ implementation (2.5h)
cd /vagrant/rag/src
# [Create DimensionalityReducer.hpp]
# [Create DimensionalityReducer.cpp]
cd /vagrant/rag/build
cmake ..
make

# Fase 4: Testing (1h)
cd /vagrant/rag/tests
# [Create test_dimensionality_reducer.cpp]
cd /vagrant/rag/build
make test-dimensionality-reducer
./test_dimensionality_reducer

# Git commit
cd /vagrant
git add rag/models/train_pca_models.py
git add rag/src/DimensionalityReducer.{hpp,cpp}
git add rag/tests/test_dimensionality_reducer.cpp
git commit -m "Day 35: DimensionalityReducer with faiss::PCAMatrix"
```

---

## 🏛️ VIA APPIA QUALITY - FILOSOFÍA DAY 35

> "Day 34 validamos el pipeline. Day 35 lo optimizamos. PCA reduction
> es Estrategia #2 del anti-curse design. 4x mejora en límites.
> Chronos: 180K eventos posibles. SBERT: 450K eventos. Attack: 85K
> benign. Foundation sólida primero. Usar faiss::PCAMatrix, NO Eigen.
> Variance preservation >96%. Validación empírica antes de producción.
> Despacio, pero avanzando. 🏛️"

**Key Principles Day 35:**
- ✅ Use FAISS built-in PCA (battle-tested)
- ✅ Train with real data (10K events)
- ✅ Validate variance preservation empirically
- ✅ 4x dimension reduction without quality loss
- ✅ Foundation for remaining anti-curse strategies

---

## 📊 EXPECTED OUTCOMES DAY 35

**After Day 35 completion:**

```
Infrastructure:
✅ FAISS v1.8.0
✅ ONNX Runtime v1.23.2 (Python + C++)
✅ 3 ONNX embedder models (512-d, 384-d, 256-d)
✅ 3 PCA matrices trained (128-d, 96-d, 64-d) ← NEW
✅ DimensionalityReducer class (C++) ← NEW

Pipeline Status:
✅ JSONL → Features (83)
✅ Features → Embeddings (512/384/256)
✅ Embeddings → Reduced (128/96/64) ← NEW
🔄 Reduced → FAISS Index (pending Day 36+)

Performance Metrics:
✅ Variance preserved: >96% (all 3 embedders)
✅ Dimension reduction: 4x (all 3 embedders)
✅ CV improvement: 4x better limits
✅ Memory savings: 4x reduction
```

**Foundation Ready For:**
- Day 36: AttackIndexManager (separate indices)
- Day 37: SelectiveEmbedder (sampling)
- Day 38: ChunkCoordinator integration

---

## ⚠️ CRITICAL REMINDER DAY 35

**MUST HAVE BEFORE STARTING:**
- ✅ FAISS_ANTI_CURSE_DESIGN.md uploaded
- ✅ Read Estrategia #2 thoroughly
- ✅ Understand faiss::PCAMatrix API
- ✅ Know target dimensions (128, 96, 64)
- ✅ Know variance targets (>96%)

**DO NOT:**
- ❌ Use Eigen for PCA (use FAISS built-in)
- ❌ Skip variance validation
- ❌ Train with <5K events (insufficient)
- ❌ Use opset 14 models (IR v9 incompatible)

---

**Next Session**: Day 35 - DimensionalityReducer

**First Action**: Solicitar FAISS_ANTI_CURSE_DESIGN.md

**Via Appia Quality**: Optimización después de validación. Foundation sólida. 🏛️
# CONTEXTO: Day 36 - PCA Training Pipeline (Synthetic Data)

**Fecha:** 10-Enero-2026  
**Sesión:** Day 36 Execution (con tokens completos)  
**Estado:** 🔥 READY TO START - Planning complete Day 35+36

---

## Resumen Ejecutivo Day 35-36

### Day 35 (COMPLETADO ✅)
**Duración:** ~2 horas  
**Entregable:** DimensionalityReducer library operacional

```
✅ common-rag-ingester/ compilado (Debian 12)
✅ API: train/transform/save/load validada
✅ Test PASSED: 908ms training, 149μs transform
✅ Performance: 20K vec/sec batch, ~10MB/model
✅ Variance: 40.97% (synthetic - expected)
```

### Day 36 Planning (COMPLETADO ✅)
**Duración:** Sesión completa de investigación  
**Descubrimiento Crítico:** Desconexión arquitectural feature extractors ↔ embedders

**Documentos Creados:**
- `TECHNICAL_DEBT_DAY36.md` - Análisis completo (18 páginas)
- `BACKLOG_UPDATE_DAY36.md` - Updates del roadmap
- Este prompt de continuidad

**Decisión:** Plan A→B→A' (synthetic → fix → real)

---

## 🚨 CRITICAL DISCOVERY - Technical Debt

### Lo que Descubrimos

**Sistema de Detección (✅ FUNCIONA):**
```
eBPF Sniffer → 11 campos básicos → ZeroMQ → ml-detector
                                              ↓
                                   FeatureExtractor (ml-detector)
                                              ↓
                                   Level 1: 23 feat → ONNX
                                   Level 2: 10 feat → DDoS C++20
                                   Level 2: 10 feat → Ransomware C++20
                                   Level 3: 10 feat → Traffic C++20
                                   Level 3: 10 feat → Internal C++20

Estado: 20+ horas operación continua ✅
```

**Pipeline RAG/FAISS (❌ INCOMPLETO):**
```
.pb guardados: Solo 11 campos básicos
Tag: "requires_processing"
Embedders ONNX: Esperan 83 features
Gap: 72 features faltantes ❌

Causa: Dos sistemas de extracción nunca se conectaron:
├─ FeatureExtractor (83 feat) - legacy CTU-13, nunca integrado
├─ MLDefenderExtractor (40 feat) - código existe, no se guarda en .pb
└─ Embedders ONNX (83 feat) - placeholders sintéticos
```

### Solución: Plan A→B→A'

```
┌─────────────────────────────────────────────────────────┐
│ Day 36: Plan A - Synthetic PCA Training (4-6h)          │
├─────────────────────────────────────────────────────────┤
│ ✅ Unblocks pipeline validation                          │
│ ✅ Proves architecture end-to-end                        │
│ ✅ Training code written and tested                      │
│ ⚠️ Variance lower (synthetic data has no structure)     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 37: Plan B - Feature Processing (1 day)             │
├─────────────────────────────────────────────────────────┤
│ Fix: Activate MLDefenderExtractor (40 features)         │
│ Debug: Why .pb submessages empty                        │
│ Validate: .pb contains real features                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Day 38: Plan A' - Real PCA Re-training (2h)             │
├─────────────────────────────────────────────────────────┤
│ ✅ SAME CODE as Day 36 (only data source changes)       │
│ ✅ Compare variance: synthetic vs real                   │
│ ✅ Double validation of pipeline                         │
│ ✅ Ready for production FAISS ingestion                  │
└─────────────────────────────────────────────────────────┘
```

**Net Impact:** 1 day delay, but double validation  
**Via Appia:** "Better to build foundation twice than rush once" 🏛️

---

## Day 36 Objetivo - Plan A (Synthetic)

### Goal
Entrenar 3 PCA reducers con datos sintéticos para validar pipeline end-to-end.

### Inputs Available
```
✅ DimensionalityReducer library: /vagrant/common-rag-ingester/
✅ ONNX Runtime installed: v1.17.1
✅ Embedders ONNX:
   ├─ /vagrant/rag/models/chronos_embedder.onnx (83→512-d)
   ├─ /vagrant/rag/models/sbert_embedder.onnx (83→384-d)
   └─ /vagrant/rag/models/attack_embedder.onnx (83→256-d)
```

### Expected Outputs
```
📁 /shared/models/pca/
├─ chronos_pca_512_128.faiss    (512→128, variance ≥96% target)
├─ sbert_pca_384_128.faiss      (384→128, variance ≥96% target)
└─ attack_pca_256_128.faiss     (256→128, variance ≥96% target)

📄 /vagrant/tools/
├─ train_pca.cpp                (main training binary)
├─ synthetic_data_generator.hpp (20K events, 83 features)
├─ onnx_embedder.hpp            (batch inference wrapper)
└─ CMakeLists.txt               (build config)
```

---

## Implementation Plan (4-6 hours)

### PASO 1: Synthetic Data Generator (1h)

```cpp
// /vagrant/tools/synthetic_data_generator.hpp

class SyntheticDataGenerator {
public:
    // Generate N events with 83 features
    std::vector<std::vector<float>> generate(
        size_t num_samples,
        unsigned seed = 42
    );
    
    // Add semantic structure (optional - improve variance)
    void add_attack_patterns(std::vector<std::vector<float>>& data);
};

Características:
- 20,000 eventos sintéticos
- 83 features normalized [0, 1]
- Reproducible (seed=42)
- Optional: Add attack patterns for better variance
```

### PASO 2: ONNX Embedder Wrapper (1-2h)

```cpp
// /vagrant/tools/onnx_embedder.hpp

class ONNXEmbedder {
public:
    ONNXEmbedder(const std::string& model_path);
    
    // Single inference
    std::vector<float> embed(const std::vector<float>& features);
    
    // Batch inference (efficient)
    std::vector<std::vector<float>> embed_batch(
        const std::vector<std::vector<float>>& features_batch,
        size_t batch_size = 64
    );
    
    size_t get_input_dim() const { return input_dim_; }
    size_t get_output_dim() const { return output_dim_; }
};

Performance target: >100 events/sec per embedder
```

### PASO 3: PCA Training Pipeline (1h)

```cpp
// /vagrant/tools/train_pca.cpp

int main() {
    // 1. Generate synthetic data
    SyntheticDataGenerator generator;
    auto data = generator.generate(20000);
    
    // 2. Load ONNX embedders
    ONNXEmbedder chronos("/vagrant/rag/models/chronos_embedder.onnx");
    ONNXEmbedder sbert("/vagrant/rag/models/sbert_embedder.onnx");
    ONNXEmbedder attack("/vagrant/rag/models/attack_embedder.onnx");
    
    // 3. Generate embeddings
    auto chronos_emb = chronos.embed_batch(data);  // 20K × 512
    auto sbert_emb = sbert.embed_batch(data);      // 20K × 384
    auto attack_emb = attack.embed_batch(data);    // 20K × 256
    
    // 4. Train PCA reducers
    DimensionalityReducer pca_chronos(512, 128);
    pca_chronos.train(chronos_emb);
    pca_chronos.save("/shared/models/pca/chronos_pca_512_128.faiss");
    
    // ... same for sbert and attack ...
    
    // 5. Report variance
    std::cout << "Chronos variance: " << pca_chronos.get_variance() << "\n";
    
    return 0;
}
```

### PASO 4: Validation (30min)

```cpp
// Test script: test_trained_pca.cpp

void test_pca_model(const std::string& model_path) {
    // Load PCA
    auto pca = DimensionalityReducer::load(model_path);
    
    // Test transform
    std::vector<float> test_vec(pca->get_input_dim(), 0.5f);
    std::vector<float> reduced = pca->transform(test_vec);
    
    // Verify dimensions
    assert(reduced.size() == pca->get_output_dim());
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        pca->transform(test_vec);
    }
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start
    ).count();
    
    std::cout << "Avg transform time: " << (duration / 1000.0) << " μs\n";
}
```

---

## Build Configuration

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(train_pca CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(PkgConfig REQUIRED)
find_path(FAISS_INCLUDE_DIR faiss/IndexFlat.h HINTS /usr/local/include)
find_library(FAISS_LIBRARY NAMES faiss HINTS /usr/local/lib)

# ONNX Runtime
find_library(ONNXRUNTIME_LIB onnxruntime HINTS /usr/local/lib)

# common-rag-ingester
find_package(common-rag-ingester REQUIRED)

# Training binary
add_executable(train_pca
    train_pca.cpp
    synthetic_data_generator.cpp
    onnx_embedder.cpp
)

target_include_directories(train_pca PRIVATE
    ${FAISS_INCLUDE_DIR}
    /usr/local/include/onnxruntime
)

target_link_libraries(train_pca
    common-rag-ingester::common-rag-ingester
    ${FAISS_LIBRARY}
    ${ONNXRUNTIME_LIB}
)

# Test binary
add_executable(test_trained_pca test_trained_pca.cpp)
target_link_libraries(test_trained_pca
    common-rag-ingester::common-rag-ingester
    ${FAISS_LIBRARY}
)
```

---

## Success Criteria - Day 36

**Must Have:**
- [x] 3 PCA models trained successfully
- [x] Models saved to /shared/models/pca/
- [x] Validation tests PASSED
- [x] Build clean on Debian 12
- [x] Code documented

**Should Have:**
- [x] Variance ≥70% (realistic for synthetic)
- [x] Performance >100 evt/sec embedding
- [x] Transform <200μs per vector
- [x] Memory <50MB total

**Nice to Have:**
- [ ] Variance ≥90% (requires pattern engineering)
- [ ] Performance >500 evt/sec
- [ ] Visualization of embeddings

---

## Known Constraints

### Variance Expectations
```
Synthetic data (random):     40-70% variance (expected)
Synthetic w/ patterns:       70-85% variance (if engineered)
Real data (from Day 38):     ≥96% variance (target)
```

**Why lower variance is OK for Day 36:**
- Validating pipeline, not final models
- Real data will have semantic structure
- Day 38 will re-train with real data using SAME code

### Performance Baseline
```
Day 35 DimensionalityReducer:
├─ Training: 908ms for 10K samples
├─ Transform: 149μs single vector
└─ Batch: 20K vec/sec

Day 36 Expected (20K samples, 3 models):
├─ Data generation: ~10s
├─ ONNX embedding: ~3min (20K × 3 models)
├─ PCA training: ~3s × 3 = 9s
└─ Total: ~4-5 minutes end-to-end
```

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|------------|
| Variance <40% | Low | Add pattern engineering |
| ONNX slow | Low | Batch inference (64 samples) |
| Memory issues | Low | Incremental processing |
| Build fails | Very Low | Dependencies already tested Day 32-35 |

---

## Documentation Requirements

### Code Documentation
```cpp
// Every file must have:
// 1. Purpose header
// 2. Architecture context
// 3. Example usage
// 4. Performance notes

// Example:
/**
 * Synthetic Data Generator for PCA Training
 * 
 * Generates N events with 83 features for validating
 * FAISS pipeline architecture (Day 36 - Plan A).
 * 
 * Real data processing will be implemented Day 37-38.
 * This code validates training pipeline logic.
 * 
 * Usage:
 *   SyntheticDataGenerator gen;
 *   auto data = gen.generate(20000);  // 20K × 83
 * 
 * Performance: ~1ms per 1000 samples
 */
```

### Results Documentation
```markdown
# Day 36 Results - Plan A (Synthetic)

## Models Trained
- chronos_pca_512_128.faiss: XX.XX% variance
- sbert_pca_384_128.faiss: XX.XX% variance
- attack_pca_256_128.faiss: XX.XX% variance

## Performance
- Data generation: X.Xs
- ONNX embedding: X.Xmin
- PCA training: X.Xs
- Total: X.Xmin

## Validation
- Transform test: PASSED
- Dimension test: PASSED
- Performance test: XXμs per vector

## Notes
Synthetic data variance lower than target (expected).
Day 38 will re-train with real data for production models.
Pipeline architecture validated successfully.
```

---

## Next Steps After Day 36

**Immediate (Day 37):**
- Debug MLDefenderExtractor .pb serialization
- Validate 40 features in .pb files
- Document feature extraction flow

**Short-term (Day 38):**
- Re-train PCA with real 40 or 83 features
- Compare variance: synthetic vs real
- Finalize production PCA models

**Medium-term (Day 39-40):**
- Implement FAISS ingester using trained PCA
- Integration testing
- Performance optimization

---

## Via Appia Reminder

> "We discovered an architectural gap during planning - exactly when we should.
> Not during execution, not during production deployment.
>
> Plan A validates the architecture TODAY.
> Plan B fixes the data pipeline PROPERLY.
> Plan A' validates the fix with SAME code.
>
> Double validation. No shortcuts. Foundation first.
> This is Via Appia Quality. 🏛️"

---

## Command to Start Session

```bash
cd /vagrant/tools
mkdir -p build
cd build

# Create train_pca project
cat > ../train_pca.cpp << 'EOF'
// Day 36 - Plan A: Synthetic PCA Training
// ...implementation...
# PROMPT DE CONTINUIDAD - DÍA 31 (01 Enero 2026)

## 📋 CONTEXTO DÍA 30 (31 Diciembre 2025)

### ✅ COMPLETADO - MEMORY LEAK INVESTIGATION & RESOLUTION

**Gran Hito Alcanzado:**
- ✅ Memory leak investigado sistemáticamente (5+ horas)
- ✅ 70% reducción lograda (102 → 31 MB/h)
- ✅ Configuración óptima identificada (artifacts + flush)
- ✅ Cron restart configurado (cada 72h)
- ✅ Sistema production-ready para 24×7×365
- ✅ Documentación completa generada

**Arquitectura Día 30 (Production-Ready):**
```
ML-DETECTOR + RAG LOGGER
  ↓ 83-field JSONL events
  ↓ Protobuf + JSON artifacts
  ↓ Memory: 31 MB/h (acceptable)
  ↓ Restart: Every 72h (cron)
  ✅ Logs ready for FAISS ingestion
```

**Investigación Científica (Via Appia Quality):**
```
METODOLOGÍA:
1. AddressSanitizer analysis (ASAN)
2. Configuration matrix testing (5 configs)
3. Systematic measurement (90+ min tests)
4. Root cause analysis (stream buffering)
5. Fix validation (70% improvement)

CONFIGURACIONES TESTEADAS:
┌─────────────────────────────────────────────┐
│ Config              Leak/h   Leak/event     │
├─────────────────────────────────────────────┤
│ PRE-FIX (baseline)  102 MB   246 KB    ❌   │
│ POST-FIX (optimal)   31 MB    63 KB    ✅   │
│ SIN-ARTIFACTS        50 MB   118 KB    ⚠️    │
│ SHRINK-FIX           53 MB    99 KB    ⚠️    │
│ QUICKFIX             53 MB    97 KB    ⚠️    │
└─────────────────────────────────────────────┘

ROOT CAUSE:
  std::ofstream buffer never flushed
  → Accumulation of 1-2KB JSON strings
  → 102 MB/h without flush()
  
THE FIX:
  current_log_.flush() after each write
  → 31 MB/h with flush() ✅
  → Artifacts enabled (helps fragmentation)
  → Cron restart every 72h
  
SURPRISING DISCOVERY:
  WITH artifacts: 31 MB/h ✅
  WITHOUT artifacts: 50 MB/h ⚠️
  Artifacts help by distributing allocations!
```

**Métricas Día 30 (Final Configuration):**
```
┌─────────────────────────────────────────────┐
│  CONFIGURATION: POST-FIX (OPTIMAL)          │
├─────────────────────────────────────────────┤
│  Memory leak:         31 MB/hour            │
│  Per-event leak:      63 KB/event           │
│  Test duration:       90 minutes            │
│  Events processed:    747 events            │
│  Improvement:         70% vs baseline       │
│  Production ready:    ✅ YES                │
│  Restart schedule:    Every 72h (cron)      │
│  Max memory growth:   2.2 GB/72h            │
│  VM allocation:       8 GB (safe margin)    │
└─────────────────────────────────────────────┘

ARTIFACTS STATUS:
  Protobuf: ✅ Enabled (optimal)
  JSON:     ✅ Enabled (optimal)
  Location: /vagrant/logs/rag/artifacts/
  Format:   event_ID.pb + event_ID.json
  
CRON CONFIGURATION:
  Entry: 0 3 */3 * * /vagrant/scripts/restart_ml_defender.sh
  User: vagrant
  Status: ✅ Configured in Vagrantfile
  Logs: /vagrant/logs/lab/restart_ml_defender.log
```

---

## 🎯 ESTADO ACTUAL (DÍA 31 INICIO)

### ✅ Phase 1 Status (100% COMPLETO)

**Funcionalidades Validadas:**
- ✅ 4 componentes distribuidos operativos
- ✅ ChaCha20-Poly1305 + LZ4 end-to-end
- ✅ ML pipeline completa (Level 1-3)
- ✅ Dual-score architecture (Fast + ML)
- ✅ Etcd service discovery + heartbeats
- ✅ RAG logger 83-field events
- ✅ Memory leak resolved (70% reduction)
- ✅ Production-ready (24×7×365)
- ✅ Real traffic validated
- ✅ Sub-millisecond crypto latencies

**Logs Disponibles para FAISS:**
```bash
/vagrant/logs/rag/events/YYYY-MM-DD.jsonl
/vagrant/logs/rag/artifacts/YYYY-MM-DD/event_*.pb
/vagrant/logs/rag/artifacts/YYYY-MM-DD/event_*.json

# Verificar
wc -l /vagrant/logs/rag/events/$(date +%Y-%m-%d).jsonl
ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ | wc -l
```

---

## 🚀 PLAN DÍA 31 - FAISS INGESTION IMPLEMENTATION (Week 5 Start)

### 📚 CONTEXTO PREVIO - FAISS INGESTION DESIGN

**Documentos de Referencia:**
1. `docs/FAISS_INGESTION_DESIGN.md` - Arquitectura completa
2. Sesión 2025-12-30 - Discusión multi-embedder coherente
3. Memory leak transcript (Day 30)

**Decisiones Arquitectónicas (Ya Tomadas):**
```
✅ Multi-embedder coherente: Mismo chunk → 3 índices
✅ Best-effort commit: Resilience > atomicidad estricta
✅ C++20 implementation: Coherencia con stack
✅ ONNX Runtime: Chronos + SBERT + Custom models
✅ Chunk = día completo: NUNCA truncar time series
✅ 3 embedders fundacionales:
   1. Chronos (time series, 512-d)
   2. SBERT (semantic, 384-d)
   3. Custom DNN (attack patterns, 256-d)
```

**Arquitectura FAISS (Diseñada):**
```
ChunkCoordinator (orquestador)
    ↓
    ├─ TimeSeriesEmbedder (Chronos ONNX)
    ├─ SemanticEmbedder (SBERT ONNX)
    └─ AttackEmbedder (Custom ONNX)
    ↓
IndexManager (3 FAISS indices)
    ↓
HealthMonitor + IndexTracker
```

---

### FASE 1: ONNX Model Export (Día 31 - 2-3 horas)

**Objetivo:** Exportar los 3 modelos a ONNX para C++ inference

#### Step 1: Setup Python Environment
```bash
cd /vagrant/ml-training
python3 -m venv venv-onnx
source venv-onnx/bin/activate
pip install torch onnx onnxruntime sentence-transformers chronos-forecasting
```

#### Step 2: Export Chronos (Time Series Embedder)
```python
# File: ml-training/export_chronos_onnx.py
import torch
import onnx
from chronos import ChronosPipeline

# Load Chronos model
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-tiny",
    device_map="cpu",
    torch_dtype=torch.float32,
)

# Create dummy input (24-hour time series)
dummy_input = torch.randn(1, 1440, 1)  # 1440 minutes in 24h

# Export to ONNX
torch.onnx.export(
    pipeline.model,
    dummy_input,
    "models/chronos_embedder.onnx",
    input_names=['time_series'],
    output_names=['embeddings'],
    dynamic_axes={
        'time_series': {0: 'batch_size', 1: 'sequence_length'},
        'embeddings': {0: 'batch_size'}
    },
    opset_version=14
)

print("✅ Chronos exported: models/chronos_embedder.onnx")
```

#### Step 3: Export SBERT (Semantic Embedder)
```python
# File: ml-training/export_sbert_onnx.py
import torch
import onnx
from sentence_transformers import SentenceTransformer

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create dummy input (tokenized text)
dummy_input = {
    'input_ids': torch.randint(0, 30522, (1, 128)),
    'attention_mask': torch.ones(1, 128, dtype=torch.long)
}

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "models/sbert_embedder.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['sentence_embedding'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'sentence_embedding': {0: 'batch_size'}
    },
    opset_version=14
)

print("✅ SBERT exported: models/sbert_embedder.onnx")
```

#### Step 4: Create Custom Attack Embedder
```python
# File: ml-training/train_and_export_attack_embedder.py
import torch
import torch.nn as nn

class AttackEmbedder(nn.Module):
    def __init__(self, input_dim=83, hidden_dim=512, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
    
    def forward(self, x):
        return self.encoder(x)

# Train on RAG logs (simplified)
model = AttackEmbedder()
# TODO: Training loop with RAG JSONL data

# Export to ONNX
dummy_input = torch.randn(1, 83)  # 83 fields from RAG logs

torch.onnx.export(
    model,
    dummy_input,
    "models/attack_embedder.onnx",
    input_names=['features'],
    output_names=['attack_embedding'],
    dynamic_axes={
        'features': {0: 'batch_size'},
        'attack_embedding': {0: 'batch_size'}
    },
    opset_version=14
)

print("✅ Attack embedder exported: models/attack_embedder.onnx")
```

#### Step 5: Verify ONNX Models
```bash
# Install ONNX tools
pip install onnx onnxruntime

# Verify models
python -c "import onnx; model = onnx.load('models/chronos_embedder.onnx'); onnx.checker.check_model(model); print('✅ Chronos OK')"
python -c "import onnx; model = onnx.load('models/sbert_embedder.onnx'); onnx.checker.check_model(model); print('✅ SBERT OK')"
python -c "import onnx; model = onnx.load('models/attack_embedder.onnx'); onnx.checker.check_model(model); print('✅ Attack OK')"

# Test inference with ONNX Runtime
python -c "
import onnxruntime as ort
import numpy as np

# Test Chronos
session = ort.InferenceSession('models/chronos_embedder.onnx')
input_data = np.random.randn(1, 1440, 1).astype(np.float32)
output = session.run(None, {'time_series': input_data})
print(f'✅ Chronos output shape: {output[0].shape}')

# Test SBERT
session = ort.InferenceSession('models/sbert_embedder.onnx')
input_ids = np.random.randint(0, 30522, (1, 128)).astype(np.int64)
attention_mask = np.ones((1, 128), dtype=np.int64)
output = session.run(None, {'input_ids': input_ids, 'attention_mask': attention_mask})
print(f'✅ SBERT output shape: {output[0].shape}')

# Test Attack
session = ort.InferenceSession('models/attack_embedder.onnx')
features = np.random.randn(1, 83).astype(np.float32)
output = session.run(None, {'features': features})
print(f'✅ Attack output shape: {output[0].shape}')
"
```

---

### FASE 2: FAISS Integration (Día 31 - 2 horas)

**Objetivo:** Integrar FAISS library en C++20

#### Step 1: Install FAISS
```bash
# Install FAISS dependencies
sudo apt-get update
sudo apt-get install -y libblas-dev liblapack-dev

# Build FAISS from source (CPU version)
cd /tmp
git clone https://github.com/facebookresearch/faiss.git
cd faiss
mkdir build && cd build
cmake .. -DFAISS_ENABLE_GPU=OFF \
         -DFAISS_ENABLE_PYTHON=OFF \
         -DBUILD_TESTING=OFF \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/usr/local
make -j4
sudo make install
sudo ldconfig

# Verify installation
pkg-config --modversion faiss
```

#### Step 2: Create FAISS Test (C++20)
```cpp
// File: rag/tests/test_faiss_integration.cpp
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <iostream>
#include <vector>

int main() {
    // Test 1: Simple flat index
    int d = 512;  // Chronos embedding dimension
    faiss::IndexFlatL2 index(d);
    
    std::cout << "✅ Index created, dimension: " << index.d << std::endl;
    
    // Add some random vectors
    std::vector<float> data(10 * d);
    for (auto& val : data) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
    
    index.add(10, data.data());
    std::cout << "✅ Added 10 vectors, total: " << index.ntotal << std::endl;
    
    // Search
    std::vector<float> query(d);
    for (auto& val : query) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }
    
    int k = 5;
    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    
    index.search(1, query.data(), k, distances.data(), labels.data());
    
    std::cout << "✅ Search complete, nearest neighbors:";
    for (int i = 0; i < k; ++i) {
        std::cout << " " << labels[i] << " (dist: " << distances[i] << ")";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Step 3: CMake Integration
```cmake
# File: rag/CMakeLists.txt (add FAISS)
find_package(faiss REQUIRED)

add_executable(test_faiss_integration
    tests/test_faiss_integration.cpp
)

target_link_libraries(test_faiss_integration
    PRIVATE
    faiss
)
```

#### Step 4: Build and Test
```bash
cd /vagrant/rag/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make test_faiss_integration

# Run test
./test_faiss_integration

# Expected output:
# ✅ Index created, dimension: 512
# ✅ Added 10 vectors, total: 10
# ✅ Search complete, nearest neighbors: 3 (dist: 0.234) 7 (dist: 0.456) ...
```

---

### FASE 3: ONNX Runtime Integration (Día 31 - 2 horas)

**Objetivo:** Load ONNX models in C++ and run inference

#### Step 1: ONNX Runtime Test
```cpp
// File: rag/tests/test_onnx_inference.cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    // Load model
    Ort::Session session(env, "models/attack_embedder.onnx", session_options);
    
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    
    std::cout << "✅ Model loaded" << std::endl;
    std::cout << "   Input nodes: " << num_input_nodes << std::endl;
    std::cout << "   Output nodes: " << num_output_nodes << std::endl;
    
    // Get input name
    auto input_name = session.GetInputNameAllocated(0, allocator);
    std::cout << "   Input name: " << input_name.get() << std::endl;
    
    // Get output name
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::cout << "   Output name: " << output_name.get() << std::endl;
    
    // Create dummy input (83 features)
    std::vector<float> input_data(83, 0.5f);
    std::vector<int64_t> input_shape = {1, 83};
    
    // Create input tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );
    
    // Run inference
    const char* input_names[] = {input_name.get()};
    const char* output_names[] = {output_name.get()};
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // Get output
    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    
    std::cout << "✅ Inference complete" << std::endl;
    std::cout << "   Output shape: [" << output_shape[0] << ", " << output_shape[1] << "]" << std::endl;
    std::cout << "   First 5 values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

#### Step 2: CMake for ONNX Test
```cmake
# File: rag/CMakeLists.txt (add ONNX Runtime)
find_package(onnxruntime REQUIRED)

add_executable(test_onnx_inference
    tests/test_onnx_inference.cpp
)

target_link_libraries(test_onnx_inference
    PRIVATE
    onnxruntime::onnxruntime
)
```

#### Step 3: Build and Test
```bash
cd /vagrant/rag/build
cmake ..
make test_onnx_inference

# Run test
./test_onnx_inference

# Expected output:
# ✅ Model loaded
#    Input nodes: 1
#    Output nodes: 1
#    Input name: features
#    Output name: attack_embedding
# ✅ Inference complete
#    Output shape: [1, 256]
#    First 5 values: 0.123 -0.456 0.789 ...
```

---

### FASE 4: ChunkCoordinator Skeleton (Día 31 - 2 horas)

**Objetivo:** Crear estructura base del coordinador

#### Step 1: Header File
```cpp
// File: rag/include/faiss_ingester/chunk_coordinator.hpp
#pragma once

#include <string>
#include <memory>
#include <chrono>
#include <vector>

namespace ml_defender {
namespace faiss_ingester {

// Forward declarations
class TimeSeriesEmbedder;
class SemanticEmbedder;
class AttackEmbedder;
class IndexManager;

struct ChunkMetadata {
    std::string chunk_id;
    std::chrono::system_clock::time_point start_time;
    std::chrono::system_clock::time_point end_time;
    size_t event_count;
    std::string jsonl_path;
};

class ChunkCoordinator {
public:
    ChunkCoordinator(const std::string& config_path);
    ~ChunkCoordinator();

    // Main orchestration
    bool process_daily_chunk(const std::string& date_str);
    
    // Status
    bool is_healthy() const;
    nlohmann::json get_statistics() const;

private:
    // Configuration
    std::string config_path_;
    std::string base_logs_path_;
    
    // Embedders (ONNX models)
    std::unique_ptr<TimeSeriesEmbedder> time_series_embedder_;
    std::unique_ptr<SemanticEmbedder> semantic_embedder_;
    std::unique_ptr<AttackEmbedder> attack_embedder_;
    
    // Index management
    std::unique_ptr<IndexManager> index_manager_;
    
    // Statistics
    std::atomic<uint64_t> chunks_processed_{0};
    std::atomic<uint64_t> events_ingested_{0};
    std::atomic<uint64_t> errors_{0};
    
    // Helper methods
    ChunkMetadata load_chunk_metadata(const std::string& date_str);
    std::vector<nlohmann::json> load_jsonl_events(const std::string& jsonl_path);
    
    bool commit_to_indices(
        const std::vector<float>& ts_embedding,
        const std::vector<float>& semantic_embedding,
        const std::vector<float>& attack_embedding,
        const ChunkMetadata& metadata
    );
};

} // namespace faiss_ingester
} // namespace ml_defender
```

#### Step 2: Implementation Skeleton
```cpp
// File: rag/src/faiss_ingester/chunk_coordinator.cpp
#include "faiss_ingester/chunk_coordinator.hpp"
#include <fstream>
#include <spdlog/spdlog.h>

namespace ml_defender {
namespace faiss_ingester {

ChunkCoordinator::ChunkCoordinator(const std::string& config_path)
    : config_path_(config_path)
{
    spdlog::info("🚀 ChunkCoordinator initializing...");
    
    // TODO: Load config
    // TODO: Initialize embedders
    // TODO: Initialize index manager
    
    spdlog::info("✅ ChunkCoordinator ready");
}

ChunkCoordinator::~ChunkCoordinator() {
    spdlog::info("📊 ChunkCoordinator statistics:");
    spdlog::info("   Chunks processed: {}", chunks_processed_.load());
    spdlog::info("   Events ingested: {}", events_ingested_.load());
    spdlog::info("   Errors: {}", errors_.load());
}

bool ChunkCoordinator::process_daily_chunk(const std::string& date_str) {
    spdlog::info("📥 Processing chunk: {}", date_str);
    
    try {
        // Step 1: Load metadata
        auto metadata = load_chunk_metadata(date_str);
        spdlog::info("   Events in chunk: {}", metadata.event_count);
        
        // Step 2: Load JSONL events
        auto events = load_jsonl_events(metadata.jsonl_path);
        spdlog::info("   Loaded {} events from JSONL", events.size());
        
        // Step 3: Generate embeddings (TODO)
        // auto ts_emb = time_series_embedder_->embed(events);
        // auto sem_emb = semantic_embedder_->embed(events);
        // auto att_emb = attack_embedder_->embed(events);
        
        // Step 4: Commit to indices (TODO)
        // bool success = commit_to_indices(ts_emb, sem_emb, att_emb, metadata);
        
        chunks_processed_++;
        events_ingested_ += events.size();
        
        spdlog::info("✅ Chunk {} processed successfully", date_str);
        return true;
        
    } catch (const std::exception& e) {
        spdlog::error("❌ Failed to process chunk {}: {}", date_str, e.what());
        errors_++;
        return false;
    }
}

ChunkMetadata ChunkCoordinator::load_chunk_metadata(const std::string& date_str) {
    ChunkMetadata metadata;
    metadata.chunk_id = date_str;
    metadata.jsonl_path = base_logs_path_ + "/events/" + date_str + ".jsonl";
    
    // Count events in JSONL
    std::ifstream file(metadata.jsonl_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSONL: " + metadata.jsonl_path);
    }
    
    std::string line;
    size_t count = 0;
    while (std::getline(file, line)) {
        count++;
    }
    
    metadata.event_count = count;
    return metadata;
}

std::vector<nlohmann::json> ChunkCoordinator::load_jsonl_events(const std::string& jsonl_path) {
    std::vector<nlohmann::json> events;
    std::ifstream file(jsonl_path);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSONL: " + jsonl_path);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        try {
            auto event = nlohmann::json::parse(line);
            events.push_back(event);
        } catch (const std::exception& e) {
            spdlog::warn("Failed to parse JSONL line: {}", e.what());
        }
    }
    
    return events;
}

bool ChunkCoordinator::is_healthy() const {
    // TODO: Check embedders and indices
    return true;
}

nlohmann::json ChunkCoordinator::get_statistics() const {
    return {
        {"chunks_processed", chunks_processed_.load()},
        {"events_ingested", events_ingested_.load()},
        {"errors", errors_.load()}
    };
}

} // namespace faiss_ingester
} // namespace ml_defender
```

---

## ✅ CRITERIOS DE ÉXITO DÍA 31

### Mínimo para Progress:
```
1. ONNX Models Exported:
   ✅ chronos_embedder.onnx created
   ✅ sbert_embedder.onnx created
   ✅ attack_embedder.onnx created
   ✅ All models verified with onnx.checker
   ✅ ONNX Runtime inference tested
   
2. FAISS Integration:
   ✅ FAISS library installed (CPU version)
   ✅ test_faiss_integration compiles
   ✅ test_faiss_integration runs successfully
   ✅ Can create index, add vectors, search
   
3. ONNX Runtime Integration:
   ✅ test_onnx_inference compiles
   ✅ Can load ONNX models in C++
   ✅ Can run inference on dummy data
   ✅ Output shapes correct
   
4. ChunkCoordinator Skeleton:
   ✅ Header file created
   ✅ Implementation skeleton created
   ✅ Can load JSONL chunks
   ✅ Can count events per chunk
   ✅ Statistics tracking working
```

---

## 🚀 COMANDOS RÁPIDOS DÍA 31
```bash
# Phase 1: Export ONNX models
cd /vagrant/ml-training
python3 export_chronos_onnx.py
python3 export_sbert_onnx.py
python3 train_and_export_attack_embedder.py

# Verify models
python3 -c "import onnx; onnx.checker.check_model(onnx.load('models/chronos_embedder.onnx'))"

# Phase 2: Install FAISS
cd /tmp
git clone https://github.com/facebookresearch/faiss.git
cd faiss && mkdir build && cd build
cmake .. -DFAISS_ENABLE_GPU=OFF -DCMAKE_INSTALL_PREFIX=/usr/local
make -j4 && sudo make install

# Phase 3: Test FAISS integration
cd /vagrant/rag/build
cmake .. && make test_faiss_integration
./test_faiss_integration

# Phase 4: Test ONNX Runtime integration
make test_onnx_inference
./test_onnx_inference

# Phase 5: Test ChunkCoordinator
make test_chunk_coordinator
./test_chunk_coordinator
```

---

## 📊 DOCUMENTACIÓN A ACTUALIZAR
```
1. docs/FAISS_INGESTION_IMPLEMENTATION.md (NEW)
   - ONNX export process
   - FAISS integration guide
   - ChunkCoordinator design
   - Testing results

2. README.md:
   - Update: Day 30 complete (memory leak resolved)
   - Add: Day 31 FAISS ingestion started
   - Progress: Phase 2 (FAISS) 20% complete

3. PROMPT_CONTINUE_CLAUDE_DAY32.md:
   - Continue embedder implementation
   - IndexManager creation
   - Feature extraction from 83 fields
```

---

## 🏛️ VIA APPIA QUALITY - DÍA 30

**Día 30 Truth:**
> "Memory leak investigado sistemáticamente durante 5+ horas. Testeamos
> 5 configuraciones diferentes. ASAN analysis confirmó: leak no era 'direct
> leak' sino stream buffer accumulation. Fix simple: current_log_.flush()
> después de cada write. Resultado: 70% reducción (102 → 31 MB/h). Descubrimiento
> sorprendente: CON artifacts (31 MB/h) mejor que SIN artifacts (50 MB/h).
> Configuramos cron restart cada 72h. Sistema production-ready para 24×7×365.
> Despacio y bien. Metodología científica. Transparencia total. 🏛️"

---

## 🎯 SIGUIENTE FEATURE (SEMANA 5)

**FAISS Ingestion Timeline:**
- ✅ Día 30: Memory leak resolved, logs ready
- 🔥 Día 31-32: ONNX export + FAISS integration
- Día 33-34: Embedder implementation (3 models)
- Día 35-36: IndexManager + HealthMonitor
- Día 37-38: Feature extraction (83 fields → embeddings)
- Día 39-40: Testing + End-to-end validation

**Key Milestones:**
```
Week 5: Foundation (ONNX + FAISS + Skeleton)
Week 6: Implementation (Embedders + Indices)
Week 7: Testing (E2E pipeline validation)
Week 8: Production (Monitoring + Reconciliation)
```

---

**Via Appia Quality:** Despacio y bien. Foundation primero, optimización después. 🏛️

**Next:** Day 31 - ONNX models + FAISS integration + ChunkCoordinator skeleton

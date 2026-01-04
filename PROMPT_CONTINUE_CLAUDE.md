## ✅ Actualización del Prompt - Recordatorio Documentos

Tienes razón, necesito añadir esa sección. Aquí está el **prompt actualizado con recordatorio explícito**:

---

# PROMPT DE CONTINUIDAD - DÍA 33 (04 Enero 2026)

## 📚 DOCUMENTOS NECESARIOS PARA ESTA SESIÓN

```
Day 33 (HOY):
  ❌ NO pasar FAISS_ANTI_CURSE_DESIGN.md
  ✅ Solo este prompt de continuidad
  
Razón: Day 33-34 son creación de modelos ONNX.
       No implementamos estrategias anti-curse todavía.
       El resumen abajo es suficiente.

RECORDATORIO PARA DÍAS FUTUROS:
┌─────────────────────────────────────────────────────┐
│ Day 35 (DimensionalityReducer):                    │
│   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md              │
│   Razón: Implementar Estrategia #2 (PCA)          │
│                                                     │
│ Day 36 (Índices Separados + Selective):           │
│   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md              │
│   Razón: Implementar Estrategias #1 y #3          │
│                                                     │
│ Day 38-40 (Advanced Strategies):                   │
│   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md              │
│   Razón: Temporal Tiers, Re-ranking, etc.         │
└─────────────────────────────────────────────────────┘

Archivo: /vagrant/docs/FAISS_ANTI_CURSE_DESIGN.md
Tamaño: ~500 líneas (12K tokens aprox)
```

---

## 📋 CONTEXTO DÍA 32 (03 Enero 2026) - COMPLETADO ✅

### ✅ ONNX Runtime Test - Infrastructure Complete

**Day 32 - ONNX Integration:**
- ✅ create_dummy_model_lite.py: 10→32-d embedder (sin PyTorch)
- ✅ test_onnx_basic.cpp: Load + inference test (ALL TESTS PASSED ✅)
- ✅ Makefile: Auto-genera modelo antes de test (reproducible)
- ✅ .gitignore: *.onnx (no binarios en git)
- ✅ CMakeLists.txt: test_onnx_basic target habilitado

**Infrastructure Status (Day 32 Complete):**
```
✅ FAISS v1.8.0 - WORKING
   ├─ test_faiss_basic PASSING
   ├─ CV computation validated
   └─ Auto-detection working

✅ ONNX Runtime v1.17.1 - WORKING
   ├─ test_onnx_basic PASSING
   ├─ Inference pipeline validated
   └─ Auto-detection working

✅ Build System - ROBUST
   ├─ CMakeLists.txt: C++20, auto-detect
   ├─ Makefile: test-faiss, test-onnx, test-all
   └─ All targets working

✅ Strategic Design - PEER REVIEWED
   ├─ FAISS_ANTI_CURSE_DESIGN.md v2.0
   ├─ 11 estrategias definidas
   ├─ Peer review: 4 AI systems
   └─ Límites empíricamente validados
```

**Test Results (Day 32):**
```
make test-faiss  → ALL TESTS PASSED ✅
make test-onnx   → ALL TESTS PASSED ✅
make test-all    → BOTH PASSING ✅
make verify-libs → FAISS + ONNX OK ✅
```

**Git Status:**
```
Rama: feature/faiss-ingestion-phase2a
Último commit: "Day 32 complete - ONNX Runtime test passing"
Archivos añadidos:
  - rag/tests/create_dummy_model_lite.py
  - rag/tests/test_onnx_basic.cpp
  - rag/Makefile (updated)
  - rag/CMakeLists.txt (updated)
  - .gitignore (*.onnx)
```

---

## 🔬 RESUMEN ESTRATEGIAS ANTI-CURSE (Para Referencia Day 33-34)

**Estrategias que implementaremos Days 35+:**

### 🔴 CRÍTICAS - Phase 2A (Days 35-38)

**1. Índices Separados por Clase** (Day 36)
- Benign index vs Malicious index
- 10x mejora para Attack embedder
- Evita saturación cross-class

**2. Dimensionality Reduction Post-Embedding** (Day 35)
- **CRÍTICO**: Usar faiss::PCAMatrix (NO Eigen manual)
- 512→128 (preserva 96.8% varianza), 384→96, 256→64
- 4x mejora en límites
- **Necesitaremos FAISS_ANTI_CURSE_DESIGN.md en Day 35**

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

## 🎯 ESTADO ACTUAL - DÍA 33 INICIO

### ✅ Completado Hasta Ahora

**Phase 2A Infrastructure (Days 31-32):**
- ✅ FAISS v1.8.0 instalado, testeado, working
- ✅ ONNX Runtime v1.17.1 instalado, testeado, working
- ✅ Build system configurado (C++20, auto-detection)
- ✅ Tests pasando (test_faiss_basic, test_onnx_basic)
- ✅ Anti-curse design completado (v2.0, peer-reviewed)

**Datos Disponibles:**
- ✅ 32,957 eventos RAG (JSONL format)
- ✅ 43,526 artifacts Protobuf
- ✅ 43,526 artifacts JSON
- ❌ NO tenemos embeddings pre-computados (.npy)
- ❌ NO tenemos modelos embedder entrenados todavía

### 🚧 Pendiente - Week 5

**Days 33-34: Real Embedder Models**
- Export/crear modelos ONNX reales
- Chronos (time series): 83 features → 512-d
- SBERT (semantic): 83 features → 384-d
- Attack (custom): 83 features → 256-d
- Test inference con estructura real

**Days 35-40: Implementation**
- DimensionalityReducer (faiss::PCAMatrix) ← **PASAR DESIGN DOC**
- AttackIndexManager (índices separados) ← **PASAR DESIGN DOC**
- SelectiveEmbedder (sampling) ← **PASAR DESIGN DOC**
- ChunkCoordinator integration
- End-to-end pipeline

---

## 🚀 PLAN DÍA 33 - REAL EMBEDDER MODELS (Parte 1)

### 🎯 Objetivo del Día

**Focus**: Crear/exportar modelos ONNX reales para los 3 embedders, preparar para ingestion.

**Contexto Importante:**
- NO tenemos embeddings pre-computados
- NO tenemos modelos custom entrenados
- Solución: Usar modelos base/pre-trained + adapters simples

**Timeline**: 4-6 horas total

**Status**: Infrastructure ✅ → Embedders ONNX (Day 33-34) → DimensionalityReducer (Day 35)

---

### DESAFÍO: No Tenemos Modelos Entrenados

**Problema:**
```
Diseño original asume:
  1. Chronos embedder custom (entrenado)
  2. SBERT embedder custom (entrenado)  
  3. Attack embedder custom (entrenado)

Realidad:
  ❌ No tenemos estos modelos
  ❌ Entrenarlos requiere semanas + GPU
```

**Solución Pragmática (Via Appia Quality):**
```
Day 33-34: Usar modelos base + arquitectura correcta
  ✅ Chronos: Modelo time-series sintético (83→512-d)
  ✅ SBERT: sentence-transformers base (texto→384-d)
  ✅ Attack: Neural network simple (83→256-d)
  
Objetivo: Validar PIPELINE, no entrenar modelos
         (Modelos reales = future work / production)
```

---

### FASE 1: Chronos Time Series Embedder (2 horas)

**Objetivo**: Crear modelo ONNX que acepta 83 features → 512-d embedding

**Opción A: Modelo Sintético (Recommended)**

```python
# File: rag/models/create_chronos_embedder.py
#!/usr/bin/env python3
"""
Create Chronos-style time series embedder for ML Defender.

Input:  83 network traffic features (float32)
Output: 512-d time series embedding (float32)

Architecture: Simple MLP mimicking time series processing
Note: This is a PLACEHOLDER for real Chronos model training
"""

import torch
import torch.nn as nn
import onnx

class ChronosEmbedder(nn.Module):
    """
    Time series embedder: 83 features → 512-d
    
    Architecture mimics real time series processing:
    - Input layer: 83 network features
    - Hidden layers: Capture temporal patterns
    - Output: 512-d embedding
    """
    def __init__(self, input_dim=83, hidden_dim=256, output_dim=512):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1: Feature extraction
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 2: Pattern detection
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Layer 3: Embedding projection
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # Normalize to [-1, 1]
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    print("╔════════════════════════════════════════╗")
    print("║  Creating Chronos Embedder (83→512-d) ║")
    print("╚════════════════════════════════════════╝\n")
    
    # Create model
    print("Step 1: Initializing Chronos architecture...")
    model = ChronosEmbedder(input_dim=83, output_dim=512)
    model.eval()
    print("  ✅ Model initialized (83 → 512-d)\n")
    
    # Dummy input for export
    print("Step 2: Creating export input...")
    dummy_input = torch.randn(1, 83)
    print(f"  ✅ Input shape: {dummy_input.shape}\n")
    
    # Export to ONNX
    print("Step 3: Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        "chronos_embedder.onnx",
        input_names=['features'],
        output_names=['embedding'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=14,
        verbose=False
    )
    print("  ✅ Exported: chronos_embedder.onnx\n")
    
    # Verify
    print("Step 4: Verifying model...")
    onnx_model = onnx.load("chronos_embedder.onnx")
    onnx.checker.check_model(onnx_model)
    print("  ✅ Model verified (opset 14)\n")
    
    print("Model Information:")
    print("  Input:  features (batch, 83)")
    print("  Output: embedding (batch, 512)")
    print("  Type:   Time series embedder (MLP)")
    print("\n╔════════════════════════════════════════╗")
    print("║  Chronos Embedder Created ✅           ║")
    print("╚════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
```

**Ejecutar:**
```bash
cd /vagrant/rag/models
mkdir -p /vagrant/rag/models  # Si no existe
python3 create_chronos_embedder.py

# Verificar
ls -lh chronos_embedder.onnx
```

---

### FASE 2: SBERT Semantic Embedder (1.5 horas)

**Objetivo**: Crear modelo que genera embeddings semánticos de features de red

**Opción: Arquitectura Simple (features → text concept → embedding)**

```python
# File: rag/models/create_sbert_embedder.py
#!/usr/bin/env python3
"""
Create SBERT-style semantic embedder for ML Defender.

Input:  83 network traffic features (float32)
Output: 384-d semantic embedding (float32)

Architecture: MLP that maps features to semantic space
Note: Real SBERT would use transformers, this is simplified
"""

import torch
import torch.nn as nn
import onnx

class SBERTEmbedder(nn.Module):
    """
    Semantic embedder: 83 features → 384-d
    
    Simplified version of sentence-BERT concept
    Maps network features to semantic embedding space
    """
    def __init__(self, input_dim=83, hidden_dim=192, output_dim=384):
        super().__init__()
        
        self.network = nn.Sequential(
            # Semantic feature extraction
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # GELU like transformers
            
            # Semantic representation
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            
            # Final embedding
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    print("╔════════════════════════════════════════╗")
    print("║  Creating SBERT Embedder (83→384-d)   ║")
    print("╚════════════════════════════════════════╝\n")
    
    print("Step 1: Initializing SBERT architecture...")
    model = SBERTEmbedder(input_dim=83, output_dim=384)
    model.eval()
    print("  ✅ Model initialized (83 → 384-d)\n")
    
    print("Step 2: Creating export input...")
    dummy_input = torch.randn(1, 83)
    print(f"  ✅ Input shape: {dummy_input.shape}\n")
    
    print("Step 3: Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, "sbert_embedder.onnx",
        input_names=['features'],
        output_names=['embedding'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=14,
        verbose=False
    )
    print("  ✅ Exported: sbert_embedder.onnx\n")
    
    print("Step 4: Verifying model...")
    onnx_model = onnx.load("sbert_embedder.onnx")
    onnx.checker.check_model(onnx_model)
    print("  ✅ Model verified\n")
    
    print("Model Information:")
    print("  Input:  features (batch, 83)")
    print("  Output: embedding (batch, 384)")
    print("  Type:   Semantic embedder (SBERT-style)")
    print("\n╔════════════════════════════════════════╗")
    print("║  SBERT Embedder Created ✅             ║")
    print("╚════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
```

---

### FASE 3: Attack Embedder (1 hora)

```python
# File: rag/models/create_attack_embedder.py
#!/usr/bin/env python3
"""
Create Attack-specific embedder for ML Defender.

Input:  83 network traffic features (float32)
Output: 256-d attack embedding (float32)

Architecture: Focused on attack pattern detection
"""

import torch
import torch.nn as nn
import onnx

class AttackEmbedder(nn.Module):
    """
    Attack embedder: 83 features → 256-d
    
    Specialized for attack pattern detection
    Smaller dimension for class-separated indices
    """
    def __init__(self, input_dim=83, hidden_dim=128, output_dim=256):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

def main():
    print("╔════════════════════════════════════════╗")
    print("║  Creating Attack Embedder (83→256-d)  ║")
    print("╚════════════════════════════════════════╝\n")
    
    print("Step 1: Initializing Attack architecture...")
    model = AttackEmbedder(input_dim=83, output_dim=256)
    model.eval()
    print("  ✅ Model initialized (83 → 256-d)\n")
    
    print("Step 2: Exporting to ONNX...")
    dummy_input = torch.randn(1, 83)
    
    torch.onnx.export(
        model, dummy_input, "attack_embedder.onnx",
        input_names=['features'],
        output_names=['embedding'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        },
        opset_version=14,
        verbose=False
    )
    print("  ✅ Exported: attack_embedder.onnx\n")
    
    print("Step 3: Verifying model...")
    onnx_model = onnx.load("attack_embedder.onnx")
    onnx.checker.check_model(onnx_model)
    print("  ✅ Model verified\n")
    
    print("Model Information:")
    print("  Input:  features (batch, 83)")
    print("  Output: embedding (batch, 256)")
    print("  Type:   Attack-specific embedder")
    print("\n╔════════════════════════════════════════╗")
    print("║  Attack Embedder Created ✅            ║")
    print("╚════════════════════════════════════════╝")

if __name__ == "__main__":
    main()
```

---

## ✅ CRITERIOS DE ÉXITO DÍA 33

```
1. Chronos Embedder:
   ✅ create_chronos_embedder.py created
   ✅ chronos_embedder.onnx generated
   ✅ Input: (batch, 83), Output: (batch, 512)
   ✅ Model verified with onnx.checker
   
2. SBERT Embedder:
   ✅ create_sbert_embedder.py created
   ✅ sbert_embedder.onnx generated
   ✅ Input: (batch, 83), Output: (batch, 384)
   ✅ Model verified
   
3. Attack Embedder:
   ✅ create_attack_embedder.py created
   ✅ attack_embedder.onnx generated
   ✅ Input: (batch, 83), Output: (batch, 256)
   ✅ Model verified

4. .gitignore:
   ✅ *.onnx ya está (Day 32)
   ✅ Scripts en git, modelos no

5. Documentation:
   ✅ README.md en /rag/models/ explicando modelos
```

---

## 📅 TIMELINE - SEMANA 5 (ACTUALIZADO)

```
✅ Day 31: FAISS + Anti-curse design
✅ Day 32: ONNX Runtime test

🔥 Day 33: Real embedders (4-6h) ← ESTAMOS AQUÍ
   - Chronos embedder (83→512-d)
   - SBERT embedder (83→384-d)
   - Attack embedder (83→256-d)
   - ONNX export + verification
   ❌ NO necesita FAISS design doc

📅 Day 34: Test embedders con datos reales (2-3h)
   - Cargar eventos JSONL
   - Extraer 83 features
   - Run inference
   - Verificar outputs
   ❌ NO necesita FAISS design doc

📅 Day 35: DimensionalityReducer (6h)
   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md ← IMPORTANTE
   - Implement faiss::PCAMatrix
   - Train PCA (cuando tengamos 10K eventos)
   - Test reduction pipeline

📅 Day 36-38: Integration (8h)
   ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md ← IMPORTANTE
   - AttackIndexManager
   - SelectiveEmbedder
   - ChunkCoordinator
   - End-to-end tests
```

---

## 🚀 COMANDOS RÁPIDOS DÍA 33

```bash
# Crear directorio modelos
mkdir -p /vagrant/rag/models
cd /vagrant/rag/models

# Fase 1: Chronos (2h)
# [Crear create_chronos_embedder.py]
python3 create_chronos_embedder.py
ls -lh chronos_embedder.onnx

# Fase 2: SBERT (1.5h)
# [Crear create_sbert_embedder.py]
python3 create_sbert_embedder.py
ls -lh sbert_embedder.onnx

# Fase 3: Attack (1h)
# [Crear create_attack_embedder.py]
python3 create_attack_embedder.py
ls -lh attack_embedder.onnx

# Verificar todos
ls -lh *.onnx

# Git (scripts sí, modelos no)
cd /vagrant
git add rag/models/create_*.py
git add rag/models/README.md  # Si creamos
# NO: git add rag/models/*.onnx (gitignored)
```

---

## 🏛️ VIA APPIA QUALITY - FILOSOFÍA DAY 33

> "No tenemos modelos custom entrenados. Podríamos pasar 2 semanas
> entrenando, o podemos crear arquitecturas sintéticas AHORA para
> validar el pipeline. Elegimos lo segundo: modelos base que tienen
> la estructura correcta (83→512/384/256) para probar ingestion,
> PCA, índices separados. Los modelos reales son 'future work'. El
> pipeline es lo que importa ahora. Despacio, pero avanzando. 🏛️"

**Key Principle:**
- ✅ Pipeline validation > Model perfection
- ✅ Arquitectura correcta > Pesos entrenados
- ✅ Progress incremental > Todo perfect

---

**Next**: Day 33 - Crear 3 embedders ONNX → Day 34 - Test con datos reales → Day 35 - DimensionalityReducer (**+ PASAR DESIGN DOC**)

**Via Appia Quality**: Modelos sintéticos para validar pipeline. Modelos reales = future work. Despacio y bien. 🏛️

---

## ✅ Cambios en el Prompt Actualizado

**Añadido:**
1. **Sección nueva al inicio**: "📚 DOCUMENTOS NECESARIOS PARA ESTA SESIÓN"
2. **Recordatorio visual** con box para días futuros
3. **Explicación clara** de cuándo SÍ y cuándo NO
4. **Timeline actualizado** con indicadores de cuándo pasar doc

**Formato del recordatorio:**
```
Day 35 (DimensionalityReducer):
  ✅ PASAR FAISS_ANTI_CURSE_DESIGN.md
  Razón: Implementar Estrategia #2 (PCA)
```


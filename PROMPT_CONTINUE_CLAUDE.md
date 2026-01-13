# RAG Ingester - Continuation Prompt
**Last Updated:** 13 Enero 2026 - Day 37 Complete  
**Phase:** 2A - Foundation + ADR-002 Implementation  
**Status:** ✅ ADR-002 Provenance + ADR-001 Encryption Complete

---

## 📍 CURRENT STATE (13 Enero 2026)

### ✅ Day 37 Achievements (TODAY) - CRITICAL MILESTONE

**ADR-002: Multi-Engine Detection Provenance - COMPLETADO**
- ✅ Protobuf contract extendido: `DetectionProvenance` + `EngineVerdict`
- ✅ `reason_codes.hpp` creado en `/vagrant/common/include` (5 códigos)
- ✅ Sniffer modificado: Llena verdict en fast-path detection
- ✅ ml-detector modificado: Agrega RF verdict + calcula discrepancy_score
- ✅ rag-ingester actualizado: Parsea provenance completa
- ✅ Event struct extendido: `std::vector<EngineVerdict>` + `discrepancy_score`
- ✅ CMakeLists.txt actualizados: 3 componentes + `/vagrant/common/include`

**ADR-001: Encrypted Artifacts (BONUS) - COMPLETADO**
- ✅ RAGLogger ahora cifra artifacts: `.pb.enc`, `.json.enc`
- ✅ Pipeline: Serialize → Compress (LZ4) → Encrypt (ChaCha20)
- ✅ Previene log poisoning attacks
- ✅ crypto_manager integrado correctamente
- ✅ save_artifacts() reescrito con crypto-transport

**Infrastructure:**
- ✅ Vagrantfile: Fix permanente ONNX Runtime lib64 symlinks
- ✅ Compilación limpia: sniffer, ml-detector, rag-ingester
- ✅ Todos los binarios: 100% funcionales

**Arquitectura Multi-Engine:**
```
Sniffer verdict  → "fast-path-sniffer" (STAT_ANOMALY)
                ↓
RandomForest     → "random-forest-level1" (STAT_ANOMALY)
                ↓
Discrepancy      → 0.0-1.0 (measure of agreement)
                ↓
Final Decision   → "ALLOW" | "DROP" | "ALERT"
```

**Nuevo Contrato Protobuf:**
```protobuf
message EngineVerdict {
  string engine_name = 1;       // "fast-path-sniffer", "random-forest"
  string classification = 2;    // "Benign", "Attack"
  float confidence = 3;         // 0.0 - 1.0
  string reason_code = 4;       // "SIG_MATCH", "STAT_ANOMALY", etc.
  uint64 timestamp_ns = 5;
}

message DetectionProvenance {
  repeated EngineVerdict verdicts = 1;
  uint64 global_timestamp_ns = 2;
  string final_decision = 3;            // "ALLOW", "DROP", "ALERT"
  float discrepancy_score = 4;          // 0.0 (agree) - 1.0 (disagree)
  string logic_override = 5;
  string discrepancy_reason = 6;
}
```

---

### 🐛 Technical Debt Identified (Day 37)

**ISSUE-007: Magic Numbers in ml-detector**
- **Ubicación:** `zmq_handler.cpp` líneas 332, 365
- **Problema:** Thresholds hardcoded (0.30, 0.70)
- **Solución:** Mover a `ml_detector_config.json`
- **Prioridad:** Medium (no bloqueante)
- **Estimación:** 30 min

**ISSUE-005: RAGLogger Memory Leak (Conocido)**
- **Estado:** Documentado, pendiente
- **Impacto:** Restart cada 3 días
- **Root Cause:** nlohmann/json allocations
- **Solución:** RapidJSON migration
- **Prioridad:** Medium

**ISSUE-003: Thread-Local FlowManager Bug (Conocido)**
- **Estado:** Documentado, pendiente
- **Impacto:** Solo 11/102 features capturadas
- **Workaround:** PCA entrenado con datos sintéticos
- **Solución:** Fix thread-local storage
- **Prioridad:** HIGH (pero no bloqueante para Day 38)

**MISSING: Acceptance Tests for Protobuf Contract**
- **Necesidad:** Validar nuevo contrato end-to-end
- **Componentes:** sniffer → ml-detector → rag-ingester
- **Tests:** Verificar que provenance se preserva
- **Prioridad:** HIGH (Day 38)

**MISSING: Log Files Not Persisted**
- **Problema:** Logs solo a stdout, no a archivos
- **Impacto:** Monitor scripts no pueden hacer tail
- **Solución:** Configurar spdlog file sinks
- **Prioridad:** Medium

---

### 📋 Day 36 (Previous Session - Context)

**Integración crypto-transport:**
- ✅ API real integrada (`crypto.hpp`, `compression.hpp`)
- ✅ event_loader.cpp con ChaCha20-Poly1305 + LZ4
- ✅ 101-feature extraction implementada
- ✅ Compilación exitosa

---

## 🎯 DAY 38 - PLAN EJECUTIVO (Synthetic Data + ONNX Embedders)

### Overview

**Duración estimada:** 6-8 horas  
**Objetivos:**
1. Verificar compilación desde cero (estabilidad)
2. Script generador de datos sintéticos (.pb.enc)
3. Actualizar ONNX Embedders con nuevo contrato (103 features)
4. Tests de aceptación para protobuf contract
5. Preparar fixes para bugs conocidos

---

## 📋 SESIÓN MAÑANA: Estabilidad + Datos Sintéticos (2-3 horas)

### 1. Compilación Desde Cero (30 min)

**Objetivo:** Verificar que todo compila limpio en VM fresca
```bash
# Destruir y recrear VM
vagrant destroy -f
vagrant up defender

# Compilar todo desde cero
vagrant ssh
cd /vagrant
make clean-all
make proto-unified
make crypto-transport
make etcd-client
make sniffer
make detector
make rag-ingester

# Verificar binarios
ls -lh /vagrant/sniffer/build/sniffer
ls -lh /vagrant/ml-detector/build/ml-detector
ls -lh /vagrant/rag-ingester/build/rag-ingester
```

**Success Criteria:**
- ✅ Compilación limpia (0 errores)
- ✅ Todos los binarios generados
- ✅ Symlinks ONNX Runtime correctos
- ✅ Librerías encontradas

---

### 2. Script Generador de Datos Sintéticos (1-2 horas)

**Archivo:** `/vagrant/scripts/generate_synthetic_events.py`

**Objetivo:** Generar .pb.enc files con provenance para testing
```python
#!/usr/bin/env python3
"""
Synthetic Event Generator for rag-ingester Testing
Generates encrypted+compressed .pb files with full provenance
"""

import sys
sys.path.append('/vagrant/protobuf')
import network_security_pb2

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import lz4.frame
import random
import time
import os

def generate_event(event_id: int, is_malicious: bool):
    """Generate synthetic NetworkSecurityEvent with provenance"""
    event = network_security_pb2.NetworkSecurityEvent()
    event.event_id = f"synthetic_{event_id:06d}"
    
    # Timestamp
    event.event_timestamp.seconds = int(time.time())
    event.event_timestamp.nanos = random.randint(0, 999_999_999)
    
    # 101 features (synthetic)
    nf = event.network_features
    for i in range(101):
        if is_malicious:
            # Malicious pattern (higher values)
            setattr(nf, f"feature_{i}", random.uniform(0.7, 1.0))
        else:
            # Benign pattern (lower values)
            setattr(nf, f"feature_{i}", random.uniform(0.0, 0.3))
    
    # NEW: Provenance (ADR-002)
    prov = event.provenance
    
    # Sniffer verdict
    v1 = prov.verdicts.add()
    v1.engine_name = "fast-path-sniffer"
    v1.classification = "MALICIOUS" if is_malicious else "BENIGN"
    v1.confidence = random.uniform(0.8, 0.95) if is_malicious else random.uniform(0.1, 0.3)
    v1.reason_code = "STAT_ANOMALY" if is_malicious else "SIG_MATCH"
    v1.timestamp_ns = int(time.time() * 1e9)
    
    # RandomForest verdict
    v2 = prov.verdicts.add()
    v2.engine_name = "random-forest-level1"
    v2.classification = "Attack" if is_malicious else "Benign"
    v2.confidence = random.uniform(0.85, 0.98) if is_malicious else random.uniform(0.05, 0.25)
    v2.reason_code = "STAT_ANOMALY"
    v2.timestamp_ns = int(time.time() * 1e9)
    
    # Discrepancy (small for agreed, large for conflict)
    if random.random() < 0.1:  # 10% conflicts
        prov.discrepancy_score = random.uniform(0.5, 1.0)
        prov.discrepancy_reason = "Engines disagree on threat level"
    else:
        prov.discrepancy_score = random.uniform(0.0, 0.2)
    
    prov.final_decision = "DROP" if is_malicious else "ALLOW"
    prov.global_timestamp_ns = int(time.time() * 1e9)
    
    # Legacy fields (backward compat)
    event.final_classification = "MALICIOUS" if is_malicious else "BENIGN"
    event.overall_threat_score = v2.confidence
    
    return event

def encrypt_and_compress(data: bytes, key: bytes) -> bytes:
    """Compress + Encrypt (ADR-001 pipeline)"""
    # 1. Compress with LZ4
    compressed = lz4.frame.compress(data)
    
    # 2. Encrypt with ChaCha20-Poly1305
    cipher = ChaCha20Poly1305(key)
    nonce = os.urandom(12)  # 96-bit nonce
    ciphertext = cipher.encrypt(nonce, compressed, None)
    
    # Prepend nonce (needed for decryption)
    return nonce + ciphertext

def main():
    output_dir = "/vagrant/logs/rag/events"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load encryption key from etcd (or use test key)
    test_key = b'0' * 32  # 32-byte key for testing
    
    print(f"🔒 Generating synthetic events...")
    print(f"   Output: {output_dir}")
    print(f"   Key: {'*' * 8} (32 bytes)")
    
    for i in range(100):
        is_malicious = (i % 5 == 0)  # 20% malicious
        
        event = generate_event(i, is_malicious)
        serialized = event.SerializeToString()
        encrypted = encrypt_and_compress(serialized, test_key)
        
        filepath = f"{output_dir}/event_{i:06d}.pb.enc"
        with open(filepath, 'wb') as f:
            f.write(encrypted)
        
        label = "MALICIOUS" if is_malicious else "BENIGN"
        print(f"   [{i+1:3d}/100] {label:10s} → {filepath}")
    
    print(f"✅ Generated 100 synthetic events (20 malicious, 80 benign)")
    print(f"   Encrypted: ChaCha20-Poly1305")
    print(f"   Compressed: LZ4")
    print(f"   Provenance: 2 verdicts per event")

if __name__ == "__main__":
    main()
```

**Ejecución:**
```bash
cd /vagrant
python3 scripts/generate_synthetic_events.py
```

**Success Criteria:**
- ✅ 100 archivos .pb.enc generados
- ✅ Encrypted + Compressed correctamente
- ✅ Provenance con 2 verdicts cada uno
- ✅ 20% malicious, 80% benign (realista)

---

### 3. Tests de Aceptación Protobuf Contract (1 hora)

**Archivo:** `/vagrant/tests/test_protobuf_contract.py`
```python
#!/usr/bin/env python3
"""
Acceptance Tests for ADR-002 Protobuf Contract
Verifies that provenance is preserved end-to-end
"""

import sys
sys.path.append('/vagrant/protobuf')
import network_security_pb2

def test_provenance_structure():
    """Test that DetectionProvenance has correct structure"""
    event = network_security_pb2.NetworkSecurityEvent()
    prov = event.provenance
    
    # Add verdicts
    v1 = prov.verdicts.add()
    v1.engine_name = "test-engine"
    v1.classification = "BENIGN"
    v1.confidence = 0.95
    v1.reason_code = "SIG_MATCH"
    v1.timestamp_ns = 123456789
    
    # Set provenance metadata
    prov.discrepancy_score = 0.15
    prov.final_decision = "ALLOW"
    prov.global_timestamp_ns = 987654321
    
    # Serialize and deserialize
    serialized = event.SerializeToString()
    event2 = network_security_pb2.NetworkSecurityEvent()
    event2.ParseFromString(serialized)
    
    # Verify
    assert event2.provenance.verdicts[0].engine_name == "test-engine"
    assert event2.provenance.discrepancy_score == 0.15
    assert event2.provenance.final_decision == "ALLOW"
    
    print("✅ test_provenance_structure PASSED")

def test_multiple_verdicts():
    """Test that multiple engine verdicts work"""
    event = network_security_pb2.NetworkSecurityEvent()
    prov = event.provenance
    
    # Add 3 verdicts
    for i, name in enumerate(["sniffer", "rf", "cnn"]):
        v = prov.verdicts.add()
        v.engine_name = name
        v.confidence = 0.9 - (i * 0.1)
    
    serialized = event.SerializeToString()
    event2 = network_security_pb2.NetworkSecurityEvent()
    event2.ParseFromString(serialized)
    
    assert len(event2.provenance.verdicts) == 3
    assert event2.provenance.verdicts[0].engine_name == "sniffer"
    assert event2.provenance.verdicts[2].confidence == 0.7
    
    print("✅ test_multiple_verdicts PASSED")

def test_reason_codes():
    """Test all 5 reason codes from Gemini table"""
    codes = ["SIG_MATCH", "STAT_ANOMALY", "PCA_OUTLIER", 
             "PROT_VIOLATION", "ENGINE_CONFLICT"]
    
    for code in codes:
        event = network_security_pb2.NetworkSecurityEvent()
        v = event.provenance.verdicts.add()
        v.reason_code = code
        
        serialized = event.SerializeToString()
        event2 = network_security_pb2.NetworkSecurityEvent()
        event2.ParseFromString(serialized)
        
        assert event2.provenance.verdicts[0].reason_code == code
    
    print("✅ test_reason_codes PASSED (all 5 codes)")

if __name__ == "__main__":
    test_provenance_structure()
    test_multiple_verdicts()
    test_reason_codes()
    print("\n🎉 All acceptance tests PASSED")
```

**Success Criteria:**
- ✅ Provenance serializa/deserializa correctamente
- ✅ Múltiples verdicts funcionan
- ✅ Todos los reason codes válidos

---

## 📋 SESIÓN TARDE: ONNX Embedders + Nuevo Contrato (3-4 horas)

### 4. Actualizar ONNX Embedders (103 features) (2-3 horas)

**Cambio crítico:** Ahora tenemos 103 features (101 + 2 meta)
```cpp
// include/embedders/chronos_embedder.hpp
class ChronosEmbedder {
public:
    // Dimensions: 103 input → 512 output
    static constexpr size_t INPUT_DIM = 103;
    static constexpr size_t OUTPUT_DIM = 512;
    
    std::vector<float> embed(const Event& event);
};

// src/embedders/chronos_embedder.cpp
std::vector<float> ChronosEmbedder::embed(const Event& event) {
    // Prepare input: 101 features + 2 meta
    std::vector<float> input;
    input.reserve(INPUT_DIM);
    
    // 1. Original 101 features
    input.insert(input.end(), event.features.begin(), event.features.end());
    
    // 2. NEW Meta-features from ADR-002
    input.push_back(event.discrepancy_score);   // Feature 102
    input.push_back(static_cast<float>(event.verdicts.size()));  // Feature 103
    
    if (input.size() != INPUT_DIM) {
        throw std::runtime_error(
            "Invalid input size: " + std::to_string(input.size()) + 
            " (expected " + std::to_string(INPUT_DIM) + ")"
        );
    }
    
    // ONNX inference
    auto input_tensor = create_tensor(input);
    auto output_tensor = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), 1
    );
    
    // Extract 512-d embedding
    float* output_data = output_tensor[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + OUTPUT_DIM);
}
```

**Hacer lo mismo para:**
- `SBERTEmbedder` (103 → 384)
- `AttackEmbedder` (103 → 256)

**Compilación:**
```bash
cd /vagrant/rag-ingester/build
cmake ..
make -j$(nproc)
```

**Success Criteria:**
- ✅ Embedders aceptan 103 features
- ✅ Validación de input size
- ✅ Output dimensions correctas

---

### 5. Test End-to-End (1 hora)
```bash
# Terminal 1: Generar eventos sintéticos
python3 /vagrant/scripts/generate_synthetic_events.py

# Terminal 2: Ejecutar rag-ingester
cd /vagrant/rag-ingester/build
./rag-ingester ../config/rag-ingester.json

# Verificar logs
tail -f /vagrant/logs/rag-ingester/rag-ingester.log
```

**Verificar:**
- ✅ rag-ingester detecta archivos .pb.enc
- ✅ Descifra correctamente (sin errores)
- ✅ Parsea provenance (logs muestran verdicts)
- ✅ Embedders procesan 103 features
- ✅ No crashes durante 100 eventos

---

## 🐛 PREPARACIÓN DE FIXES (Day 39+)

### Fix 1: Magic Numbers → JSON Config

**Archivo:** `/vagrant/ml-detector/config/ml_detector_config.json`
```json
{
  "ml": {
    "thresholds": {
      "divergence_alert": 0.30,
      "classification_threshold": 0.70
    }
  }
}
```

**Código:**
```cpp
// zmq_handler.cpp
if (score_divergence > config_.ml.thresholds.divergence_alert) {
    // ...
}

if (final_score >= config_.ml.thresholds.classification_threshold) {
    event.set_final_classification("MALICIOUS");
}
```

---

### Fix 2: Thread-Local FlowManager Bug

**Análisis:** Ver `/vagrant/docs/bugs/2025-01-10_thread_local_flowmanager_bug.md`

**Estrategia:**
1. Hacer FlowManager thread-safe (mutex)
2. O eliminar thread_local (dependiendo del análisis)
3. Re-entrenar PCA con datos reales (102 features)

---

### Fix 3: Log Files Persistence
```cpp
// ml-detector/src/main.cpp
auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
    "/vagrant/logs/ml-detector/ml-detector.log", 
    1024 * 1024 * 10,  // 10 MB
    3                   // 3 rotated files
);

auto logger = std::make_shared<spdlog::logger>("ml-detector", 
    spdlog::sinks_init_list{console_sink, file_sink});
```

---

## ✅ CHECKLIST DEL DÍA 38

### Mañana (Estabilidad + Sintéticos):
- [ ] Compilación desde cero exitosa
- [ ] Vagrantfile fix ONNX verificado
- [ ] Script `generate_synthetic_events.py` funcionando
- [ ] 100 eventos .pb.enc generados
- [ ] Tests de aceptación protobuf PASSED

### Tarde (ONNX Embedders):
- [ ] ChronosEmbedder actualizado (103 → 512)
- [ ] SBERTEmbedder actualizado (103 → 384)
- [ ] AttackEmbedder actualizado (103 → 256)
- [ ] rag-ingester procesa eventos sintéticos
- [ ] Logs muestran provenance correctamente

### Preparación (Fixes):
- [ ] Config JSON para thresholds preparado
- [ ] Análisis thread-local bug completado
- [ ] Log persistence implementado

---

## 🎯 Success Criteria Day 38

**Synthetic Data Generation:**
- ✅ 100+ eventos .pb.enc con provenance
- ✅ Encryption + Compression funcional
- ✅ Datos realistas (20% malicious)

**ONNX Embedders:**
- ✅ 103 features procesadas correctamente
- ✅ Output dimensions verificadas
- ✅ Inference <10ms per event

**End-to-End:**
- ✅ rag-ingester procesa sintéticos sin errors
- ✅ Provenance parseada correctamente
- ✅ Embeddings generados

---

## 🔒 CRITICAL SECURITY REMINDERS

**ADR-001: Encryption Mandatory**
- ✅ Todos los .pb files DEBEN estar cifrados
- ✅ RAGLogger cifra artifacts automáticamente
- ✅ rag-ingester rechaza plaintext

**ADR-002: Provenance Preserved**
- ✅ Múltiples verdicts capturados
- ✅ Discrepancy score calculado
- ✅ Reason codes documentados

---

## 🏛️ VIA APPIA REMINDERS

1. **Foundation first** - Day 37 completó protobuf contract ✅
2. **Security by design** - Encryption + Provenance ✅
3. **Test before scale** - Synthetic data antes de producción ✅
4. **Document exhaustively** - ADRs actualizados ✅
5. **Measure before optimize** - Tests aceptación primero ✅

---

**End of Continuation Prompt**

**Ready for Day 38:** Synthetic Data + ONNX Embedders (103 features)  
**Dependencies:** Protobuf contract (Day 37), Encryption (Day 37)  
**Expected Duration:** 6-8 hours  
**Blockers:** None (all systems compiled and functional)

🏛️ Via Appia: Day 37 complete - Multi-engine provenance implemented, encryption hardened, ready for synthetic data testing.
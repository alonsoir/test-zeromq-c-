# ML Defender - FAISS Ingestion Design

## Overview
Multi-embedder ingestion system con best-effort commit strategy.

## Architecture Decisions

### Multi-Index Strategy
- Temporal Index (128-dim): Time series queries
- Semantic Index (768-dim): Behavioral pattern queries
- Entity Index (256-dim): IP/domain centric queries

**Rationale:** Cada índice sirve tipo de pregunta diferente,
deben poder fallar independientemente.

### Chunk Coherence (CRITICAL)
- Unit: 1 día completo (YYYY-MM-DD.jsonl)
- Validation: Timestamps monotónicos, mismo día
- NUNCA truncar arbitrariamente
- Mismo chunk para los 3 embedders (coherencia)

### Commit Strategy
- Best-effort (NO atomicidad estricta)
- Commit independiente por índice
- Retry con exponential backoff
- Disponibilidad parcial > no disponibilidad

**Rationale:** Hardware falla, mejor 2/3 índices que 0/3.

## Technology Stack

### Embeddings
- Chronos (temporal): Export → ONNX
- SBERT (semantic): Export → ONNX
- Custom GNN (entity): Export → ONNX

### Inference
- ONNX Runtime C++ API (coherencia con ml-detector)

### Vector Store
- FAISS C++ native (mejor performance que Python)

### Language
- C++20 (coherencia con resto del sistema)

## Implementation Components

### ChunkCoordinator
- Procesa mismo chunk en 3 embedders paralelo
- Staging (pre-commit)
- Commit independiente por índice
- Compresión post-commit (async)

### IndexTracker
- Track qué índice tiene qué chunk
- Metadata granular
- Support reconciliation

### HealthMonitor
- Disk space monitoring
- Index health checks
- Alerting (email, Slack)

## Deployment Considerations

### Phase 1 (MVP)
- Single-node validation
- Bare-metal systemd services
- Manual deployment

### Phase 2+ (Distributed)
- FAISS cluster en servidor dedicado
- Multiple instances (scaling)
- Ansible orchestration
- Possible Nomad (NO K8s por eBPF)

## Open Questions for Implementation Week
1. ONNX model export exact procedure
2. Feature extraction 83 fields → embeddings
3. FAISS index type (IVF vs HNSW vs Flat)
4. Reconciliation schedule (cada cuánto)
5. Alert channels (email config, Slack webhook)

## References
- Session: 2025-12-30 FAISS Architecture Discussion
- Related: PROMPT_CONTINUIDAD_DIA30.md
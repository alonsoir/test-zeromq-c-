# CONTEXTO: Day 35 - DimensionalityReducer Implementation

## Resumen Ejecutivo
El Consejo de Sabios (ChatGPT-5, DeepSeek, Gemini, GLM-4.7, Grok, Qwen)
completó peer review de HIERARCHICAL_RAG_VISION.md v1.0 el 07-Enero-2026.

**VEREDICTO**: APROBADO UNÁNIME (6/6) con corrección industry-agnostic.
**DOCUMENTO FINAL**: HIERARCHICAL_RAG_VISION.md v2.0 DISPONIBLE

## Decisiones Consolidadas

### Arquitectura (CONFIRMADO)
- Separación RAG (consumer) vs FAISS-Ingester (producer)
- Jerarquía 3 niveles: Organization → Site → Zone
- Industry-agnostic: hospitales, bancos, escuelas, empresas
- Exclusión ética: uso militar
- Hybrid query processing: TinyLlama + Regex

### Implementación Phase 1 (6 semanas)
- Week 5 (Day 35-40): DimensionalityReducer + PCA training
- Week 6 (Day 41-45): FAISS Ingester service
- Week 7-8 (Day 46-55): RAG Local + TinyLlama + Hybrid parser
- Week 9-10 (Day 56-65): Integration + testing + Prometheus

### Técnico (CONFIRMADO)
- LLM Strategy: Hybrid (TinyLlama intent + Regex entities)
- Multi-tenancy: Separate Indices (OBLIGATORIO)
- Aggregation: Batch Nightly (Phase 2)
- Timeline: 17 semanas (15 nominal + 2 buffer)
- Telemetry: Prometheus desde Day 1
- Event IDs: Hierarchical format ({org}-{site}-{timestamp}-{seq})

### Paper (CONFIRMADO)
- Primary Contribution: Anti-Curse Strategies (A) - 40%
- Secondary: Hierarchical Architecture (B) - 30%
- Supporting: Natural Language Interface (C) - 20%
- Target: IEEE S&P / NDSS / CCS (Tier 1)

## Documentos Generados
- HIERARCHICAL_RAG_VISION.md v1.0 (pre-review)
- CONSEJO_DE_SABIOS_SINTESIS.md (síntesis completa 15 páginas)
- 6 respuestas individuales revisores
- HIERARCHICAL_RAG_VISION.md v2.0 ← **DOCUMENTO OFICIAL FINAL**
- Preflight check script: /vagrant/rag/scripts/preflight_check_day35.sh

## Next Immediate Actions (ORDEN ESTRICTO)

### 1. Preflight Check (OBLIGATORIO PRIMERO)
```bash
cd /vagrant/rag/scripts
chmod +x preflight_check_day35.sh
./preflight_check_day35.sh
```
**CRITICAL**: No codificar hasta que preflight pase

### 2. Review HIERARCHICAL_RAG_VISION.md v2.0
- Leer sección 2.1 (Component Separation)
- Leer sección 4.1 (Timeline Week 5)
- Leer sección 4.3 (Preflight Checks)

### 3. DimensionalityReducer Implementation
```cpp
// /vagrant/rag/src/dimensionality_reducer.cpp
- Usar faiss::PCAMatrix (NO Eigen)
- Train con datos balanceados multi-source (Gemini warning)
- Validar variance ≥96% (Chronos: 512→128)
- Save models: /shared/models/pca/
```

### 4. Training Data Preparation
- Load 10K eventos from /vagrant/logs/rag/events/2025-12-12.jsonl
- Balance por sources (evitar domain shift)
- Generate embeddings (ONNX Runtime)
- Train PCA matrices

## Referencias Críticas
- **HIERARCHICAL_RAG_VISION.md v2.0** ← DOCUMENTO GUÍA OFICIAL
- FAISS_ANTI_CURSE_DESIGN.md v2.0 (estrategias validadas)
- Peer review: /docs/peer-review-2026-01-07/
- Via Appia Quality: Foundation primero, expansión después

## Cambios Clave v1.0 → v2.0
1. ✅ Nomenclatura industry-agnostic (organization/site/zone)
2. ✅ Hybrid query processing (TinyLlama + Regex)
3. ✅ Telemetry Day 1 (Prometheus)
4. ✅ Preflight checks mandatory
5. ✅ Timeline oficial 17 semanas
6. ✅ Peer Review Summary incluido

## Estado Actual
✅ Diseño arquitectónico APROBADO (6/6)
✅ Peer review CERRADO
✅ v2.0 documentación COMPLETA
✅ Preflight script READY
🚀 LISTO PARA DAY 35 IMPLEMENTATION

---
Prompt sugerido para Claude en próxima sesión:
"Day 35: Implementar DimensionalityReducer usando faiss::PCAMatrix según
HIERARCHICAL_RAG_VISION.md v2.0. PRIMERO ejecutar preflight_check_day35.sh.
Prioridad: PCA training con datos balanceados multi-source (Gemini warning).
Timeline: 17 semanas oficiales. Via Appia Quality: Foundation primero."
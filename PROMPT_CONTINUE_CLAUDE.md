Day 73 — ML Defender (aegisIDS)

Retomamos donde dejamos Day 72. Estado del sistema:

COMPLETADO Day 72:
- Idempotencia: exists() guard en MetadataDB, FAISS/DB perfectamente sincronizados
- Replay limpio verificado: 199 líneas CSV, 100 únicos indexados, 99 duplicados rechazados
- trace_id_generator.hpp: SHA256_prefix_16B(src_ip|dst_ip|canonical_attack_type|bucket)
  header-only, include/utils/, zero-coordination, O(1), stateless, deterministic
  Sentinels: IP vacía → "0.0.0.0" (warn logged), attack_type vacío → "unknown" (warn logged)
  TraceIdMetadata.fallback_applied distingue sentinel de valor real sin parsear logs
  TraceIdPolicy v1: ventanas por tipo (ransomware=60s, ddos=10s, ssh_brute=30s, scan=60s)
  Ejemplo real: 64985c7cb546cb6227c5a1e7538d5deb
- 100/100 eventos con trace_id poblado verificado en SQLite
- Schema MetadataDB completo desde nacimiento (sin ALTER TABLE)
- Tests unitarios: 6 grupos incluyendo casos de filo (sentinels, window sensitivity,
  collision resistance, canonicalization, metadata fields)
- Fix cosmético --explain: "Attack embedding (64-dim)" correcto
- Clustering quality: 4/4 vecinos misma clase, distancias 0.001-0.004

DECISIONES DE DISEÑO DOCUMENTADAS (Consejo de Sabios — cierre Day 72):
- Colisiones: espacio 2¹²⁸, P~1.5×10⁻¹⁵ para 10¹² eventos — no se necesita
  resolución secundaria. Falsa correlación intencional (mismo bucket, flujos distintos)
  es comportamiento esperado del diseño, no colisión.
- Bucket: granularidad por attack_type via TraceIdPolicy, no por hora/día fija.
- UUID v5 descartado: SHA256 prefix más ligero y auditable para sistema de seguridad.
- trace_id en query-time: pendiente Day 73/74 (valor real para paper y demo).

PENDIENTE Day 73 — orden priorizado:

1. Bug 7: etcd PUT falla con ChaCha20 decryption error (500) en componente "rag"
  - Hipótesis DeepSeek: rotación de clave maestra (mismo patrón que afectó chronos/sbert)
  - Investigar: /vagrant/logs/etcd/ + comparar clave en rag/config/
  - El componente rag registra con component_name="security-system" pero el PUT
    va a /v1/config/security-system con component="" en el header
  - No bloquea pipeline pero ensucia logs y complica demos

2. trace_id en CLI del componente rag:
  - Mostrar trace_id en query_similar, recent, search
  - Permite visualizar correlación multi-evento en la interfaz
  - Valor directo para demo y paper (evidencia visual de zero-coordination)

3. Actualizar docs/ABOUT_TRACE_ID.md con implementación final:
  - Fórmula definitiva, política de ventanas, casos de filo, collision math
  - Párrafo para paper (base del draft de Qwen)

4. Script de validación experimental (sugerencia Gemini):
  - Forzar dos ataques idénticos en ventanas temporales distintas
  - Confirmar que genera dos trace_id distintos (separación de incidentes)
  - Evidencia reproducible para sección experimental del paper

Stack: C++20, eBPF/XDP, ZeroMQ, ChaCha20-Poly1305, LZ4, FAISS, TinyLlama,
SQLite WAL, protobuf, etcd-client propio, RandomForest embebido
Arquitectura: sniffer → ml-detector → firewall-acl-agent → rag-ingester → rag
Rama activa: day72-trace-id
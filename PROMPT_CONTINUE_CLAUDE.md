Day 73 — ML Defender (aegisIDS)

Retomamos donde dejamos Day 72. Estado del sistema:

COMPLETADO Day 72:
- Idempotencia: exists() guard en MetadataDB, FAISS/DB perfectamente sincronizados
- Replay limpio verificado: 199 líneas CSV, 100 únicos indexados, 99 duplicados rechazados
- trace_id_generator.hpp: SHA256(src_ip|dst_ip|canonical_attack_type|bucket), header-only
  Sentinels: IP vacía → "0.0.0.0" (warn logged), attack_type vacío → "unknown" (warn logged)
  TraceIdMetadata.fallback_applied distingue sentinel de valor real
- 100/100 eventos con trace_id poblado verificado en SQLite
- Schema MetadataDB completo desde nacimiento (sin ALTER TABLE)
- Tests unitarios: 6 grupos incluyendo casos de filo
- Fix cosmético --explain: "Attack embedding (64-dim)" correcto
- Clustering quality: 4/4 vecinos misma clase, distancias 0.001-0.004

PENDIENTE Day 73:
- Bug 7: etcd PUT falla con ChaCha20 decryption error (500) en componente "rag"
  El componente rag registra con component_name="security-system" pero el PUT
  va a /v1/config/security-system con component="" en el header.
  Hipótesis: el etcd-client del rag no pasa el component_name correcto al
  CryptoManager en el momento del PUT — la key derivada no coincide con la
  que usó el etcd-server para cifrar. No bloquea el pipeline principal.

- Preparación paper: transición de desarrollo a escritura académica (Q1 2026)

Stack: C++20, eBPF/XDP, ZeroMQ, ChaCha20-Poly1305, LZ4, FAISS, TinyLlama,
SQLite WAL, protobuf, etcd-client propio, RandomForest embebido
Arquitectura: sniffer → ml-detector → firewall-acl-agent → rag-ingester → rag
# ML Defender — Day 71 (continúa desde Day 70)

## Estado al cierre de Day 70

### rag-ingester — FUNCIONAL ✅
- 100 eventos en MetadataDB con source_ip, dest_ip, timestamp_ms poblados
- attack.faiss: 26K, 100 vectores persistidos
- chronos.faiss / sbert.faiss: 45 bytes (vacíos — path .pb.enc inutilizable por rotación de clave)
- replay_on_start=true: procesa CSVs históricos al arrancar
- Checkpoint cada 100 eventos funcionando
- Firewall correlation: 0 matches (desfase temporal entre datasets sintéticos — por diseño, no bug)

### Bugs corregidos en Day 70
1. FAISS no persistía: checkpoint ausente en CSV callback
2. Replay sólo buscaba today.csv: corregido a iteración de todo el directorio
3. MetadataDB vacía: INSERT violaba NOT NULL de `timestamp` silenciosamente
4. source_ip/dest_ip no parseadas: cols 2/3 ausentes en parse_section1
5. spdlog en csv_dir_watcher rompía el test: eliminado

### Siguiente — rag-local
Arrancar y validar el servicio rag-local (query engine sobre FAISS + MetadataDB).
Ruta probable: /vagrant/rag-local o /vagrant/rag
Verificar que lee los índices de /vagrant/shared/indices/
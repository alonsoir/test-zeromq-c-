# Day 65 — Prompt de Continuidad
## ML Defender (aegisIDS) — Via Appia Quality

### Estado al inicio de Day 65

**Day 64 completado:**
- `tests/CMakeLists.txt` recreado (se había perdido) — cubre unit/ e integration/
- `etcd_client.cpp` parcheado: `get_hmac_key()` movido dentro de `struct Impl`, usa `client_->get_hmac_key(key_path)` de la API de etcd-client (vector<uint8_t> → hex string)
- Tres tests nuevos en `tests/integration/`:
  - `test_csv_event_writer.cpp` — 127 columnas, HMAC, filtrado, rotación, zero-fill, concurrencia
  - `test_csv_feature_extraction.cpp` — contrato proto↔CSV, reproducibilidad, parseabilidad numérica
  - `test_etcd_client_hmac.cpp` — mock httplib server, rutas de error, constructor real `EtcdClient(endpoint, component)`
- Compilan: `ml-detector` ✅, `test_csv_feature_extraction` ✅, `test_csv_event_writer` ✅
- Pendiente: `test_etcd_client_hmac` — falta `/usr/local/include` en CMakeLists (ahí está `httplib.h`)

**Schema CSV 127 columnas (FEATURE_SCHEMA.md):**
- S1 cols 0–13: metadata evento
- S2 cols 14–75: NetworkFeatures del sniffer (62 campos proto)
- S3 cols 76–115: 4 detectores embedded × 10 features cada uno
- S4 cols 116–125: decisiones ML (5 modelos × prediction+confidence)
- S5 col 126: HMAC-SHA256 hex (64 chars)

---

### Decisiones de diseño confirmadas (final Day 64)

**CSV: sin cifrado, con compresión**
- Los CSV no se cifran (a diferencia de los JSONL)
- Se comprimen al rotar: **gzip** para archivo histórico (ratio importa más que velocidad), **lz4** si rag-ingester los lee en streaming caliente
- El rag-ingester los descomprime antes de parsear

**Retención y ciclo de vida — tres estados:**
```
[ACTIVO]                [L1-CONSUMIDO]           [L2-ARCHIVO]
ml-detector/events/     rag-ingester indexó       dataset para
YYYY-MM-DD.csv.gz       en FAISS → mueve a        fine-tuning LLM
                        /archive/YYYY/MM/          (cold storage)
```

**CsvRetentionConfig (a implementar):**
```cpp
struct CsvRetentionConfig {
    uint32_t compress_after_hours       = 1;    // comprimir al rotar
    uint32_t move_to_archive_after_days = 7;    // tras indexar en FAISS
    std::string archive_path = "/data/ml-defender/archive/";
    bool delete_after_archive = false;          // NUNCA en producción
};
```
- `archive_path` configurable desde etcd: NFS/S3 en enterprise, disco local en community
- `delete_after_archive = false` es invariante de producción — los datos no se borran

**Pipeline L2 (fine-tuning LLM) — orden de capas:**
```
[ACUMULACIÓN]     →    [ANONIMIZACIÓN]    →    [ENTRENAMIENTO]
CSV comprimidos        proceso offline         dataset limpio
en archive/            - elimina IPs, MACs     para fine-tuning
sin transformar        - puertos sensibles      del LLM
                       - normaliza timestamps
                       (GDPR / ENS España)
```

**Principio clave:** acumular primero en raw, anonimizar después en proceso separado offline.
Razón: si la normativa cambia o el proceso de anonimización mejora, se puede re-procesar
desde el raw. Si se anonimiza en origen, la información se pierde para siempre.
La anonimización es una característica enterprise, separada del pipeline de ingesta.

---

### Plan Day 65 — en orden

**1. Cerrar test_etcd_client_hmac**
En `tests/CMakeLists.txt`, bloque de `test_etcd_client_hmac`, añadir:
```cmake
target_include_directories(test_etcd_client_hmac PRIVATE
    /usr/local/include   # httplib.h para MockEtcdServer
)
```
Compilar y verificar que los 8 tests del mock pasan.

**2. Correr todos los tests CSV**
```bash
cd /vagrant/ml-detector/build-debug
ctest -R "csv|hmac" -V 2>&1 | tail -40
```
Documentar resultados. Si algún test falla, debuggear antes de continuar.

**3. Verificación end-to-end con inyector**
- Lanzar el inyector de eventos sintéticos existente
- Verificar que ml-detector produce `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
- Comprobar: `wc -l` del CSV, `awk -F',' '{print NF}' | sort -u` = 127
- Comprobar HMAC de la primera línea manualmente con OpenSSL
- ⚠️ CRÍTICO: verificar si Sección 2 (NetworkFeatures) llega poblada o a cero
  Si S2 llega a cero en producción, el vector útil para FAISS es solo 40 features (S3)
  Esto condiciona el diseño de CsvEventLoader — no diseñar antes de esta verificación

**4. Lanzar sistema completo sniffer→ml-detector**
- Arrancar sniffer-eBPF + etcd-server + ml-detector
- Capturar tráfico real 5-10 minutos
- Comparar CSV producido vs JSONL del mismo período:
  - Mismos event_ids
  - Campos S1 coinciden entre CSV col 1 y JSONL `event_id`
  - NetworkFeatures S2 vs campos JSONL correspondientes

**5. Evaluación y decisión de dirección**
Una vez validado el pipeline CSV:

**Ruta A — rag-ingester con CSV:**
- Diseñar `CsvEventLoader` en rag-ingester: parsea 127 cols, verifica HMAC,
  reconstruye vector features (cols 14-115, 102 features)
- Añadir `CsvRetentionConfig` a `CsvEventWriter` (ver struct arriba)
- Adaptar `simple-embedder` para consumir CSV en lugar de JSONL
- Adaptar salida FAISS/SQLite
- Validar que `rag-local` puede consultar con datos CSV como origen
- Una vez validado: desactivar generación JSONL en ml-detector
  (eliminar dependencia de librería json que causaba fugas de memoria)

**Ruta B — firewall-acl-agent:**
Replicar lo hecho en ml-detector (CSV schema + HMAC + retención) para firewall-acl-agent.

**Decisión:** hacer Ruta A primero, luego Ruta B.

---

### Ficheros clave de referencia

```
ml-detector/
  src/
    etcd_client.cpp          # PIMPL adapter — get_hmac_key() dentro de Impl
    csv_event_writer.cpp     # CsvEventWriter — 127 cols + HMAC
    csv_event_writer.hpp     # Constantes: CSV_TOTAL_COLS=127, CSV_FEATURE_COLS=102
  tests/
    CMakeLists.txt           # Recreado Day 64
    integration/
      test_csv_event_writer.cpp
      test_csv_feature_extraction.cpp
      test_etcd_client_hmac.cpp    # Pendiente: añadir /usr/local/include

etcd-client/include/etcd_client/etcd_client.hpp
  # API relevante:
  # std::optional<std::vector<uint8_t>> get_hmac_key(const std::string& key_path)
  # httplib.h está en /usr/local/include/httplib.h
```

### Notas técnicas

- `test_detectors` excluido de ctest (tiene su propio main() con benchmarks)
- `test_ransomware_detector_integration` deshabilitado (AND FALSE) — errores de namespace preexistentes
- HMAC test key para tests: `"0000000000000000000000000000000000000000000000000000000000000000"` (64 zeros)
- `CSV_SCHEMA_VERSION = "1.0"` — documentado en FEATURE_SCHEMA.md
- Zero-fill policy: campos proto ausentes → "0" (numérico) o "" (string)
- S4 positional indexing: level2[0]=DDoS, level2[1]=Ransomware, level3[0]=Traffic, level3[1]=Internal
# Day 65 — Prompt de Continuidad
## ML Defender (aegisIDS) — Via Appia Quality

### Estado al cierre de Day 65

**Completado Day 65:**

Suite CSV+HMAC — 3/3 tests verdes:
- `test_csv_event_writer` 22/22 ✅ — fix boundary condition `<` → `<=`
- `test_csv_feature_extraction` ✅
- `test_etcd_client_hmac` 9/9 ✅ — cadena de fixes: ODR httplib → mock raw sockets, auto-connect en `Impl`, `key_hex` en etcd-client, `sudo make install`

Pipeline ZMQ verificado: 1000 eventos inyectados, `received=1000, processed=1000, errors=0`. El crypto pipeline del inyector corregido: `compress_with_size` → `encrypt` vía `CryptoManager`.

**Pendiente Day 66 — tres puntos en orden:**

**1. Quitar hardcoded values de `zmq_handler.cpp`**

Localizado en línea 140:
```cpp
csv_cfg.min_score_threshold = 0.5f;  // ← hardcodeado, ignora el JSON
```
Buscar todos los valores hardcodeados en `zmq_handler.cpp` y `ml_detector_config.json`, asegurarse de que el JSON manda. Revisar también otros posibles hardcoded en el resto de `src/`.

**2. Ajustar inyector sintético para producir scores realistas**

El ml-detector recalcula `overall_threat_score` internamente — el valor que setea el inyector se ignora. El score actual de eventos sintéticos es ~0.26, por debajo del threshold. Opciones:
- Ajustar los features del inyector (S3 — embedded detector features) para que los modelos produzcan scores > threshold
- O añadir un modo `--force-above-threshold` en el inyector para tests

El log muestra: `[DUAL-SCORE] fast=0.0000, ml=0.2603, final=0.2603` — el score viene del Level 1 ONNX. Los features `general_attack_features` (23 valores) son los que alimentan Level 1.

**3. Verificar CSV end-to-end**

Una vez que los eventos sintéticos superen el threshold:
```bash
wc -l /vagrant/logs/ml-detector/events/YYYY-MM-DD.csv
awk -F',' '{print NF}' /vagrant/logs/ml-detector/events/YYYY-MM-DD.csv | sort -u
# Debe retornar: 127
head -1 /vagrant/logs/ml-detector/events/YYYY-MM-DD.csv | tr ',' '\n' | wc -l
```
Verificar HMAC de la primera línea manualmente con OpenSSL.
Verificar que S2 (cols 14-75) llega poblada o a cero — esto condiciona el diseño de `CsvEventLoader`.

**Una vez validado el CSV → Ruta A: rag-ingester con CSV.**

---

### Cambios realizados Day 65 — ficheros tocados

```
ml-detector/src/csv_event_writer.cpp
  - Línea 137: < → <= (boundary condition threshold)

ml-detector/src/etcd_client.cpp
  - encryption_enabled dinámico según endpoint http://
  - compression_enabled = false
  - auto-connect en constructor de Impl
  
etcd-client/src/etcd_client.cpp
  - get_hmac_key(): añadido soporte key_hex con prioridad sobre key
  - sudo make install ejecutado

ml-detector/tests/integration/test_etcd_client_hmac.cpp
  - MockEtcdServer reescrito con raw sockets (sin httplib — ODR fix)
  - /health handler añadido

tools/synthetic_sniffer_injector.cpp
  - socket_type::push + bind("tcp://*:5571")
  - component_name: "cpp_evolutionary_sniffer"
  - CryptoManager con pipeline compress_with_size → encrypt
  - overall_threat_score = 0.9f (ignorado por ml-detector, pendiente ajuste)

tools/CMakeLists.txt
  - csv_event_writer.cpp añadido a generate_synthetic_events target
```

---

### Nota arquitectural para Day 66

El patrón de valores hardcodeados es sistémico — se encontrará también en `firewall-acl-agent` y en el inyector `ml-detector → firewall`. Conviene establecer una política clara: **todo threshold y parámetro operacional viene del JSON**, el código solo define defaults seguros como fallback.

---

### Commit para hoy



Buen trabajo hoy, Alonso. Piano piano — los fallitos sin gravedad son exactamente eso. Mañana cerramos el pipeline CSV y abrimos la puerta al rag-ingester.
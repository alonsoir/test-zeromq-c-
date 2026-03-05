## Estado DAY 76 → Continuidad DAY 77

### Pipeline
6/6 componentes RUNNING y estables. ml-detector sobrevive 60s+ sin crash.
SIGSEGV ByteSizeLong eliminado definitivamente.

### Feature actual
Cerrar: feature/rag-firewall-hmac-security → merge a main
Abrir:  feature/ring-consumer-real-features

### Objetivo DAY 77
Reemplazar los 0.5f sentinel de init_embedded_sentinels() con valores
reales extraídos del tráfico de red, para poder validar F1-score contra
CTU-13 con datos correctos.

### Arquitectura actual (ring_consumer.cpp)

Ruta principal: populate_protobuf_event()
1. Llama ml_extractor_.populate_ml_defender_features(flow_stats, proto_event)
   → Si flow existe en ShardedFlowManager: rellena los 40 campos reales ✅
   → Si flow NO existe: cae al sentinel 0.5f (fallback)
2. Llama run_ml_detection() → infiere pero NO escribe resultados al proto (TODO)
3. Llama init_embedded_sentinels() → sobrescribe con 0.5f ← PROBLEMA

El bug actual: init_embedded_sentinels() se llama DESPUÉS de
populate_ml_defender_features(), sobrescribiendo los valores reales.

### Primer análisis DAY 77
Verificar si populate_ml_defender_features() realmente rellena los 40 campos:

vagrant ssh -c "grep -n 'populate_ml_defender_features\|mutable_ddos\|mutable_ransomware_embedded\|mutable_traffic\|mutable_internal' /vagrant/sniffer/src/userspace/ml_defender_features.cpp | head -30"

Luego ver ml_defender_features.cpp completo para entender qué extrae
del FlowStatistics y qué queda vacío.

### Lógica correcta a implementar
En populate_protobuf_event():
1. populate_ml_defender_features() — valores reales si flow existe
2. init_embedded_sentinels() SOLO si los submensajes siguen vacíos
   (usar has_ddos_embedded() para no sobrescribir valores reales)

En send_fast_alert() y send_ransomware_features():
→ No hay flow context → sentinels 0.5f son permanentes (Phase 2)

### Estado de run_ml_detection()
Función incompleta — infiere los 4 modelos pero no escribe resultados
de vuelta al proto_event. Los TODO están documentados en el código.
Esto es trabajo DAY 77+ también.

### Ficheros clave
/vagrant/sniffer/src/userspace/ring_consumer.cpp
/vagrant/sniffer/src/userspace/ml_defender_features.cpp
/vagrant/sniffer/include/ml_defender_features.hpp

### Tests de regresión — no tocar
sniffer/tests/test_proto3_embedded_serialization.cpp     3/3 ✅
ml-detector/tests/unit/test_rag_logger_artifact_save.cpp 3/3 ✅

### Objetivo final de la feature
Que el pcap relay con CTU-13 produzca F1-score válido:
make test-replay-neris   (492K eventos, botnet Neris)
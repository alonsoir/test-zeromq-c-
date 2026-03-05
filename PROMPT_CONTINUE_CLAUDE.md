### Segundo trabajo DAY 77 — crear la feature branch
```bash
git checkout -b feature/ring-consumer-real-features
```

### Objetivo de la feature
Reemplazar NaN sentinels con valores reales extraídos del tráfico
para validar F1-score contra CTU-13 Neris (492K eventos).

### Primer diagnóstico — verificar el extractor
```bash
vagrant ssh -c "grep -n 'set_' /vagrant/sniffer/src/userspace/ml_defender_features.cpp | wc -l"
# debe ser ~40 setters

vagrant ssh -c "grep -n 'set_' /vagrant/sniffer/src/userspace/ml_defender_features.cpp"
```

### Problema arquitectónico a resolver
En populate_protobuf_event(), el orden actual es incorrecto:

1. populate_ml_defender_features()  ← valores reales si flow existe
2. run_ml_detection()
3. init_embedded_sentinels()        ← SOBRESCRIBE con NaN ❌

Fix correcto (ChatGPT5):
```cpp
auto* ddos = net->mutable_ddos_embedded();
if (ddos->ByteSizeLong() == 0) init_ddos_sentinels(ddos);
// idem para ransomware, traffic, internal
```

ByteSizeLong()==0 → submensaje vacío → aplicar sentinel
ByteSizeLong()>0  → populate_ml_defender_features() ya escribió → no tocar

### Trabajo pendiente en run_ml_detection()
Función incompleta — infiere los 4 modelos pero los resultados
no se escriben al proto_event. Los TODO están documentados.
Completar esto es parte de la feature.

### Tests de regresión — no tocar
sniffer/tests/test_proto3_embedded_serialization.cpp     3/3 ✅
ml-detector/tests/unit/test_rag_logger_artifact_save.cpp 3/3 ✅

### Validación final
make test-replay-neris   # CTU-13 Neris botnet, 492K eventos
# Objetivo: F1-score > 0.90 en detección DDoS + Ransomware
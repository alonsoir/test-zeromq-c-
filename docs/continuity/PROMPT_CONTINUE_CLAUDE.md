# CONTINUIDAD DAY262 — Arqueología del contrato DDoS HECHA. Siguiente: cerrar grieta A (geo en el sniffer), luego MEDIR grieta B.

## Estado (medido DAY261, verificar al retomar)
main = 2c5f8f4a (PR #141 gate merged). Ramas del gate borradas. scratch/ en
.git/info/exclude. Sin commitear: docs/BACKLOG.md (lleva DEBT-GATE-BARE-CTEST ya
pegada, con FORMATO ## a arreglar → negrita) + docs/continuity/PROMPT_CONTINUE_CLAUDE.md.
Untracked de antes: "Cierre day255 reparacion ddos.md", seed-repro/*.json.

## HECHO DAY261 (arqueología, no código de producción)
Mapa de los TRES contratos DDoS del protobuf: ddos_features=100 FANTASMA (nadie
escribe/consume), ddos_embedded=112 VIVO (la cabeza de 9 de Fase 1), 9 de DDOSFeatures.py.
Circuito de ddos_embedded medido extremo a extremo (sniffer escribe ml_defender_features.cpp:730;
DOS predicts — ring_consumer.cpp:1554 y zmq_handler.cpp:568 —; consume 9 reindexadas).
Las 9 features SON el invariante DDoS (mecanismo, no técnica) — medido en
ml_defender_features.cpp:23-120. Detalle completo en HALLAZGOS-DAY261.

## PENDIENTE (plan de 3, en orden)
1. GRIETA A: quitar geo de extract_ddos_features del sniffer (ml_defender_features.cpp:34),
   dejar 9. Termina el reindex de Fase 1 en el PRODUCTOR (Fase 1 solo tocó el consumidor).
   Rama fuera de main. Manivela en la VM, push desde host. Árbitro: compilador + el
   test de serialización (el 0.5f de geo debe caer).
2. GRIETA B (LA de diseño): source_ip_dispersion NO es función del flujo — sale del
   aggregator (ventana 30s del sensor). Covariate shift latente para Fase 2. MEDIR
   distribución dataset-offline vs serve-en-vivo antes de decidir (a) replicar ventana
   / (b) sacarla / (c) reformular. Consejo de Sabios + material de paper.
3. CICDDoS2019 por las 9 (menos lo que caiga en el paso 2). Sobre raíl sano. Verificar
   URL como el binetflow de Neris. Demostrar por medición train==serve, no asumir.

## SOSPECHADO (sin medir)
Dos level1 conviven: config → level1_attack_detector.onnx, plugin → xgboost_cicids2017.ubj.
Cuál decide en serve = sin medir. Y level1 tiene su propia grieta P0 (features[14]
Init_Win=0.0f muerto; Init_Win no está en el protobuf) — INDEPENDIENTE de DDoS, es
higiene del level1, no prerequisito de la cabeza DDoS.

## Aviso honesto
Raíl sano ≠ detector útil. Las 9 se entrenan sobre Betas sintéticas. La inequivocidad
se gana en el paso 3, no en la fontanería. Sin tag nuevo.
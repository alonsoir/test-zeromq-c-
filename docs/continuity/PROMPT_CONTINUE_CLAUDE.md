# CONTINUIDAD DAY263 — Grieta A REFUTADA (fantasma medido). Plan DDoS 3→2. Siguiente: MEDIR grieta B (source_ip_dispersion / aggregator).

## Estado (verificar al retomar)
Rama fase2/sniffer-reindex-ddos — OJO el nombre describe trabajo que resultó FANTASMA;
renombrar (git branch -m) o anotarlo en el cuerpo del PR antes de mergear. main = 2c5f8f4a
(PR #141 gate merged), PROTEGIDA (PR only). scratch/ en .git/info/exclude. Untracked de
antes, no tocar: "Cierre day255 reparacion ddos.md", seed-repro/*.json.
Commits de DAY262 (docs; si ya disparados, verde): (1) housekeeping BACKLOG — DEBT-GATE-BARE-
CTEST formato ##→negrita + newline final; (2) DEBT-DDOS-GEO-DEADCODE nueva; (3) continuity —
grieta A = fantasma, plan 3→2. Si NO disparados: docs/BACKLOG.md + este fichero sin commitear.

## HECHO DAY262 (medición, no código de producción)
GRIETA A REFUTADA de punta a punta. DAY261 leyó "el sniffer emite 10 con geo" confundiendo
DOS funciones homónimas extract_ddos_features:
- ml_defender_features.cpp:23 (flow, DDoSFeatures*) = PRODUCTOR del submensaje protobuf;
  setea los 10 campos POR NOMBRE (set_syn_ack_ratio … set_geographical_concentration:34 …).
  Enriquecimiento. Vivo, se queda.
- ring_consumer.cpp:1457 (proto_event) = EXTRACTOR del vector del modelo; lee del submensaje
  y OMITE geo por nombre → struct de 9 (idx7=traffic_escalation, idx8=resource_saturation)
  → ddos_detector_.predict() (misma clase DDoSDetector de 9 de DAY255, ring_consumer.hpp:211).
  Los dos lados leen por NOMBRE de campo protobuf → orden irrelevante, skew posicional IMPOSIBLE.
  Fase 1 estaba completa en productor Y consumidor desde DAY255; el productor nunca necesitó
  reindex porque el consumidor no llama al getter de geo. No había nada que arreglar.
  Deuda derivada nacida: DEBT-DDOS-GEO-DEADCODE (BACKLOG) — geo pasa a write-only-dead cuando
  se deprecue el artefacto RAG (col 83 csv_event_writer + rag_logger son sus últimos lectores);
  borrado coordinado (productor + reserved 8 en el proto + col 83 + fixtures) gated en ese
  evento. Ortogonal a la cabeza DDoS. NO tocar antes: geo sigue viva mientras exista col 83.

## PENDIENTE (plan de 2, en orden)
1. GRIETA B (LA de diseño). source_ip_dispersion NO es función del flujo. Medido en
   ml_defender_features.cpp:65-73: param flow comentado, sale del aggregator_ (ventana 30s
   del sensor); if(!aggregator_) return MISSING_FEATURE_SENTINEL. A DIFERENCIA de geo, SÍ la
   consume la cabeza (1 de las 9) → con aggregator frío el sentinel entra VIVO a predict(),
   en un slot sobre el que el bosque parte. Es el skew tipo-geo pero con dientes.
   PRIMER movimiento = MEDIR, no diseñar: distribución de source_ip_dispersion en dos
   regímenes lado a lado — dataset-offline (CICDDoS2019, flujo a flujo) vs serve-en-vivo
   (ventana 30s). El número decide entre (a) replicar la ventana sobre el dataset /
   (b) sacarla del contrato train+serve como se hizo con geo (pero geo no la comía el modelo
   y esta sí — cuidado) / (c) reformular como función del flujo. Consejo de Sabios + paper.
   Prerequisitos antes de arrancar: (i) acceso/URL CICDDoS2019 verificado como el binetflow
   de Neris; (ii) cómo replicar la ventana 30s del aggregator sobre flujos offline — lo que
   hace la feature no-trivial: offline hay flujos, no estado de sensor; (iii) el test de
   "misma distribución" a correr (KS, o histogramas + percentiles). El número decide, no la
   corazonada.
2. CICDDoS2019 por las 9 (menos lo que caiga en el paso 1). El trabajo grande, sobre raíl
   sano. Demostrar POR MEDICIÓN que cada feature mide lo mismo en train y serve, no asumir.

## SOSPECHADO (sin medir, INDEPENDIENTE de DDoS)
Dos level1 conviven: config → level1_attack_detector.onnx, plugin → xgboost_cicids2017.ubj.
Cuál decide en serve = sin medir. level1 tiene su propia grieta P0 (features[14]
Init_Win_bytes_forward=0.0f hardcodeado; Init_Win NO está en el protobuf NetworkFeatures) —
higiene del level1, NO prerequisito de la cabeza DDoS.

## Aviso honesto (paper, no sobre-vender)
Raíl sano ≠ detector útil. Las 9 se entrenan hoy sobre Betas sintéticas y se sirven sobre
tráfico sin DDoS etiquetado. Contrato sano = condición NECESARIA, no suficiente. La
inequivocidad se gana en el paso 2 (ground-truth real por el mismo extractor), no en la
fontanería. Hallazgo de método fuerte para la tesis: el patrón de skew de Fase 1 GENERALIZA
(reapareció como sospecha en el sniffer y en el level1); el método lo caza incluso cuando la
caza concluye "aquí no había bug" — grieta A refutada por medición es medir ganándole a
votar tanto como un bug encontrado. Sin tag nuevo.

## Invariantes
main PROTEGIDA (PR only). Un commit una idea. add explícito por fichero. git grep o fichero
concreto — NUNCA grep -rn desde raíz (site/search/search_index.json y build/ te comen la
noche). No encadenar salidas grandes en un bloque. Manivela en la VM, push desde el HOST.
Compilador/ctest = árbitro. Revert → recompilar. vboxsf: sha host↔VM antes de compilar.
sed BSD peligroso: contar el match o editar con Python.
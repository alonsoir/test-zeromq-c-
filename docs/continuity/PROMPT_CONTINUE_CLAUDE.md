# CONTINUIDAD DAY264 — Grieta B MEDIDA de punta a punta (definición serve + acople ransomware). Siguiente: CONSTRUIR el análogo offline sobre CICDDoS2019 y comparar distribuciones.

## Estado (verificar al retomar)
Sesión DAY263+ fue PURA MEDICIÓN (arqueología por código y validación de dataset): CERO
commits de producción, cero código nuevo. main PROTEGIDA (PR only), sin cambios. Rama
`fase2/sniffer-reindex-ddos` sigue describiendo trabajo fantasma (grieta A) — renombrar
(`git branch -m`) o anotar en el cuerpo del PR antes de mergear. Los hallazgos de esta
sesión están en dos docs (pendientes de committear en rama → PR): `grieta-b-source-ip-
dispersion.md` y la nota DAY261-263 añadida a `reparacion-cabeza-ddos.md`. Dataset ya en
disco: `ml-training/datasets/CICDDoS2019/` (01-12 = 11 CSV ~22 GB / 50 063 112 filas;
03-11 = 7 CSV ~8.7 GB / 20 364 525 filas), validado fila-a-fila contra CIC.

## HECHO DAY263 (medición, no código de producción)
GRIETA B = `source_ip_dispersion` — definición operacional del lado serve CERRADA:
- `ml_defender_features.cpp:64-73`. Fórmula `min(log2(uniq+1)/log2(ev+2), 1.0)`;
  `ev==0`→`0.0f`; `!aggregator_`→`MISSING_FEATURE_SENTINEL(-9999)`.
- Ventana 30 s SLIDING global (sin clave de agrupación).
- El estadístico cuenta `src_ip ∪ dst_ip` en un solo set (`time_window_aggregator.cpp:60-61`)
  — NO solo origen, pese al nombre. (Trampa cazada: no construir el offline sobre Source IP a secas.)
- `add_event` = solo `push_back` al ring buffer, SIN filtro whitelist/DNS. Cap
  `max_events=10000` (ambos lados). Bajo flood el cap MUERDE → ventana efectiva = últimos
  10000 eventos, no 30 s de reloj.
  ACOPLE ESTRUCTURAL (hallazgo fuerte): la feature DDoS NO tiene ventana propia. Lee de un
  `TimeWindowAggregator` que POSEE el processor de RANSOMWARE, inyectado lazy en
  `ring_consumer.cpp:819` (`set_aggregator(ransomware_processor_->get_aggregator())`, misma
  instancia, sin copia, incondicional al tráfico ransomware). El ÚNICO `add_event` del repo
  está en la ruta ransomware. → source_ip_dispersion solo lleva señal viva si la ruta
  ransomware está construida y alimentando. Mismo patrón "decisiones razonables por
  separado, incoherentes juntas" de grieta A, ahora en el CABLEADO. Generaliza: ~12 features
  DDoS leen del mismo aggregator.
  GRANULARIDAD: aggregator ALIMENTADO por PAQUETE (`ring_consumer.cpp:621` =
  `ransomware_processor_->process_packet`; `tw_event.bytes = event.packet_len`); feature
  LEÍDA por flujo (`process_event_features`→`get_flow_stats_copy`→`populate_ml_defender_features`).
  DATASET CICDDoS2019 validado: 88 cols CICFlowMeter-V3, mismo hash cabecera `46e0bdae` en
  los 18. Timestamp formato ÚNICO `AAAA-DD-MM HH:MM:SS.ffffff` (día-mes, año 2018).
  `Infinity` confinado a cols 22-23. Cols necesarias presentes (Source IP=3, Dest IP=5,
  Timestamp=8, Flow ID=2).

## PENDIENTE (en orden)
1. CONSTRUIR el análogo offline (script nuevo en `ml-training/scripts/ddos_detection/`,
   untracked hasta que pase). Por cada flujo a `t` (Timestamp de fin): ventana global
   `[t-30s, t] ∩ últimos 10000 eventos`, `Source IP ∪ Destination IP` en un set (uniq),
   `event_count` = eventos en ventana, aplicar `min(log2(uniq+1)/log2(ev+2), 1)`.
    - Serve cuenta por PAQUETE → expandir cada fila a `Total Fwd + Bwd Packets`, o declarar
      sesgo acotado (decidir por medición, no por corazonada).
    - Parser de fecha EXPLÍCITO: `pd.to_datetime(col, format='%Y-%d-%m %H:%M:%S.%f')`
      (pandas por defecto invierte día-mes).
    - Empezar por el CSV más pequeño (`03-11/Portmap.csv`, 75 MB) para iterar barato.
2. COMPARAR distribuciones offline vs serve (KS o histogramas+percentiles). El número
   decide la horquilla: (a) replicar la ventana offline / (b) sacarla del contrato como
   geo — pero geo NO la comía el modelo y esta SÍ / (c) reformular como función del flujo
   — casi muerta por construcción (dispersión necesita un conjunto). Fork vivo = (a) vs
   (b). Consejo de Sabios + paper.
3. Etiquetar la asimetría al llevarlo al Consejo/paper: cada CSV es un ataque casi puro →
   el offline es "ventana dentro de un ataque puro", no "producción mezclada". Válido para
   comparar la FORMA, no el régimen.
4. Paso grande (después): CICDDoS2019 por las 9, demostrar POR MEDICIÓN que cada feature
   mide lo mismo en train y serve, sobre raíl sano.

## SOSPECHADO (sin medir)
- Residual de granularidad: confirmar que `process_packet` se llama 1×/paquete en el mismo
  dispatch (`ring_consumer.cpp` 615-645). Solo importa si el factor paquetes/flujo pesa.
- level1: su propia grieta P0 (features[14] `Init_Win_bytes_forward=0.0f` hardcodeado; no
  está en el protobuf `NetworkFeatures`). CICDDoS2019 col 74 lo trae → medible cuando le
  toque a level1. NO prerequisito de la cabeza DDoS.

## Aviso honesto (paper, no sobre-vender)
Raíl sano ≠ detector útil. Contrato sano = necesario, no suficiente. El método de la tenaza
GENERALIZA: grieta A refutada por medición vale tanto como un bug encontrado; hoy se
cazaron además dos fantasmas (filtro de whitelist inexistente, y "feature muerta a 0.0f
siempre" refutada). Y se destapó un acople real que sin medir no salía.

## Deudas nuevas (BACKLOG, correlacionar con la tarea que las ejecuta)
- `ml_defender_features.cpp.bak.day79` TRACKEADO ensucia `git grep` → `git rm`.
- Dos formas de contar IPs conviven en `TimeWindowAggregator` (global src∪dst en cpp:60-61
  vs selectiva `count_unique_ips` cpp:445, solo-destino con flag true). No mezclar.
- (Diseño, no bug) Acople DDoS↔ransomware vía aggregator compartido: documentar como
  dependencia conocida; decidir si se quiere aggregator propio para la cabeza DDoS.

## Invariantes
main PROTEGIDA (PR only). Un commit una idea. `add` explícito por fichero. `git grep` o
fichero concreto — NUNCA `grep -rn` desde raíz. No encadenar salidas grandes en un bloque.
Manivela en la VM, push desde el HOST. Compilador/ctest = árbitro. Revert → recompilar.
vboxsf: sha host↔VM antes de compilar. sed BSD peligroso: contar el match o editar con
Python. Para grieta B específicamente: evento=paquete, cap 10000, src∪dst (no src),
parser de fecha `%Y-%d-%m` explícito, Infinity en cols 22-23.
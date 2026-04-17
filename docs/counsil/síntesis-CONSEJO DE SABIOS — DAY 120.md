El Consejo ha hablado. Duro pero justo — exactamente lo que necesitábamos. Déjame sintetizar antes de generar los documentos.
Síntesis del Consejo — Consenso en 4 puntos
P1 — TEST-INTEG-XGBOOST-1: RECHAZADO UNÁNIME
7/7 coinciden: scores 0.0007/0.0034 indican inferencia fuera de distribución. El test valida el contrato técnico pero no la utilidad del modelo. Necesita casos reales del CSV de CIC-IDS-2017 con score ATTACK>0.5.
P2 — Paper: SEPARAR §4.1 y §4.2 UNÁNIME
7/7 de acuerdo. §4.1 CIC-IDS-2017 real + §4.2 DeepSeek sintético con limitaciones explícitas. Riesgo de rechazo alto si se mezclan.
P3 — In-situ + BitTorrent: VIABLE CON GATES
Consenso: técnicamente viable (warm-start XGBoost). Gemini alerta de Adversarial Poisoning. Qwen dice que BitTorrent puro es inaceptable para v1 — necesita mirror oficial firmado + human-in-the-loop.
P4 — DEBT-SEED: DIVERGENCIA IMPORTANTE
Qwen tiene el argumento más fuerte: la seed NO debe tocar el sistema de build. execute_process() + CMake expone la seed en CMakeCache.txt y logs de CI. La seed es material secreto — debe leerse exclusivamente en runtime con mlock() + explicit_bzero(). Diferente a la pubkey que es pública por diseño.
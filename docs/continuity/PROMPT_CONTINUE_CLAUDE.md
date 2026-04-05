Soy Alonso (aRGus NDR, ML Defender, DAY 109). Lee el transcript de DAY 108.

DAY 108 CERRADO:
- PASO 1 ✅ swap CTX_ETCD_TX/RX confirmado necesario (ADR-027)
- PASO 2 ✅ invariant fail-fast en 3 adaptadores etcd_client.cpp:
  std::terminate() si encryption_enabled && component_config_path.empty()
- PASO 3 ✅ provision.sh formalizado (9 fixes: permisos, seed maestro,
  symlinks JSON, libsodium so.23→so.26, install_shared_libs, tmux, libsnappy,
  etcd-client, libcrypto_transport rebuild)
- PASO 4 ✅ vagrant destroy && vagrant up → 6/6 RUNNING sin intervención manual
- ADR-026 (P2P Fleet) + ADR-027 (CTX swap) commiteados
- Consejo DAY 108: 5/5 respondieron

ORDEN DAY 109 (no saltarse):

ANTES de PHASE 2b — dos fixes rápidos del Consejo:

FIX-A) MLD_ALLOW_UNCRYPTED en 3 adaptadores etcd_client.cpp:
Sustituir std::terminate() por escape hatch explícito:
if (getenv("MLD_ALLOW_UNCRYPTED")) { cerr << "FATAL[DEV]..."; return false; }
else { std::terminate(); }
Verificar 6/6 tras rebuild.

FIX-B) provision.sh: mkdir -p /vagrant/rag-security/config
+ symlink JSON en sección de rag-security
+ eliminar warning "Config dir no existe aún"

PHASE 2b:
plugin_process_message() en rag-ingester
Contrato Consejo: READ-ONLY (ctx_readonly.payload = nullptr; ctx_readonly.length = 0)
El plugin decide accept/reject — no modifica payload antes de FAISS
Gate: TEST-INTEG-4b (MessageContext, result_code=0)
Patrón: igual que firewall-acl-agent DAY 105
Archivos: rag-ingester/src/main.cpp, rag-ingester/CMakeLists.txt,
rag-ingester/config/rag-ingester.json

DEUDA PENDIENTE (no bloqueante):
- Unificar sniffer bajo SeedClient (eliminar get_encryption_seed manual)
- ADR-025 (Plugin Integrity Ed25519) — post PHASE 2 completa
- TEST-PROVISION-1 como gate CI formal (post PHASE 2b)
- ADR-028: RAG Ingestion Trust Model (antes de write-capable plugins)
- arXiv submit/7438768 pendiente moderación

NOTA QWEN: accede vía chat.qwen.ai, tiene pesos propios. Se auto-identifica
como DeepSeek — comportamiento de entrenamiento, no identidad real. Registrar
como Qwen en todas las actas del Consejo.

CONSEJO DE SABIOS (7 miembros):
Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), DeepSeek, Qwen (Alibaba),
Gemini (Google), Parallel.ai. Todos: revisores rigurosos e implacables.
Demostrar problemas con tests compilables o matemáticas antes de proponer fixes.
Verificar que el test de vulnerabilidad retorna negativo tras aplicar el fix.
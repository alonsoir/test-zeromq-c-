## Prompt de continuidad — Día 75

**Estado al cierre del Día 74:**
pipeline-start y pipeline-stop funcionan perfectamente.
make pipeline-stop → tmux limpio, sin procesos huérfanos confirmado.
6/6 componentes arrancan, se registran en etcd, heartbeats 200 OK.

**Bugs cerrados Día 74:**
- Bug 5 sniffer: register_component() antes de put_config() ✅
- Bug LZ4 sniffer: compression_min_size=0 ✅
- pipeline-status via tmux has-session ✅
- pipeline-stop via tmux kill-session ✅
- firewall-start paths corregidos ✅
- Typo ppipeline-start → pipeline-start ✅

**Pendiente Día 75 (en orden):**
1. Verificar heartbeat del sniffer llega a 200 (solo vimos CRYPTO, no heartbeat confirmado)
2. Conectar a sesión tmux de rag-security para lanzar comandos interactivos
   - Investigar cómo expone su CLI (probablemente stdin de la sesión tmux)
3. trace_id en CLI (P2 del Día 72)
4. Test de stress con CTU-13 pcap replay:
   - Arrancar las dos VMs del Vagrantfile (server + client)
   - make pipeline-start → make test-replay-neris (492K events)
   - Dejar corriendo varias horas y medir:
      * Latencia end-to-end sniffer → ml-detector → firewall
      * Tasa de eventos/seg sostenida
      * Estabilidad de memoria (sin leaks)
      * Heartbeats continuos durante todo el test
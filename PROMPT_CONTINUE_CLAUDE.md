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

Buenas respuestas del consejo. Coinciden en lo esencial y no hay contradicciones entre ellos. Mi síntesis para arrancar:

**El orden correcto para hoy es exactamente el tuyo**, con un matiz: el punto 1 (heartbeat sniffer) y el punto 2 (CLI rag-security) son verificaciones rápidas de 10 minutos cada una — hazlas primero antes de lanzar el stress test, porque si hay un problema con el heartbeat bajo carga intensiva del replay, necesitas saberlo en estado limpio, no mezclado con 492K eventos.

El stress test es el plato fuerte y merece atención completa. Gemini tiene razón en preparar telemetría antes de lanzarlo — añade este target al Makefile antes de arrancar:

```makefile
stress-monitor:
	@vagrant ssh -c "while true; do \
		echo \"=== \$$(date) ===\"; \
		ps aux | grep -E 'sniffer|ml-detector|firewall|rag' | grep -v grep | awk '{print \$$11, \"CPU:\"\$$3\"%\", \"MEM:\"\$$4\"%\"}'; \
		echo '---'; \
		sleep 300; \
	done >> /vagrant/logs/stress_monitor.log 2>&1 &"
```

Así tienes un log con snapshot cada 5 minutos durante toda la noche.

Arranca:
```bash
make pipeline-start
# verificar heartbeat sniffer
# verificar rag-security CLI
make stress-monitor
make test-replay-neris
```
## Prompt de continuidad — Día 74

**Estado al cierre del Día 73:**

`pipeline-start` arranca correctamente **etcd-server** y **rag-security** desde el Makefile del Mac. Ambos sobreviven y mandan heartbeat. El resto de componentes (**ml-detector, firewall-acl-agent, sniffer**) no tienen targets en `pipeline-start` todavía.

**Bugs cerrados hoy:**
- Bug 5 (register_component antes de put_config) ✅
- Bug LZ4 (compression_min_size=0) ✅
- rag-security bucle infinito en daemon mode ✅
- etcd-server SIGTERM sin exit(0) ✅
- Makefile pipeline-start/stop/status funcional ✅

**Pendiente Día 74:**
1. Añadir ml-detector, firewall-acl-agent, sniffer a `pipeline-start` con sus paths correctos (`build-debug/`)
2. Verificar que rag-security recibe el seed de cifrado de etcd (el `component=` vacío en PUT puede seguir siendo problema)
3. Fix path hardcodeado en etcd-server-start — usar `$(ETCD_SERVER_BUILD_DIR)` en lugar de `build-debug`
4. Añadir rag-ingester al pipeline
5. Una vez pipeline estable: trace_id en CLI (P2 del Día 72)

**Archivos modificados:**
- `/vagrant/rag/src/etcd_client.cpp`
- `/vagrant/rag/src/main.cpp`
- `/vagrant/etcd-server/src/main.cpp`
- `/vagrant/Makefile`
- `/vagrant/rag/config/rag-config.json`
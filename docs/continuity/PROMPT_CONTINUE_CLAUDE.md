# PROMPT DE CONTINUIDAD — DAY 141

---

Soy Alonso Isidoro Román, fundador de aRGus NDR, sistema open-source C++20 de detección y respuesta a intrusiones de red para infraestructura crítica (hospitales, escuelas, municipios). Trabajo en modo "solopreneur" con un Consejo de Sabios de 8 modelos de IA como equipo de revisión adversarial.

**Estado repo:** branch `feature/variant-b-libpcap` @ `f6dcb56b`
**Tag main:** `v0.6.0-hardened-variant-a` @ `737ba0d5`
**arXiv:** 2604.04952 — Draft v18 subido (Cornell procesando)
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
**FEDER deadline:** 22-Sep-2026 | **Go/no-go:** 1-Ago-2026

---

## COMPLETADO DAY 140

Sesión de ~10 horas. EMECAS verde. Cierre completo de `DEBT-COMPILER-WARNINGS-CLEANUP-001`.

**Tareas completadas:**

- **TAREA-05:** `feature_extractor.cpp:231` — `static_cast<float>` en `total_flags`. Commit `ef18e45d`.
- **TAREA-06/07:** `-Wtype-limits` y `-Wswitch-unreachable` — ya limpios de sesiones anteriores.
- **TAREA-08:** unused var/param en `rag_logger.cpp`, `ring_consumer.cpp`, `component_registration.cpp`. Commit `c4af9650`.
- **TAREA-11:** 17 stubs `ml_defender_features.cpp` — `/*flow*/`. Commit `8d29ce4c`.
- **TAREA-09 (ODR):** `PROFILE=production` con LTO — **sin violations**. Build tarda ~45 min en VM por LTO.
- **TAREA-10 (`-Werror`):** Activado en `CXX_WARNINGS`. Destapó warnings ocultos en tests, rag, etcd-server, rag-ingester — todos corregidos. **192 → 0 warnings con `-Werror` como invariante permanente.** Commit `f2852de2`.
- **TAREA-DOC-PROFILES:** Build profiles (debug/production/tsan/asan) documentados en README y Makefile help. Commit `f1cb6e9b`.
- **DEBT-EMECAS-AUTOMATION-001:** Registrada. Commit `b943b6f5`.
- **BACKLOG-ZMQ-TUNING-001 + BACKLOG-BENCHMARK-CAPACITY-001:** Documentados en `docs/adr/`. Commit `a7334911`.
- **ml-training/.venv:** Eliminado del repo (557 ficheros, alerta Dependabot resuelta en rama). Commit `8926689f`.
- **Consejo DAY 140 (8/8):** 5 preguntas, feedback completo de los 8 modelos. Veredictos aplicados.
- **4 DEBTs nuevas:** `DEBT-LLAMA-API-UPGRADE-001`, `DEBT-ODR-CI-GATE-001`, `DEBT-GENERATED-CODE-CI-001`, `DEBT-MAYBE-UNUSED-MIGRATION-001`.
- **`docs/THIRDPARTY-MIGRATIONS.md`:** Creado — registro de APIs deprecated suprimidas.
- **`Jenkinsfile`:** Skeleton completo para cuando haya servidor CI/CD (FEDER hardware).
- **BACKLOG.md + README.md:** Actualizados con todas las decisiones del Consejo DAY 140.

**make test-all:** ALL TESTS COMPLETE (KNOWN-FAIL-001 pre-existente — safe_path en dev VM).

---

## PRIMER PASO DAY 141

```bash
vagrant destroy -f && vagrant up && make bootstrap && make test-all
```

Luego verificar estado:
```bash
git log --oneline -5
make all 2>&1 | grep -c 'warning:'  # debe ser 0
make all 2>&1 | grep -c 'error:'    # debe ser 0
```

---

## DEUDAS P0 activas — orden de prioridad

### `DEBT-PCAP-CALLBACK-LIFETIME-DOC-001` — trivial, 10 min

Añadir comentario de contrato de lifetime en `sniffer/include/pcap_backend.hpp`:
- `PcapCallbackData` válido durante toda la sesión de captura
- No destruir `PcapBackend` durante `pcap_dispatch()` activo
- Señalización asíncrona no soportada

```bash
vagrant ssh -c "grep -n 'PcapCallbackData\|struct Pcap' /vagrant/sniffer/include/pcap_backend.hpp | head -10"
```

### `DEBT-VARIANT-B-CONFIG-001` — siguiente, 1 sesión

JSON propio `sniffer-libpcap.json`. Ver estado actual:
```bash
vagrant ssh -c "cat /vagrant/sniffer/config/sniffer.json | head -30"
```

Campos a eliminar del JSON (hardcodeados en binario): `zmq_sender_threads=1`, `io_thread_pools=1`, `ring_buffer.*`, `snaplen=65535`, `promiscuous=1`.
Campos a añadir: `capture.buffer_size_mb`, `capture.sampling.mode/rate`.
Test e2e con `pcap_open_dead()` + inject (CI, sin root).

### `DEBT-IRP-NFTABLES-001` — 2 sesiones

`argus-network-isolate` con nftables transaccional. Protocolo aprobado Consejo DAY 138 (8/8).

---

## ACCIONES PENDIENTES (no técnicas)

- **Lunes:** Enviar email a Andrés Caro — hardware FEDER (RPi5 + N100, deadline 15 Junio)
- **Lunes:** Enviar email a Andrés Caro — scope NDR standalone vs federado (antes Julio)
- Los borradores están listos en la conversación DAY 140.

---

## REGLAS PERMANENTES

- REGLA EMECAS: `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
- macOS: nunca `sed -i` sin `-e ''` — usar `python3 << 'PYEOF'` heredoc o `vagrant ssh -c "python3 << 'PYEOF'"`
- Makefile es la única fuente de verdad — nunca cmake/make directo en VM
- **`-Werror` activo** — cualquier warning nuevo rompe el build. `make all 2>&1 | grep -c 'warning:'` = 0 es invariante permanente
- `PROFILE=production all` antes de cualquier merge a main (gate ODR)
- Código de terceros deprecated → suprimir por fichero + `docs/THIRDPARTY-MIGRATIONS.md`
- Variant B es monohilo por diseño — no configurable, no negociable
- ODR violations = P0 bloqueante para cualquier tag posterior
- `make detector-clean ml-detector` para rebuild limpio sin destruir pipeline

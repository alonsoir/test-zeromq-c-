# PROMPT DE CONTINUIDAD — DAY 140

---

Soy Alonso Isidoro Román, fundador de aRGus NDR, sistema open-source C++20 de detección y respuesta a intrusiones de red para infraestructura crítica (hospitales, escuelas, municipios). Trabajo en modo "solopreneur" con un Consejo de Sabios de 8 modelos de IA como equipo de revisión adversarial.

**Estado repo:** branch `feature/variant-b-libpcap` @ `91281005`
**Tag main:** `v0.6.0-hardened-variant-a` @ `737ba0d5`
**arXiv:** 2604.04952 — Draft v18 subido (Cornell procesando)
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
**FEDER deadline:** 22-Sep-2026 | **Go/no-go:** 1-Ago-2026

---

## COMPLETADO DAY 139

EMECAS dev PASSED. Limpieza repo: untrack build artifacts (`6494d490`, `189acf52`, `de36c3ce`). `DEBT-COMPILER-WARNINGS-CLEANUP-001` iniciada y ejecutada: **192 → ~67 warnings** en una sesión de ~5 horas.

Tareas completadas:

- **TAREA-01** (`-Wreorder`): `ZMQHandler`, `RingBufferConsumer`, `DualNICManager` — initializer lists reordenados para coincidir con orden de declaración. UB silencioso eliminado. Commit `2f947170`.
- **TAREA-02** (OpenSSL deprecated): `SHA256_Init/Update/Final` → `EVP_DigestInit_ex/Update/Final_ex` en `rag_logger.cpp`. Commit `bdd1567c`.
- **TAREA-03** (`-Wsign-conversion`): 52 instancias propias en `rag_logger.cpp`, `zmq_handler.cpp`, `csv_event_writer.cpp`, `contract_validator.cpp`, `ransomware_detector.cpp`. Supresión CMake para generados: `network_security.pb.cc` (protobuf) + `ddos/traffic/internal_detector.cpp` (XGBoost trees). Commit `d73ed2fb`.
- **TAREA-04** (`-Wconversion`/`-Wfloat-conversion`): 20 instancias en `feature_extractor.cpp` + 1 en `zmq_handler.cpp`. Commit `91281005`.
- **Makefile help**: `sniffer-libpcap` añadido con descripción Variant B.

**1 warning residual pendiente:** `feature_extractor.cpp:231` — `uint32_t → float` no incluido en el set de targets de TAREA-04.

---

## PRIMER PASO DAY 140

```bash
vagrant destroy -f && vagrant up && make bootstrap && make test-all
```

Luego verificar estado:
```bash
git log --oneline -8
git status
make all 2>&1 | grep -c 'warning:'
```

---

## DEUDAS P0 activas — orden de prioridad

### `DEBT-COMPILER-WARNINGS-CLEANUP-001` — EN CURSO (192 → ~67)

**Pendiente DAY 140 (en orden):**

1. **TAREA-05** — 1 warning residual `feature_extractor.cpp:231`:
   ```bash
   vagrant ssh -c "sed -n '229,234p' /vagrant/ml-detector/src/feature_extractor.cpp"
   ```
   Fix: `static_cast<float>(nf.xxx())` en línea 231.

2. **TAREA-06** — `-Wtype-limits` unsigned >= 0 (2 instancias):
   ```bash
   make all 2>&1 | grep 'Wtype-limits'
   ```

3. **TAREA-07** — `-Wswitch-unreachable` (7 instancias):
   ```bash
   make all 2>&1 | grep 'Wswitch-unreachable' | grep -oE '/vagrant/[^:]+:[0-9]+'
   ```

4. **TAREA-08** — Conversiones menores restantes (`-Wconversion` W-13, unused vars W-21/22)

5. **TAREA-09** — ODR verification con LTO:
   ```bash
   # Añadir a CMakeLists de sniffer y ml-detector:
   # target_compile_options(... PRIVATE -flto=thin)
   # target_link_options(... PRIVATE -flto=thin -Wodr)
   # Rebuild y capturar output del linker
   ```

6. **TAREA-10** — Activar `-Werror` en todos los CMakeLists del pipeline

7. **TAREA-11** — `-Wunused-parameter` (64 instancias, cosmético, al final)

**Test de cierre final:**
```bash
make all 2>&1 | grep -c 'warning:'  # debe ser 0 (o solo libtool ignorados)
make test-all  # 8/8 PASSED
```

---

### `DEBT-VARIANT-B-CONFIG-001` — siguiente tras cleanup

JSON propio `sniffer-libpcap.json`. Campos multihilo HARDCODEADOS en binario (no en JSON): `zmq_sender_threads=1`, `io_thread_pools=1`, `ring_buffer.*` eliminado, `snaplen=65535`, `promiscuous=1`. Añadir: `capture.buffer_size_mb`, `capture.sampling.mode/rate`. Test e2e con `pcap_open_dead()` + inject (CI). 1 sesión.

### `DEBT-IRP-NFTABLES-001` — `argus-network-isolate`

nft -f transaccional. Snapshot previo + rollback 300s + fallback `ip link down`. Protocolo aprobado Consejo 8/8 DAY 138. 2 sesiones.

### `DEBT-PCAP-CALLBACK-LIFETIME-DOC-001` — trivial, 10 min

Comentario contrato lifetime en `pcap_backend.hpp`.

---

## PLAN DAY 140-141

- **DAY 140:** Cerrar DEBT-COMPILER-WARNINGS-CLEANUP-001 completamente (TAREA-05 al 11). Validar con `-Werror` activo.
- **DAY 141:** pcap relay de validación — Variant B (libpcap) bajo x86 dev + AppArmor enforce. Verificar que los static_cast no introducen regresiones de comportamiento. Luego `DEBT-VARIANT-B-CONFIG-001`.

---

## REGLAS PERMANENTES

- REGLA EMECAS: `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
- macOS: nunca `sed -i` sin `-e ''` — usar `python3 << 'PYEOF'` heredoc
- Makefile es la única fuente de verdad — nunca cmake/make directo en VM
- Hardened VM ssh: `cd vagrant/hardened-x86 && vagrant ssh -c '...'`
- Variant B es monohilo por diseño de libpcap — no configurable, no negociable
- ODR violations en C++20 = UB = P0 bloqueante para cualquier tag posterior
- make all se lanza desde macOS, no desde la VM — rutas /vagrant/ aparecen en output porque la compilación ocurre en la VM
- `make detector-clean ml-detector` para rebuild limpio de ml-detector sin destruir el pipeline completo
- Supresión CMake para código generado (protobuf, XGBoost trees): `set_source_files_properties(...COMPILE_OPTIONS "-Wno-...")`
- Fix manual con `static_cast` para código propio — nunca suprimir warnings en código nuestro
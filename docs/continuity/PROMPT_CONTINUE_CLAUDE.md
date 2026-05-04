# PROMPT DE CONTINUIDAD — DAY 142

---

Soy Alonso Isidoro Román, fundador de aRGus NDR, sistema open-source C++20 de detección y respuesta a intrusiones de red para infraestructura crítica (hospitales, escuelas, municipios). Trabajo en modo "solopreneur" con un Consejo de Sabios de 8 modelos de IA como equipo de revisión adversarial.

**Estado repo:** branch `feature/variant-b-libpcap` @ `63a37d9d` (último commit DAY 141)
**Tag main:** `v0.6.0-hardened-variant-a` @ `737ba0d5`
**arXiv:** 2604.04952 — Draft v18 (Cornell procesando)
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
**FEDER deadline:** 22-Sep-2026 | **Go/no-go:** 1-Ago-2026

---

## COMPLETADO DAY 141

- **Bug Makefile seed-client-build:** `firewall` y `pipeline-build` no declaraban `seed-client-build` como dependencia explícita. Fix commit `63a37d9d`. EMECAS verde: 0 errors, 15 warnings third-party (FAISS/libtool documentados en `docs/THIRDPARTY-MIGRATIONS.md`).
- **DEBT-PCAP-CALLBACK-LIFETIME-DOC-001:** Comentario de contrato en `sniffer/include/pcap_backend.hpp` — lifetime de `PcapCallbackData`, prohibición de destruir PcapBackend durante `pcap_dispatch()` activo. ✅ CERRADA.
- **DEBT-VARIANT-B-CONFIG-001:** `sniffer/config/sniffer-libpcap.json` creado. `main_libpcap.cpp` acepta `-c <config_path>`, lee interface y ZMQ endpoint desde JSON, SeedClient recibe `sniffer-libpcap.json`. Stats periódicas 30s con `send_failures` + `drop_rate_alert`. 9/9 tests PASSED, 0 warnings. ✅ CERRADA.
- **Consejo DAY 141 (8/8):** 4 preguntas. Veredictos aplicados.
- **2 DEBTs nuevas:** `DEBT-VARIANT-B-BUFFER-SIZE-001` (P1 pre-FEDER), `DEBT-VARIANT-B-MUTEX-001` (P1 pre-FEDER).
- **BACKLOG-BUILD-WARNING-CLASSIFIER-001:** Actualizado a script grep/awk determinista (no TinyLlama).
- **Emails a Andrés Caro Lindo:** Hardware FEDER (RPi5+N100+switch, ~400-600€) + scope NDR standalone vs federado. Pendiente respuesta.
- **BACKLOG.md + THIRDPARTY-MIGRATIONS.md + README.md:** Actualizados con todas las decisiones del Consejo DAY 141.

**make test-all:** ALL TESTS COMPLETE (9/9 sniffer Variant B PASSED).

---

## PRIMER PASO DAY 142

```bash
vagrant destroy -f && vagrant up && make bootstrap && make test-all
```

Verificar:
```bash
grep -c 'warning:' protocol-EMECAS-output-05-05-2026.md
grep 'warning:' protocol-EMECAS-output-05-05-2026.md | grep -v 'defender:'  # debe ser 0
grep -c 'error:' protocol-EMECAS-output-05-05-2026.md  # debe ser 0
git log --oneline -5
```

---

## DEUDAS P0 activas — orden de prioridad

### `DEBT-IRP-NFTABLES-001` — 3 sesiones, arrancar hoy

`argus-network-isolate` con nftables transaccional. Protocolo aprobado Consejo DAY 138 (8/8).

Ver estado actual:
```bash
vagrant ssh -c "cat /etc/systemd/system/argus-apt-integrity.service | grep -A3 ExecStartPre"
vagrant ssh -c "ls -la /usr/local/bin/argus-network-isolate 2>/dev/null || echo 'NOT FOUND'"
vagrant ssh -c "nft --version"
```

Protocolo (6 pasos):
1. Snapshot: `nft list ruleset > /tmp/argus-backup-$$.nft`
2. Generar reglas de aislamiento
3. Validar: `nft -c -f /tmp/argus-isolate-$$.nft`
4. Aplicar atómico: `nft -f /tmp/argus-isolate-$$.nft`
5. Timer rollback 300s
6. Fallback: `ip link set eth0 down`

### `DEBT-VARIANT-B-BUFFER-SIZE-001` — 1 sesión, pre-benchmark ARM64

Refactorizar `PcapBackend::open()` de `pcap_open_live()` a `pcap_create()+pcap_set_buffer_size()+pcap_activate()`. El campo `capture.buffer_size_mb` en `sniffer-libpcap.json` existe pero no se aplica aún.

### `DEBT-VARIANT-B-MUTEX-001` — 1 sesión, Nivel 1

Script bash/python en Makefile. Antes de arrancar cualquier variante sniffer: consultar etcd, si hay otro sniffer registrado → warn + stop ambos + exit 1. La lógica NO entra en los binarios.

---

## REGLAS PERMANENTES

- REGLA EMECAS: `vagrant destroy -f && vagrant up && make bootstrap && make test-all`
- macOS: nunca `sed -i` sin `-e ''` — usar `python3 << 'PYEOF'` heredoc
- Makefile es la única fuente de verdad
- **`-Werror` activo** — `make all 2>&1 | grep -c 'warning:'` = 0 invariante permanente
- Verificación EMECAS: `grep 'warning:' output.md | grep -v 'defender:'` debe ser 0
- `PROFILE=production all` antes de cualquier merge a main (gate ODR)
- Variant B es monohilo por diseño — no configurable
- Variant A y Variant B nunca simultáneas en el mismo hardware (DEBT-VARIANT-B-MUTEX-001)
- buffer_size_mb variable por diseño para trazar curva de optimización
- Warning classifiers: script grep/awk determinista, no LLM
- Código terceros deprecated → `docs/THIRDPARTY-MIGRATIONS.md`

---

## IDEAS REGISTRADAS — pendiente diseño formal

### DEBT-VARIANT-B-PCAP-RELAY-001 — pcap relay E2E pre-merge
Añadir al gate de merge de `feature/variant-b-libpcap` un relay real con pcap versionado (CTU-13 Neris subset + SHA-256). `tcpreplay` en loopback en VM → `sniffer-libpcap` captura → ml-detector recibe. Gate: `packets_sent > 0`, `send_failures = 0`, al menos 1 evento en ml-detector. Aplica también cuando exista Jenkins en producción/preproducción — es política de CI, no solo de merge manual.
**Pendiente:** decidir si bloquea merge de esta feature o entra como fase del EMECAS general (DEBT-EMECAS-AUTOMATION-001).

### IDEA-SEED-PLUGIN-001 — Plugin de obtención de semillas por profile
La política de dónde buscar las seeds es una decisión de despliegue, no lógica de negocio. Candidato natural para plugin ADR-012:
- Profile DEV → seed.bin en host (`/etc/ml-defender/*/seed.bin`) — mecanismo actual, sin cambios
- Profile PROD/PRE-PROD/TESTING/INTEGRATION → Vault via Jenkinsfile

**Tensión de bootstrap a resolver antes de diseñar:** el plugin se verifica vía Ed25519 (ADR-025), pero la seed es necesaria para verificar el plugin. ¿Quién firma al firmante? Posible resolución: seed de bootstrap para verificar el plugin de semillas viene del Vault directamente via Jenkinsfile, no del plugin mismo. Requiere Consejo antes de abrir feature.
**Target:** post-FEDER, junto a DEBT-CRYPTO-MATERIAL-STORAGE-001 y DEBT-JENKINS-SEED-DISTRIBUTION-001.
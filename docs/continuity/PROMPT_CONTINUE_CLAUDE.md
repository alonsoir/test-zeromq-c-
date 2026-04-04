Soy Alonso (aRGus NDR, ML Defender, DAY 108).
Lee el transcript de DAY 107 antes de responder nada.

CONTEXTO DAY 107 (sesión de troubleshooting intenso):
Pipeline 6/6 RUNNING. Root cause resuelto: component_config_path no
seteado en etcd_client.cpp → tx_ null → datos LZ4 sin cifrar enviados
como octet-stream → MAC failure garantizado en servidor.

FIXES EN CÓDIGO FUENTE (commiteados en feature/plugin-crypto):
1. ml-detector/src/etcd_client.cpp
   → component_config_path = "/etc/ml-defender/ml-detector/ml_detector_config.json"
2. sniffer/src/userspace/etcd_client.cpp  
   → component_config_path = "/etc/ml-defender/sniffer/sniffer.json"
   → get_encryption_seed() reescrito para leer seed.bin local
3. firewall-acl-agent/src/core/etcd_client.cpp
   → component_config_path = "/etc/ml-defender/firewall-acl-agent/firewall.json"
4. etcd-server/src/component_registry.cpp
   → swap CTX_ETCD_TX/RX (rx_ usa TX, tx_ usa RX)
   → PENDIENTE VERIFICAR si era necesario (ver PASO 1)

FIXES SOLO EN VM (NO en provision.sh — estado frágil):
- chmod 755 /etc/ml-defender/{6 componentes}/
- chmod 640 + chown root:vagrant en seed.bin de todos
- Seeds sincronizados manualmente (cp etcd-server/seed.bin → otros 5)
- Symlinks JSON /etc/ml-defender/*/  → /vagrant/*/config/
- ln -sf libsodium.so.26 libsodium.so.23 + ldconfig
- libcrypto_transport.so reconstruida (era feb 16, ahora abr 4)

SI HACES vagrant destroy && vagrant up HOY → pipeline NO arranca.
provision.sh genera seeds independientes, deja dirs drwx------ root,
no crea symlinks JSON, no sincroniza seeds.

ORDEN OBLIGATORIO DAY 108 (no saltarse pasos):

PASO 1 — Verificar swap CTX (crítico, 5 min):
Revertir en etcd-server/src/component_registry.cpp:
rx_ → CTX_ETCD_RX (original)
tx_ → CTX_ETCD_TX (original)
rm -rf etcd-server/build-debug && make pipeline-stop && make pipeline-start
sleep 10 && make pipeline-status
Si 6/6 RUNNING → swap era innecesario, dejarlo revertido
Si MAC falla → restaurar swap, documentar como ADR-026

PASO 2 — Añadir invariant fail-fast (los 3 adaptadores etcd_client.cpp):
if (config_.encryption_enabled && !tx_) {
std::terminate(); // FATAL: component_config_path no seteado
}

PASO 3 — Formalizar provision.sh (gate obligatorio antes de PHASE 2b):
a) Un solo seed maestro → distribuir a 6 componentes
b) chmod 755 directorios, 640 seed.bin, chown root:vagrant
c) Symlinks JSON automáticos para los 6 componentes
d) ln -sf libsodium.so.26 libsodium.so.23 + ldconfig
e) Rebuild libcrypto_transport si fecha < hoy

PASO 4 — Gate de calidad:
vagrant destroy && vagrant up
make pipeline-start && sleep 15 && make pipeline-status
Debe dar 6/6 RUNNING sin intervención manual.
Si no → volver a PASO 3.

PASO 5 — Solo si PASO 4 verde:
PHASE 2b: plugin_process_message() en rag-ingester
Gate: TEST-INTEG-4b (MessageContext, result_code=0)
Patrón: igual que firewall-acl-agent DAY 105

DEUDA PENDIENTE (no bloqueante):
- Unificar sniffer bajo SeedClient (eliminar get_encryption_seed manual)
- SeedClient v2: aceptar 640 con warning en dev, 600 en prod
- ADR-025 (Plugin Integrity Ed25519) — post PHASE 2b
- arXiv submit/7438768 — pendiente moderación
- Feedback Consejo DAY 107: ya recibido y procesado

Consejo DAY 107 unánime en:
- swap CTX: revertir y verificar (mayoría)
- doble path seed: deuda técnica, unificar en SeedClient
- permisos 640: aceptar en dev con warning, 600 en prod
- invariant fail-fast: añadir urgente
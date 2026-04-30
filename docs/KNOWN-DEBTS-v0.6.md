# aRGus NDR — KNOWN-DEBTS-v0.6.md
# Deudas técnicas documentadas — pre-merge feature/adr030-variant-a
# Fecha: 2026-04-29 (DAY 136)

## DEBT-COMPILER-WARNINGS-001
**Severidad:** 🟡 Media — no bloqueante para FEDER  
**Componente:** etcd-client, ml-detector, sniffer, rag-ingester  
**Descripción:**
- `test_etcd_client_hmac_grace_period` DISABLED en etcd-client (requires GTest)
- Warnings LTO/ODR `tree_N`: InternalNode vs TrafficNode en ml-detector y sniffer
- Linker: libsodium.so.26 vs libsodium.so.23 (conflict warning, no error)
- SHA256_Init/Update/Final deprecated since OpenSSL 3.0 en rag_logger.cpp
- Posibles tests adicionales deshabilitados no catalogados (log EMECAS voluminoso)
  **Corrección:** post-merge, pre-FEDER  
  **Decisión:** Consejo pendiente

---

## DEBT-SEEDS-SECURE-TRANSFER-001
**Severidad:** 🔴 Alta — mitigado en Vagrant, inaceptable en producción real  
**Componente:** scripts/prod-deploy-seeds.sh  
**Descripción:** Seeds pasan por Mac host vía /vagrant durante el despliegue.
Actualmente solo válido para entorno Vagrant dev/test. En producción real
los seeds deben generarse directamente en el nodo hardened sin salir del HSM
o del canal seguro.  
**Corrección:** post-FEDER (protocolo de distribución out-of-band)  
**Decisión:** D2 Consejo DAY 134

---

## DEBT-SEEDS-LOCAL-GEN-001
**Severidad:** 🟡 Media  
**Componente:** scripts/prod-deploy-seeds.sh  
**Descripción:** Los seeds se generan en la dev VM y se extraen al Mac host
antes de instalarse en hardened. En producción real deben generarse
localmente en cada nodo sin atravesar ningún host intermediario.  
**Corrección:** post-FEDER

---

## DEBT-SEEDS-BACKUP-001
**Severidad:** 🟡 Media  
**Componente:** /etc/ml-defender/*/seed.bin  
**Descripción:** No existe protocolo de backup/recovery para seeds. Si un nodo
falla catastróficamente los seeds se pierden y el pipeline no puede
arrancar. Requiere procedimiento de regeneración documentado.  
**Corrección:** post-FEDER

---

## DEBT-IRP-NFTABLES-001
**Severidad:** 🔴 Alta — post-merge  
**Componente:** /usr/local/bin/argus-network-isolate (pendiente implementar)  
**Descripción:** ExecStartPre de argus-apt-integrity.service referencia
argus-network-isolate pero el script no está implementado. El aislamiento
actual vía `ip link set down` es insuficiente — se necesita drop-all con
nftables (enmienda E1 ADR-042).  
**Corrección:** primera iteración post-merge, antes de demo FEDER  
**ADR relacionado:** ADR-042 IRP enmienda E1

---

## DEBT-IRP-QUEUE-PROCESSOR-001
**Severidad:** 🔴 Alta — post-merge  
**Componente:** ADR-042 IRP  
**Descripción:** La cola irp-queue no tiene límites de tamaño ni procesador
systemd dedicado. Sin límites puede crecer sin control bajo ataque sostenido.
Requiere unidad systemd irp-queue-processor con límites explícitos
(enmienda E3 ADR-042).  
**Corrección:** primera iteración post-merge  
**ADR relacionado:** ADR-042 IRP enmienda E3

---


---
## KNOWN-TEST-FAILURES (expected — no action required)

### KNOWN-FAIL-001: test_config_parser (rag-ingester, dev VM)
**Severidad:** ℹ️ Informativo — comportamiento correcto por diseño
**Componente:** rag-ingester / safe_path guard
**Descripción:** El test falla en dev VM porque carga el config desde
`/vagrant/rag-ingester/config/rag-ingester.json`, path fuera del prefix
permitido `/etc/ml-defender/`. El safe_path guard emite SECURITY VIOLATION
y aborta — exactamente como debe comportarse.
En hardened VM el config vive en `/etc/ml-defender/` y el test pasaría.
**Acción:** Ninguna. No es regresión. No confundir con un bug.
**Referencia:** ADR-028, DEBT-SAFE-PATH-TEST-PRODUCTION-001
**Descubierto:** DAY 137 — 2026-04-30

## Notas
- Deudas 🔴 Alta post-merge se abordan antes de demo FEDER (deadline 1 agosto 2026)
- Deudas 🟡 Media se abordan antes de deadline FEDER (22 septiembre 2026)
- Ver también: BACKLOG.md sección DEBT para deudas anteriores a v0.6
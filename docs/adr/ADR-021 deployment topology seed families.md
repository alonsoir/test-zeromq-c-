# ADR-021 — Deployment Topology SSOT and Channel Seed Families

**Estado:** Aprobado (FASE 3 — post-arXiv)
**Fecha:** 2026-03-28 (DAY 100)
**Autores:** Alonso Isidoro Roman + Consejo de Sabios (Grok/xAI, Claude/Anthropic)
**Relacionado:** ADR-013 (seed distribution), ADR-022 (threat model), ADR-001 (deployment stack)

---

## Contexto

En DAY 98–99 se completó la migración completa a `CryptoTransport` (ADR-013 PHASE 2).
Los contextos HKDF son ahora simétricos por canal gracias a `contexts.hpp`.
El siguiente paso lógico es escalar a topologías distribuidas reales con múltiples
instancias de cada componente.

La pregunta es: ¿cómo se gestiona la distribución de seeds cuando hay N sniffers,
P ml-detectors, R firewalls en hosts distintos?

La solución de FASE 1 (seed único por componente en instancia única) no escala
a topologías multi-nodo. Se necesita un modelo de **familias de canal**.

---

## Decisión

### 1. deployment.yml como SSOT de topología

Se introduce `deployment.yml` como **única fuente de verdad** de la topología
distribuida de Argus. Define instancias, hosts, y pertenencia a familias de canal.

```yaml
topology:
  sniffer:
    instances: [sniffer1]
    host: 192.168.56.10
    seed_families: [family_A]
  ml-detector:
    instances: [ml-detector1, ml-detector2]
    hosts: [192.168.56.11, 192.168.56.12]
    seed_families: [family_A, family_B, family_C]
  firewall:
    instances: [firewall1]
    host: 192.168.56.13
    seed_families: [family_B, family_C]
  rag-ingester:
    instances: [rag-ingester1]
    host: 192.168.56.14
    seed_families: [family_C]
  rag-local:
    instances: [rag-local1]
    host: 192.168.56.15
    seed_families: []
```

### 2. Familias de canal (seed families)

Cada canal lógico tiene su propia familia de seeds. Un componente puede pertenecer
a varias familias y recibe un `seed.bin` distinto por familia, en paths separados.

| Familia | Canal lógico | Miembros |
|---------|-------------|----------|
| family_A | captura → detección | sniffer + ml-detector |
| family_B | detección → enforcement | ml-detector + firewall |
| family_C | artefactos → RAG | ml-detector + firewall + rag-ingester |

**Principio:** la clave HKDF derivada es compartida **solo** entre los miembros
de la misma familia. Un sniffer comprometido no puede derivar las claves del
canal RAG.

### 3. provision.sh refactorizado para leer deployment.yml

En FASE 3, `tools/provision.sh` leerá `deployment.yml` y:
1. Generará un `seed.bin` por familia
2. Distribuirá cada seed solo a los miembros de esa familia
3. Copiará vía SSH/Ansible a `/etc/ml-defender/{component}/seeds/{family}/seed.bin`

### 4. Versioning de contextos HKDF

Los strings de contexto en `contexts.hpp` ya incluyen sufijo de versión (`:v1`).
**Política de versioning:**

- Cambio de topología (nuevos nodos) → **no** requiere bump de versión de contexto
- Cambio de semántica del canal → bump a `:v2`
- Migración: ambas versiones coexisten durante un período de transición documentado

```cpp
// Ejemplo de coexistencia durante migración
constexpr const char* CTX_SNIFFER_TO_ML_V1 = "ml-defender:sniffer-to-ml-detector:v1";
constexpr const char* CTX_SNIFFER_TO_ML_V2 = "ml-defender:sniffer-to-ml-detector:v2";
```

---

## Consecuencias

**Positivas:**
- Topología completamente declarativa — el operador define quién habla con quién
- Principio de mínimo privilegio a nivel de canal criptográfico
- `provision.sh` se convierte en herramienta de infraestructura completa
- Compatible con Ansible (FASE 3 CI/CD)

**Negativas / restricciones:**
- Requiere que `deployment.yml` exista antes de provisionar
- Un componente en múltiples familias recibe múltiples seeds — la selección
  del seed correcto por canal es responsabilidad del componente
- Aumenta la superficie de gestión de secrets en despliegues grandes

---

## Estado de implementación

| Elemento | Estado |
|----------|--------|
| `contexts.hpp` — contextos v1 | ✅ DAY 99 |
| `deployment.yml` — schema y ejemplo | ⏳ FASE 3 |
| `provision.sh` — lectura de manifiesto | ⏳ FASE 3 |
| Multi-seed por componente | ⏳ FASE 3 |
| Vagrantfile multi-VM | ⏳ FASE 3 |

**Pre-requisito:** arXiv submission → milestone gate antes de FASE 3.

---

## Revisión del Consejo de Sabios

Esta ADR fue propuesta durante la revisión de cierre DAY 99.
FASE 3 completa pasará por nueva revisión del Consejo antes de implementarse.

*DAY 100 — 2026-03-28*
# ML Defender — Prompt de Continuidad DAY 95
## 23 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 33/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7, sniffer 1/1)
**Rama activa:** `feature/plugin-loader-adr012`
**Último tag:** DAY92

---

## Lo que se hizo en DAY 94

### Sesión de diseño — ADR-017 al Consejo de Sabios

**Consulta enviada al Consejo:** `docs/consejo/CONSEJO_ADR017_CONSULTA.md`
**Respuestas recibidas:** ChatGPT5, DeepSeek, Gemini, Grok, Qwen (5/7 — Parallel.ai pendiente)
**Resultado:** Unanimidad total en las 5 preguntas.

### ADR-017 — Plugin Interface Hierarchy (ACEPTADO)

Decisiones del Consejo (6/6 unánimes):

| Pregunta | Decisión |
|---|---|
| P1 — Contextos | Opción A — tipado fuerte por familia |
| P2 — Función entrada | Separación semántica por familia + `plugin_subtype()` |
| P3 — Plugins futuros | Opción C — solo primero-partido PHASE 2, terceros PHASE 3 |
| P4 — Skills rag-security | NO unificar — mecanismo separado |
| P5 — eBPF plugins | ADR-018 separado |

**Enriquecimientos aprobados:**
- `plugin_subtype()` añadido al contrato base (ChatGPT5)
- `NetworkTuple` struct base compartida entre contextos (DeepSeek)
- `EbpfUprobePlugin` como excepción documentada, no subfamilia formal
- Familias futuras documentadas: PostProcessorPlugin, FlowPlugin, UprobePlugin

**Jerarquía validada:**
```
PluginBase (identidad + ciclo de vida)
├── SnifferPlugin       → plugin_process_packet(SnifferContext*)
│   └── PacketPlugin    → DPI: ja4, dns_dga, http_inspect [PHASE 2]
├── MlDetectorPlugin
│   ├── InferencePlugin → plugin_predict(MlDetectorContext*)
│   └── EnrichmentPlugin→ plugin_enrich(MlDetectorContext*)
├── RagIngesterPlugin   → plugin_process_event(RagIngesterContext*) [PHASE 3]
└── [EbpfKernelPlugin]  → ADR-018
```

### ADR-018 — eBPF Kernel Plugin Loader (PROPUESTO)

Mismo modelo mental que ADR-017 pero para stack eBPF:
- `libbpf` en lugar de `dlopen`
- JSON por programa eBPF (`kt_bpf_prog_load_v1.json`)
- HMAC en dos capas: build-time (ADR-015) + provision-time (ADR-018)
- Naming: `kt_{funcion}_{version}.bpf.o`
- `EbpfPluginLoader` análogo al `PluginLoader` de ADR-012

### ADR-019 — OS Hardening y Secure Deployment (PROPUESTO)

Seis capas de hardening para producción:
1. LUKS2 — cifrado de disco en reposo
2. SO mínimo — Debian 12 / Ubuntu 24.04 LTS, solo dependencias necesarias
3. ufw — todos los puertos cerrados excepto SSH; ZMQ solo en loopback
4. AppArmor — perfil por componente Y por plugin (blast radius mínimo)
5. kernel sysctl — no forwarding, no core dumps, kptr_restrict=2
6. SSH hardening — solo claves, no passwords, solo usuario de gestión

**Nota crítica para DAY 95:** `provision.sh` debe diseñarse compatible con
AppArmor desde el principio. Los paths de claves (`/etc/ml-defender/`) deben
ser los paths que los perfiles AppArmor permitirán. Si no, habrá que reescribir.

### ADR-003 — ML Autonomous Evolution (ACTUALIZADO)

Cambio principal: los nuevos modelos RF reentrenados son plugins
`InferencePlugin` (`libmodel_*.so`), no ficheros `.hpp` compilados en el core.
El core legacy permanece **FROZEN** — Strangler Fig Pattern.
Los validadores verify_A..F son herramientas CI, no plugins de producción.

### Principios añadidos por Alonso (incorporados en ADR-017)

- **JSON is the law — también para plugins:** cada plugin tiene su propio JSON
  de contrato. Fallo explícito si falta cualquier clave. Sin defaults silenciosos.
- **La clave HMAC nunca en el JSON:** derivada del intercambio de keypairs en
  provisioning — nunca en texto claro en ficheros de configuración.
- **Keypairs para plugins:** mismo modelo que componentes. `provision.sh` genera
  keypairs solo para plugins declarados en el JSON del componente.
- **Versionado es identidad criptográfica:** v1 y v2 tienen keypairs distintos.
  Actualizar un plugin requiere re-provisioning explícito.

---

## Objetivo principal DAY 95 — scripts/provision.sh + libs/seed-client

### Tarea A — scripts/provision.sh

El script bash que genera y distribuye keypairs y seeds. Compatible con
AppArmor desde el primer día (ADR-019).

**Responsabilidades:**
```bash
# Para cada componente del pipeline (6 componentes):
1. Generar keypair Ed25519 (privada + pública)
2. Generar seed ChaCha20 (32 bytes aleatorios)
3. Cifrar seed con clave pública del receptor
4. Escribir en /etc/ml-defender/{componente}/ con chmod 0600

# Para cada plugin declarado en los JSONs de componentes:
5. Generar keypair Ed25519 por plugin
6. Cifrar seed con clave pública del plugin
7. Intercambiar claves públicas componente↔plugin
8. Solo si el plugin está en plugins.enabled del componente

# Para cada plugin eBPF declarado en kernel_telemetry.json:
9. Generar keypair Ed25519 por programa eBPF
10. Firmar HMAC del .bpf.o con clave del componente
```

**Paths de claves (fijos, AppArmor-compatible):**
```
/etc/ml-defender/{componente}/         → claves de componentes
/etc/ml-defender/plugins/              → claves de plugins userspace
/etc/ml-defender/ebpf-plugins/         → claves de plugins eBPF
```

### Tarea B — libs/seed-client

Mini-componente al estilo `crypto-transport`:

```cpp
// seed_client.hpp — interfaz mínima
class SeedClient {
public:
    explicit SeedClient(const std::string& config_json_path);
    void load();                                    // lee y descifra seed.enc
    const std::array<uint8_t, 32>& seed() const;   // seed listo para crypto-transport
    bool seed_rotated() const;                      // para rotación futura
private:
    std::string seed_path_;
    std::string private_key_path_;
    std::array<uint8_t, 32> seed_;
    bool loaded_ = false;
};
```

**seed-client NO hace:** comunicación de red, generación de seeds, distribución.

### Tarea C (si queda tiempo DAY 95)

- Integrar plugin-loader en sniffer (Tarea A pendiente de DAY 94)
- Test suite plugin-loader (Tarea B pendiente de DAY 94)

---

## Secuencia de diagnóstico al arrancar DAY 95

```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker
git status
git log --oneline -5

# Verificar que los ADR nuevos están commiteados
ls docs/adr/ADR-01*.md

# Verificar artefactos en VM
vagrant ssh defender -c "ls -lh /usr/local/lib/libplugin_loader.so* \
    /usr/lib/ml-defender/plugins/"
```

---

## Backlog activo — estado actualizado DAY 94

| ID | Descripción | Estado |
|---|---|---|
| **ADR-017** | Plugin Interface Hierarchy | ✅ DONE — DAY 94 |
| **ADR-018** | eBPF Kernel Plugin Loader | ✅ PROPUESTO — DAY 94 |
| **ADR-019** | OS Hardening y Secure Deployment | ✅ PROPUESTO — DAY 94 |
| **ADR-003** | ML Autonomous Evolution (actualizado) | ✅ DONE — DAY 94 |
| **ADR-012 PHASE 1b** | Integración sniffer + test suite | **P1 — DAY 95** |
| **provision.sh** | Script bash keypairs + seeds | **P1 — DAY 95-96** |
| **seed-client** | Mini-componente libs/seed-client | **P1 — DAY 95-96** |
| **SYN-1/2** | rst_ratio + syn_ack_ratio | ✅ DONE — DAY 92 |
| **DEBT-SMB-001** | MIN_SYN_THRESHOLD empírico | DAY 97+ |
| **SYN-3..7** | Sintético + reentrenamiento + F1 | DAY 97+ |
| **DEBT-FD-001** | Fast Detector Path A → JSON | PHASE2 |
| **ADR-007** | AND-consensus firewall | PHASE2 |

---

## arXiv — Estado

- Paper draft v4: `docs/Ml defender paper draft v4.md`
- Email enviado a Sebastian Garcia — **esperando respuesta**
- Deadline: DAY 96 — si no responde, email a Yisroel Mirsky (Tier 2)

---

## Constantes del proyecto

```
Raíz:          /Users/aironman/CLionProjects/test-zeromq-docker
VM:            vagrant ssh defender
Logs:          /vagrant/logs/lab/
Plugin dir:    /usr/lib/ml-defender/plugins/
Keys dir:      /etc/ml-defender/  (a crear en DAY 95)
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
```

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*DAY 94 — 22 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus NDR)*
*Acta Consejo #3: ChatGPT5 · DeepSeek · Gemini · Grok · Qwen (unanimidad 5/5)*
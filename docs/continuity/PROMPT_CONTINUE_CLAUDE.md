# ML Defender — Prompt de Continuidad DAY 93
## 21 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 33/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7, sniffer 1/1 NEW)
**Rama activa:** `feature/plugin-loader-adr012`
**Último tag:** DAY92

---

## Lo que se hizo en DAY 92
Hace escasos minutos, hemos creado ADR-15 y ADR-16, pero vamos a trabajar hoy en feature/plugin-loader-adr012
### Flujo A — Documentación

**Rama `feature/documentation-contracts`** — mergeada a main.

Tres documentos vivos en `docs/contracts/`:
- `protobuf-contract.md` — `network_security.proto` documentado completo
- `json-contracts.md` — contratos JSON de los 5 componentes
- `rag-security-commands.md` — whitelist completo de comandos rag-security

### Flujo B — SMB Detection Features

**Rama `feature/smb-detection-features`** — mergeada a main.

**Proto (`protobuf/network_security.proto`):**
- Nuevo mensaje `SMBScanFeatures` (líneas 87-97)
- `NetworkFeatures.smb_scan = field 116`
- Comentario ambigüedad `RansomwareFeatures` vs `RansomwareEmbeddedFeatures` (línea 663)

**Extractor (`sniffer/src/userspace/ml_defender_features.cpp`, líneas 939-950):**
```cpp
auto* smb = net_features->mutable_smb_scan();
const float syn_count = static_cast<float>(flow.syn_count);
if (syn_count > 0.0f) {
    smb->set_rst_ratio(rst_count / syn_count);
    smb->set_syn_ack_ratio(ack_count / syn_count);
} else {
    smb->set_rst_ratio(MISSING_FEATURE_SENTINEL);    // -9999.0f
    smb->set_syn_ack_ratio(MISSING_FEATURE_SENTINEL);
}
```

**Tests (`sniffer/tests/test_smb_scan_features.cpp`):** 3/3 ✅

---

## Decisiones del Consejo de Sabios — Acta DAY 92

### Convergencia unánime (5/5 modelos)

**ADR-012 constraints acordados por el Consejo:**

1. **`dlopen`/`dlsym` lazy loading** (Grok, Gemini, Qwen, DeepSeek, ChatGPT5) — no eager loading. En hardware restringido (N100, RPi) la diferencia entre 80MB y 200MB de RAM es crítica.

2. **Plugins limitados a extracción de features — nunca a decisión de bloqueo** (Qwen, Gemini) — un crash en un plugin no puede derribar la cadena de decisión. Separación estricta de responsabilidades.

3. **Interfaz plugin: struct con punteros a funciones** (Qwen) — `init`, `extract_feature`, `destroy`. Sin vtables, sin herencia, sin excepciones cruzando el boundary del plugin. Máxima portabilidad y estabilidad.

4. **`MISSING_FEATURE_SENTINEL` compartido desde cabecera común** (Gemini) — no puede ser que cada plugin lo redefina. Propuesta: `libs/common/sentinel.hpp`.

5. **Sin dependencias crypto en el loader base** — solo carga + resolución de símbolos. Crypto llega en DAY 95-96 con `libs/seed-client`.

### DEBT-SMB-001 — Caso frontera syn_flag_count bajo (DeepSeek)

Con `syn_flag_count = 1`, el ratio colapsa a 0.0 o 1.0 — valores extremos estadísticamente no fiables. El sentinel actual solo cubre `syn_count == 0`, no cubre bajo volumen.

**Mitigación pendiente (DAY 97+ tras datos sintéticos):**
```cpp
constexpr uint32_t MIN_SYN_THRESHOLD = 5;  // valor empírico pendiente de validar
if (syn_count >= MIN_SYN_THRESHOLD) {
    smb->set_rst_ratio(rst_count / syn_count);
    smb->set_syn_ack_ratio(ack_count / syn_count);
} else {
    smb->set_rst_ratio(MISSING_FEATURE_SENTINEL);
    smb->set_syn_ack_ratio(MISSING_FEATURE_SENTINEL);
}
```
El valor correcto de `MIN_SYN_THRESHOLD` se determina empíricamente con SYN-3 (dataset sintético).

### Sugerencias para el paper (Gemini, DeepSeek)

- §4: "Ratios de Fracaso" como descripción intuitiva de `rst_ratio`/`syn_ack_ratio` — WannaCry no falla en conectar, *fracasa en completar*.
- §8: Decision stump sobre `rst_ratio` + `syn_ack_ratio` solos como ablation study baseline (Grok).
- §10 limitaciones: `flow_duration_min_ms` < 50ms es dato empírico — añadir solo tras validación real con SYN-3, no antes.
- Tabla coherencia ética (Qwen): mapeo valor→manifestación técnica, útil para §1 introducción.

### README — badge sugerido (Grok)

```markdown
📜 Living contracts: [Protobuf schema](docs/contracts/protobuf-contract.md) · [Pipeline configs](docs/contracts/json-contracts.md) · [RAG API](docs/contracts/rag-security-commands.md)
```
Añadir en DAY 93 — 3 líneas, alto impacto para colaboradores externos.

---

## Objetivo principal DAY 93 — ADR-012 plugin-loader minimalista

### Contexto

Plugin-loader minimalista **sin seed-client todavía** — documentado explícitamente como "sin autenticación hasta seed-client (DAY 95-96)".

### Spec mínima acordada por el Consejo

```
Interface:
  struct PluginInterface {
      int  (*init)(const char* config_path);
      int  (*extract_feature)(const FlowFeatures* in, float* out, int out_size);
      void (*destroy)(void);
  };

Loader:
  - dlopen() lazy (RTLD_LAZY)
  - dlsym() para resolver init/extract_feature/destroy
  - Sin crypto, sin seed — solo carga + resolución de símbolos
  - MISSING_FEATURE_SENTINEL desde libs/common/sentinel.hpp (compartido)

Restricciones PHASE 1:
  - Plugins: solo feature extraction
  - Decisión de bloqueo: NUNCA en un plugin
  - Crash isolation: PHASE 2 (proceso separado + watchdog)
```

### Punto de partida

```bash
# Verificar si ADR-012 ya existe como fichero
ls /Users/aironman/CLionProjects/argus/docs/adr/
cat docs/adr/ADR-012-*.md 2>/dev/null || echo "ADR-012 pendiente de redactar"
```

### Secuencia acordada

```
DAY 93-94 — ADR-012 plugin-loader + libs/common/sentinel.hpp
            README badge + docs/contracts actualización menor
DAY 95-96 — scripts/provision.sh bash + libs/seed-client
DAY 97+   — refactor etcd-server + datos sintéticos WannaCry (SYN-3)
            DEBT-SMB-001: MIN_SYN_THRESHOLD empírico
```

---

## Backlog activo — estado actualizado

| ID | Descripción | Estado |
|---|---|---|
| **SYN-1** | `rst_ratio` extractor en sniffer | ✅ DONE — DAY 92 |
| **SYN-2** | `syn_ack_ratio` extractor en sniffer | ✅ DONE — DAY 92 |
| **ADR-012** | plugin-loader minimalista (sin seed-client) | **P1 — DAY 93-94** |
| **DEBT-SMB-001** | `MIN_SYN_THRESHOLD` empírico para rst/syn_ack ratios | DAY 97+ tras SYN-3 |
| provision.sh | Script bash generación keypairs + seeds | DAY 95-96 |
| seed-client | Mini-componente libs/seed-client | DAY 95-96 |
| etcd refactor | Eliminar responsabilidades criptográficas | DAY 97+ |
| SYN-3 | Generador sintético Python/Scapy | DAY 97+ |
| SYN-4 | Validación CSV contra spec §9 | tras SYN-3 |
| SYN-5 | Reentrenamiento Random Forest | tras SYN-4 |
| SYN-6 | Validación en CTU-13 Neris hold-out | tras SYN-5 |
| SYN-7 | Actualizar f1_replay_log.csv | tras SYN-6 |
| SYN-8 | `dst_port_445_ratio` extractor | P2 |
| SYN-8b | `flow_duration_min` extractor | P2 |
| SYN-9 | `port_diversity_ratio` extractor | P2 |
| SYN-10 | FEAT-WINDOW-2 (60s secundaria) | P2 |
| SYN-11 | `flow_duration_std_ms` para NotPetya (Grok) | P2 |
| DEBT-FD-001 | Fast Detector Path A — leer sniffer.json | PHASE2 |
| ADR-007 | AND-consensus firewall — implementación | PHASE2 |

---

## arXiv — Estado

- Paper draft v4 listo: `docs/Ml defender paper draft v4.md`
- Email enviado a Sebastian Garcia — **recibido, esperando respuesta**
- Deadline: DAY 96 — si no responde, email a Yisroel Mirsky (Tier 2)
- Limitaciones a añadir en §10 (pendiente):
  - Killswitch DNS no detectable en capa 3/4
  - Generalización SMB: Recall estimado 0.70–0.85 sin reentrenamiento
  - `flow_duration_min_ms` < 50ms — añadir solo tras validación con SYN-3

---

## Constantes del proyecto

```
Raíz:          /Users/aironman/CLionProjects/argus
VM:            vagrant ssh defender
Logs:          /vagrant/logs/lab/
F1 log:        /vagrant/docs/experiments/f1_replay_log.csv
Paper:         /vagrant/docs/Ml defender paper draft v4.md
Proto:         /vagrant/protobuf/network_security.proto
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
```

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*DAY 92 — 20 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus EDR)*
*Acta Consejo #2 incorporada: Grok · Gemini · Qwen · DeepSeek · ChatGPT5*
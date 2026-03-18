# ML Defender — Prompt de Continuidad DAY 91
## 19 marzo 2026

---

## Estado del sistema

**Pipeline:** 6/6 RUNNING (etcd-server, rag-security, rag-ingester, ml-detector, sniffer, firewall)
**Test suite:** 31/31 ✅ (crypto 3/3, etcd-hmac 12/12, ml-detector 9/9, rag-ingester 7/7)
**Rama activa:** main
**Último tag:** pendiente commit DAY 90

---

## Lo que se hizo en DAY 90

### Flujo A — Código
- Test suite verificada: 31/31 ✅ — nada que arreglar
- Aclaración: el "72/72" del plan era incorrecto. 31 es el conteo real de CTest.
  Los 46 son assertions internas dentro del test_trace_id, no tests de CTest.

### Flujo B — Documentación
- **ADR-007 consolidado y listo** para `docs/adr/ADR-007-and-consensus-firewall.md`
  - Base: documento DAY 82 (más rico)
  - Adiciones: alternativas descartadas, DEBT-FD-001 como prerequisito,
    estado actualizado a ACEPTADO, formalizado DAY 90
  - Fichero en outputs: `ADR-007-and-consensus-firewall.md`

### Flujo C — Consejo de Sabios
- **Consulta #1 completada** — 2 rondas, 6 modelos (Qwen pendiente de respuesta genuina)
- Decisiones finales documentadas en `consejo_consulta_1_decisiones_finales.md`

---

## Decisiones del Consejo #1 (WannaCry/NotPetya features)

### UNANIMIDAD
1. **`rst_ratio` → P1 INMEDIATO** — implementar antes de cualquier dataset sintético
2. **`syn_ack_ratio` → P1 INMEDIATO** — ídem
3. **Modelo actual NO generaliza a ransomware SMB** sin reentrenamiento
  - Recall estimado: 0.70–0.85 sin retraining
  - F1 > 0.90 requiere datos sintéticos SMB + reentrenamiento

### MAYORÍA CLARA
4. **Ventana 10s:** suficiente WannaCry, insuficiente NotPetya
  - Decisión: mantener 10s + backlog FEAT-WINDOW-2 (60s secundaria)
5. **Killswitch DNS:** NO detectable en capa 3/4 sin DPI — limitación honesta
  - La firma compuesta DNS→SMB fue analizada y descartada:
    FPR inaceptable (patrón ocurre en tráfico Windows legítimo),
    y la propagación ya está cubierta por rst_ratio sin necesitar DNS
6. **dns_query_count:** P3 — valor solo en correlación, no primaria

### Roadmap de features (resultado Consejo)
- **P1:** rst_ratio, syn_ack_ratio (implementar DAY 91-92)
- **P2:** port_diversity_ratio, new_dst_ip_rate, dst_port_445_ratio, FEAT-WINDOW-2
- **P3:** dns_query_count, smb_connection_burst, wmi_activity_proxy
- **Descartadas:** ICMP_unreachable_rate, firma compuesta DNS→SMB

---

## Primer objetivo DAY 91 — ARCHITECTURE.md

**Documento técnico completo del pipeline para colaboradores externos.**
Es el trabajo principal de hoy. No hay código que tocar primero.

Path destino: `ARCHITECTURE.md` (raíz del repo)

Secciones a cubrir:
1. Visión general — qué es ML Defender, para quién, por qué
2. Componentes (6) — descripción técnica de cada uno
3. Flujo de datos — sniffer → ml-detector → firewall → rag-ingester → rag-security
4. Decisiones arquitectónicas clave — ADRs referenciados
5. Deployment — Vagrant para experimentos, bare-metal para producción
6. Diagrama ASCII del pipeline (expandir el del README)
7. Stack tecnológico — C++20, eBPF/XDP, ZeroMQ, ChaCha20, FAISS, protobuf, ONNX

**Referencia:** ARCHITECTURE.md v5.1.0 ya existe en el repo (DAY 62).
Hay que revisar qué tiene antes de escribir desde cero.

---

## Segundo objetivo DAY 91 — Spec sintético WannaCry

Path: `docs/design/synthetic_data_wannacry_spec.md`

Contenido base ya definido por el Consejo:
- Port 445 burst: rst_ratio > 0.5, connection_rate > 100/s
- unique_dst_ips_count: miles en ventana 10s
- syn_ack_ratio: < 0.1 (handshakes fallidos)
- Killswitch DNS: NO incluir como señal (no detectable)
- Control negativo: tráfico administrativo Windows (WSUS, backups, SCCM)

---

## Tercer objetivo DAY 91 — Integrar Consejo #1

- Compartir `consejo_consulta_1_decisiones_finales.md` con el Consejo
- Esperar respuesta genuina de Qwen para completar el registro

---

## Constantes del proyecto

```
Raíz:          /Users/aironman/CLionProjects/test-zeromq-docker
VM:            vagrant ssh defender
Logs:          /vagrant/logs/lab/
F1 log:        docs/experiments/f1_replay_log.csv
Paper:         docs/Ml defender paper draft v4.md
macOS CRÍTICO: NUNCA usar sed -i sin -e '' — usar Python3 o editar en VM
```

---

## Backlog activo (P1)

| ID | Descripción | Estado |
|---|---|---|
| DEBT-FD-001 | Fast Detector Path A hardcoded (ignora sniffer.json) | Pendiente PHASE2 |
| ADR-007 | AND-consensus firewall | ACEPTADO — implementación PHASE2 |
| FEAT-NET-1 | DNS/DGA detection | P1 PHASE2 |
| FEAT-NET-2 | Threat intel feeds | P1 PHASE2 |
| rst_ratio | Implementar desde sentinel | **P1 — próxima sesión con VM** |
| syn_ack_ratio | Implementar desde sentinel | **P1 — próxima sesión con VM** |
| FEAT-WINDOW-2 | Agregador 60s para NotPetya | P2 PHASE2 |

---

## Estado arXiv

- Paper draft v4 listo en `docs/Ml defender paper draft v4.md`
- Esperando respuesta de Sebastian Garcia (endorser arXiv)
- Si no responde en DAY 96 → email Yisroel Mirsky (Tier 2)
- Limitaciones a añadir en §10 (resultado Consejo #1):
  - Killswitch DNS no detectable
  - Generalización a ransomware SMB: Recall 0.70–0.85 sin retraining

---

*Co-authored-by: Alonso Isidoro Román + Claude (Anthropic)*
*DAY 90 — 18 marzo 2026*
*Consejo de Sabios — ML Defender (aRGus EDR)*
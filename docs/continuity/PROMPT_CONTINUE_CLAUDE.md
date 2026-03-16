# ML Defender — Prompt de Continuidad DAY 88
**Generado:** Cierre DAY 87 (15 marzo 2026)
**Branch activa:** `main`
**Estado del pipeline:** 6/6 componentes RUNNING ✅
**Tests:** crypto 3/3 ✅ | etcd-hmac 12/12 ✅ | ml-detector 9/9 ✅ | trace_id 46/46 ✅

---

## Logros DAY 87

### TAREA 1 — Paper draft v4 ✅
- Fichero: `docs/Ml defender paper draft v4.md`
- Cambio principal: **§8.9 Throughput Stress Test and Resource Utilization** añadido
- §9.3 actualizado con techo medido (no solo teórico)
- §9.4 Queue Stability: validación empírica directa documentada
- §11.11 actualizado: stress test virtualizado completado, bare-metal = P1
- Abstract y Conclusion actualizados con resultados del stress test
- **Commit pendiente** (ver comando abajo)

### TAREA 2 — Stress Test DAY 87 completado ✅
Resultados definitivos:

| Run | Dataset | Pedido | Rated real | Paquetes | Failed | ML errors |
|-----|---------|--------|-----------|----------|--------|-----------|
| 1 | Neris | 10 Mbps | 9.60 Mbps | 320,524 | 2,630† | 0 |
| 2 | Neris | 25 Mbps | 11.28 Mbps | 320,524 | 2,630† | 0 |
| 3 | bigFlows | 50 Mbps | 34.68 Mbps | 791,615 | 0 | 0 |
| 4 | bigFlows | 100 Mbps (×3) | 33.16 Mbps | 2,374,845 | 0 | 0 |

† Artefactos de pcap, no saturación del pipeline.

**Recursos bajo 100 Mbps (top -b -d 5):**
- ml-detector: ~315–320% CPU, 164 MB RES (estable)
- sniffer: ~88–108% CPU, 55–57 MB RES
- Sistema total: ~65–73% user, ~23–30% idle
- RAM total: 1,271–1,289 MB (~18 MB drift en 8 min = negligible)
- Post-replay: sniffer cae a <2% CPU; ml-detector mantiene ~315% drenando cola ZMQ → validación empírica queue stability

**Conclusión:** cuello de botella = NIC virtual VirtualBox (~33–38 Mbps), no el pipeline.

---

## Tarea DAY 88: Integrar feedback v4 → draft v5

### Feedback pendiente de integrar
Alonso tiene feedback sobre el draft v4. Hay que:
1. Recibir el feedback de Alonso al inicio de la sesión
2. Evaluar qué cambios son correcciones críticas vs mejoras menores
3. Si hay feedback del Consejo de Sabios (otros modelos), integrarlo también
4. Generar draft v5

### Commit DAY 87 pendiente
```bash
cd /Users/aironman/CLionProjects/test-zeromq-docker

git add "docs/Ml defender paper draft v4.md"
git commit -m "docs: paper arXiv v4 — stress test DAY 87 integrated

§8.9 Throughput Stress Test: VirtualBox NIC ceiling ~33-38 Mbps
- 2,374,845 packets processed (loop=3 bigFlows), 0 failures, 0 errors
- ml-detector: ~315-320% CPU sustained, 164 MB RES stable
- System: ~65-73% user, ~23-30% idle — pipeline is NOT the bottleneck
- RAM: 1,271-1,289 MB, ~18 MB drift over 8 min (negligible)
- Post-replay drain: ZeroMQ queue stability empirically validated

§9.3 Throughput: theoretical + measured ceiling documented
§9.4 Queue stability: direct empirical validation cited
§11.11 Updated: virtualized stress test complete, bare-metal = P1
Abstract + Conclusion updated with stress test results

Co-authored-by: Claude (Anthropic) <claude@anthropic.com>"

git push origin main
```

---

## Estado general del proyecto

### Paper
- **Draft v4** en `docs/Ml defender paper draft v4.md` — commit pendiente
- Objetivo final: versión LaTeX para arXiv (cs.CR)
- Venue: arXiv → RAID 2026 / USENIX Security Fall 2026
- Endorser objetivo: Sebastian Garcia (CTU Prague, autor CTU-13)
- Estructura: Abstract | §1 Intro | §2 Related Work | §3 Threat Model | §4 Architecture | §5 Implementation | §6 Consejo de Sabios | §7 Formal Model | §8 Evaluation | §9 Performance | §10 Limitations | §11 Future Work | §12 Conclusion | §13 Reproducibility | §14 Acknowledgments | References

### Métricas definitivas (paper)
- F1=0.9985, Precision=0.9969, Recall=1.0000 (CTU-13 Neris, DAY 86, 4 runs estables)
- FPR=0.0002% neris | FPR Fast Detector=6.61% bigFlows | FP reduction ~500×
- 2 FP identificados: artefactos VirtualBox (multicast + broadcast), no existen en bare-metal
- Latencia: 0.24 μs (DDoS) a 1.06 μs (Ransomware)
- Stress test: ~33–38 Mbps techo VirtualBox NIC, 0 failed, 0 errors, RAM estable

### Deuda técnica (post-paper)
| Item | Prioridad | Estado |
|---|---|---|
| Paper arXiv — draft v5 (feedback v4) | **P0** | DAY 88 |
| Paper arXiv — versión LaTeX | P0 | post-v5 |
| Endorsers arXiv — contacto (Sebastian Garcia + alternativas) | P0 | post-v5 |
| Bare-metal throughput stress test | P1 | post-paper |
| DNS payload parsing real | P2 | post-paper |
| DEBT-FD-001: FastDetector Path A → JSON | P1-PHASE2 | post-paper |
| ADR-007: Consenso AND firewall | P1-PHASE2 | post-paper |

---

## Nuevo backlog: Detección de vectores de entrada (FEAT-ENTRY-*)

### Contexto (DAY 87 — conversación de cierre)
ML Defender detecta y detiene la **propagación** del ransomware (movimiento lateral SMB).
Lo que NO cubre actualmente es la **infiltración** — el momento en que el atacante entra.
Alonso quiere documentar esto honestamente en el paper (§11 Future Work) y añadirlo
al backlog open source. Todas estas features serán MIT license.

**Frase clave para el paper:**
*"ML Defender detiene la propagación. Lo que sigue es detectar la infiltración."*

### Features a añadir al §11 Future Work del paper (draft v5)

**FEAT-NET-1 (P1 — extensión natural del pipeline actual)**
DNS anomaly detection:
- DGA (Domain Generation Algorithm) detection — entropía alta en subdominios
- Dominios de registro reciente como proxy de C2
- Volumen de queries DNS anómalo por host
- No requiere cambios arquitectónicos — nuevo detector en la capa de features del sniffer

**FEAT-NET-2 (P1 — extensión natural)**
Threat intelligence feed integration:
- IPs/dominios maliciosos conocidos consultados contra flows capturados en tiempo real
- Encaja con el RAG subsystem como fuente de datos adicional
- Feeds públicos: abuse.ch, Emerging Threats, AlienVault OTX

**FEAT-AUTH-1 (P2 — nuevo input al pipeline)**
Ingesta de logs de autenticación:
- Syslog / Windows Event Log forwarding al pipeline
- Detección de credential stuffing: múltiples logins fallidos desde misma IP externa
- Login exitoso desde geolocalización o IP anómala
- Múltiples logins simultáneos con mismas credenciales
- Arquitectura: nueva fuente de input a ZeroMQ, sin reescribir componentes existentes

**FEAT-AUTH-2 (P2)**
Behavioral authentication anomaly detection:
- Acceso a recursos sensibles fuera de horario laboral
- Patrón de acceso a shares de red inusual (reconocimiento pre-ransomware)
- Escalada de privilegios anómala

**FEAT-EDR-1 (P3 — salto arquitectónico, convierte NIDS en EDR completo)**
Agente ligero en endpoints Windows/Linux:
- Eventos de proceso: procesos hijos anómalos de apps legítimas (Word → PowerShell)
- PowerShell / cmd con comandos codificados en base64
- Filesystem: creación/modificación masiva de ficheros (firma de cifrado)
- Enumeración de network shares (net view, net use)
- Reporta al pipeline central vía ZeroMQ
- Nota: este feature cierra el loop completo — ML Defender ya es EDR en nombre, aquí lo sería en implementación

### Tabla resumen para §11 Future Work
| Feature | Descripción | Prioridad | Vector cubierto |
|---|---|---|---|
| FEAT-NET-1 | DNS anomaly / DGA detection | P1 | C2 communication |
| FEAT-NET-2 | Threat intelligence feeds | P1 | Known malicious IPs/domains |
| FEAT-AUTH-1 | Auth log ingestion + brute force detection | P2 | RDP / credential attacks |
| FEAT-AUTH-2 | Behavioral auth anomaly | P2 | Stolen credentials |
| FEAT-EDR-1 | Lightweight endpoint agent | P3 | Phishing payload execution |

### Nota sobre coste del atacante vs efectividad (DAY 87)
- Botnet 200 Mbps / 1h: ~5-20 USD en booters
- Pipeline bloquea núcleo de botnet: ~30-60 segundos
- Atacante nota que el ataque no funciona: ~60-180 segundos
- Asimetría clave: el coste de defender no escala con el número de ataques; el coste del atacante sí
- Para ransomware: ML Defender es efectivo si lo detecta en fase de propagación lateral (ventana típica: horas antes del cifrado masivo en ataques modernos)
- Punto ciego actual: low-and-slow infiltration, phishing, RDP brute force → cubierto por FEAT-* arriba

### Para §8.9 v5: párrafo adicional sobre eficiencia de recursos
Añadir interpretación de que 1.28 GB RAM (incluyendo TinyLlama) + 3.2 cores
es compatible con hardware existente en hospitales/escuelas — no requiere
servidor dedicado nuevo. Un servidor de administración de 5 años con 16 GB RAM
puede correr ML Defender sin interferir con otros servicios.

---

## Infraestructura permanente

- **macOS (BSD sed):** Nunca `sed -i`. Usar Python3 inline **desde dentro de la VM**.
- **JSON sniffer:** `sniffer/config/sniffer.json` (NO build-debug/)
- **JSON ml-detector:** `ml-detector/config/ml_detector_config.json`
- **VM:** `defender` (no `server`)
- **CSV ml-detector:** `/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv`
- **CSV firewall:** `/vagrant/logs/firewall_logs/firewall_blocks.csv`
- **F1 calculator:** `python3 scripts/calculate_f1_neris.py <sniffer.log> --total-events 19135`
- **Flujo correcto test:**
  ```bash
  make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15
  make test-replay-neris
  python3 scripts/calculate_f1_neris.py /vagrant/logs/lab/sniffer.log --total-events 19135
  ```

---

## Consejo de Sabios
Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), DeepSeek, Qwen (Alibaba), Gemini (Google), Parallel.ai — 7 modelos.

---

## Secuencia DAY 88

1. Commit DAY 87 (comando arriba)
2. Recibir feedback v4 del Consejo de Sabios
3. Integrar feedback + FEAT-ENTRY-* en §11 + párrafo §8.9 recursos → draft v5
4. **README.md raíz** — actualizar DESPUÉS de cerrar v5, no antes
  - Números definitivos del paper
  - Stress test results
  - Hardware recomendado
  - Roadmap con FEAT-ENTRY-*
  - Instrucciones de reproducción
  - Sección de financiación / colaboración (ENISA, INCIBE, Horizon Europe)

*DAY 88 arranca con: commit DAY 87 + feedback v4 → draft v5 → README*
*Test suite: 100% ✅ | Stress test: completado ✅ | Paper: draft v4 ✅*
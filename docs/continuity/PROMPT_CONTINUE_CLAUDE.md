# PROMPT DE CONTINUIDAD — DAY 143

---

Soy Alonso Isidoro Román, fundador de aRGus NDR, sistema open-source C++20
de detección y respuesta a intrusiones de red para infraestructura crítica
(hospitales, escuelas, municipios). Trabajo en modo "solopreneur" con un
Consejo de Sabios de 8 modelos de IA como equipo de revisión adversarial.

**Estado repo:** branch `feature/variant-b-libpcap` @ `9458a90d` (último commit DAY 142)
**Tag main:** `v0.6.0-hardened-variant-a` @ `737ba0d5`
**arXiv:** 2604.04952 — Draft v18 (Cornell procesando)
**Keypair activo:** `b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa`
**FEDER deadline:** 22-Sep-2026 | **Go/no-go:** 1-Ago-2026

---

## COMPLETADO DAY 142

- **EMECAS verde:** 15 warnings third-party, 0 warnings propios, 0 errors.
- **Regresión cerrada:** `test_config_parser` — safe_path (ADR-037) bloqueaba
  `/vagrant/...` correctamente. Fix: usar path de producción
  `/etc/ml-defender/rag-ingester/rag-ingester.json`. Commit `4bbc98ee`.
- **DEBT-IRP-NFTABLES-001 sesiones 1/3 y 2/3:** ✅
    - Binario `argus-network-isolate` C++20 en `tools/argus-network-isolate/`.
    - Pasos 1-3: snapshot selectivo (solo tabla `argus_isolate`), generate_rules,
      validate_dry_run (`nft -c`).
    - Pasos 4-6: apply atómico (`nft -f`), timer `systemd-run` idempotente (300s),
      rollback robusto (elimina tabla, no toca tablas iptables-managed).
    - Ciclo verificado en dev VM (eth2): NORMAL→ISOLATED→STATUS→ROLLBACK→NORMAL.
    - SSH sobrevivió (eth0 + whitelist). Forense JSONL operativo.
    - Commits `6480e234` + `e8928612`.
- **Reproducibilidad EMECAS garantizada:** `e3f5f9c4`
    - Vagrantfile: `nftables` declarado explícitamente.
    - `provision.sh`: instala `isolate.json` en
      `/etc/ml-defender/firewall-acl-agent/` + crea `/var/log/argus/`.
    - Makefile: `argus-network-isolate-build` + `argus-network-isolate-install`
      en `pipeline-build`. `check-system-deps` verifica nftables + binario.
- **DEBT-VARIANT-B-BUFFER-SIZE-001 CERRADA:** `7c4dba58`
    - `PcapBackend::open()` refactorizado: `pcap_open_live()` →
      `pcap_create()+pcap_set_buffer_size()+pcap_activate()`.
    - `CaptureBackend` interfaz actualizada.
    - Verificado: `[pcap] Variant B opened on eth1 buffer=8MB`.
- **DEBT-VARIANT-B-MUTEX-001 CERRADA (Nivel 1):** `9458a90d`
    - `scripts/check-sniffer-mutex.sh` via sesiones tmux.
    - Makefile: `sniffer-start` y `sniffer-libpcap-start` llaman al mutex.
    - Verificado: Variant B activa + intento Variant A → violación detectada,
      exit 1.
    - **NOTA:** Esta implementación es provisional (Nivel 1). Ver
      `DEBT-MUTEX-ROBUST-001` post-FEDER.

**make test-all:** verde pre-commits (9/9 sniffer Variant B PASSED).

---

## PRIMER PASO DAY 143

```bash
vagrant destroy -f && vagrant up && make bootstrap && make test-all
```

Verificar:
```bash
grep -c 'warning:' protocol-EMECAS-output-06-05-2026.md
grep 'warning:' protocol-EMECAS-output-06-05-2026.md | grep -v 'defender:'
grep -c 'error:' protocol-EMECAS-output-06-05-2026.md
git log --oneline -5
vagrant ssh -c "ls -la /usr/local/bin/argus-network-isolate"
vagrant ssh -c "ls -la /etc/ml-defender/firewall-acl-agent/isolate.json"
vagrant ssh -c "ls -la /var/log/argus/"
vagrant ssh -c "sudo /usr/sbin/nft --version"
```

---

## DEUDA PRIORITARIA DAY 143

### `DEBT-IRP-NFTABLES-001` — sesión 3/3

**Decisiones de diseño aprobadas DAY 142 (Consejo 8/8 + founder):**

- `auto_isolate: true` por defecto — instalar y funcionar sin leer el manual.
- Criterio de disparo: score `>= 0.95` NO es señal suficiente sola. Se necesita
  señal multi-componente. Para FEDER: `threat_score >= 0.95 AND event_type IN
  (ransomware, lateral_movement, c2_beacon)` mínimo dos condiciones AND.
- Contexto hospitalario: monitores de quirófano y equipos médicos pueden estar
  en intranet/DMZ — `firewall-acl-agent` corriendo en esos nodos tiene sentido.
  El aislamiento nunca puede basarse en señal única — un falso positivo sobre
  un ventilador mecánico es inaceptable.
- `fork() + execv()` — el firewall NO puede morir. Operación atómica.
- AppArmor: `enforce` por defecto. Combinar perfiles de Gemini + Kimi mañana.
- Rollback: diseño actual suficiente para FEDER.

**Trabajo sesión 3:**
1. Añadir a `isolate.json`: `auto_isolate`, `threat_score_threshold` (0.95),
   `auto_isolate_event_types` (ransomware, lateral_movement, c2_beacon).
2. En `firewall-acl-agent`: detectar umbral + tipo superado → `fork()+execv()`
   a `argus-network-isolate isolate --interface <iface>`.
3. Test de integración con evento sintético score >= 0.95 + tipo correcto.
4. AppArmor profile `enforce` para `argus-network-isolate` (combinar Gemini +
   Kimi).
5. Instalar binario en `provision.sh` + `pipeline-build` para hardened VM.

---

## NUEVAS DEUDAS REGISTRADAS DAY 142

### DEBT-MUTEX-ROBUST-001 — Mutex robusto entre variantes sniffer
**Severidad:** 🟡 P1 post-FEDER
**Estado:** ABIERTO — DAY 142
La implementación actual via sesiones tmux es Nivel 1 provisional. No es
robusta en producción (depende de herramienta de usuario). Alternativas a
evaluar: `flock` (lockfile), PID file, o mecanismo etcd cuando esté en HA.
La solución definitiva depende de `DEBT-ETCD-HA-QUORUM-001`.

### DEBT-ETCD-HA-QUORUM-001 — etcd-server en HA con quorum
**Severidad:** 🔴 Alta — P0 post-FEDER (OBLIGATORIO, no opcional)
**Estado:** ABIERTO — DAY 142
etcd-server actual es single-node. Si cae, ningún componente puede
registrarse ni coordinarse. Diseño requerido:
- Múltiples instancias etcd-server.
- Quorum con líder elegido (Raft o equivalente).
- Componentes se registran ante el primer etcd disponible.
- Al recuperarse un nodo etcd, se une al quorum y sincroniza estado.
- Si el líder cae, quorum inmediato para elegir nuevo líder.
- Estado compartido garantizado: todos los componentes registrados y vivos.
  **Nota:** No es deuda "eterna" — es deuda crítica que hay que cerrar. Es
  prerequisito de `DEBT-MUTEX-ROBUST-001` y de cualquier coordinación fiable
  entre componentes.

### DEBT-IRP-MULTI-SIGNAL-001 — Criterio de disparo multi-señal
**Severidad:** 🟡 P1 post-FEDER
**Estado:** ABIERTO — DAY 142
Para FEDER: dos condiciones AND mínimas (score + event_type). Para
producción hospitalaria: señal más rica. Monitores de quirófano y equipos
médicos en intranet/DMZ son activos críticos — el criterio de disparo debe
ser explicable, auditable y resistente a falsos positivos. Registrado como
`IDEA-IRP-DECISION-MATRIX-001`.

### DEBT-IRP-LAST-KNOWN-GOOD-001 — Rollback con estado persistente
**Severidad:** 🟢 Baja post-FEDER
**Estado:** ABIERTO — DAY 142
Para entornos con rulesets nftables propios del cliente.
`/etc/ml-defender/firewall-acl-agent/last-known-good.nft`.
No bloqueante para FEDER — diseño actual (eliminar solo `argus_isolate`)
es correcto y suficiente para el MVP.

---

## REGLAS PERMANENTES

- REGLA EMECAS: `vagrant destroy -f && vagrant up && make bootstrap && make
  test-all`
- macOS: nunca `sed -i` sin `-e ''` — usar `python3 << 'PYEOF'` o
  `vagrant ssh << 'SSHEOF'` para código con emojis o caracteres especiales
- zsh intercepta `!` en heredocs — siempre `vagrant ssh << 'SSHEOF'`
- Makefile es la única fuente de verdad
- **`-Werror` activo** — 0 warnings propios invariante permanente
- Verificación EMECAS: `grep 'warning:' output.md | grep -v 'defender:'` = 0
- `PROFILE=production all` antes de cualquier merge a main (gate ODR)
- Variant B es monohilo por diseño — no configurable
- Variant A y Variant B nunca simultáneas (DEBT-VARIANT-B-MUTEX-001 Nivel 1)
- Warning classifiers: script grep/awk determinista, no LLM
- Código terceros deprecated → `docs/THIRDPARTY-MIGRATIONS.md`
- `argus-network-isolate`: usar `--dry-run` en dev VM, apply solo en hardened

---

## IDEAS REGISTRADAS — pendiente diseño formal

### DEBT-VARIANT-B-PCAP-RELAY-001
Relay real con pcap versionado (CTU-13 Neris subset + SHA-256) como gate de
merge de `feature/variant-b-libpcap`. Pendiente: decidir si bloquea merge o
entra como fase EMECAS general.

### IDEA-SEED-PLUGIN-001
Plugin de obtención de semillas por profile (DEV→seed.bin local,
PROD→Vault via Jenkinsfile). Tensión de bootstrap a resolver antes de
diseñar. Target: post-FEDER junto a DEBT-CRYPTO-MATERIAL-STORAGE-001 y
DEBT-JENKINS-SEED-DISTRIBUTION-001.
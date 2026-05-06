Soy Alonso Isidoro Román, fundador de aRGus NDR, sistema open-source C++20
de detección y respuesta a intrusiones de red para infraestructura crítica.

Estado repo: branch feature/variant-b-libpcap @ f00b1809 (último commit DAY 143)
Tag main: v0.6.0-hardened-variant-a @ 737ba0d5
arXiv: 2604.04952 — Draft v18 (Cornell procesando)
Keypair activo: b5b6cbdf67dad75cdd7e3169d837d1d6d4c938b720e34331f8a73f478ee85daa
FEDER deadline: 22-Sep-2026 | Go/no-go: 1-Ago-2026

COMPLETADO DAY 143:
- EMECAS verde: DEBT-BOOTSTRAP-ORDER-001 resuelta, regresión firma
  PcapBackend::open() corregida en 5 test files.
- DEBT-IRP-NFTABLES-001 sesión 3/3 CERRADA:
    * isolate.json: auto_isolate, threat_score_threshold, auto_isolate_event_types,
      isolate_interface
    * firewall-acl-agent: IrpConfig, should_auto_isolate() (función pura),
      check_auto_isolate() fork()+execv()
    * Bug IEEE 754 detectado por tests y corregido (tolerancia 1e-6)
    * test_auto_isolate: 12/12 PASSED
    * AppArmor argus.argus-network-isolate — 7/7 perfiles enforce hardened VM
- Consejo 8/8 — 5 deudas nuevas registradas (ver abajo)

PRIMER PASO DAY 144:
vagrant destroy -f && vagrant up && make bootstrap && make test-all

DEUDAS P0 PRE-MERGE (bloqueantes para merge a main):

DEBT-IRP-SIGCHLD-001 — SA_NOCLDWAIT
fork()+execv() sin wait() acumula zombies.
Fix: sigaction(SIGCHLD, SA_NOCLDWAIT) en main.cpp del firewall-acl-agent.
El kernel recoge hijos automáticamente. Una línea.
Test: N disparos IRP en loop → ps aux | grep defunct = 0.

DEBT-IRP-AUTOISO-FALSE-001 — auto_isolate: false por defecto
REEMPLAZA la regla DAY 142 ("auto_isolate: true por defecto").
Consejo 8/8 unánime: false en hospitales. Un FP sobre ventilador mecánico
es inaceptable sin onboarding explícito.
Fix: isolate.json default false + WARNING prominente al arrancar
firewall-acl-agent con IRP desactivado.

DEBT-IRP-BACKUP-DIR-001 — /tmp peligroso
Migrar artefactos nftables de /tmp/ a:
- /run/argus/irp/ (tmpfs, volátil) para backup + ruleset temporal
- /var/lib/argus/irp/ (persistente) para estado IRP
  Permisos: 0700 argus:argus. Actualizar AppArmor + Falco vigila ambas rutas.

DESPUÉS DEL MERGE (orden):
A) make PROFILE=production all (gate ODR — invariante pre-merge)
B) git merge --no-ff feature/variant-b-libpcap → main
C) tag v0.7.0-variant-b
D) ADR-029 benchmark Variant A vs B (contribución científica paper)

DEUDAS P1 PRE-FEDER (no bloqueantes para merge):
DEBT-IRP-FLOAT-TYPES-001 — Unificar tipos score float/double.
Investigar qué tipo produce exactamente el ml-detector antes de decidir.
La tolerancia 1e-6 es parche correcto pero no solución de raíz.

DEUDAS P1 POST-FEDER:
DEBT-IRP-PROB-CONJUNTA-001 — Función probabilidad conjunta multi-señal.
No topología por quirófano (inviable). Todas las señales disponibles con
pesos → probabilidad conjunta → decisión auditable + publicable.
DEBT-PROTO-DETECTION-TYPES-001 — Ampliar enum post-MITRE/CTF.

## EXPERIMENTO POST-MERGE — aRGus vs Suricata vs Zeek

**Acordado DAY 143 — primer tema del Consejo post-merge.**

**Contexto:** Suricata (2009), Zeek/Bro (1994), Snort (1998) llevan décadas operando.
No hay que tenerles miedo — hay que entender empíricamente dónde están ellos y
dónde estamos nosotros. La verdad, sea cual sea, es más valiosa que la narrativa
que queremos contar.

**Hipótesis a demostrar (no afirmar):**
- Ellos detectan lo CONOCIDO mejor que nadie — reglas ET, firmas CVE, patrones catalogados.
- aRGus detecta lo DESCONOCIDO mejor que ellos — comportamiento anómalo sin firma previa.
- F1=0.9985 Recall=1.0000 sobre CTU-13 Neris es el baseline. ¿Lo igualan o superan?
- Si el experimento dice que Suricata también lo detecta — mejor saberlo ahora que en FEDER.

**Diseño del experimento (a desarrollar con el Consejo):**
- Tráfico base: CTU-13 Neris (baseline conocido, resultados aRGus ya documentados)
- Tráfico adversarial: MITRE ATT&CK (primer escenario real sin firma conocida)
- Los tres sistemas sobre el mismo tráfico simultáneamente
- Métricas: qué detecta cada uno, solapamiento, qué solo ve uno de los tres
- Contribución científica: complementariedad de detectores para paper arXiv v19

**Orden DAY 144+:**
1. EMECAS + deudas P0 pre-merge (SIGCHLD, AUTOISO-FALSE, BACKUP-DIR)
2. PROFILE=production gate ODR
3. Merge feature/variant-b-libpcap → main → tag v0.7.0-variant-b
4. Diseño experimento aRGus + Suricata + Zeek (Consejo 8/8)
5. 
REGLAS PERMANENTES:
- REGLA EMECAS: vagrant destroy -f && vagrant up && make bootstrap && make test-all
- macOS: nunca sed -i sin -e '' — usar python3 << 'PYEOF' o vagrant ssh << 'SSHEOF'
- zsh intercepta ! en heredocs — siempre vagrant ssh << 'SSHEOF'
- Makefile es la única fuente de verdad
- -Werror activo — 0 warnings propios invariante permanente
- PROFILE=production all antes de cualquier merge a main (gate ODR)
- Variant B es monohilo por diseño — no configurable
- Variant A y Variant B nunca simultáneas

RECORDATORIO: Si Andrés Caro Lindo no ha respondido antes del viernes 8 Mayo,
enviar WhatsApp sobre los dos emails (hardware FEDER + scope NDR).
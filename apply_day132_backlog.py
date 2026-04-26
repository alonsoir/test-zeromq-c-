#!/usr/bin/env python3
# apply_day132_backlog.py
# Aplica los cambios de DAY 132 al BACKLOG.md
# Uso: python3 apply_day132_backlog.py
# macOS safe — sin sed, sin heredoc complejo

import re

with open('docs/BACKLOG.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Actualizar header de fecha
content = content.replace(
    '*Última actualización: DAY 130 — 25 Abril 2026*',
    '*Última actualización: DAY 132 — 26 Abril 2026*'
)

# 2. Añadir sección CERRADO DAY 132 después de "## ✅ CERRADO DAY 130"
# (antes de "## ✅ CERRADO DAY 129")
CERRADO_132 = '''
---

## ✅ CERRADO DAY 132

### DEBT-PROD-COMPAT-BASELINE-001
- **Status:** ✅ CERRADO DAY 132
- **Fix:** `docs/HARDWARE-REQUIREMENTS.md` — especificaciones mínimas y recomendadas, tabla de plataformas commodity (~150–200 USD), compatibilidad XDP por driver NIC, paquetes runtime vs paquetes prohibidos en producción, relación con BSR axiom (ADR-039).
- **Commit:** `9b3438fb` en `feature/adr030-variant-a`

### vagrant/hardened-x86/Vagrantfile — ADR-030 Variant A (inicio)
- **Status:** ✅ INICIADO DAY 132 (skeleton)
- **Fix:** VM Debian 12 + AppArmor enforcing + sin compilador + verificación BSR en provisioner + post_up_message con pasos siguientes.
- **Commit:** `9b3438fb` en `feature/adr030-variant-a`
- **Pendiente:** Makefile targets de producción (DAY 133)

### Paper Draft v17 — §6.5, §6.8, §6.10, §6.12
- **Status:** ✅ GENERADO DAY 132 — pendiente arXiv
- **Fix:** +315 líneas en §6: RED→GREEN gate, Fuzzing como tercera capa, CWE-78 execv(), BSR axiom. Nuevas referencias: `cwe78` + `thompson1984`.
- **Commit:** `b7d38d1f` en `main`
- **Pendiente:** métricas reales de fuzzing (§6.8) + corrección frase "misses nothing" + métricas VM hardened (§6.12) antes de arXiv

### README — Prerequisites
- **Status:** ✅ CERRADO DAY 132
- **Fix:** Sección `## 🔧 Prerequisites` con instrucciones de instalación de Vagrant + VirtualBox + make para macOS y Linux.
- **Commit:** `18d8e101` en `main`

'''

content = content.replace(
    '\n---\n\n## ✅ CERRADO DAY 130\n',
    CERRADO_132 + '\n---\n\n## ✅ CERRADO DAY 130\n'
)

# 3. Añadir nuevas deudas abiertas en la sección de deuda abierta de producción
NEW_DEBTS = '''
---

## 🔴 DEUDA ABIERTA — Seguridad imagen de producción (ADR-030)

### DEBT-PROD-APPARMOR-COMPILER-BLOCK-001
**Severidad:** 🔴 Alta | **Bloqueante:** Sí (pre-producción) | **Target:** feature/adr030-variant-a
**Origen:** DAY 132 — Consejo 8/8 + decisión founder
**Descripción:** AppArmor es la primera línea de defensa del BSR axiom, no el check de dpkg. Los perfiles AppArmor de la imagen de producción deben bloquear explícitamente:
- Instalación de compiladores (`apt install gcc` → DENIED)
- Ejecución de compiladores desde cualquier path (`/usr/bin/gcc`, `/usr/local/bin/gcc`, `/tmp/gcc`, etc. → DENIED)
- Escritura en paths de sistema fuera del pipeline (`/usr/bin/`, `/usr/sbin/` → DENIED para el usuario de servicio)

El check `check-prod-no-compiler` (dpkg + command -v) es evidencia auditable, no la defensa real. AppArmor es la defensa real.

**Fix:** Perfiles AppArmor específicos por componente del pipeline, más un perfil global que deniegue ejecución de compiladores en toda la imagen. Verificar con `aa-status` + test de intento de instalación en VM hardened.

**Test de cierre:** `sudo apt-get install -y gcc` en VM hardened → DENIED por AppArmor (log en `/var/log/syslog`). `aa-status` muestra perfil en enforce mode.

---

### DEBT-PROD-FALCO-EXOTIC-PATHS-001
**Severidad:** 🔴 Alta | **Bloqueante:** Sí (pre-producción) | **Target:** feature/adr030-variant-a
**Origen:** DAY 132 — decisión founder
**Descripción:** Falco como vigilancia runtime de paths exóticos en la imagen de producción. AppArmor previene; Falco detecta y alerta cuando algo intenta evadir AppArmor.

Paths a vigilar:
- Escritura en `/tmp`, `/var/tmp`, `/dev/shm` por procesos del pipeline
- Ejecución desde `/tmp`, `/var/tmp`, `/opt`, `/home` (nunca debería ocurrir)
- Cualquier `execve()` de binarios fuera de `/usr/bin`, `/usr/sbin`, `/usr/local/bin`
- Cambios en `/etc/apt/sources.list` o `/etc/apt/sources.list.d/`
- `ptrace()` sobre procesos del pipeline

**Fix:** Reglas Falco específicas para aRGus. Falco en modo `--daemon` con output a `/var/log/ml-defender/falco-alerts.log`. Integración con rag-ingester para correlación de eventos de seguridad.

**Test de cierre:** `echo "" > /tmp/test` desde proceso del pipeline → alerta Falco en log. `execve("/tmp/evil", ...)` → alerta Falco + DENIED AppArmor.

---

### DEBT-PROD-FS-MINIMIZATION-001
**Severidad:** 🔴 Alta | **Bloqueante:** Sí (pre-producción) | **Target:** feature/adr030-variant-a
**Origen:** DAY 132 — decisión founder
**Descripción:** La imagen de producción solo tiene acceso (lectura/escritura/ejecución) a los paths estrictamente necesarios para el funcionamiento del pipeline. Todo lo demás, denegado.

Inventario de paths necesarios por componente:
- `/etc/ml-defender/` — configs (lectura)
- `/usr/lib/ml-defender/plugins/` — plugins (lectura + ejecución)
- `/usr/lib/ml-defender/models/` — modelos XGBoost (lectura)
- `/var/log/ml-defender/` — logs (escritura)
- `/run/ml-defender/` — sockets ZeroMQ (lectura/escritura)
- `/usr/bin/`, `/usr/sbin/` — binarios del sistema necesarios (ejecución)
- Binarios propios del pipeline en `/usr/local/bin/` (ejecución)

Todo lo demás: `noexec`, `nodev`, `nosuid` en mount options + AppArmor `deny` explícito.

Directorios exóticos (`/tmp`, `/var/tmp`, `/dev/shm`, `/opt`, `/home`, `/root`): acceso mínimo o nulo. Montados con `noexec,nodev,nosuid`.

**Fix:** `fstab` de la imagen hardened con opciones de montaje correctas. Perfiles AppArmor con deny explícito sobre paths no necesarios. Documentar inventario en `docs/PROD-FS-MAP.md`.

**Test de cierre:** Proceso del pipeline no puede escribir en `/tmp`. No puede ejecutar desde `/home`. `mount | grep noexec` muestra todos los paths exóticos con `noexec`.

---

### DEBT-PROD-APT-SOURCES-INTEGRITY-001
**Severidad:** 🔴 Crítica | **Bloqueante:** Sí (pre-producción) | **Target:** feature/adr030-variant-a
**Origen:** DAY 132 — decisión founder
**Descripción:** Los ficheros de fuentes apt (`/etc/apt/sources.list`, `/etc/apt/sources.list.d/`) son controlados por el proyecto. Solo se permiten actualizaciones desde repositorios autorizados y verificados.

Si se detecta modificación de estos ficheros respecto al hash SHA-256 conocido en el momento de la imagen cocinada:
- **Opción A (fail-closed):** El pipeline no arranca. Log explícito: "SECURITY: apt sources modified. Pipeline start refused. Contact system administrator."
- **Opción B (fail-warn):** El pipeline arranca con banner de alerta visible en cada log: "WARNING: apt sources integrity check FAILED. System may be compromised. Contact administrator."

**Decisión founder:** Opción A (fail-closed) como default. Opción B configurable para entornos donde la continuidad del servicio es crítica (hospitales en horario de emergencias).

**Fix:**
1. En la imagen cocinada, calcular SHA-256 de `sources.list` y `sources.list.d/*` y almacenar en `/etc/ml-defender/apt-sources.sha256` (fichero firmado con Ed25519 del proyecto).
2. En el boot check (antes de arrancar el pipeline): verificar SHA-256 actual vs firmado. Si no coincide → fail-closed o fail-warn según configuración.
3. AppArmor deny de escritura en `/etc/apt/` para todos los procesos del pipeline.
4. Falco alerta si cualquier proceso escribe en `/etc/apt/`.

**Test de cierre:** Modificar manualmente `sources.list` en VM hardened → pipeline no arranca + log "apt sources modified". Restaurar → pipeline arranca normalmente.

---

### DEBT-DEBIAN13-UPGRADE-001
**Severidad:** 🟡 Media | **Bloqueante:** No | **Target:** post-FEDER
**Origen:** DAY 132 — Consejo 8/8 unánime
**Descripción:** Documentar y validar el upgrade path de Debian 12 (bookworm) a Debian 13 (trixie) para bare-metal hospitalario.

El Vagrantfile usa `debian/bookworm64` por reproducibilidad y estabilidad de box. El target de producción bare-metal es Debian 13 cuando esté disponible.

**Fix:** Script `scripts/upgrade-to-trixie.sh` + sección en `docs/HARDWARE-REQUIREMENTS.md` + validación de que los binarios compilados en Debian 12 son compatibles con Debian 13 (glibc 2.36 vs 2.37+).

**Test de cierre:** `make test-all` verde en VM Debian 13 con binarios compilados en Debian 12.

---

### DEBT-PAPER-FUZZING-METRICS-001
**Severidad:** 🟡 Media | **Bloqueante:** Sí (pre-arXiv v17) | **Target:** DAY 133
**Origen:** DAY 132 — Consejo 8/8 (6/8 señalan falta de números) + decisión founder
**Descripción:** El §6.8 del paper (Fuzzing as the Third Testing Layer) necesita:
1. Números reales de DEBT-FUZZING-LIBFUZZER-001 (ya cerrada): 2.4M runs, 0 crashes, corpus 67 ficheros, 30 segundos. Añadir tabla al paper.
2. Corrección de la frase "Fuzzing misses nothing within CPU time" — técnicamente incorrecta según Claude y 5 modelos más. Pedir al Consejo DAY 133 que explique la frase y proponga reformulación para aprender todos juntos.
3. Métricas de VM hardened para §6.12 (BSR axiom): `dpkg -l | wc -l` en dev vs hardened, tamaño imagen, CVEs. Obtener en DAY 133 al arrancar la hardened VM.

**Fix:** Actualizar `docs/latex/main.tex` con tabla de fuzzing + corrección de frase + tabla BSR metrics. Commit a `main` antes de subir a arXiv.

**Test de cierre:** §6.8 y §6.12 tienen datos cuantitativos reales con procedimiento reproducible documentado. Frase "misses nothing" eliminada o reformulada con precisión científica.

'''

# Insertar las nuevas deudas antes de la sección "## 🔵 BACKLOG — Deuda de seguridad crítica"
content = content.replace(
    '\n## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)\n',
    NEW_DEBTS + '\n## 🔵 BACKLOG — Deuda de seguridad crítica (pre-producción)\n'
)

# 4. Añadir nuevas decisiones a la tabla de decisiones
NEW_DECISIONS = '''| **AppArmor como primera línea BSR** | AppArmor bloquea compiladores en producción. check-prod-no-compiler es auditoría, no defensa. | DAY 132 — decisión founder |
| **Falco para paths exóticos** | Falco vigila runtime ejecución/escritura en paths no autorizados. AppArmor previene; Falco detecta. | DAY 132 — decisión founder |
| **FS de producción mínimo** | Imagen de producción: acceso solo a paths estrictamente necesarios. /tmp, /var/tmp, /opt con noexec+nosuid. | DAY 132 — decisión founder |
| **apt sources integrity check** | SHA-256 de sources.list firmado en imagen. Si cambia: fail-closed (default) o fail-warn configurable. Pipeline no arranca si comprometido. | DAY 132 — decisión founder |
| **Makefile raíz con prefijo prod-** | Targets de producción en Makefile raíz con prefijo prod-, guard _check-dev-env. No Makefile.production separado. | Consejo 8/8 · DAY 132 |
| **debian/bookworm64 en Vagrantfile** | Reproducibilidad sobre novedad. Trixie cuando tenga box estable. Upgrade path documentado en HARDWARE-REQUIREMENTS.md. | Consejo 8/8 · DAY 132 |
| **Dos capas BSR check** | dpkg + command -v. No find /. La defensa real es AppArmor+Falco+FS minimizado. | Consejo 8/8 · DAY 132 |
| **Método científico puro para paper** | Medir, publicar lo que salga con el procedimiento. Sin adornar. Números de fuzzing ya existen (2.4M runs, 0 crashes). | DAY 132 — decisión founder |
| **RED→GREEN se mantiene** | Formalización de Kimi rechazada por claridad. RED→GREEN es comprensible por un CISO de hospital. | DAY 132 — decisión founder |
'''

content = content.replace(
    '| **Plugin unload vía mensaje firmado** | Emergency Patch Protocol:',
    NEW_DECISIONS + '| **Plugin unload vía mensaje firmado** | Emergency Patch Protocol:'
)

# 5. Actualizar progress bars — añadir entradas DAY 132
PROGRESS_132 = '''DEBT-PROD-COMPAT-BASELINE-001:          ████████████████████ 100% ✅  DAY 132
vagrant/hardened-x86/Vagrantfile:       ████████░░░░░░░░░░░░  40% 🟡  DAY 132 (skeleton)
Paper Draft v17 (§6.5/6.8/6.10/6.12):  ████████████████░░░░  80% 🟡  DAY 132 (pre-arXiv)
DEBT-PROD-APPARMOR-COMPILER-BLOCK-001:  ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr030-variant-a
DEBT-PROD-FALCO-EXOTIC-PATHS-001:       ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr030-variant-a
DEBT-PROD-FS-MINIMIZATION-001:          ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr030-variant-a
DEBT-PROD-APT-SOURCES-INTEGRITY-001:    ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr030-variant-a
DEBT-DEBIAN13-UPGRADE-001:              ░░░░░░░░░░░░░░░░░░░░   0% ⏳ post-FEDER
DEBT-PAPER-FUZZING-METRICS-001:         ░░░░░░░░░░░░░░░░░░░░   0% ⏳ DAY 133
'''

content = content.replace(
    'DEBT-SAFE-PATH-RESOLVE-MODEL-001:       ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr038-acrl',
    PROGRESS_132 + 'DEBT-SAFE-PATH-RESOLVE-MODEL-001:       ░░░░░░░░░░░░░░░░░░░░   0% ⏳ feature/adr038-acrl'
)

# 6. Actualizar footer
content = content.replace(
    '*DAY 130 — 25 Abril 2026 · Tag activo: v0.5.2-hardened · commit aab08daa · main limpio*',
    '*DAY 132 — 26 Abril 2026 · Tag activo: v0.5.2-hardened · main @ 18d8e101 · feature/adr030-variant-a @ 9b3438fb*'
)

# 7. Añadir notas del Consejo DAY 132
NOTAS_132 = '''
## 📝 Notas del Consejo de Sabios — DAY 132 (8/8)

> "DAY 132 es una sesión de consolidación documental con decisiones arquitectónicas de alto impacto.
> La superficie de ataque de la imagen de producción se define hoy como un principio estructural,
> no como una lista de checks.
>
> Decisiones vinculantes:
> - D1 (8/8): Makefile raíz con prefijo prod- y guard _check-dev-env. No Makefile.production separado.
> - D2 (8/8): debian/bookworm64 en Vagrantfile. Trixie documentada como upgrade path.
> - D3 (8/8): Dos capas BSR check (dpkg + command -v). AppArmor+Falco es la defensa real.
> - D4 (8/8): Paper Draft v17 no sube a arXiv hasta tener métricas reales de VM hardened.
>
> Decisiones del founder (no votadas, vinculantes por diseño):
> - AppArmor es la primera línea de defensa, no dpkg.
> - Falco entra en el stack de producción (DEBT-PROD-FALCO-EXOTIC-PATHS-001).
> - Minimización radical del FS: solo lo imprescindible para el pipeline.
> - apt sources: SHA-256 firmado en imagen. Si cambia → fail-closed. Pipeline no arranca comprometido.
> - Ninguna herramienta que pueda corromper la imagen entra en producción. Sin excepciones.
>
> Pregunta abierta para DAY 133 (Consejo):
> Explicar la frase "Fuzzing misses nothing within CPU time" del §6.8.
> El founder no la entiende. Si el autor del paper no la entiende, se reformula o se elimina.
> Puro método científico.
>
> 'La superficie de ataque mínima no es una aspiración. Es una decisión de diseño.' — Consejo 8/8
> 'Haremos nuestro trabajo minimizando al máximo la superficie de ataque,
>  con humildad para reaccionar cuando los problemas aparezcan.' — Founder"
> — Consejo de Sabios (8/8) · DAY 132

'''

# Insertar antes de las notas del DAY 129
content = content.replace(
    '\n## 📝 Notas del Consejo de Sabios — DAY 129 (8/8)\n',
    NOTAS_132 + '\n## 📝 Notas del Consejo de Sabios — DAY 129 (8/8)\n'
)

with open('docs/BACKLOG.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("docs/BACKLOG.md actualizado correctamente")
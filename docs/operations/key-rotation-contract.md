# Recovery Contract — Rotación de Claves (OQ-6 ADR-024)

**Estado:** PHASE 3 (single-node, reinicio controlado)
**Última actualización:** DAY 117 — 14 Abril 2026
**Próxima revisión:** FASE 4 (hot-reload sin downtime)

---

## Cuándo ejecutar este procedimiento

- Seed con antigüedad > 30 días (el RAG alertará al admin)
- Compromiso de claves detectado o sospechado
- Rotación planificada por política de seguridad
- Post-incidente de seguridad

---

## Procedimiento (downtime estimado: ~30 segundos)

**Paso 1 — Parar el pipeline**
```bash
make pipeline-stop
```

**Paso 2 — Rotar todas las claves**
```bash
echo 'RESET' | sudo bash tools/provision.sh --reset
```
→ Genera nuevas keypairs Ed25519 + seed_family compartido para los 6 componentes
→ Backup automático de claves anteriores (máx. 2 por componente)
→ Nueva pubkey MLD_PLUGIN_PUBKEY_HEX registrada en logs

**Paso 3 — Re-firmar plugins**
```bash
make sign-plugins
```

**Paso 4 — Arrancar y verificar**
```bash
make pipeline-start
make test-all
```
→ Criterio de éxito: 6/6 RUNNING + test-all verde

**Paso 5 — Registrar la rotación**
Anotar en el log de operaciones (o commit de git):
- Fecha y hora
- Motivo de la rotación
- Nueva pubkey MLD_PLUGIN_PUBKEY_HEX
- Resultado de make test-all

---

## Invariantes post-rotación

- INVARIANTE-SEED-001: `make test-invariant-seed` PASSED
- 6/6 componentes con seed idéntico
- Todos los plugins firmados con la nueva clave

---

## Problemas conocidos / lecciones aprendidas

| Problema | Causa | Solución |
|---|---|---|
| HKDF MAC fail post-reset | Seeds independientes por componente | INVARIANTE-SEED-001: seed_family compartido (DAY 116) |
| Pipeline fail-closed tras reset | Plugins firmados con clave antigua | Siempre ejecutar `make sign-plugins` tras `--reset` |
| Binarios no recompilados | libplugin_loader.so cambiado | `make pipeline-build` si hay cambios en libs (DAY 114) |

---

## Roadmap

- **PHASE 3 (actual):** Rotación con reinicio controlado. ~30s downtime.
- **FASE 4 (futuro):** Hot-reload sin downtime. Ventana de solapamiento dual-key.
  Coordinación vía RAG subsystem. Ver ENT-4 en backlog.


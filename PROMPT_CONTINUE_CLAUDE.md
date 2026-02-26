Prompt de continuidad actualizado — Day 70:

---

### Prompt de Continuidad — Day 70

**Contexto del proyecto:** ML Defender (aegisIDS) — Day 70. Sistema de seguridad de red open-source, arquitectura distribuida C++20.

**Estado al inicio de Day 70 — checklist de arranque:**

**1. Verificar índices antes de arrancar rag-local:**
```bash
# ¿WAL activo?
sqlite3 /vagrant/shared/indices/metadata.db "PRAGMA journal_mode;"
# Si devuelve "delete" → activar WAL antes de continuar:
# sqlite3 /vagrant/shared/indices/metadata.db "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;"

# ¿FAISS persistido con timestamp de hoy?
ls -lh /vagrant/shared/indices/
```

**2. Verificar rag-local:**
- ¿Atiende a los dos índices? FAISS + MetadataDB (14 columnas Day 69)
- ¿Abre SQLite con WAL explícito en su código?
- ¿Respeta lista blanca antes de emitir recomendaciones de bloqueo?

**3. Verificar firewall-acl-agent lista blanca:**
```bash
find /vagrant/firewall-acl-agent -name "whitelist*" -o -name "*allowlist*" 2>/dev/null
grep -r "whitelist\|allowlist\|never_block" /vagrant/firewall-acl-agent/config/ 2>/dev/null
```

**Flujo completo a validar Day 70:**
```
rag-local recibe consulta
  → FAISS similarity search
  → enriquece con MetadataDB (clasificación, score, IPs, firewall_action)
  → verifica lista blanca firewall-acl-agent
  → emite recomendación solo si IP no está en lista blanca
```

**Estado de índices al cierre Day 69:**
- MetadataDB: 14 columnas, ~200 eventos, `/vagrant/shared/indices/metadata.db`
- FAISS: checkpoint cada 100 eventos — verificar que se disparó con los 100 del smoke test
- Shared indices path: `/vagrant/shared/indices/`

**Nota científica para el paper (no bloquea desarrollo):**
Modelo L2/L3 entrenado con dataset sintético DeepSeek confrontado con injector sintético distinto → divergencia sistemática `fast=0.9, ml~=0.2`. Tricapa DIVERGENCE funciona correctamente. Hipótesis abierta para paper: *synthetic-A trained vs synthetic-B tested*. Validación futura: tcpreplay + PCAP académico.

**Pendiente técnico documentado:**
- trace_id Day 7z: UUID v7, nace en ml-detector, propaga a firewall-acl-agent → correlación limpia MetadataDB
- Dataset DeepSeek: localizar en `/vagrant/ml-training/` para paper

**Rutas clave:**
```
/vagrant/rag-ingester/
/vagrant/rag-local/          — objetivo Day 70
/vagrant/ml-detector/
/vagrant/firewall-acl-agent/
/vagrant/logs/ml-detector/events/YYYY-MM-DD.csv
/vagrant/logs/firewall_logs/firewall_blocks.csv
/vagrant/shared/indices/metadata.db
```

**Namespaces:** `rag_ingester::` (rag-ingester), `ml_defender::` (firewall loader), `rag::` (MetadataDB)

---


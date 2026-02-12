Day 56 Session - Cooldown Window Implementation (COMPLETE)

## ✅ Estado Final: ADR-004 IMPLEMENTADO Y DOCUMENTADO

### ✅ Objetivos Day 56 Cumplidos

1. **Cooldown Window Implementation** ✅
   - `min_rotation_interval_seconds = grace_period_seconds = 300`
   - Garantía matemática: máximo 2 claves concurrentes
   - HTTP 429 + `Retry-After` header

2. **Emergency Override** ✅
   - Parámetro `?force=true` bypassa cooldown
   - Logging `WARN: EMERGENCY rotation`
   - Sin MFA (deferred to enterprise phase)

3. **Testing Complete** ✅
   - 5/5 unit tests passing
   - 4 HTTP integration tests verified
   - Logs confirman comportamiento correcto

4. **ADR-004 Documentation** ✅
   - ~700 líneas, formato académico
   - Toda la historia del Consejo de Sabios
   - ASCII art diagrams
   - Evidencia empírica completa
   - Listo para papers futuros

---

## Verificación Funcional Day 56

### Tests Unitarios (5/5 PASS)

```
Test 1: Generate and get HMAC key... PASS
Test 2: Rotation with grace period... PASS
Test 3: Grace period expiry... PASS
Test 4: Grace period configuration... PASS
Test 5: Cooldown enforcement (ADR-004)... PASS

🎉 ALL 5 TESTS PASSED!
```

**Evidencia clave Test 5**:
```
[warning] Rotation REJECTED for test_cooldown - cooldown active (4s remaining)
[warning] EMERGENCY rotation for test_cooldown (force=true) - cooldown bypassed
```

### Tests HTTP (4/4 VERIFIED)

**Test 1: Primera rotación**
```bash
curl -X POST /secrets/rotate/test-component
→ 200 OK, valid_keys_count: 1
```

**Test 2: Rotación prematura (cooldown activo)**
```bash
curl -X POST /secrets/rotate/test-component  # 2s después
→ HTTP 429 Too Many Requests
→ Retry-After: 291
→ "Rotation too soon, retry in 291s"
```

**Test 3: Emergency override**
```bash
curl -X POST /secrets/rotate/test-component?force=true
→ 200 OK, forced: true, valid_keys_count: 2
→ Server log: "EMERGENCY rotation (force=true) - cooldown bypassed"
```

**Test 4: Claves válidas**
```bash
curl /secrets/valid/test-component
→ Exactamente 2 claves (activa + grace)
→ Active: expires_at: 2262 (nunca)
→ Grace: expires_at: 2026-02-12T08:33:56Z (+5 min)
```

---

## Cambios Técnicos Day 56

### Archivos Modificados

1. **`/vagrant/etcd-server/include/etcd_server/secrets_manager.hpp`**
   - Añadido `int min_rotation_interval_seconds = 300` a Config
   - Añadido `const int min_rotation_interval_seconds_` (private)
   - Añadido `std::map<string, time_point> last_rotation_` (private)
   - Modificado `rotate_hmac_key()` signature: `bool force = false`

2. **`/vagrant/etcd-server/src/secrets_manager.cpp`**
   - Constructor: Lee `min_rotation_interval_seconds` de JSON
   - Constructor: Valida `min_rotation_interval >= grace_period` (CRITICAL)
   - `rotate_hmac_key()`: Implementa cooldown check
   - `rotate_hmac_key()`: Lanza exception si cooldown no elapsed
   - `rotate_hmac_key()`: Permite override con `force=true`
   - `rotate_hmac_key()`: Actualiza `last_rotation_[component] = now`

3. **`/vagrant/etcd-server/src/etcd_server.cpp`**
   - Endpoint `POST /secrets/rotate/{component}`: Lee parámetro `?force=true`
   - Endpoint: Catch `std::runtime_error` → HTTP 429 si cooldown violation
   - Endpoint: Extrae retry seconds de exception message
   - Endpoint: Header `Retry-After` con tiempo calculado dinámicamente
   - Response JSON: Campo `"forced": true/false`

4. **`/vagrant/etcd-server/src/main.cpp`**
   - Config hardcoded: Añadido `"min_rotation_interval_seconds": 300`

5. **`/vagrant/etcd-server/tests/test_secrets_manager_simple.cpp`**
   - Test 5 añadido: `test_cooldown_enforcement()`
   - Tests 1-4: Actualizados para incluir `min_rotation_interval_seconds` en config
   - Main: Cambio de "4 TESTS PASSED" → "5 TESTS PASSED"

### Archivos Creados

6. **`/vagrant/docs/adr/ADR-004-key-rotation-cooldown.md`**
   - Documento completo (~700 líneas)
   - Sección 9: Toda la interacción del Consejo de Sabios
   - Apéndice B: ASCII art diagrams
   - Sección 5: Evidencia empírica completa (logs reales)
   - Sección 7: Limitaciones conocidas (persistencia)
   - Sección 8: Future work (MFA, persistencia distribuida)

---

## Decisiones Técnicas Day 56

### ✅ Cooldown = Grace Period

**Decisión**: `MIN_ROTATION_INTERVAL = GRACE_PERIOD = 300s`

**Rationale**:
- Garantía matemática: máximo 2 claves concurrentes
- Mental model simple: "Una clave entra, una sale"
- NIST SP 800-57 compliant

**Proof**:
```
Si t_n - t_(n-1) >= grace_period:
  Key_(n-1) expira en: t_(n-1) + 300s
  Key_n creada en: t_n >= t_(n-1) + 300s
  
  Por tanto: Key_(n-1) expira ANTES o CUANDO Key_n se crea
  
  Claves válidas en t_n: {Key_n (active), Key_(n-1) (grace si t_n = t_(n-1) + 300 exacto)}
  
  Máximo: 2 claves ∎
```

### ✅ Emergency Override Simplificado

**Decisión**: Parámetro `?force=true` sin MFA

**Rationale**:
- MFA añade complejidad operativa innecesaria para hospitales/escuelas
- Logging `WARN` proporciona audit trail suficiente
- MFA es feature enterprise para cuando aegisIDS alcance producción a escala
- "Ojalá" lleguemos a ese punto

**Implementación Futura MFA**:
```http
POST /secrets/rotate/component?force=true
X-MFA-Token: 123456
X-MFA-User: admin@hospital.org

→ Verify TOTP/HOTP
→ Log: "EMERGENCY rotation by admin@hospital.org (MFA verified)"
→ Alert security team
```

### ✅ Validación Estricta en Constructor

**Decisión**: `throw std::runtime_error` si config inválido

**Código**:
```cpp
if (min_rotation_interval_seconds_ < grace_period_seconds_) {
    logger_->critical("UNSAFE CONFIG: min_rotation_interval ({}) < grace_period ({})",
                      min_rotation_interval_seconds_, grace_period_seconds_);
    throw std::runtime_error("Invalid config: min_rotation_interval must be >= grace_period");
}
```

**Rationale**: Fail-fast. Mejor crash al arrancar que comportamiento inseguro en producción.

---

## Consejo de Sabios - Votación ADR-004

| Miembro | Voto | Contribución Clave |
|---------|------|---------------------|
| **DeepSeek (Qwen)** | ✅ APROBAR | Proof matemático de max 2 claves, NIST compliance |
| **Claude** | ✅ APROBAR | HTTP 429 semántica, force=true sin MFA, logging |
| **Gemini** | ✅ APROBAR | Thread safety, forensic logging, validación |
| **ChatGPT** | ✅ APROBAR | Simplicity emphasis, ASCII diagrams |
| **Alonso** | ✅ APROBAR | "Piano piano, muy sencillo, máximo 2 claves" |

**Resultado**: 5/5 aprobación unánime

**Dissent**: Ninguno

**Abstenciones**: Gemini (inicial), resuelta tras clarificación MFA

---

## Limitaciones Conocidas

### 1. Sin Persistencia de `last_rotation_`

**Problema**:
```bash
t=0s:   Rotar clave → last_rotation_[component] = t0
t=60s:  Reiniciar etcd-server
t=61s:  last_rotation_ map vacío (in-memory)
t=62s:  Rotar clave → Permitido (cooldown bypassed)
```

**Impacto**: Atacante con capacidad de reiniciar servidor puede bypass cooldown

**Mitigación Corto Plazo**:
- Log all restarts con `WARN`
- Monitor restart frequency (>1/hour sospechoso)
- aegisIDS en entornos controlados (hospitales/escuelas)

**Solución Largo Plazo** (Day 57+):

| Opción | Pros | Cons | Recomendado Para |
|--------|------|------|------------------|
| File local `/var/lib/etcd-server/rotation.state` | Simple, sin deps | Single point of failure | Small deployments |
| etcd cluster (real etcd) | Distributed, resilient | Operational complexity | Medium/large |
| Consul | Service discovery + KV | Extra infrastructure | Cloud |
| SQLite | Queryable, transactional | File locking | Dev/testing |

**Schema Propuesto** (JSON):
```json
{
  "version": 1,
  "last_updated": "2026-02-12T08:28:56Z",
  "rotations": {
    "rag-ingester": {
      "last_rotation_time": "2026-02-12T08:00:00Z",
      "rotation_count": 42
    },
    "ml-detector": {
      "last_rotation_time": "2026-02-12T07:30:00Z",
      "rotation_count": 38
    }
  }
}
```

### 2. Sin Coordinación Distribuida (Multi-Server)

**Problema**: En HA setup con múltiples `etcd-server` replicas:
- Server A puede permitir rotación
- Server B puede rechazarla
- Race condition en `last_rotation_` tracking

**Impacto**: Comportamiento inconsistente si load-balanced

**Mitigación**: aegisIDS actualmente single-server (hospitales/escuelas)

**Solución Futuro**: Distributed lock (etcd locks, Consul sessions, Redis SETNX)

### 3. Audit Trail No Centralizado

**Problema**: En deployment distribuido, logs de emergency rotations no visibles centralmente

**Impacto**: Incomplete audit trail para forensics

**Solución**: ELK stack, Splunk, Datadog + alertas en >N emergency rotations/24h

---

## Backlog Day 57+

### 🎯 Alta Prioridad (Day 57-58)

#### 1. Verificar etcd-client Actualizado para HMAC

**Estado Actual**:
- etcd-client tiene soporte ChaCha20 seeds ✅ (Day 53)
- etcd-client soporte HMAC secrets? ⚠️ **VERIFICAR**

**Tareas**:
```bash
# Verificar que etcd-client puede:
1. GET /secrets/{component} - Obtener clave HMAC activa
2. GET /secrets/valid/{component} - Obtener claves válidas (activa + grace)
3. Usar claves para generar HMAC-SHA256 en logs
4. Validar HMAC con múltiples claves (activa + grace)
```

**Tests Requeridos**:
- `test_etcd_client_hmac.cpp`: Unit tests para HMAC operations
- Integration test: etcd-client ↔ etcd-server HMAC roundtrip

**Archivos a Modificar**:
- `/vagrant/etcd-client/include/etcd_client/etcd_client.hpp`
- `/vagrant/etcd-client/src/etcd_client.cpp`
- `/vagrant/etcd-client/tests/test_etcd_client.cpp`

---

#### 2. Configurar etcd-server.json para Todos los Componentes

**Estado Actual**:
```json
{
  "secrets": {
    "keys": {
      "rag_log_hmac": {
        "path": "/secrets/rag/log_hmac_key",
        "auto_generate": true
      },
      "ml_detector_hmac": {
        "path": "/secrets/ml-detector/log_hmac_key",
        "auto_generate": false  ← NO configurado para auto-gen
      },
      "firewall_hmac": {
        "path": "/secrets/firewall/log_hmac_key",
        "auto_generate": false  ← NO configurado para auto-gen
      }
    }
  }
}
```

**Problema**: Solo `rag_log_hmac` se auto-genera. Otros componentes no tienen claves al arrancar.

**Solución**: Actualizar config para todos los componentes:

```json
{
  "secrets": {
    "keys": {
      "rag_log_hmac": {
        "path": "/secrets/rag/log_hmac_key",
        "algorithm": "hmac-sha256",
        "purpose": "HMAC validation for RAG log files",
        "auto_generate": true,
        "rotation_enabled": true
      },
      "ml_detector_hmac": {
        "path": "/secrets/ml-detector/log_hmac_key",
        "algorithm": "hmac-sha256",
        "purpose": "HMAC validation for ML detector event logs",
        "auto_generate": true,  ← CAMBIAR a true
        "rotation_enabled": true
      },
      "firewall_hmac": {
        "path": "/secrets/firewall/log_hmac_key",
        "algorithm": "hmac-sha256",
        "purpose": "HMAC validation for firewall ACL agent logs",
        "auto_generate": true,  ← CAMBIAR a true
        "rotation_enabled": true
      }
    }
  }
}
```

**Implementación en SecretsManager**:
```cpp
// En constructor, si auto_generate_on_startup = true:
for (auto& [key_name, key_config] : config["secrets"]["keys"]) {
    if (key_config["auto_generate"].get<bool>()) {
        std::string component = extract_component_from_path(key_config["path"]);
        generate_hmac_key(component);
        logger_->info("Auto-generated HMAC key for: {}", component);
    }
}
```

**Archivos a Modificar**:
- `/vagrant/etcd-server/config/etcd_server.json`
- `/vagrant/etcd-server/src/secrets_manager.cpp` (constructor)

---

#### 3. Integración firewall-acl-agent → rag-ingester (Primera Dupla HMAC)

**Objetivo**: Primera integración end-to-end con HMAC operativo

**Flujo Completo**:

```
┌──────────────────────┐
│ firewall-acl-agent   │
├──────────────────────┤
│ 1. Genera logs CSV   │──┐
│ 2. Cifra con ChaCha20│  │
│ 3. Genera HMAC-SHA256│  │ CSV cifrado + HMAC
└──────────────────────┘  │
                          │
                          v
                   ┌──────────────┐
                   │ Filesystem   │
                   │ /logs/       │
                   └──────┬───────┘
                          │
                          v
┌──────────────────────┐  │
│ rag-ingester         │  │
├──────────────────────┤  │
│ 1. Lee CSV cifrado   │<─┘
│ 2. GET /secrets/valid/firewall  → [key_active, key_grace]
│ 3. Intenta HMAC validation:
│    - Try key_active → ✓ Success → Ingest
│    - Try key_grace  → ✓ Success → Ingest (old key)
│    - Both fail      → ❌ LOG POISONING DETECTED
│ 4. Descifra ChaCha20
│ 5. Parse CSV
│ 6. Ingesta a RAG
└──────────────────────┘
```

**Componentes a Modificar**:

**firewall-acl-agent**:
1. `/vagrant/firewall-acl-agent/src/firewall_logger.cpp`
   - Añadir HMAC generation después de cifrado
   - GET /secrets/firewall → obtener clave activa
   - Calcular HMAC-SHA256 del CSV cifrado
   - Guardar HMAC en archivo `.hmac` (mismo nombre que CSV)

**rag-ingester**:
2. `/vagrant/rag-ingester/src/rag_logger.cpp` (o equivalente)
   - Antes de descifrar, GET /secrets/valid/firewall
   - Leer archivo `.hmac`
   - Validar con cada clave en `valid_keys`
   - Si todas fallan → `logger_->critical("LOG POISONING DETECTED")`
   - Si alguna pasa → descifrar y procesar

**Tests End-to-End**:
```bash
# Test 1: Rotación sin downtime
1. firewall-acl-agent genera log con key_A
2. Rotar clave: POST /secrets/rotate/firewall
3. firewall-acl-agent genera log con key_B
4. rag-ingester procesa AMBOS logs sin errores
   - Log viejo valida con key_A (grace)
   - Log nuevo valida con key_B (active)

# Test 2: Log poisoning detection
1. Editar manualmente archivo CSV (modificar 1 byte)
2. rag-ingester intenta procesar
3. HMAC validation falla con TODAS las claves
4. Log: "LOG POISONING DETECTED - file rejected"
```

**Archivos Nuevos**:
- `/vagrant/firewall-acl-agent/tests/test_hmac_generation.cpp`
- `/vagrant/rag-ingester/tests/test_hmac_validation.cpp`

**Criterios de Éxito**:
- ✅ firewall-acl-agent genera CSV + HMAC
- ✅ rag-ingester valida HMAC antes de ingerir
- ✅ Rotación de clave NO rompe pipeline (grace period funciona)
- ✅ Log poisoning detectado y rechazado

---

#### 4. firewall-acl-agent: Cifrado CSV + HMAC

**Objetivo**: Asegurar integridad de logs CSV desde firewall

**Implementación**:

**Estructura de Archivos**:
```
/var/log/aegisids/firewall/
├── firewall_2026-02-12_08-00-00.csv.enc  ← CSV cifrado (ChaCha20)
└── firewall_2026-02-12_08-00-00.csv.hmac ← HMAC-SHA256 (hex)
```

**Código en firewall-acl-agent**:
```cpp
// firewall_logger.cpp
void FirewallLogger::write_log_with_hmac(const std::string& csv_data) {
    // 1. Cifrar CSV
    auto encrypted = crypto_manager_->encrypt(csv_data);
    
    // 2. Obtener clave HMAC activa
    auto hmac_key = etcd_client_->get_secret("/secrets/firewall");
    
    // 3. Calcular HMAC del CSV cifrado
    auto hmac = calculate_hmac_sha256(encrypted, hmac_key);
    
    // 4. Guardar archivos
    std::string timestamp = format_timestamp(std::chrono::system_clock::now());
    std::string csv_path = "/var/log/aegisids/firewall/firewall_" + timestamp + ".csv.enc";
    std::string hmac_path = csv_path + ".hmac";
    
    write_file(csv_path, encrypted);
    write_file(hmac_path, bytes_to_hex(hmac));
    
    logger_->info("Log written with HMAC: {}", csv_path);
}
```

**Función HMAC**:
```cpp
std::vector<uint8_t> calculate_hmac_sha256(
    const std::string& data, 
    const std::vector<uint8_t>& key
) {
    std::vector<uint8_t> hmac(SHA256_DIGEST_LENGTH);
    
    HMAC(EVP_sha256(),
         key.data(), key.size(),
         reinterpret_cast<const uint8_t*>(data.data()), data.size(),
         hmac.data(), nullptr);
    
    return hmac;
}
```

**Archivos a Crear/Modificar**:
- `/vagrant/firewall-acl-agent/include/firewall/hmac_utils.hpp` (new)
- `/vagrant/firewall-acl-agent/src/hmac_utils.cpp` (new)
- `/vagrant/firewall-acl-agent/src/firewall_logger.cpp` (modify)

---

### 🔧 Media Prioridad (Day 59-60)

#### 5. Leer etcd_server.json desde Archivo (No Hardcoded)

**Estado Actual**: Config hardcoded en `main.cpp`:
```cpp
nlohmann::json config = {
    {"secrets", {
        {"grace_period_seconds", 300},
        // ...
    }}
};
```

**Objetivo**: Leer de `/vagrant/etcd-server/config/etcd_server.json`

**Implementación**:
```cpp
// main.cpp
nlohmann::json load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }
    
    nlohmann::json config;
    file >> config;
    return config;
}

int main(int argc, char** argv) {
    std::string config_path = argc > 1 ? argv[1] : "/vagrant/etcd-server/config/etcd_server.json";
    
    try {
        auto config = load_config(config_path);
        g_secrets_manager = std::make_shared<etcd_server::SecretsManager>(config);
        // ...
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << std::endl;
        return 1;
    }
}
```

**Test**:
```bash
./etcd-server                                           # Default path
./etcd-server /custom/path/etcd_server.json            # Custom path
./etcd-server /nonexistent.json                         # Should fail gracefully
```

---

#### 6. Persistencia de `last_rotation_` (File-Based)

**Implementación Mínima** (Day 60):

```cpp
// secrets_manager.cpp
void SecretsManager::persist_rotation_state() {
    nlohmann::json state = {
        {"version", 1},
        {"last_updated", format_time(std::chrono::system_clock::now())},
        {"rotations", {}}
    };
    
    for (auto& [component, timestamp] : last_rotation_) {
        state["rotations"][component] = {
            {"last_rotation_time", format_time(timestamp)}
        };
    }
    
    std::ofstream file("/var/lib/etcd-server/rotation.state");
    file << state.dump(2);
}

void SecretsManager::load_rotation_state() {
    std::ifstream file("/var/lib/etcd-server/rotation.state");
    if (!file.is_open()) {
        logger_->info("No rotation state file found (first run)");
        return;
    }
    
    nlohmann::json state;
    file >> state;
    
    for (auto& [component, data] : state["rotations"].items()) {
        auto timestamp = parse_time(data["last_rotation_time"]);
        last_rotation_[component] = timestamp;
        logger_->info("Restored rotation state for {}: {}", 
                      component, format_time(timestamp));
    }
}
```

**Llamar en**:
- Constructor: `load_rotation_state()`
- `rotate_hmac_key()`: `persist_rotation_state()` después de rotación exitosa

---

### 📚 Baja Prioridad (Day 61+)

7. **Tests Completos SecretsManager** - Reescribir `test_secrets_manager.cpp` completo
8. **Métricas de Grace Period** - Contador rotaciones, claves activas vs grace
9. **OpenAPI Spec** - Documentar API /secrets/* formalmente
10. **Auto-rotation Scheduler** - Rotar automáticamente cada `rotation_interval_hours`
11. **MFA Integration** - Cuando aegisIDS alcance producción enterprise

---

## Criterios de Éxito Day 57

✅ etcd-client soporta HMAC (GET /secrets, validación multi-clave)
✅ etcd-server.json configura auto-generate para todos los componentes
✅ firewall-acl-agent genera CSV + HMAC
✅ rag-ingester valida HMAC antes de ingerir
✅ Test end-to-end: rotación durante pipeline activo (zero downtime)
✅ Log poisoning detectado y rechazado

## Filosofía Via Appia - Day 56 Reflexión

**✅ Lo que Hicimos Bien:**
- Piano piano: Cooldown simple, max 2 claves, funcionó a la primera
- Documentación completa: ADR-004 listo para academia
- Testing dual: Unit + HTTP integration
- Consenso del Consejo: 5/5 aprobación unánime
- Evidencia empírica: Logs reales, no assumptions

**📝 Aprendizajes:**
- Colaboración humano-IA produce arquitecturas robustas
- Documentar MIENTRAS desarrollamos (no después) es más eficiente
- ASCII art ayuda a entender decisiones complejas
- Limitaciones conocidas son features, no bugs (transparencia)

**🎯 Principios Mantenidos:**
- Cada fase 100% testeada
- Código compila en cada paso
- Decisiones basadas en evidencia
- Transparencia absoluta (open source, auditable)

---

## Próxima Sesión Day 57

**Objetivo**: Integración HMAC end-to-end firewall-acl-agent → rag-ingester

**Comenzar con**:
1. Verificar etcd-client tiene métodos HMAC
2. Actualizar etcd_server.json (auto-generate para todos)
3. Implementar HMAC generation en firewall-acl-agent

**Piano piano - un componente a la vez.**

**Transcript Day 56**: /mnt/transcripts/2026-02-12-[timestamp]-day56-cooldown-complete.txt

---

Co-authored-by: Claude (Anthropic)
Co-authored-by: Alonso 
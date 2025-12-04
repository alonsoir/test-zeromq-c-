# **PROMPT DE CONTINUIDAD: FASE 2 - INTEGRACIÓN ETCD-SERVER Y CIFRADO UNIFICADO**

## **🎯 Contexto Actual (Día 8 Completado)**
```
✅ DUAL-NIC VALIDADO: Kernel-userspace metadata pipeline operacional
✅ libbpf 1.4.6: Bug crítico resuelto, iface_configs map funciona
✅ 43+ paquetes con metadata dual-NIC, latencia 59.63μs avg
✅ Pipeline ML Defender: eBPF → Ring Buffer → Protobuf → 4 modelos ML
```

## **📋 PRÓXIMOS OBJETIVOS (Días 9-12)**

### **1. 🎯 OBJETIVO PRINCIPAL: Centralización de Configuración y Cifrado**
**Meta:** Convertir etcd-server en el hub central de gestión para todos los componentes del ML Defender.

### **2. 🏗️ ARQUITECTURA PROPUESTA**
```
┌─────────────────────────────────────────────────────────────┐
│                   etcd-server (Central Hub)                  │
│  ├─ /config/sniffer/json       (configuración del sniffer)   │
│  ├─ /config/detector/json      (configuración del detector)  │
│  ├─ /config/firewall/json      (configuración del firewall)  │
│  ├─ /keys/encryption/seed      (semilla de cifrado común)    │
│  ├─ /keys/encryption/rotation  (rotación programada)         │
│  └─ /status/components/*       (estado de componentes)       │
└─────────────────────────────────────────────────────────────┘
         ▲            ▲                ▲
         │            │                │
┌────────┴────┐ ┌────┴────────┐ ┌─────┴─────────┐
│   Sniffer   │ │   Detector   │ │    Firewall   │
│  (etcd-client) │  (etcd-client) │   (etcd-client) │
└─────────────┘ └──────────────┘ └───────────────┘
```

### **3. 🔧 IMPLEMENTACIÓN PASO A PASO**

#### **FASE 2.1: Análisis del Cliente etcd Existente en RAG**
```bash
# Examinar la implementación actual en RAG
cd /vagrant/rag
grep -r "etcd" --include="*.cpp" --include="*.hpp"
cat src/etcd_client.cpp  # Si existe
```

#### **FASE 2.2: Crear Biblioteca Compartida de etcd-client**
```
/vagrant/common/etcd-client/
├── CMakeLists.txt
├── include/
│   ├── etcd_client.hpp
│   └── config_manager.hpp
├── src/
│   ├── etcd_client.cpp
│   └── config_manager.cpp
└── examples/
    ├── basic_usage.cpp
    └── config_watcher.cpp
```

**Características clave del cliente compartido:**
```cpp
class UnifiedEtcdClient {
public:
    // 1. Conexión automática con reconexión
    bool connect(const std::string& endpoints = "127.0.0.1:2379");
    
    // 2. Gestión de configuración JSON
    bool put_config(const std::string& component, const nlohmann::json& config);
    nlohmann::json get_config(const std::string& component);
    
    // 3. Gestión de claves de cifrado
    std::string get_encryption_seed();
    bool update_encryption_seed(const std::string& new_seed);
    
    // 4. Watch/notificaciones de cambios
    void watch_config(const std::string& component, 
                      std::function<void(nlohmann::json)> callback);
    
    // 5. Health checks y métricas
    bool is_healthy();
    std::map<std::string, std::string> get_metrics();
};
```

#### **FASE 2.3: Integración en Cada Componente**

**A. Sniffer Integration:**
```cpp
// sniffer/src/etcd_integration.cpp
class SnifferEtcdIntegration {
private:
    UnifiedEtcdClient etcd_client_;
    std::string encryption_seed_;
    
public:
    void init() {
        // 1. Conectar a etcd
        etcd_client_.connect();
        
        // 2. Subir configuración actual
        nlohmann::json config = load_current_config();
        etcd_client_.put_config("sniffer", config);
        
        // 3. Obtener semilla de cifrado
        encryption_seed_ = etcd_client_.get_encryption_seed();
        
        // 4. Configurar watcher para cambios
        etcd_client_.watch_config("sniffer", [this](auto new_config) {
            this->on_config_updated(new_config);
        });
    }
    
    void on_config_updated(const nlohmann::json& new_config) {
        // Aplicar nueva configuración en caliente
        apply_configuration(new_config);
        LOG_INFO("[ETCD] Configuración actualizada en tiempo real");
    }
};
```

**B. Detector Integration:**
```cpp
// ml-detector/src/etcd_integration.cpp
class DetectorEtcdIntegration {
public:
    void init() {
        // Obtener thresholds desde etcd
        auto config = etcd_client_.get_config("detector");
        update_model_thresholds(config["thresholds"]);
        
        // Sincronizar estado del modelo
        publish_model_status();
    }
    
    void publish_model_status() {
        nlohmann::json status = {
            {"model_version", current_model_version_},
            {"inference_time", avg_inference_time_},
            {"accuracy", current_accuracy_}
        };
        etcd_client_.put_key("/status/detector/model", status.dump());
    }
};
```

**C. Firewall Integration:**
```cpp
// firewall-acl-agent/src/etcd_integration.cpp
class FirewallEtcdIntegration {
public:
    void init() {
        // Sincronizar reglas de firewall
        sync_firewall_rules();
        
        // Publicar estadísticas de bloqueo
        start_metrics_publisher();
    }
    
    void sync_firewall_rules() {
        auto rules = etcd_client_.get_config("firewall/rules");
        apply_iptables_rules(rules);
    }
};
```

#### **FASE 2.4: Sistema de Cifrado Unificado**

**Estructura de claves en etcd:**
```json
{
  "/keys/encryption/current": {
    "seed": "a1b2c3d4e5f67890123456789abcdef0",
    "algorithm": "chacha20-poly1305",
    "created_at": "2025-12-04T10:30:00Z",
    "expires_at": "2025-12-11T10:30:00Z"
  },
  "/keys/encryption/previous": [
    {
      "seed": "old_seed_1",
      "expired_at": "2025-12-03T10:30:00Z"
    }
  ],
  "/keys/encryption/rotation_schedule": {
    "interval_hours": 168,
    "next_rotation": "2025-12-11T10:30:00Z"
  }
}
```

**Implementación del cifrado:**
```cpp
class UnifiedEncryption {
public:
    static std::vector<uint8_t> encrypt(const std::string& plaintext) {
        auto seed = etcd_client_.get_encryption_seed();
        auto key = derive_key(seed, "ml-defender-encryption");
        return chacha20_poly1305_encrypt(plaintext, key);
    }
    
    static std::string decrypt(const std::vector<uint8_t>& ciphertext) {
        auto seed = etcd_client_.get_encryption_seed();
        auto key = derive_key(seed, "ml-defender-encryption");
        return chacha20_poly1305_decrypt(ciphertext, key);
    }
};
```

#### **FASE 2.5: Makefile y Sistema de Build Unificado**

**Actualizar /vagrant/Makefile principal:**
```makefile
# ============================================
# ETCD-CLIENT COMMON LIBRARY
# ============================================
ETCD_CLIENT_DIR = $(COMMON_DIR)/etcd-client
ETCD_CLIENT_INCLUDE = $(ETCD_CLIENT_DIR)/include
ETCD_CLIENT_SRC = $(wildcard $(ETCD_CLIENT_DIR)/src/*.cpp)
ETCD_CLIENT_OBJ = $(ETCD_CLIENT_SRC:.cpp=.o)
ETCD_CLIENT_LIB = $(LIB_DIR)/libetcdclient.a

$(ETCD_CLIENT_LIB): $(ETCD_CLIENT_OBJ)
	@echo "[ETCD] Creando librería compartida..."
	@mkdir -p $(LIB_DIR)
	@ar rcs $@ $^

# ============================================
# COMPONENTES CON ETCD INTEGRATION
# ============================================
SNIFFER_ETCD_SRC = $(SNIFFER_DIR)/src/etcd_integration.cpp
DETECTOR_ETCD_SRC = $(DETECTOR_DIR)/src/etcd_integration.cpp
FIREWALL_ETCD_SRC = $(FIREWALL_DIR)/src/etcd_integration.cpp

# Reglas para construir con etcd-client
build-with-etcd: $(ETCD_CLIENT_LIB) build-sniffer-etcd build-detector-etcd build-firewall-etcd

build-sniffer-etcd: $(ETCD_CLIENT_LIB)
	@echo "[BUILD] Compilando sniffer con etcd-client..."
	cd $(SNIFFER_DIR) && make ETCD_ENABLED=1

# ============================================
# DEPLOYMENT Y CONFIGURACIÓN
# ============================================
deploy-etcd-config:
	@echo "[ETCD] Desplegando configuraciones a etcd-server..."
	@python3 scripts/deploy_configs_to_etcd.py
```

### **4. 🧪 PLAN DE PRUEBAS Y VALIDACIÓN**

#### **Test 1: Conectividad Básica**
```bash
# Verificar que todos los componentes pueden conectar a etcd
cd /vagrant
make test-etcd-connectivity

# Salida esperada:
# [OK] etcd-server: listening on 127.0.0.1:2379
# [OK] sniffer: connected to etcd, version: 3.5.0
# [OK] detector: connected to etcd, config retrieved
# [OK] firewall: connected to etcd, encryption seed obtained
```

#### **Test 2: Sincronización de Configuración**
```bash
# Prueba de actualización en caliente
cd /vagrant/scripts
python3 test_hot_reload.py

# 1. Modificar configuración en etcd
# 2. Verificar que sniffer aplica cambios sin reiniciar
# 3. Validar que detector actualiza thresholds
# 4. Confirmar que firewall actualiza reglas
```

#### **Test 3: Cifrado End-to-End**
```bash
# Validar que el cifrado funciona entre componentes
cd /vagrant
make test-encryption-pipeline

# Proceso:
# 1. Sniffer cifra datos con semilla de etcd
# 2. Datos viajan por ZMQ cifrados
# 3. Detector descifra con misma semilla
# 4. Firewall aplica reglas sobre datos descifrados
```

### **5. 📊 MÉTRICAS Y MONITOREO (Actualizar Script)**

**Actualizar /vagrant/scripts/monitor_lab.sh:**
```bash
# Nueva sección para etcd
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}🗄️  ETCD-Server Status & Metrics${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verificar claves almacenadas
etcd_keys=$(etcdctl get --prefix /config 2>/dev/null | wc -l)
echo -e "Config keys stored: ${GREEN}${etcd_keys}${NC}"

# Verificar conexiones de clientes
echo -e "Connected clients: ${YELLOW}$(netstat -an | grep 2379 | grep ESTABLISHED | wc -l)${NC}"

# Mostrar última rotación de clave
last_rotation=$(etcdctl get /keys/encryption/current --print-value-only 2>/dev/null | jq -r '.created_at')
echo -e "Last key rotation: ${CYAN}${last_rotation}${NC}"
```

### **6. 🚀 PLAN DE IMPLEMENTACIÓN POR DÍAS**

**Día 9 (Con Claude):**
- [ ] Recap relay con MAWI dataset en Gateway Mode
- [ ] Validar eth3 captura tráfico transit
- [ ] Benchmark performance dual-NIC

**Día 10 (Contigo):**
- [ ] Analizar etcd-client del RAG existente
- [ ] Diseñar interfaz común UnifiedEtcdClient
- [ ] Crear biblioteca compartida en /vagrant/common/

**Día 11:**
- [ ] Integrar etcd-client en Sniffer
- [ ] Implementar hot-reload de configuración
- [ ] Pruebas de conectividad y sincronización

**Día 12:**
- [ ] Integrar etcd-client en Detector y Firewall
- [ ] Implementar sistema de cifrado unificado
- [ ] Pruebas end-to-end con rotación de claves

### **7. ⚠️ CONSIDERACIONES CRÍTICAS**

1. **Backward Compatibility:** Los componentes deben funcionar sin etcd como fallback
2. **Seguridad:** Semillas de cifrado nunca en logs, rotación automática
3. **Performance:** Conexiones persistentes a etcd, no abrir/cerrar por transacción
4. **Resiliencia:** Reconexión automática si etcd se cae
5. **Observabilidad:** Métricas detalladas de cada interacción con etcd

### **8. 📁 ESTRUCTURA FINAL PROPUESTA**
```
/vagrant/
├── common/etcd-client/           # Biblioteca compartida
├── scripts/deploy_configs_to_etcd.py
├── scripts/encryption_key_rotator.py
├── tests/etcd_integration_tests/
│   ├── test_connectivity.cpp
│   ├── test_hot_reload.cpp
│   └── test_encryption.cpp
└── docs/etcd-integration-guide.md
```

---

**¿Listo para comenzar?** Cuando termines el recap relay con Claude, podemos:

1. Examinar el etcd-client existente en RAG
2. Diseñar la interfaz común
3. Crear la biblioteca compartida
4. Integrar progresivamente en cada componente

**Pregunta clave:** ¿Prefieres comenzar por el componente más simple (firewall) o por el más complejo (sniffer) para la integración?
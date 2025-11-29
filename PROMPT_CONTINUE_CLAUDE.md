# 🔄 Prompt de Continuidad - Day 7: Watcher System

## 📋 Contexto Actual (Day 6.5 Completado)

### ✅ Estado del Proyecto ML Defender (aegisIDS)
**Phase 1 Progress:** 6.5/12 días (54% completo)

**Componentes Operacionales:**
1. ✅ **Sniffer (eBPF/XDP)** - Captura paquetes, extrae 40+ features
2. ✅ **ML Detector (Tricapa)** - 4 detectores C++20 embebidos (<1.06μs)
3. ✅ **Firewall ACL Agent** - Bloqueo autónomo vía IPSet/IPTables
4. ✅ **ETCD-Server** - Hub centralizado con validación
5. ✅ **RAG Security System** - LLAMA real (TinyLlama-1.1B)
6. ✅ **Async Logger** - JSON + Protobuf dual-format (Day 6.5)

**Pipeline End-to-End:** FUNCIONAL (8,871+ eventos procesados, 0 errores)

### 🎯 Logro Day 6.5: Async Logger
- **Implementado:** Logger asíncrono production-ready
- **Formato:** JSON (metadata) + Protobuf (payload completo)
- **Performance:** <10μs per log, 1K-5K eventos/seg
- **Tests:** 5/6 pasando (83% success rate)
- **Integración:** Completa en `zmq_subscriber.cpp`
- **Bloqueador identificado:** Modelos demasiado buenos (clasifican todo como BENIGN)
- **Solución:** Phase 2 - PCAP replay con tráfico real de malware

**Decisión Arquitectónica (Via Appia):**
- ❌ NO usar eventos fake
- ❌ NO bajar thresholds artificialmente
- ✅ Esperar validación con PCAPs reales (Phase 2)
- ✅ Logger listo para producción

### 📂 Archivos Clave Creados/Modificados
```
firewall-acl-agent/
├── include/firewall/logger.hpp          (220 líneas - nuevo)
├── src/utils/logger.cpp                 (400 líneas - nuevo)
├── src/api/zmq_subscriber.cpp           (+80 líneas - modificado)
├── tests/unit/test_logger.cpp           (320 líneas - nuevo)
├── CMakeLists.txt                       (+10 líneas - modificado)
```

**Total:** ~1,000 líneas de C++20 production-ready

---

## 🎯 Próxima Prioridad: Day 7 - Watcher System

### **Objetivo**
Implementar sistema de configuración dinámica (hot-reload) para TODOS los componentes, usando etcd-server como hub central.

### **Scope del Watcher**
**Componentes que DEBEN tener Watcher:**
1. ✅ **Sniffer** - Recargar perfiles, interfaces, filtros BPF
2. ✅ **ML Detector** - Hot-reload de thresholds sin reiniciar
3. ✅ **Firewall** - Actualizar ipsets, timeouts, configuración
4. ✅ **RAG** - Ya tiene integración con etcd (extender)

### **Requisitos Técnicos**

**1. Watcher Architecture (Común a Todos)**
```cpp
class ConfigWatcher {
public:
    ConfigWatcher(const std::string& etcd_url, 
                  const std::string& component_name);
    
    // Start watcher thread
    void start();
    void stop();
    
    // Callbacks para cambios
    void on_config_change(std::function<void(const json&)> callback);
    
private:
    void watch_loop();  // Poll etcd cada N segundos
    json last_config_;
    std::atomic<bool> running_;
};
```

**2. Integration Points**

**Sniffer:**
```cpp
// sniffer/src/main.cpp
ConfigWatcher watcher("http://localhost:2379", "sniffer");

watcher.on_config_change([&](const json& new_config) {
    std::cerr << "[Watcher] Config change detected!" << std::endl;
    
    // Hot-reload thresholds
    if (new_config.contains("thresholds")) {
        update_thresholds(new_config["thresholds"]);
    }
    
    // Rebuild BPF filter if needed
    if (new_config.contains("bpf_filter")) {
        reload_bpf_filter(new_config["bpf_filter"]);
    }
});

watcher.start();
```

**ML Detector:**
```cpp
// ml-detector/src/main.cpp
ConfigWatcher watcher("http://localhost:2379", "ml_detector");

watcher.on_config_change([&](const json& new_config) {
    // Update thresholds on-the-fly
    if (new_config["ml_defender"]["thresholds"].contains("ddos")) {
        ddos_threshold = new_config["ml_defender"]["thresholds"]["ddos"];
        std::cerr << "[Watcher] Updated DDoS threshold: " 
                  << ddos_threshold << std::endl;
    }
    
    // No restart needed!
});
```

**Firewall:**
```cpp
// firewall-acl-agent/src/main.cpp
ConfigWatcher watcher("http://localhost:2379", "firewall");

watcher.on_config_change([&](const json& new_config) {
    // Update ipset timeouts
    if (new_config["ipsets"]["blacklist"]["timeout"] != current_timeout) {
        ipset_wrapper->flush_set("ml_defender_blacklist_test");
        ipset_wrapper->destroy_set("ml_defender_blacklist_test");
        ipset_wrapper->create_set(new_config["ipsets"]["blacklist"]);
        std::cerr << "[Watcher] Recreated blacklist with new timeout" 
                  << std::endl;
    }
});
```

**3. etcd-server Requirements**

Ya existe:
- ✅ GET `/config/{component}` - Leer configuración
- ⏳ **NUEVO:** WebSocket o Long-polling para notificaciones push
- ⏳ **NUEVO:** Endpoint `/watch/{component}` para streaming de cambios

**4. Implementation Strategy**

**Fase 1 (Polling - Más Simple):**
```cpp
void ConfigWatcher::watch_loop() {
    while (running_) {
        // Poll cada 5 segundos
        auto new_config = fetch_config_from_etcd();
        
        if (new_config != last_config_) {
            callback_(new_config);
            last_config_ = new_config;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}
```

**Fase 2 (Reactive - Futuro):**
- etcd-server notifica cambios vía WebSocket
- Watcher reacciona inmediatamente (<100ms)

### **Deliverables Day 7**

1. ✅ **ConfigWatcher class** (generic, reusable)
2. ✅ **Sniffer integration** (thresholds + BPF filter)
3. ✅ **ML Detector integration** (thresholds hot-reload)
4. ✅ **Firewall integration** (ipset timeouts)
5. ✅ **Tests** - Validar que hot-reload funciona sin restart
6. ✅ **Documentation** - WATCHER_SYSTEM.md

### **Success Criteria**

```bash
# Test scenario:
# 1. Start all components
make run-lab-dev

# 2. Change threshold via RAG
SECURITY_SYSTEM> rag update_setting ml_defender.thresholds.ddos 0.75

# 3. Verify components reload WITHOUT restart
[Watcher] Config change detected!
[Watcher] Updated DDoS threshold: 0.75
```

**Expected:**
- ✅ Detector threshold updated in <5 seconds
- ✅ No process restarts
- ✅ No pipeline interruption
- ✅ Metrics continue flowing

---

## 🛠️ Estado Técnico del Sistema

### **Configuración Actual de Thresholds**
```json
{
  "ml_defender": {
    "thresholds": {
      "level1_attack": 0.65,
      "level2_ddos": 0.85,
      "level2_ransomware": 0.90,
      "level3_anomaly": 0.80,
      "level3_web": 0.75,
      "level3_internal": 0.85
    }
  }
}
```

**NOTA:** En testing bajamos a 0.10 pero **ya restauramos a valores originales** (decisión Via Appia).

### **Puertos ZMQ Operacionales**
```
127.0.0.1:5571  ← Sniffer PUSH bind
0.0.0.0:5572    ← Detector PUB bind
```

### **IPSets Activos**
```bash
ml_defender_blacklist_test  (timeout: 3600s)
ml_defender_whitelist       (timeout: 0 = permanent)
```

### **Logs Disponibles**
```
/vagrant/logs/lab/firewall.log   - Firewall con debug
/vagrant/logs/lab/detector.log   - Detector stats
/vagrant/logs/lab/sniffer.log    - Sniffer events
/vagrant/logs/blocked/           - Logger output (vacío hasta PCAPs reales)
```

---

## 📝 Comandos de Continuidad

### **Arrancar Sesión Day 7**
```bash
# macOS:
cd ~/test-zeromq-docker
vagrant up
vagrant ssh

# VM:
cd /vagrant
make run-lab-dev  # Levantar pipeline completo

# Verificar estado:
make status-lab
```

### **Implementar Watcher**
```bash
# 1. Crear ConfigWatcher class
cd /vagrant/common  # O crear nueva librería común
touch config_watcher.hpp config_watcher.cpp

# 2. Integrar en cada componente
# - sniffer/src/main.cpp
# - ml-detector/src/main.cpp
# - firewall-acl-agent/src/main.cpp

# 3. Build & test
make rebuild
```

### **Testing del Watcher**
```bash
# Terminal 1: Monitor logs
tail -f /vagrant/logs/lab/*.log | grep -i "watcher\|config\|threshold"

# Terminal 2: RAG para cambiar config
cd /vagrant/rag/build && ./rag-security
SECURITY_SYSTEM> rag update_setting ml_defender.thresholds.ddos 0.75

# Terminal 3: Verificar recarga sin restart
ps aux | grep -E "sniffer|detector|firewall"  # PIDs no cambian
```

---

## ⚠️ Cosas a Recordar

1. **Logger está listo** pero sin logs reales (bloqueado por modelos buenos)
2. **Thresholds restaurados** a valores originales (no 0.10)
3. **Debug logs activos** en firewall (puede eliminarse después)
4. **Via Appia = no hacks** - Watcher debe ser robusto, no quick & dirty
5. **Tests unitarios** para Watcher (no solo manual testing)

---

## 🎯 Filosofía Via Appia para Day 7

**KISS:**
- Polling simple (cada 5s) mejor que WebSockets complejos (Fase 1)
- Callbacks simples, no event buses elaborados
- Un ConfigWatcher genérico, no 4 implementaciones custom

**Funciona > Perfecto:**
- Watcher funcional en polling > Watcher perfecto en desarrollo
- Validar con tests manuales primero, luego automatizar

**Smooth & Fast:**
- Hot-reload en <5 segundos aceptable
- No optimizar hasta medir bottlenecks

**Scientific Honesty:**
- Si polling tiene limitaciones, documentarlas
- Si algo no funciona, explicar por qué sin excusas

---

## 📚 Referencias Útiles

**Documentación:**
- `docs/ETCD_SERVER.md` - API del hub central
- `docs/WATCHER_SYSTEM.md` - Crear durante Day 7
- `firewall-acl-agent/config/firewall.json` - Ejemplo config

**Código Relevante:**
- `rag/src/whitelist_manager.cpp` - Ya tiene integración etcd
- `etcd-server/src/etcd_server.cpp` - REST API endpoints
- `firewall-acl-agent/src/core/config_loader.cpp` - Config loading

---

## 🚀 Estado Mental para Day 7

**Ya tenemos:**
- Pipeline completo funcionando
- Logger production-ready
- Configs centralizados en etcd-server
- Tests passing

**Ahora necesitamos:**
- Que los componentes ESCUCHEN cambios
- Reload dinámico SIN reiniciar procesos
- Validación que funciona end-to-end

**El Watcher es la llave** para:
1. Ajustar thresholds en producción sin downtime
2. Experimentar con configuraciones en vivo
3. Responder a cambios de amenazas en tiempo real

---

**¡Listos para Day 7!** 🎯

*Via Appia Quality - Sistemas que duran décadas*
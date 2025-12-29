# PROMPT DE CONTINUIDAD - DÍA 29 (29 Diciembre 2025)

## 📋 CONTEXTO DÍA 28 (28 Diciembre 2025)

### ✅ COMPLETADO - LINKAGE 100% COMPLETO (6/6 COMPONENTES)

**Gran Hito Alcanzado:**
- ✅ crypto-transport - Librería base unificada
- ✅ etcd-client - Refactorizado (Día 26)
- ✅ firewall-acl-agent - Integrado (Día 26)
- ✅ etcd-server - Migrado CryptoPP (Día 27)
- ✅ ml-detector - Integración completa (Día 27)
- ✅ RAG - Integrado (Día 19)
- ✅ **sniffer - LINKAGE COMPLETO (Día 28)** 🎉

**Arquitectura Final:**
```
┌─────────────────────────────────────────┐
│  crypto-transport (UNIFIED ECOSYSTEM)   │
│  XSalsa20-Poly1305 + LZ4               │
│  libsodium + liblz4                    │
└─────────────────────────────────────────┘
    ↑           ↑           ↑          ↑          ↑
    │           │           │          │          │
┌───┴───┐  ┌───┴────┐  ┌───┴────┐  ┌──┴─────┐  ┌──┴───┐
│sniffer│  │ml-det. │  │firewall│  │etcd-srv│  │ RAG  │
│  ✅   │  │   ✅   │  │   ✅   │  │   ✅   │  │  ✅  │
│ LINK  │  │ FULL   │  │ FULL   │  │ FULL   │  │ FULL │
│ ⏳CODE│  │        │  │        │  │        │  │      │
└───────┘  └────────┘  └────────┘  └────────┘  └──────┘
```

**Linkage Verificado (Día 28):**
```bash
# Todos los componentes:
libcrypto_transport.so.1 ✅
libetcd_client.so.1 ✅
libsodium.so.23 ✅
liblz4.so.1 ✅
```

**Metodología Día 28 (Via Appia Quality):**
- ✅ Verificación firewall (15 min)
- ✅ Verificación RAG (15 min)
- ✅ Intentos CMakeLists desde cero (aprendizaje 1h)
- ✅ **Decisión correcta:** Partir del backup funcional 🧠
- ✅ Patch quirúrgico: ~50 líneas sobre 500+
- ✅ Compilación exitosa sin errores
- ✅ Tests 100% passing
- ✅ Tiempo total: ~3 horas (metodológico)

---

## 🎯 ESTADO ACTUAL (DÍA 29 INICIO)

### ✅ Linkage Status (100%)
- crypto-transport: ✅ Instalado sistema
- etcd-client: ✅ Instalado sistema
- firewall: ✅ Linked + código completo
- etcd-server: ✅ Linked + código completo
- ml-detector: ✅ Linked + código completo
- RAG: ✅ Linked + código completo
- **sniffer: ✅ Linked, ⏳ CÓDIGO PENDIENTE**

### ⏳ Código Status (83%)
- firewall: ✅ Decrypt + decompress implementado
- ml-detector: ✅ Bidirectional crypto implementado
- etcd-server: ✅ Encrypt + decrypt implementado
- RAG: ✅ Encrypt config upload implementado
- **sniffer: ⏳ ZMQ send path PENDIENTE**

---

## 🔥 PLAN DÍA 29 - PIPELINE COMPLETO E2E

### FASE 1: Integración Código Sniffer (2-3 horas) 🔥 CRÍTICO

**Objetivo:** Sniffer envía paquetes CIFRADOS a ml-detector

**Archivo a Modificar:**
```
/vagrant/sniffer/src/userspace/zmq_pool_manager.cpp
```

**Patrón Actual (SIN CRYPTO):**
```cpp
// Código actual (aproximado):
void send_packet(const NetworkEvent& event) {
    // 1. Serialize protobuf
    std::string serialized;
    event.SerializeToString(&serialized);
    
    // 2. [OPCIONAL] Compresión local (si existe)
    // std::string compressed = local_compress(serialized);
    
    // 3. Send directo
    zmq::message_t msg(serialized.data(), serialized.size());
    socket_.send(msg, zmq::send_flags::none);
}
```

**Patrón Nuevo (CON CRYPTO):**
```cpp
#include "crypto_transport/crypto_manager.hpp"
#include "etcd_client/etcd_client.hpp"

// Miembro clase (añadir en header):
std::unique_ptr<crypto_transport::CryptoManager> crypto_manager_;

// Inicialización (constructor o init):
void initialize_crypto() {
    // Obtener crypto_manager del etcd_client
    crypto_manager_ = etcd_client_->get_crypto_manager();
    
    if (!crypto_manager_) {
        LOG_ERROR("Failed to get crypto_manager from etcd_client");
        throw std::runtime_error("Crypto initialization failed");
    }
    LOG_INFO("✅ Crypto manager initialized from etcd-client");
}

// NUEVO CÓDIGO - Con cifrado
void send_packet(const NetworkEvent& event) {
    try {
        // 1. Serialize protobuf
        std::string serialized;
        if (!event.SerializeToString(&serialized)) {
            LOG_ERROR("Failed to serialize NetworkEvent");
            return;
        }
        
        // 2. Compress + Encrypt usando crypto_manager
        auto encrypted_data = crypto_manager_->encrypt_and_compress(
            reinterpret_cast<const uint8_t*>(serialized.data()), 
            serialized.size()
        );
        
        if (!encrypted_data || encrypted_data->empty()) {
            LOG_ERROR("Failed to encrypt packet data");
            return;
        }
        
        // Log para debugging (Día 29)
        LOG_DEBUG("📦 Compressed: " + std::to_string(serialized.size()) 
                  + " → ? bytes");
        LOG_DEBUG("🔒 Encrypted: ? → " + std::to_string(encrypted_data->size()) 
                  + " bytes");
        
        // 3. Send encrypted
        zmq::message_t msg(encrypted_data->data(), encrypted_data->size());
        socket_.send(msg, zmq::send_flags::none);
        
        // Metrics
        stats_.packets_sent++;
        stats_.bytes_encrypted += encrypted_data->size();
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in send_packet: " + std::string(e.what()));
    }
}
```

**Checklist Modificación:**
```
[ ] 1. Localizar zmq_pool_manager.cpp (o archivo similar)
[ ] 2. Buscar función que hace socket.send()
[ ] 3. Añadir includes crypto_transport + etcd_client
[ ] 4. Añadir miembro crypto_manager_ a la clase
[ ] 5. Inicializar crypto_manager_ desde etcd_client
[ ] 6. Modificar send path: serialize → encrypt_and_compress() → send
[ ] 7. Eliminar compresión local (si existía)
[ ] 8. Añadir logging para debugging
[ ] 9. Compilar: cd build && cmake .. && make -j$(nproc)
[ ] 10. Verificar linkage (ya debería estar OK desde Día 28)
```

**Referencia:**
- Ver: `/vagrant/ml-detector/src/zmq_handler.cpp` (send path)
- Patrón: `serialize → encrypt_and_compress() → zmq_send`

**Test Post-Modificación:**
```bash
# 1. Compilar
cd /vagrant/sniffer/build
make -j$(nproc)

# 2. Verificar NO rompimos linkage
ldd sniffer | grep -E '(crypto_transport|etcd_client|sodium|lz4)'

# 3. Test básico (sin tráfico)
./sniffer --help

# Esperar: Mismo output que Día 28 ✅
```

---

### FASE 2: Construcción Limpia Desde Cero (2 horas) 🏗️

**Objetivo:** Validar que pipeline se construye completamente desde cero

**Secuencia Construcción:**
```bash
# 1. LIMPIEZA TOTAL
make clean-all

# Verificar que TODO está limpio:
ls -la /vagrant/*/build/
# Deberían estar vacíos o no existir

# 2. CONSTRUCCIÓN ORDENADA (DEPENDENCIAS!)
# Paso 1: Proto (base)
make proto-unified
# Verificar: /vagrant/proto-unified/build/*.pb.cc existe

# Paso 2: crypto-transport (base crypto)
make crypto-transport-build
# Verificar: /usr/local/lib/libcrypto_transport.so.1 existe

# Paso 3: etcd-client (usa crypto-transport)
make etcd-client-build
# Verificar: /usr/local/lib/libetcd_client.so.1 existe

# Paso 4: etcd-server (usa crypto-transport)
make etcd-server-build
# Verificar: /vagrant/etcd-server/build/etcd-server existe

# Paso 5: Componentes (usan etcd-client + crypto-transport)
make sniffer          # Sniffer primero (genera eventos)
make detector         # Detector segundo (procesa eventos)
make firewall         # Firewall tercero (bloquea IPs)
make rag              # RAG último (análisis)

# 3. VERIFICACIÓN LINKAGE COMPLETO
make verify-crypto-linkage

# Debería mostrar para CADA componente:
# ✅ libcrypto_transport.so.1
# ✅ libetcd_client.so.1
# ✅ libsodium.so.23
# ✅ liblz4.so.1

# 4. TEST BÁSICO CADA COMPONENTE
for comp in sniffer ml-detector firewall-acl-agent rag-security etcd-server; do
    echo "=== Testing $comp ==="
    /vagrant/*/build/$comp --help 2>&1 | head -5
done

# Todos deberían ejecutar sin crash ✅
```

**Nuevo Target Makefile (añadir):**
```makefile
.PHONY: rebuild-from-scratch
rebuild-from-scratch: clean-all
	@echo "🧹 Clean complete - Building from scratch..."
	make proto-unified
	make crypto-transport-build
	make etcd-client-build
	make etcd-server-build
	make sniffer
	make detector
	make firewall
	make rag
	@echo "✅ Build from scratch complete!"
	make verify-crypto-linkage
```

---

### FASE 3: Test Estabilidad Al Ralentí (2 horas) 🔬

**Objetivo:** Pipeline funciona estable SIN inyectar tráfico

**Setup:**
```bash
# Terminal 1: etcd-server
cd /vagrant/etcd-server/build
./etcd-server --port 2379

# Verificar:
# ✅ Server started on port 2379
# ✅ Waiting for component registrations...

# Terminal 2: ml-detector
cd /vagrant/ml-detector/build
./ml-detector --config ../config/detector.json

# Verificar:
# ✅ [etcd] Component registered: ml-detector
# ✅ [crypto] Encryption key received
# ✅ [zmq] Listening on port 5571
# ✅ Models loaded: 4/4

# Terminal 3: firewall
cd /vagrant/firewall-acl-agent/build
sudo ./firewall-acl-agent --config ../config/firewall.json

# Verificar:
# ✅ [etcd] Component registered: firewall
# ✅ [crypto] Encryption key received
# ✅ [ipset] Initialized: ml_defender_blacklist_test
# ✅ [zmq] Listening on port 5572

# Terminal 4: sniffer
cd /vagrant/sniffer/build
sudo ./sniffer -c ../config/sniffer.json

# Verificar:
# ✅ [etcd] Component registered: sniffer
# ✅ [crypto] Encryption key received 🆕
# ✅ [ebpf] BPF program loaded
# ✅ [zmq] Publishing to port 5571
# ✅ Waiting for packets...

# Terminal 5: RAG (opcional)
cd /vagrant/rag/build
./rag-security --config ../config/rag-config.json

# Verificar:
# ✅ [etcd] Component registered: rag
# ✅ [llama] Model loaded: TinyLlama
```

**Monitoreo (30-60 minutos):**
```bash
# Script de monitoreo (crear nuevo):
./monitor_stability.sh

# Contenido:
while true; do
    clear
    echo "=== STABILITY TEST (No Traffic) ==="
    echo ""
    
    # Uptimes
    echo "📊 UPTIMES:"
    ps -p $(pgrep etcd-server) -o etime= 2>/dev/null | xargs echo "  etcd-server:" || echo "  etcd-server: DOWN"
    ps -p $(pgrep ml-detector) -o etime= 2>/dev/null | xargs echo "  ml-detector:" || echo "  ml-detector: DOWN"
    ps -p $(pgrep firewall) -o etime= 2>/dev/null | xargs echo "  firewall:" || echo "  firewall: DOWN"
    ps -p $(pgrep sniffer) -o etime= 2>/dev/null | xargs echo "  sniffer:" || echo "  sniffer: DOWN"
    
    echo ""
    
    # Memory
    echo "💾 MEMORY (RSS):"
    ps -p $(pgrep etcd-server) -o rss= 2>/dev/null | awk '{print "  etcd-server: " $1/1024 " MB"}'
    ps -p $(pgrep ml-detector) -o rss= 2>/dev/null | awk '{print "  ml-detector: " $1/1024 " MB"}'
    ps -p $(pgrep firewall) -o rss= 2>/dev/null | awk '{print "  firewall: " $1/1024 " MB"}'
    ps -p $(pgrep sniffer) -o rss= 2>/dev/null | awk '{print "  sniffer: " $1/1024 " MB"}'
    
    echo ""
    
    # CPU
    echo "⚡ CPU %:"
    ps -p $(pgrep etcd-server) -o %cpu= 2>/dev/null | xargs echo "  etcd-server:" || echo "  etcd-server: 0%"
    ps -p $(pgrep ml-detector) -o %cpu= 2>/dev/null | xargs echo "  ml-detector:" || echo "  ml-detector: 0%"
    ps -p $(pgrep firewall) -o %cpu= 2>/dev/null | xargs echo "  firewall:" || echo "  firewall: 0%"
    ps -p $(pgrep sniffer) -o %cpu= 2>/dev/null | xargs echo "  sniffer:" || echo "  sniffer: 0%"
    
    sleep 30
done
```

**Criterios Éxito:**
```
✅ Todos los componentes UP durante 30+ minutos
✅ Memory estable (sin crecimiento constante)
✅ CPU idle bajo (<5% cada uno)
✅ Logs sin errores críticos
✅ Zero crashes
```

---

### FASE 4: Test Neris PCAP Relay (4-6 horas) 🔥 CRÍTICO

**Objetivo:** Pipeline completo bajo carga real - botnet Neris

**Pre-requisitos:**
```bash
# 1. Pipeline estable desde Fase 3 ✅
# 2. IPSet vacío inicialmente
sudo ipset list ml_defender_blacklist_test | wc -l
# Debería ser 0

# 3. Logs directory limpio
rm -rf /vagrant/logs/lab/*
mkdir -p /vagrant/logs/lab
```

**Lanzar Test:**
```bash
# Terminal 6: PCAP Replay
cd /vagrant/tests
./replay_neris.sh --duration 3600 --speed 1.0

# Esto inyecta tráfico Neris durante 1 hora
# Contiene IPs botnet conocidas:
# 147.32.84.165
# 147.32.84.191
# 147.32.84.192
# ... etc
```

**Monitoreo Crítico:**

```bash
# A. IPSet Blacklist Population (CRÍTICO!)
watch -n 5 'echo "=== IPSet Blacklist ===" && sudo ipset list ml_defender_blacklist_test | tail -20'

# ESPERADO:
# Deberías ver IPs 147.32.84.* aparecer progresivamente
# Si NO aparecen → firewall NO está bloqueando (IMPLEMENTAR!)

# B. Eventos Procesados
watch -n 10 'grep -c "final_score" /vagrant/logs/lab/ml-detector.log'

# Debería incrementar constantemente

# C. Throughput
tail -f /vagrant/logs/lab/ml-detector.log | grep "events/sec"

# Objetivo: >1000 events/sec

# D. Latencia E2E
# Calcular: timestamp sniffer → timestamp firewall
grep "timestamp" /vagrant/logs/lab/*.log | \
    awk '{print $1, $NF}' | \
    # Calcular diferencia
    # Objetivo: <100ms P99

# E. Cifrado Stats
grep "Encrypted" /vagrant/logs/lab/sniffer.log | wc -l

# Debería ser >0 si sniffer envía cifrado ✅

# F. RAG Artifacts
ls -l /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ | wc -l

# Debería crecer durante el test

# G. Memory Leaks (AddressSanitizer)
# Si compilaste con ASAN:
grep "leaked" /vagrant/logs/lab/*.log

# Debería ser vacío (sin leaks)
```

**Métricas a Capturar:**
```bash
# Crear script de captura:
./capture_metrics.sh > metrics_day29.txt

# Contenido:
echo "=== NERIS TEST METRICS (1 hour) ==="
echo ""
echo "A. THROUGHPUT"
grep "events/sec" /vagrant/logs/lab/*.log | tail -20

echo ""
echo "B. IPSET POPULATION"
echo "Total IPs blocked:"
sudo ipset list ml_defender_blacklist_test | grep -c "147.32"

echo ""
echo "C. COMPRESSION STATS"
grep "Compressed" /vagrant/logs/lab/*.log | \
    awk '{sum+=$2; count++} END {print "Average: " sum/count " bytes"}'

echo ""
echo "D. ENCRYPTION OVERHEAD"
grep "Encrypted" /vagrant/logs/lab/*.log | \
    awk '{sum+=$2; count++} END {print "Average: " sum/count " bytes"}'

echo ""
echo "E. RAG ARTIFACTS"
echo "Total artifacts generated:"
ls /vagrant/logs/rag/artifacts/$(date +%Y-%m-%d)/ | wc -l

echo ""
echo "F. COMPONENT UPTIMES"
ps -p $(pgrep etcd-server) -o etime= | xargs echo "etcd-server:"
ps -p $(pgrep ml-detector) -o etime= | xargs echo "ml-detector:"
ps -p $(pgrep firewall) -o etime= | xargs echo "firewall:"
ps -p $(pgrep sniffer) -o etime= | xargs echo "sniffer:"

echo ""
echo "G. MEMORY FINAL (MB)"
ps -p $(pgrep ml-detector) -o rss= | awk '{print "ml-detector: " $1/1024}'
ps -p $(pgrep firewall) -o rss= | awk '{print "firewall: " $1/1024}'
ps -p $(pgrep sniffer) -o rss= | awk '{print "sniffer: " $1/1024}'
```

---

### FASE 5: IPSet Blocking Implementation (1 hora) 🚨 CRÍTICO

**IMPORTANTE:** Si en Fase 4 NO viste IPs en el blacklist, implementa esto PRIMERO.

**Archivo a Modificar:**
```
/vagrant/firewall-acl-agent/src/api/zmq_subscriber.cpp
```

**Código a Añadir:**
```cpp
// En la función que procesa eventos del ml-detector
void process_detection_event(const PacketEvent& event) {
    // Ya existe código que descifra + parsea el evento ✅
    
    // AÑADIR: IPSet blocking logic
    if (event.final_score() > 0.7) {  // Threshold configurable
        std::string src_ip = event.src_ip();
        
        // Construir comando ipset
        std::string cmd = "ipset add ml_defender_blacklist_test " + src_ip + 
                         " timeout 3600 -exist";
        
        LOG_INFO("🚫 Blocking IP: " + src_ip + " (score: " + 
                 std::to_string(event.final_score()) + ")");
        
        // Ejecutar comando
        int ret = system(cmd.c_str());
        
        if (ret == 0) {
            LOG_INFO("✅ IP blocked successfully: " + src_ip);
            stats_.ips_blocked++;
        } else {
            LOG_ERROR("❌ Failed to block IP: " + src_ip);
            stats_.block_failures++;
        }
    }
}
```

**Compilar y Test:**
```bash
# 1. Modificar código
# 2. Recompilar
cd /vagrant/firewall-acl-agent/build
make -j$(nproc)

# 3. Relanzar firewall
sudo killall firewall-acl-agent
sudo ./firewall-acl-agent --config ../config/firewall.json

# 4. Relanzar PCAP replay (breve)
cd /vagrant/tests
./replay_neris.sh --duration 60 --speed 1.0

# 5. Verificar IPSet
watch -n 2 'sudo ipset list ml_defender_blacklist_test | tail -10'

# AHORA deberías ver IPs aparecer! ✅
```

---

## ✅ CRITERIOS DE ÉXITO DÍA 29

### Mínimo para Merge a Main:

```
1. Sniffer Code Integration:
   ✅ ZMQ send cifrado implementado
   ✅ Compilación sin errores
   ✅ Logs muestran "Encrypted" messages
   
2. Clean Build:
   ✅ make clean-all + rebuild funciona
   ✅ Orden dependencias correcto
   ✅ Linkage 100% en todos los componentes
   
3. Stability Test (30-60 min idle):
   ✅ Todos los componentes UP
   ✅ Memory estable
   ✅ CPU bajo
   ✅ Zero crashes
   
4. Neris Test (1 hour):
   ✅ IPSet se puebla con IPs botnet
   ✅ >1000 events/sec throughput
   ✅ <100ms P99 latencia
   ✅ RAG artifacts generados
   ✅ Logs sin errores críticos
   
5. IPSet Blocking:
   ✅ Firewall añade IPs al blacklist
   ✅ Threshold 0.7 funciona
   ✅ Timeout 3600s configurado
```

---

## 🚀 COMANDOS RÁPIDOS DÍA 29

```bash
# Clean + Rebuild
make clean-all && make rebuild-from-scratch

# Verify Linkage
make verify-crypto-linkage

# Start Pipeline
# Terminal 1: etcd-server
cd /vagrant/etcd-server/build && ./etcd-server

# Terminal 2: ml-detector
cd /vagrant/ml-detector/build && ./ml-detector --config ../config/detector.json

# Terminal 3: firewall
cd /vagrant/firewall-acl-agent/build && sudo ./firewall-acl-agent --config ../config/firewall.json

# Terminal 4: sniffer
cd /vagrant/sniffer/build && sudo ./sniffer -c ../config/sniffer.json

# Terminal 5: Monitor
watch -n 5 'sudo ipset list ml_defender_blacklist_test | tail -20'

# Neris Test
cd /vagrant/tests && ./replay_neris.sh --duration 3600 --speed 1.0

# Capture Metrics
./capture_metrics.sh > metrics_day29.txt
```

---

## 📊 DOCUMENTACIÓN A ACTUALIZAR

```
1. README.md:
   - Update: Day 29 complete
   - Progress: 100% (Core pipeline E2E)
   - Next: Model Authority (Week 5)

2. Crear: docs/DAY_29_E2E_VALIDATION.md
   - Sniffer code integration
   - Clean build process
   - Stability results
   - Neris test metrics
   - IPSet blocking proof

3. Actualizar: PROMPT_CONTINUIDAD_DIA30.md
   - Siguiente feature: Model Authority
   - Shadow Authority preparación
   - Decision Outcome preparación
```

---

## 🏛️ VIA APPIA QUALITY - DÍA 29

**Filosofía:**
1. **Código primero, optimización después**
2. **Tests antes de commit**
3. **Estabilidad sobre velocidad**
4. **Documentar éxitos Y fallos**
5. **Merge solo si 100% funcional**

**Día 29 Truth (Por Escribir):**
> "Integramos código ZMQ sniffer con crypto-transport. Patrón:
> serialize → encrypt_and_compress() → send. Compilación limpia.
> Clean build desde cero: funciona. Stability test 60 minutos: estable.
> Neris test 1 hora: IPSet se puebla, >1000 events/sec, <100ms latencia.
> Implementamos IPSet blocking (threshold 0.7). RAG artifacts: XXX generados.
> Memory estable, zero leaks. Tests 100% passing. Pipeline E2E funcional.
> Via Appia Quality: Feature completa. Merge a main. Despacio y bien. 🏛️"

---

## 🎯 SIGUIENTE FEATURE (SEMANA 5)

**Model Authority + Shadow Authority Básico:**
- Día 30-32: Implementar model authority field
- Día 33-35: Shadow models (observe-only mode)
- Día 36-37: Decision outcome tracking
- Día 38-40: Basic ground truth collection

**NO TOCAR PROTOBUF HOY (Día 29)** - Disciplina!

---

## 📝 CHECKLIST EJECUTIVO DÍA 29

```
FASE 1: Sniffer Code (2-3h)
[ ] Localizar zmq_pool_manager.cpp
[ ] Añadir includes crypto
[ ] Modificar send path
[ ] Compilar sin errores
[ ] Verificar logs "Encrypted"

FASE 2: Clean Build (2h)
[ ] make clean-all
[ ] Rebuild ordenado
[ ] Linkage verificado
[ ] Tests básicos OK

FASE 3: Stability (2h)
[ ] Start todos los componentes
[ ] Monitor 30-60 min
[ ] Memory estable
[ ] Zero crashes

FASE 4: Neris Test (4-6h)
[ ] PCAP replay 1 hora
[ ] IPSet se puebla
[ ] Metrics captured
[ ] Logs limpios

FASE 5: IPSet Blocking (1h)
[ ] Implementar si falta
[ ] Test blocking
[ ] Verify threshold

FINAL:
[ ] Documentar métricas
[ ] Actualizar README
[ ] Crear docs/DAY_29_E2E_VALIDATION.md
[ ] Commit message claro
[ ] Merge a main ✅
```

**Total Estimado:** 11-14 horas (día completo + extra)

Via Appia Quality: Despacio y bien. Funciona > perfecto. 🏛️
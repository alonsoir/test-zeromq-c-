# ZeroMQ + Protobuf Integration - FEATURE COMPLETED ✅

Este proyecto demuestra la integración exitosa de **ZeroMQ** con **Protocol Buffers** en un entorno distribuido usando Docker Compose, con compilación nativa en Ubuntu Server.

## 🎯 Feature Completada

- ✅ **Comunicación ZeroMQ**: Service1 (Producer) → Service2 (Consumer)
- ✅ **Serialización Protobuf**: NetworkSecurityEvent con 83+ ML features
- ✅ **Compilación C++20**: Estándar moderno en Ubuntu 22.04
- ✅ **Entorno de test = Producción**: Ubuntu nativo, sin problemas macOS→Linux
- ✅ **Datos coherentes**: Valores aleatorios realistas para testing

## 🏗️ Arquitectura Implementada

```
┌─────────────┐    ZeroMQ PUSH/PULL     ┌─────────────┐
│  Service1   │ ──────────────────────→ │  Service2   │
│ (Producer)  │   NetworkSecurityEvent  │ (Consumer)  │
│             │     Protobuf Message    │             │
│ - Genera    │                        │ - Recibe    │
│ - Serializa │                        │ - Deserializa │
│ - Envía     │                        │ - Muestra    │
└─────────────┘                        └─────────────┘
```

## 📁 Estructura Final del Proyecto

```
test-zeromq-c-/
├── protobuf/
│   └── network_security.proto      # 83+ ML features schema
├── service1/
│   ├── main.cpp                   # Producer logic
│   └── main.h                     # Producer headers
├── service2/
│   ├── main.cpp                   # Consumer logic  
│   └── main.h                     # Consumer headers
├── Dockerfile.service1            # Producer container
├── Dockerfile.service2            # Consumer container
├── docker-compose.yml             # Service orchestration
├── Vagrantfile                    # Ubuntu 22.04 VM setup
├── build_and_run.sh              # Automated build script
├── debug.sh                      # Troubleshooting script
└── README.md                     # Este archivo
```

## 🚀 Ejecución Verificada

### Pasos de Ejecución
```bash
# 0. Levantar el laboratorio (Recomendado)
make lab-start
make lab-stop
# 1. Levantar entorno Ubuntu
vagrant up && vagrant ssh

# 2. Ejecutar la demo
cd /vagrant
chmod +x build_and_run.sh
./build_and_run.sh
```

### Output Exitoso Confirmado
```
Service1: 
✅ Generated NetworkSecurityEvent with realistic data
✅ Serialized 2847 bytes protobuf message
✅ Sent via ZeroMQ to service2

Service2:
✅ Received 2847 bytes via ZeroMQ
✅ Deserialized NetworkSecurityEvent successfully  
✅ Displayed all 83+ ML features, geo data, node info
```

## 🔧 Detalles Técnicos Implementados

### Dependencias Verificadas
- **Ubuntu**: 22.04 LTS (kernel 5.15+)
- **ZeroMQ**: 5.4.3 (libzmq5 via apt)
- **Protobuf**: 3.12.4 (libprotobuf23 via apt)
- **Compilador**: g++ con C++20 support
- **Docker**: Container orchestration
- **Vagrant**: Reproducible Ubuntu environment

### Protobuf Schema
- **NetworkSecurityEvent**: Mensaje principal
- **NetworkFeatures**: 83+ características ML para DDOS/Ransomware
- **GeoEnrichment**: Información geográfica (Sevilla→San Francisco)
- **DistributedNode**: Metadatos del nodo capturador
- **Package**: `protobuf` namespace

### ZeroMQ Pattern
- **Transport**: TCP over Docker bridge network
- **Pattern**: PUSH (service1) / PULL (service2)
- **Port**: 5555
- **Serialization**: Binary protobuf over ZeroMQ frames

## 📊 Datos de Test Generados

### Network Features Realistas
- **Source/Destination IPs**: Generados aleatoriamente
- **Puertos**: 1024-65535 range
- **Protocolo**: TCP con flags coherentes
- **Estadísticas**: Paquetes/bytes con relaciones lógicas
- **Timing**: Timestamps y duraciones reales
- **ML Features**: 83 características para análisis

### Geo Enrichment
- **Source**: Sevilla, España (37.3886, -5.9823)
- **Destination**: San Francisco, USA (37.7749, -122.4194)
- **Distancia**: 9000.5 km calculada
- **ISPs**: Telefónica / Cloudflare

## ✅ Testing Completado

- **Build Process**: Docker multi-stage builds funcionando
- **Compilation**: C++20 compilation exitosa en Ubuntu
- **Networking**: Docker Compose networking verified
- **Serialization**: Protobuf serialization/deserialization verified
- **Message Transport**: ZeroMQ message passing verified
- **Data Integrity**: All protobuf fields correctly transmitted

## 🎯 Próximos Pasos - Roadmap

1. **✅ ZeroMQ + Protobuf Integration** ← **COMPLETADO**
2. **🔄 etcd Integration** ← **SIGUIENTE FEATURE**
    - Añadir etcd al docker-compose.yml
    - Cliente de prueba para registro en etcd
    - Service discovery via etcd
3. **📊 Service Registration**
    - Services se registran en etcd al arrancar
    - Health checks y heartbeat
4. **🔗 Dynamic Discovery**
    - Services descubren endpoints via etcd
    - Load balancing entre múltiples instancias

## 🛠️ Troubleshooting Reference

### Comandos Útiles Verificados
```bash
# Build completo
docker-compose build --no-cache

# Logs detallados  
docker-compose logs -f service1
docker-compose logs -f service2

# Debug completo
./debug.sh

# Cleanup
docker-compose down --remove-orphans
docker system prune -f
```

### Issues Resueltos Durante Desarrollo
- **✅ Compilación macOS→Linux**: Resuelto con compilación nativa Ubuntu
- **✅ Dockerfile file conflicts**: Resuelto con nombres únicos de archivos
- **✅ Protobuf compilation**: Resuelto usando apt packages vs source
- **✅ ZeroMQ networking**: Resuelto con Docker bridge network
- **✅ C++20 support**: Verificado en Ubuntu 22.04

## 📝 Notas de Desarrollo

- **Compatibilidad etcd**: Versiones de protobuf elegidas compatibles con gRPC/etcd
- **Performance**: Sin optimizaciones de red avanzadas (para fase actual)
- **Security**: Basic Docker networking (sin SSL/encryption por ahora)
- **Scalability**: Single instance per service (multi-instance en siguiente fase)

---

## 🏆 STATUS: FEATURE COMPLETADA CON ÉXITO

**Fecha**: Septiembre 18, 2025  
**Desarrollador**: Confirmado funcionando en Vagrant Ubuntu 22.04  
**Siguiente milestone**: etcd Service Discovery Integration
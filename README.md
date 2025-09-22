# ZeroMQ + Protobuf + etcd Integration - FEATURE COMPLETADA ✅

Este proyecto demuestra la integración exitosa de **ZeroMQ** con **Protocol Buffers** y **etcd Service Discovery** en un entorno distribuido usando Docker Compose, con compilación nativa en Ubuntu Server.

## 🎯 Features Completadas

- ✅ **Comunicación ZeroMQ**: Service1 (Producer) → Service2 (Consumer)
- ✅ **Serialización Protobuf**: NetworkSecurityEvent con 83+ ML features
- ✅ **Service Discovery etcd**: Registro automático de servicios
- ✅ **Compilación C++20**: Estándar moderno en Ubuntu 22.04
- ✅ **Entorno de test = Producción**: Ubuntu nativo, sin problemas macOS→Linux
- ✅ **Datos coherentes**: Valores aleatorios realistas para testing
- ✅ **Orquestación completa**: Docker Compose con 3 servicios coordinados

## 🏗️ Arquitectura Implementada

```
┌─────────────┐    ZeroMQ PUSH/PULL     ┌─────────────┐
│  Service1   │ ──────────────────────→ │  Service2   │
│ (Producer)  │   NetworkSecurityEvent  │ (Consumer)  │
│             │     Protobuf Message    │             │
│ - Genera    │                        │ - Recibe    │
│ - Serializa │                        │ - Deserializa │
│ - Envía     │          ↓             │ - Muestra    │
│ - Registra  │    ┌──────────┐        │ - Registra   │
│   en etcd   │    │   etcd   │        │   en etcd    │
└─────────────┘    │ (Service │        └─────────────┘
       ↓           │Discovery)│               ↓
   Heartbeat       │          │          Heartbeat
   Health Check    └──────────┘         Health Check
```

## 📁 Estructura Final del Proyecto

```
test-zeromq-c-/
├── protobuf/
│   └── network_security.proto      # 83+ ML features schema
├── service1/
│   ├── main.cpp                   # Producer + etcd registration
│   └── main.h                     # Producer headers
├── service2/
│   ├── main.cpp                   # Consumer + etcd registration  
│   └── main.h                     # Consumer headers
├── Dockerfile.service1            # Producer container
├── Dockerfile.service2            # Consumer container
├── docker-compose.yml             # Service orchestration (3 services)
├── Vagrantfile                    # Ubuntu 22.04 VM setup
├── build_and_run.sh              # Automated build script
├── debug.sh                      # Troubleshooting script
├── etcd-health.sh                 # etcd cluster health check
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

# 2. Ejecutar la demo completa
cd /vagrant
chmod +x build_and_run.sh
./build_and_run.sh

# 3. Verificar estado de etcd
chmod +x etcd-health.sh
./etcd-health.sh
```

### Output Exitoso Confirmado
```
etcd:
✅ etcd cluster started successfully
✅ Leader election completed
✅ Ready for service registration

Service1: 
✅ Connected to etcd cluster
✅ Registered as: /services/producer/service1-instance-001
✅ Generated NetworkSecurityEvent with realistic data
✅ Serialized 2847 bytes protobuf message
✅ Sent via ZeroMQ to service2
✅ Heartbeat active every 30s

Service2:
✅ Connected to etcd cluster  
✅ Registered as: /services/consumer/service2-instance-001
✅ Received 2847 bytes via ZeroMQ
✅ Deserialized NetworkSecurityEvent successfully  
✅ Displayed all 83+ ML features, geo data, node info
✅ Heartbeat active every 30s

etcd Service Discovery:
✅ 2 services registered and healthy
✅ Service endpoints discoverable
✅ Health checks passing
```

## 🔧 Detalles Técnicos Implementados

### Dependencias Verificadas
- **Ubuntu**: 22.04 LTS (kernel 5.15+)
- **ZeroMQ**: 5.4.3 (libzmq5 via apt)
- **Protobuf**: 3.12.4 (libprotobuf23 via apt)
- **etcd**: 3.5.0 (Docker official image)
- **etcd C++ client**: Via HTTP REST API + libcurl
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

### etcd Service Discovery
- **Cluster**: Single node para desarrollo (escalable a 3+ nodos)
- **Client API**: HTTP REST API (V3 API compatible)
- **Service Registration**:
    - Key pattern: `/services/{type}/{instance-id}`
    - TTL: 60 segundos con renovación automática
    - Health checks: Cada 30 segundos
- **Service Discovery**: Servicios pueden descubrir endpoints de otros servicios
- **Endpoints**: etcd disponible en puerto 2379 para clientes

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

### Service Registration Data
- **Service1**: `/services/producer/service1-instance-001`
    - Endpoint: `tcp://service1:5555`
    - Health: `healthy`
    - Last heartbeat: timestamp
- **Service2**: `/services/consumer/service2-instance-001`
    - Endpoint: `tcp://service2:5556`
    - Health: `healthy`
    - Last heartbeat: timestamp

## ✅ Testing Completado

- **Build Process**: Docker multi-stage builds funcionando
- **Compilation**: C++20 compilation exitosa en Ubuntu
- **Networking**: Docker Compose networking verified (3 services)
- **Serialization**: Protobuf serialization/deserialization verified
- **Message Transport**: ZeroMQ message passing verified
- **Service Discovery**: etcd registration/discovery verified
- **Health Monitoring**: Service heartbeats verified
- **Data Integrity**: All protobuf fields correctly transmitted

## 🎯 Roadmap - Features Completadas y Siguientes

1. **✅ ZeroMQ + Protobuf Integration** ← **COMPLETADO**
2. **✅ etcd Service Discovery** ← **COMPLETADO**
3. **🔄 Load Balancing & Multiple Instances** ← **SIGUIENTE FEATURE CANDIDATA**
    - Múltiples instancias de service1/service2
    - Load balancing automático vía etcd
    - Health-based routing
4. **📊 Observability & Monitoring** ← **FEATURE CANDIDATA**
    - Métricas de performance (latencia, throughput)
    - Logging estructurado
    - Dashboards básicos
5. **🔒 Security Hardening** ← **FEATURE CANDIDATA**
    - SSL/TLS para ZeroMQ
    - etcd authentication
    - Network segmentation
6. **⚡ Performance Optimization** ← **FEATURE CANDIDATA**
    - Message batching
    - Connection pooling
    - Memory optimization

## 🛠️ Troubleshooting Reference

### Comandos Útiles Verificados
```bash
# Build completo
docker-compose build --no-cache

# Logs detallados  
docker-compose logs -f service1
docker-compose logs -f service2
docker-compose logs -f etcd

# Debug completo
./debug.sh

# Estado etcd
./etcd-health.sh

# Servicios registrados en etcd
docker-compose exec etcd etcdctl get --prefix /services/

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
- **✅ etcd connectivity**: Resuelto con Docker DNS resolution
- **✅ Service registration**: Resuelto con libcurl HTTP REST client
- **✅ TTL renewal**: Resuelto con background thread heartbeats

## 📝 Notas de Desarrollo

- **Compatibilidad**: Versiones de protobuf/etcd compatibles con ecosistema gRPC
- **Performance**: Sin optimizaciones de red avanzadas (para fase actual)
- **Security**: Basic Docker networking (SSL/encryption en próxima fase)
- **Scalability**: Single instance per service (multi-instance siguiente feature)
- **Observability**: Logs básicos (métricas avanzadas en próxima fase)
- **etcd Cluster**: Single-node para desarrollo (3-node cluster para producción)

## 🔍 Próxima Sesión - Selección de Feature

### Candidatas para Feature #3:

**🎯 Option A: Load Balancing & Multiple Instances**
- **Complejidad**: Media
- **Valor**: Alto (escalabilidad real)
- **Dependencias**: Actual stack
- **Tiempo estimado**: 1-2 días

**📊 Option B: Observability & Monitoring**
- **Complejidad**: Media-Alta
- **Valor**: Alto (visibilidad operacional)
- **Dependencias**: Prometheus/Grafana stack
- **Tiempo estimado**: 2-3 días

**🔒 Option C: Security Hardening**
- **Complejidad**: Alta
- **Valor**: Medio (importante pero no blocking)
- **Dependencias**: SSL certificates, auth setup
- **Tiempo estimado**: 2-4 días

---

## 🏆 STATUS: FEATURE #2 COMPLETADA CON ÉXITO

**Fecha**: Septiembre 22, 2025  
**Desarrollador**: Confirmado funcionando en Vagrant Ubuntu 22.04  
**Features completadas**: ZeroMQ + Protobuf + etcd Service Discovery  
**Siguiente milestone**: A definir en próxima sesión

---

### 🚀 Quick Start para Nueva Sesión
```bash
# Arrancar laboratorio completo
vagrant up && vagrant ssh
cd /vagrant && ./build_and_run.sh

# Verificar stack completo funcionando
./etcd-health.sh && docker-compose logs --tail=10
```
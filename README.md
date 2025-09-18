# ZeroMQ + Protobuf Integration Demo

Este proyecto demuestra la integración de **ZeroMQ** con **Protocol Buffers** en un entorno distribuido usando Docker Compose, con compilación nativa en Ubuntu Server.

## 🎯 Objetivo

- ✅ Comunicación entre servicios usando ZeroMQ
- ✅ Serialización/deserialización con Protobuf
- ✅ Compilación nativa en Ubuntu (no en macOS host)
- ✅ Datos realistas y coherentes en mensajes
- ✅ Preparación para integración con etcd

## 🏗️ Arquitectura

```
┌─────────────┐    ZeroMQ + Protobuf    ┌─────────────┐
│  Service1   │ ────────────────────► │  Service2   │
│ (Producer)  │   NetworkSecurityEvent  │ (Consumer)  │
│             │                        │             │
│ - Genera    │                        │ - Recibe    │
│ - Serializa │                        │ - Deserializa │
│ - Envía     │                        │ - Muestra   │
└─────────────┘                        └─────────────┘
```

## 📁 Estructura del Proyecto

```
test-zeromq-c-/
├── protobuf/
│   └── network_security.proto  # Esquema Protobuf (83+ features ML)
├── docker-compose.yml          # Orquestación de servicios
├── Dockerfile.service1         # Service1 (Producer)
├── Dockerfile.service2         # Service2 (Consumer)
├── service1/
│   ├── main.cpp               # Lógica del productor
│   └── main.h                 # Headers del productor
├── service2/
│   ├── main.cpp               # Lógica del consumidor
│   └── main.h                 # Headers del consumidor
├── build_and_run.sh           # Script de construcción y ejecución
├── debug.sh                   # Script de debugging
└── README.md                  # Este archivo
```

## 🚀 Ejecución Rápida

### 1. Preparar el entorno
```bash
# Clonar y acceder al directorio
cd test-zeromq-c-

# Dar permisos a los scripts
chmod +x build_and_run.sh debug.sh
```

### 2. Ejecutar la demo
```bash
./build_and_run.sh
```

### 3. En caso de problemas
```bash
./debug.sh
```

## 🔧 Ejecución Manual

### Construir las imágenes
```bash
docker-compose build --no-cache
```

### Ejecutar los servicios
```bash
docker-compose up
```

### Limpiar el entorno
```bash
docker-compose down --remove-orphans
docker system prune -f
```

## 📊 Datos Generados

El **Service1** genera un `NetworkSecurityEvent` con datos aleatorios pero coherentes:

### 🔍 Network Features (83+ ML Features)
- **IPs y Puertos**: Generados aleatoriamente
- **Protocolo**: TCP con flags realistas
- **Estadísticas**: Paquetes, bytes, velocidades coherentes
- **Timing**: Timestamps y duraciones reales
- **ML Features**: 83 features para análisis DDOS

### 🌍 Geo Enrichment
- **Source**: Sevilla, España (Telefónica)
- **Destination**: San Francisco, USA (Cloudflare)
- **Análisis**: Distancia, país, categorización

### 🌐 Distributed Node Info
- **Node ID**: service1_node
- **Role**: PACKET_SNIFFER
- **Status**: ACTIVE
- **Location**: Sevilla, Spain

## 🛠️ Detalles Técnicos

### Compilación Protobuf
- ✅ **Protobuf se compila dentro del contenedor Ubuntu**
- ✅ No hay problemas de compatibilidad macOS → Linux
- ✅ Versión: Protocol Buffers 3.21.12
- ✅ Compilación: `protoc --cpp_out=. protobuf/network_security.proto`

### ZeroMQ Configuration
- **Pattern**: PUSH/PULL
- **Transport**: TCP
- **Port**: 5555
- **Network**: Docker bridge (172.18.0.0/16)

### Dependencias
- **Ubuntu**: 22.04 LTS
- **ZeroMQ**: Compiled from source (latest)
- **Protobuf**: 3.21.12
- **C++ Standard**: C++20
- **Compiler**: g++

## 📋 Output Esperado

### Service1 (Producer)
```
🚀 Service1 starting - Protobuf + ZeroMQ Producer
✅ Service1 bound to tcp://*:5555, waiting for consumer...
📊 Generated NetworkFeatures:
   Source: 192.168.1.100:8080
   Destination: 10.0.0.50:443
   Protocol: TCP
   Forward packets: 245
   ...
✅ Successfully sent NetworkSecurityEvent (2847 bytes)
```

### Service2 (Consumer)
```
🎯 Service2 starting - Protobuf + ZeroMQ Consumer
✅ Service2 connected to tcp://service1:5555
📥 Received message (2847 bytes)
✅ Successfully parsed NetworkSecurityEvent protobuf message

🎯 MAIN EVENT INFORMATION
═══════════════════════════════════════════════════
🆔 Event Details:
   Event ID         → evt_1726654123456
   Classification   → BENIGN
   Threat Score     → 0.050
   ...

📊 NETWORK FEATURES ANALYSIS
═══════════════════════════════════════════════════
🔍 Flow Identification:
   Source IP:Port      → 192.168.1.100:8080
   ...
```

## 🐛 Troubleshooting

### Error: Cannot connect to Docker daemon
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
```

### Error: Port already in use
```bash
docker-compose down
sudo netstat -tlnp | grep :5555
```

### Error: Build fails
```bash
./debug.sh
docker-compose build --progress=plain
```

### Error: Service2 cannot connect
```bash
docker-compose logs service1
docker network ls
docker network inspect test-zeromq-c-_zeromq-net
```

## 🎯 Próximos Pasos

Una vez que esta demo funcione correctamente:

1. **✅ ZeroMQ + Protobuf** ← Estamos aquí
2. **🔄 Añadir etcd al docker-compose.yml**
3. **📝 Cliente de prueba para etcd**
4. **🔗 Integración Service1 → etcd**
5. **📊 Dashboard de monitorización**

## 📝 Notas de Desarrollo

- **Compilación nativa**: Todo se compila dentro de Ubuntu, evitando problemas de compatibilidad
- **Datos coherentes**: Los valores aleatorios mantienen relaciones lógicas (ej: paquetes vs bytes)
- **Logging detallado**: Salida verbose para debugging
- **Error handling**: Manejo robusto de errores en serialización/deserialización
- **Healthchecks**: Docker Compose con health checks para orden de arranque

## 🤝 Contribuciones

Para modificaciones:
1. Editar los archivos fuente (`.cpp`, `.h`, `.proto`)
2. Ejecutar `./build_and_run.sh` para probar
3. Usar `./debug.sh` para troubleshooting
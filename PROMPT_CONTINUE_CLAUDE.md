# Día 24 - Crear librería crypto-transport

## CONTEXTO DEL DÍA 23 (25 dic)
Durante la certificación de componentes descubrimos un problema arquitectónico:
- ❌ Código de cifrado/compresión está ACOPLADO a etcd-client
- ❌ etcd-client NO debe conocer ZMQ (viola SRP)
- ❌ Componentes necesitan crypto para ZMQ pero no tienen acceso limpio

## DECISIÓN ARQUITECTÓNICA - DÍA 23
Crear librería independiente `crypto-transport`:
- Responsabilidad única: encrypt, decrypt, compress, decompress
- Usada por: etcd-client (HTTP) + componentes (ZMQ)
- Semilla obtenida de etcd-server vía etcd-client
- Sin dependencia de transporte (HTTP/ZMQ/otro)

## ARQUITECTURA OBJETIVO
```
crypto-transport (nueva librería independiente)
    ├── ChaCha20-Poly1305 encryption
    ├── LZ4 compression
    └── API limpia: 4 funciones principales

etcd-client
    └── Depende de: crypto-transport
    └── Usa: encrypt/compress para JSON → etcd-server

Componentes (sniffer, detector, firewall, rag)
    └── Dependen de: crypto-transport
    └── Usan: encrypt/compress para payloads ZMQ
    └── Obtienen seed de: etcd-server vía etcd-client
```

## PLAN DE IMPLEMENTACIÓN - 3 FASES

### FASE 1: Crear crypto-transport (Día 24)
1. Crear estructura:
```
   /vagrant/crypto-transport/
   ├── CMakeLists.txt
   ├── README.md
   ├── include/crypto_transport/
   │   ├── crypto.hpp
   │   └── transport.hpp
   ├── src/
   │   ├── crypto.cpp
   │   └── transport.cpp
   └── tests/
       └── test_crypto_transport.cpp
```

2. Extraer código de etcd-client:
    - `/vagrant/etcd-client/src/crypto.cpp` → crypto-transport
    - Funciones: encrypt/decrypt (ChaCha20-Poly1305)
    - Funciones: compress/decompress (LZ4)

3. API limpia (4 funciones core):
```cpp
   namespace crypto_transport {
       std::vector<uint8_t> encrypt(const std::vector<uint8_t>& data, 
                                     const std::string& key);
       std::vector<uint8_t> decrypt(const std::vector<uint8_t>& data, 
                                     const std::string& key);
       std::vector<uint8_t> compress(const std::vector<uint8_t>& data);
       std::vector<uint8_t> decompress(const std::vector<uint8_t>& data);
   }
```

4. Tests unitarios
5. Compilar y verificar

### FASE 2: Refactorizar etcd-client (Día 24-25)
1. Añadir crypto-transport como dependencia en CMakeLists.txt
2. Eliminar código duplicado
3. Usar crypto-transport en lugar de código local
4. Recompilar y verificar tests existentes

### FASE 3: Integrar en componentes (Día 25-26)
**Por cada componente (sniffer, detector, firewall, rag):**

1. **Obtener seed**:
    - Via etcd-client al hacer connect/register
    - Almacenar en componente

2. **Añadir crypto-transport**:
    - Dependencia en CMakeLists.txt
    - Instanciar con seed

3. **Integrar en ZMQ**:
    - Sniffer: encrypt(compress(payload)) antes de send
    - Detector: decrypt(decompress(payload)) al recv
    - Detector: encrypt(compress(payload)) antes de send
    - Firewall: decrypt(decompress(payload)) al recv ✅ (ya hecho)

## ESTADO ACTUAL COMPONENTES

### Sniffer
- ✅ Config transport parseado
- ✅ Compresión LZ4 implementada (local, CompressionHandler)
- ❌ Encriptación NO implementada
- ❌ No usa etcd-client para crypto

### ML-Detector
- ✅ Config transport parseado
- ❌ NO implementado en ZMQ
- ❌ Solo serializa protobuf

### Firewall
- ✅ Config transport parseado
- ✅ Decrypt/decompress implementado (ayer)
- ⚠️ Token HARDCODED (debe obtener de etcd)

### RAG
- ❓ Pendiente certificación

## ARCHIVOS CLAVE DE REFERENCIA
- `/vagrant/etcd-client/src/crypto.cpp` - Código a extraer
- `/vagrant/sniffer/src/userspace/compression_handler.cpp` - Ref LZ4
- `/vagrant/firewall-acl-agent/src/api/zmq_subscriber.cpp` - Decrypt/decompress
- `/vagrant/ml-detector/config/ml_detector_config.json` - Config transport

## OBJETIVO DÍA 24
✅ FASE 1 completa: crypto-transport compilando con tests pasando
🎯 Empezar FASE 2: Refactorizar etcd-client

## PRINCIPIOS GUÍA
- "Despacio y bien" - Sin prisas
- Single Responsibility Principle
- Composición sobre acoplamiento
- Via Appia Quality - construir para durar décadas
- JSON is law
- Fail fast

¿Listo para empezar con FASE 1?
Day 23 - Sesión de Testing

CONTEXTO:
Ayer (24 dic) implementamos completamente la integración de
decrypt/decompress en firewall-acl-agent. El código compila
correctamente y está listo para testing.

ESTADO ACTUAL:
✅ Código implementado (decrypt_chacha20_poly1305 + decompress_lz4)
✅ CMake configurado (LZ4 + OpenSSL linkeados)
✅ JSON configurado (transport.encryption/compression.enabled = true)
✅ Compilación exitosa (2.8MB binary)
⏳ Pendiente: Testing con etcd-server

OBJETIVO DE HOY:
1. Arrancar etcd-server
2. Verificar que firewall muestra logs de transport config
3. Test de pipeline completo: sniffer → detector → firewall
4. Verificar decrypt/decompress en tiempo real

COMANDOS PARA EMPEZAR:
```bash
# 1. Arrancar etcd-server
make etcd-server-start
sleep 2

# 2. Verificar health
vagrant ssh -c "curl http://localhost:2379/health"

# 3. Arrancar firewall y verificar logs
vagrant ssh -c "cd /vagrant/firewall-acl-agent/build && \
  ./firewall-acl-agent -c ../config/firewall.json 2>&1 | head -50"

# Deberías ver:
# [INIT] 📦 LZ4 decompression ENABLED
# [INIT] 🔐 ChaCha20-Poly1305 decryption ENABLED

# 4. Pipeline completo
make run-lab-dev-day23
```

ARCHIVOS MODIFICADOS AYER:
- firewall-acl-agent/include/firewall/zmq_subscriber.hpp
- firewall-acl-agent/src/api/zmq_subscriber.cpp
- firewall-acl-agent/include/firewall/config_loader.hpp
- firewall-acl-agent/src/core/config_loader.cpp
- firewall-acl-agent/src/main.cpp
- firewall-acl-agent/CMakeLists.txt

NOTA IMPORTANTE:
El token está hardcoded temporalmente. En producción debe
obtenerse de etcd. Línea ~354 en main.cpp.
```markdown
# 🚀 ML Defender - Phase 1, Day 1: Protobuf Schema Update

## 📍 CONTEXTO ACTUAL

**Proyecto:** ML Defender - Sistema de seguridad de red con ML embebido en C++20
**Estado:** Phase 0 COMPLETADA ✅
**Hoy:** Phase 1, Day 1 - Integración sniffer-eBPF con ml-detector

### Phase 0 - Logros:
- ✅ 4 detectores C++20 embebidos integrados y testeados
- ✅ Ransomware: 1.06μs latency, 100 trees, 3,764 nodes
- ✅ DDoS: 0.24μs latency, 100 trees, 612 nodes
- ✅ Traffic: 0.37μs latency, 100 trees, 1,014 nodes (Internet vs Internal)
- ✅ Internal: 0.33μs latency, 100 trees, 940 nodes (Lateral Movement)
- ✅ Todos los tests unitarios pasando
- ✅ Makefile del host validado
- ✅ Config JSON con fail-fast validation

### Arquitectura Actual:
```
sniffer-eBPF → [protobuf] → ml-detector (4 detectores) → Alert
↑
└── NECESITA ACTUALIZACIÓN HOY
```

## 🎯 OBJETIVO DEL DÍA

**Actualizar protobuf schema** con las features necesarias para los 4 detectores del ml-detector.

**Criterio de éxito:** 
- Protobuf regenerado correctamente
- Sniffer compila sin errores
- ml-detector compila sin errores
- NO es necesario que funcione end-to-end (eso es para días siguientes)

## 📋 FEATURES POR DETECTOR

### Level 2 - DDoS (10 features):
1. syn_ack_ratio
2. packet_symmetry
3. source_ip_dispersion
4. protocol_anomaly_score
5. packet_size_entropy
6. traffic_amplification_factor
7. flow_completion_rate
8. geographical_concentration
9. traffic_escalation_rate
10. resource_saturation_score

### Level 2 - Ransomware (10 features):
1. io_intensity
2. entropy
3. resource_usage
4. network_activity
5. file_operations
6. process_anomaly
7. temporal_pattern
8. access_frequency
9. data_volume
10. behavior_consistency

### Level 3 - Traffic (10 features):
1. packet_rate
2. connection_rate
3. tcp_udp_ratio
4. avg_packet_size
5. port_entropy
6. flow_duration_std
7. src_ip_entropy
8. dst_ip_concentration
9. protocol_variety
10. temporal_consistency

### Level 3 - Internal (10 features):
1. internal_connection_rate
2. service_port_consistency
3. protocol_regularity
4. packet_size_consistency
5. connection_duration_std
6. lateral_movement_score
7. service_discovery_patterns
8. data_exfiltration_indicators
9. temporal_anomaly_score
10. access_pattern_entropy

## 📂 ARCHIVOS RELEVANTES

```bash
/vagrant/protobuf/network_security.proto  # Actualizar este
/vagrant/protobuf/generate.sh             # Regenerar con este
/vagrant/sniffer/                         # Recompilar después
/vagrant/ml-detector/                     # Recompilar después
```

## 🔧 COMANDOS INICIALES

```bash
# En el HOST (macOS):
cd ~/path/to/test-zeromq-docker

# Verificar estado
vagrant status
make status

# Si VM apagada:
vagrant up

# Empezar trabajo
vagrant ssh
cd /vagrant/protobuf

# Backup del schema actual
cp network_security.proto network_security.proto.backup_phase0

# Ver estructura actual
grep -A 50 "message NetworkFeatures" network_security.proto
```

## 🏛️ FILOSOFÍA VIA APPIA

- **Día a día:** Solo el protobuf hoy, integración mañana
- **KISS:** Añadir campos necesarios, nada más
- **Funciona > Perfecto:** Que compile es suficiente
- **Smooth & Fast:** No optimizar prematuramente

## ❓ PREGUNTAS PARA CLAUDE

1. ¿Dónde en el protobuf actual debo añadir las nuevas features?
2. ¿Cómo estructurar los mensajes para los 4 detectores?
3. ¿Algún campo existente puedo reutilizar o necesito todos nuevos?
4. Ayúdame a actualizar el .proto y regenerarlo
5. Si hay errores de compilación, ayúdame a resolverlos

## 📌 NOTAS IMPORTANTES

- Estamos en rama: `feature/sniffer-ebpf-integration` (o crear si no existe)
- El sniffer NO tiene que extraer las features aún (eso es Day 2-3)
- Solo necesitamos que el schema exista y compile
- El ml-detector ya tiene los extractores (feature_extractor.cpp)

---

**Ready to start Phase 1!** 🚀
```

---

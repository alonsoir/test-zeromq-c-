CONTEXTO PHASE 1 DAY 8 COMPLETO (Dec 4, 2025)

ESTADO ACTUAL:
✅ Dual-NIC metadata pipeline VALIDADO end-to-end
✅ [DUAL-NIC] logs confirman: ifindex=3, mode=1, wan=1, iface=if03
✅ 43 eventos capturados con metadatos completos (59.63μs avg latency)
✅ libbpf 1.4.6: Bug de struct map loading RESUELTO
✅ iface_configs BPF map operacional (FD: 4)
✅ Host-based IDS: 130K+ eventos validados (Days 1-7)
✅ Rama actual: feature/day8-mawi-gateway-validation

BLOQUEADOR CRÍTICO RESUELTO:
- Bug original: libbpf 1.1.2 no cargaba mapas con structs complejos
- Solución: Upgrade a libbpf 1.4.6
- Validación: bpftool map dump + [DUAL-NIC] logs en userspace
- Pipeline completo: eBPF ctx->ingress_ifindex → iface_configs lookup → ring buffer → protobuf

ARQUITECTURA CONFIRMADA:
- Host-based IDS (eth1): FUNCIONAL (macOS → 192.168.56.20)
- Gateway mode (eth3): PENDIENTE validación con MAWI
- Configuración dual-NIC: eth1=host-based/WAN, eth3=gateway/LAN

PRÓXIMA PRIORIDAD - DAY 9:
Gateway Mode Validation con MAWI Dataset

OBJETIVO:
1. Configurar recap relay: MAWI PCAP → eth3 (LAN interface)
2. Validar que eth3 captura tráfico de tránsito (no solo local)
3. Verificar [DUAL-NIC] ifindex=5, mode=2 (GATEWAY), wan=0
4. Benchmark: eventos/seg, latencia, drops

DATASET MAWI:
- Ubicación: /vagrant/datasets/mawi/mawi-ready.pcap
- Características: Tráfico real backbone japonés, ~100MB
- Preparación: Ya procesado (snaplen completo, sin truncar)

CONFIGURACIÓN ACTUAL:
- Sniffer attached solo a eth1 (línea ~XXX en código)
- Necesita: Attach también a eth3 para gateway mode
- IP forwarding: YA habilitado (IPv4 + IPv6)
- NAT: Deshabilitado en config (correcto para gateway IDS)

HERRAMIENTAS:
- tcpreplay: Para replay PCAP → eth3
- bpftool: Verificar attachment y stats
- Logs DEBUG: [DUAL-NIC] mantener para validación

PREGUNTAS A RESOLVER:
1. ¿El código actual attach XDP a ambas interfaces o solo eth1?
2. ¿Necesitamos modificar DualNICManager para dual-attach?
3. ¿tcpreplay necesita configuración especial para gateway testing?

ARCHIVOS CLAVE:
- /vagrant/sniffer/src/userspace/dual_nic_manager.cpp
- /vagrant/sniffer/src/kernel/sniffer.bpf.c (XDP attachment)
- /vagrant/sniffer/config/sniffer.json (deployment mode: dual)

VALIDACIÓN ESPERADA:
- Logs: [DUAL-NIC] ifindex=5 mode=2 wan=0 iface=if05
- Stats BPF: Incremento en ambos interfaces
- Events processed: Debe capturar paquetes replay MAWI

DOCUMENTACIÓN PENDIENTE:
- Day 8 postmortem: libbpf bug analysis
- MAWI testing methodology
- Gateway mode architecture diagram

Este es el estado completo. ¿Por dónde empezamos con MAWI?

Hay un problema temporal con la conectividad de Claude.

vagrant@bookworm:/vagrant/sniffer$ grep -n "DualNICManager" src/userspace/main.cpp | head -10 355:            sniffer::DualNICManager dual_nic_manager(json_root); vagrant@bookworm:/vagrant/sniffer$
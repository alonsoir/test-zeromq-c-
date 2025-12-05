# CONTINUIDAD DAY 10 - ML Defender Gateway Mode Validation

**Date**: December 6, 2025 (mañana)  
**Objetivo**: Validar gateway mode con tráfico real usando Vagrant multi-machine  
**Contexto**: Day 9 completado - dual XDP attachment funcional, falta validación con tráfico externo

---

## 📋 ESTADO AL CIERRE DE DAY 9

### ✅ COMPLETADO

**Dual-NIC Implementation**:
- Dual XDP attachment a eth1 + eth3 ✅
- BPF iface_configs map configurada correctamente ✅
- Host-based IDS validado (100+ eventos, 59.63μs latency) ✅
- Código production-ready ✅

**Hallazgos Científicos**:
- XDP Generic NO captura tráfico local/namespace/bridge ✅
- XDP Generic SOLO captura tráfico físico externo ✅
- Estrategia de validación definida: multi-machine setup ✅

### 📂 ARCHIVOS CLAVE

```
/vagrant/sniffer/
├── POSTMORTEM_DUAL_NIC_DAY9.md    ← Documentación completa Day 9
├── src/userspace/
│   ├── ebpf_loader.cpp             ← Multi-interface support
│   ├── main.cpp                    ← Dual attachment logic
│   └── dual_nic_manager.cpp        ← Config management
├── include/
│   └── ebpf_loader.hpp             ← Vector de ifindexes
└── build/
    └── sniffer                     ← Binary compilado

/vagrant/Vagrantfile                ← A MODIFICAR mañana
```

### 🎯 CONFIGURACIÓN ACTUAL

**Defender VM (eth3)**:
```bash
# Verificar attachment:
sudo bpftool net show
# Esperado:
# xdp:
# eth1(3) generic id 22
# eth3(5) generic id 22

# Verificar BPF map:
sudo bpftool map dump name iface_configs
# Esperado:
# key=3: mode=1 (HOST_BASED), wan=1
# key=5: mode=2 (GATEWAY), wan=0
```

**Namespace creado (puede limpiarse)**:
```bash
# Si existe, limpiar:
sudo ip netns del client 2>/dev/null
sudo ip link del veth-host 2>/dev/null
sudo ip link del br-lan 2>/dev/null
```

---

## 🚀 PLAN DAY 10 - GATEWAY VALIDATION

### FASE 1: Vagrant Multi-Machine Setup (30 min)

**Objetivo**: Crear Client VM que genere tráfico real hacia Defender.

**Modificar `/vagrant/Vagrantfile`**:

```ruby
Vagrant.configure("2") do |config|
  
  # ============================================================================
  # DEFENDER VM - ML Defender con Dual-NIC
  # ============================================================================
  config.vm.define "defender", primary: true do |defender|
    defender.vm.box = "debian/bookworm64"
    defender.vm.hostname = "ml-defender"
    
    # eth1: WAN (host-only) - 192.168.56.20
    defender.vm.network "private_network", 
                        ip: "192.168.56.20",
                        name: "vboxnet0"
    
    # eth2: Captura externa (bridge) - opcional
    defender.vm.network "public_network",
                        bridge: "en0: Wi-Fi"
    
    # eth3: LAN (internal network) - 192.168.100.1
    defender.vm.network "private_network",
                        ip: "192.168.100.1",
                        virtualbox__intnet: "ml_defender_lan"
    
    defender.vm.provider "virtualbox" do |vb|
      vb.name = "ML-Defender-Dual-NIC"
      vb.memory = "4096"
      vb.cpus = 4
      vb.customize ["modifyvm", :id, "--nicpromisc2", "allow-all"]  # eth1
      vb.customize ["modifyvm", :id, "--nicpromisc3", "allow-all"]  # eth2
      vb.customize ["modifyvm", :id, "--nicpromisc4", "allow-all"]  # eth3
    end
    
    # Provisioning (mantener el actual)
    defender.vm.provision "shell", path: "scripts/provision.sh"
  end
  
  # ============================================================================
  # CLIENT VM - Traffic Generator para Gateway Testing
  # ============================================================================
  config.vm.define "client", autostart: false do |client|
    client.vm.box = "debian/bookworm64"
    client.vm.hostname = "ml-client"
    
    # eth1: Conectado a internal network "ml_defender_lan"
    client.vm.network "private_network",
                      ip: "192.168.100.50",
                      virtualbox__intnet: "ml_defender_lan"
    
    client.vm.provider "virtualbox" do |vb|
      vb.name = "ML-Defender-Client"
      vb.memory = "512"   # Mínimo necesario
      vb.cpus = 1
    end
    
    # Provisioning simple
    client.vm.provision "shell", inline: <<-SHELL
      echo "🔧 Configurando ML Defender Client..."
      
      # Default gateway apunta al Defender
      ip route add default via 192.168.100.1 || true
      
      # Herramientas básicas
      apt-get update
      apt-get install -y curl wget tcpdump netcat-openbsd dnsutils
      
      # Test de conectividad
      ping -c 3 192.168.100.1 || echo "⚠️  Gateway no responde aún"
      
      echo "✅ Client configurado"
      echo "   IP: 192.168.100.50"
      echo "   Gateway: 192.168.100.1 (Defender)"
    SHELL
  end
end
```

**Levantar Client VM**:
```bash
cd /vagrant
vagrant up client  # Solo levanta el client
```

### FASE 2: Validación de Conectividad (10 min)

**Desde Client VM**:
```bash
vagrant ssh client

# 1. Verificar IP y rutas
ip addr show
ip route show

# 2. Ping al gateway (Defender)
ping -c 5 192.168.100.1

# 3. Verificar que NO hay ruta directa a Internet (debe pasar por Defender)
ip route get 8.8.8.8
# Esperado: via 192.168.100.1 dev eth1
```

**Desde Defender VM** (en paralelo):
```bash
vagrant ssh defender

# Verificar IP forwarding
cat /proc/sys/net/ipv4/ip_forward  # Debe ser 1

# Verificar iptables
sudo iptables -L -n -v | grep FORWARD
# Debe ser: Chain FORWARD (policy ACCEPT ...)

# tcpdump en eth3 para ver tráfico del client
sudo tcpdump -i eth3 -n icmp
```

### FASE 3: Gateway Mode Validation (15 min)

**En Defender**: Arrancar sniffer
```bash
cd /vagrant/sniffer/build
sudo ./sniffer -c ../config/sniffer.json
```

**Esperado en logs**:
```
✅ eBPF program attached to interface: eth1
✅ eBPF program attached to interface: eth3
🚀 Sniffer running...
```

**En Client**: Generar tráfico
```bash
# Test 1: Ping a Internet
ping -c 10 8.8.8.8

# Test 2: HTTP requests
curl http://example.com
curl http://1.1.1.1

# Test 3: DNS queries
nslookup google.com
dig @8.8.8.8 anthropic.com

# Test 4: Continuous traffic
for i in {1..20}; do curl -s http://example.com > /dev/null; sleep 0.5; done
```

**OBSERVAR LOGS DEL SNIFFER**:

✅ **SUCCESS CRITERIA**:
```
[DUAL-NIC] ifindex=5 mode=2 wan=0 iface=if05  ← ¡GATEWAY MODE!
[DUAL-NIC] ifindex=3 mode=1 wan=1 iface=if03  ← Host-based (SSH)

=== ESTADÍSTICAS ===
Paquetes procesados: 50+ (incrementando)
```

### FASE 4: Performance Benchmarking (20 min)

**Métricas a capturar**:

1. **Throughput**:
```bash
# Desde client, iperf3 si está disponible, o:
dd if=/dev/zero bs=1M count=100 | curl -T - http://example.com/upload
```

2. **Packet rate**:
```bash
# Desde client:
sudo hping3 -c 1000 --fast 8.8.8.8
```

3. **Latency**:
```bash
# Observar en logs del sniffer la latencia promedio
# grep "Latency" en los logs
```

4. **Dual-mode simultaneous**:
```bash
# Terminal 1 (client): Generar tráfico
ping 8.8.8.8

# Terminal 2 (Mac): SSH al defender
ssh vagrant@192.168.56.20

# Verificar que AMBOS ifindex aparecen en logs:
# ifindex=3 (SSH desde Mac)
# ifindex=5 (ping desde client)
```

### FASE 5: MAWI Dataset Test (OPCIONAL, 15 min)

**Si todo lo anterior funciona**:

```bash
# Desde client:
# Copiar MAWI (via shared folder si está configurado)
sudo tcpreplay -i eth1 --pps=100 /path/to/mawi-ready.pcap

# Observar estadísticas en Defender sniffer
```

---

## 📊 MÉTRICAS ESPERADAS

### Success Thresholds

| Metric                    | Target           | Acceptable    |
|---------------------------|------------------|---------------|
| Gateway events (ifindex=5)| >0               | Any           |
| Host events (ifindex=3)   | >0               | Any           |
| Dual-mode simultaneous    | Yes              | Yes           |
| Packet loss               | 0%               | <1%           |
| Latency (μs)              | <100             | <500          |
| Throughput                | >10 Mbps         | >1 Mbps       |

### Data to Collect

1. **Event Distribution**:
    - Count de eventos ifindex=3 vs ifindex=5
    - Ratio host-based / gateway

2. **Performance**:
    - Latencia promedio y p95
    - Paquetes procesados por segundo
    - CPU usage en Defender

3. **Correctness**:
    - Metadata correcta en eventos ifindex=5
    - BPF map lookup funcional
    - Sin packet drops

---

## 🐛 TROUBLESHOOTING PREVISTO

### Problema 1: Client no puede hacer ping a gateway

**Síntomas**: `ping 192.168.100.1` falla desde client

**Diagnóstico**:
```bash
# En Defender:
sudo tcpdump -i eth3 -n icmp  # ¿Ve los ICMP requests?

# En Client:
ip route get 192.168.100.1    # ¿Ruta correcta?
```

**Fix**:
```bash
# En Defender, verificar que eth3 responda:
sudo sysctl -w net.ipv4.conf.eth3.arp_accept=1
sudo sysctl -w net.ipv4.conf.eth3.proxy_arp=1
```

### Problema 2: Gateway funciona pero XDP no captura

**Síntomas**: Ping funciona pero no hay eventos ifindex=5

**Diagnóstico**:
```bash
# Verificar XDP attachment
sudo bpftool net show
# Debe mostrar eth3(5)

# Verificar BPF map
sudo bpftool map dump name iface_configs
# key=5 debe existir con mode=2
```

**Fix**: Re-attachar XDP
```bash
sudo pkill sniffer
cd /vagrant/sniffer/build
sudo ./sniffer -c ../config/sniffer.json
```

### Problema 3: Solo captura en una dirección

**Síntomas**: Ve request pero no reply (o viceversa)

**Análisis**: XDP captura en ingress, no en egress. Esto es **esperado**.

**Solución**: Agregar XDP en egress (TC-BPF) si se requiere captura bidireccional completa.

---

## 📝 CHECKLIST DAY 10

### Pre-flight

- [ ] Leer este prompt completo
- [ ] Revisar POSTMORTEM_DUAL_NIC_DAY9.md
- [ ] Verificar que Defender VM está limpia (no namespace residual)
- [ ] Backup del Vagrantfile actual

### Setup

- [ ] Modificar Vagrantfile con client VM
- [ ] `vagrant up client`
- [ ] Verificar client tiene IP 192.168.100.50
- [ ] Verificar client puede ping a 192.168.100.1

### Validation

- [ ] Arrancar sniffer en Defender
- [ ] Verificar dual XDP attachment (bpftool)
- [ ] Generar tráfico desde client
- [ ] **CONFIRMAR**: Logs muestran ifindex=5 ✅
- [ ] **CONFIRMAR**: Logs muestran ambos ifindex=3 e ifindex=5 simultáneos ✅

### Documentation

- [ ] Capturar screenshots de logs con ifindex=5
- [ ] Anotar métricas de performance
- [ ] Documentar cualquier issue encontrado
- [ ] Actualizar README con multi-machine setup

---

## 🎯 EXIT CRITERIA DAY 10

**Mínimo** (Must have):
- ✅ Al menos 1 evento capturado con ifindex=5
- ✅ Metadata correcta: mode=2, wan=0
- ✅ Dual-mode funcional simultáneamente

**Target** (Should have):
- ✅ 100+ eventos de gateway mode
- ✅ Performance acceptable (<500μs latency)
- ✅ Zero packet drops
- ✅ Documentation actualizada

**Stretch** (Nice to have):
- ✅ MAWI dataset procesado en gateway mode
- ✅ Benchmark comparativo host-based vs gateway
- ✅ Dashboard con métricas por interface

---

## 💬 PROMPT PARA CLAUDE MAÑANA

```
Hola Claude, vamos con Day 10 - Gateway Mode Validation.

Contexto rápido:
- Day 9: Implementamos dual XDP attachment (✅ funcional)
- Problema: XDP Generic no captura tráfico local/namespace
- Solución: Vagrant multi-machine con client VM real
- Objetivo hoy: Validar gateway mode con tráfico externo físico

Estado actual:
- Defender VM: Dual XDP attached (eth1 + eth3)
- BPF map: Correcta (verificado con bpftool)
- Host-based: Validado (100+ eventos)
- Gateway mode: Código listo, falta validación

Plan:
1. Modificar Vagrantfile para client VM
2. Levantar client (192.168.100.50)
3. Generar tráfico: client → defender → Internet
4. VALIDAR: Logs muestran ifindex=5 (gateway mode)

Archivos relevantes:
- /vagrant/sniffer/POSTMORTEM_DUAL_NIC_DAY9.md (contexto completo)
- /vagrant/Vagrantfile (a modificar)
- /vagrant/sniffer/build/sniffer (binary listo)

¿Empezamos con el Vagrantfile?
```

---

**Documentación completa**: `/vagrant/sniffer/POSTMORTEM_DUAL_NIC_DAY9.md`  
**Rama activa**: `feature/day9-dual-xdp-attachment`  
**Binary compilado**: `/vagrant/sniffer/build/sniffer`  
**Config**: `/vagrant/sniffer/config/sniffer.json` (deployment: dual)

**Via Appia Quality**: Day 9 documentado honestamente. Day 10 con plan claro y métricas definidas. 🏛️

---

*Preparado para mañana con contexto completo y estrategia clara.*  
*Next: Multi-machine validation → Gateway mode confirmed → Phase 1 complete.* 🚀
## 3️⃣ README.debian.md - 
```markdown
# eBPF Network Sniffer - Paquete Debian

Paquete `.deb` para instalación del sniffer eBPF con XDP y ML features.

## 📦 Instalación
```bash
    sudo dpkg -i sniffer-ebpf_0.83.0-1_amd64.deb
```
### 📁 Ubicación de Archivos
```bash
Binario: /usr/bin/sniffer_ebpf
Objeto eBPF: /usr/lib/bpf/sniffer.bpf.o
Configuración: /etc/sniffer-ebpf/config.json
Servicio systemd: /lib/systemd/system/sniffer-ebpf.service
Documentación: /usr/share/doc/sniffer-ebpf/
```

### 🚀 Inicio Rápido

# 1. Validar configuración
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json --dry-run
```

# 2. Probar manualmente
# Sin verbosity (solo estadísticas)
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json
```

# Con resumen básico
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v
```

# Con features agrupadas
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vv
```

# Con dump completo (guardar en archivo)
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vvv > /tmp/features.log 2>&1
```

# 3. Iniciar servicio
```bash
    sudo systemctl start sniffer-ebpf
    sudo systemctl status sniffer-ebpf
    sudo journalctl -u sniffer-ebpf -f
```

### 🔍 Opciones de Línea de Comandos

| Opción | Descripción |
|--------|-------------|
| `-c, --config FILE` | Archivo de configuración JSON (obligatorio) |
| `-v, --verbose` | Nivel 1: Resumen básico por paquete |
| `-vv` | Nivel 2: Features agrupadas por categoría |
| `-vvv` | Nivel 3: Dump completo de todas las features |
| `-i, --interface IFACE` | Override de interfaz de red |
| `-p, --profile PROFILE` | Override de perfil (lab/cloud/bare_metal) |
| `-d, --dry-run` | Solo validar JSON sin ejecutar |
| `-s, --show-config` | Mostrar configuración y salir |
| `-h, --help` | Mostrar ayuda |

### 🎯 Niveles de Verbosity Detallados
# Nivel 0: Sin Verbosity (Producción)
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json

    Solo estadísticas cada 30 segundos
    Máximo rendimiento
    Uso: Entorno de producción
    Output:
    === ESTADÍSTICAS ===
    Paquetes procesados: 1250
    Paquetes enviados: 1250
    Tiempo activo: 30 segundos
    Tasa: 41.67 eventos/seg
    ===================
```
# Nivel 1: Resumen Básico (-v)
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v


    Una línea por paquete
    Información básica: protocolo, IPs, puertos, tamaño
    Uso: Monitoreo en tiempo real, verificación de captura
    
    Output:
    
    [PKT #123] TCP 192.168.1.1:443 → 10.0.0.5:50123 1460B flags:ACK PSH
    [PKT #124] UDP 192.168.1.10:53 → 10.0.0.5:39845 128B
    [PKT #125] TCP 192.168.1.1:443 → 10.0.0.5:50123 60B flags:ACK
```

# Nivel 2: Features Agrupadas (-vv)
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vv

    Features organizadas por sección
    Info básica, timing, rates, TCP flags, arrays de features
    Uso: Debugging del pipeline, validación de extracción
    
    Output: ~15-20 líneas por paquete (ver README.md para ejemplo completo)
    Nivel 3: Dump Completo (-vvv)
 ```
```bash
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vvv > /tmp/full_features.log 2>&1

    TODAS las ~193 features con índice y valor
    Campos individuales + 4 arrays de features ML
    Uso: Análisis exhaustivo, training de modelos, documentación
    
    Output: ~200-300 líneas por paquete
    Estructura del dump completo:
    
    Basic Identification (Event ID, Node, Timestamp)
    Network Features (50+ campos individuales)
    Packet Statistics
    Forward/Backward Packet Length
    Inter-Arrival Times (Flow, Forward, Backward)
    TCP Flags (si es TCP)
    Headers & Bulk Transfer
    General Attack Features: 23 features (Random Forest)
    Internal Traffic Features: 4 features
    Ransomware Detection Features: 83 features
    DDoS Detection Features: 83 features
```

### 💡 Ejemplos Prácticos
# Debugging de Pipeline
# Ver features extraídas en tiempo real
```bash
  sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vv | less -R
```
# Buscar paquetes específicos
```bash
  sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v | grep "192.168.1.1"
```
# Solo paquetes TCP con flags
```bash
  sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v | grep "SYN"
```

### Análisis de Features ML
```bash
    # Capturar 5 minutos de features completas
    timeout 300 sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vvv \
      > features_$(date +%Y%m%d_%H%M%S).log 2>&1
    
    # Contar paquetes por protocolo
    grep "protocol_name" features.log | sort | uniq -c
    
    # Extraer features de ransomware
    grep -A 83 "RANSOMWARE DETECTION FEATURES" features.log > ransomware_features.txt
```

### Validación de Datos

```bash
    # Verificar que captura en eth2
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v | head -20
    
    # Ver distribución de tamaños de paquetes
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v | \
      awk '{print $NF}' | sed 's/B//' | sort -n | uniq -c
```

### ⚙️ Configuración de Interfaz de Red
```bash
    Este paquete está pre-configurado para el laboratorio Vagrant con:
    
    Interfaz: eth2
    Perfil: lab
    Endpoint: 172.18.0.3:5571
    
    Para instalación en bare metal
    Si instalas en hardware real, edita /etc/sniffer-ebpf/config.json:
```

```bash
    sudo nano /etc/sniffer-ebpf/config.json
```
Cambiar:

{
"profiles": {
    "active": "bare_metal",
    "bare_metal": {
        "description": "Bare metal production",
        "interface": "eth0"  // ← Cambiar a tu interfaz
    }
    }
}

### Reiniciar servicio:
```bash
    sudo systemctl restart sniffer-ebpf
```

### 🔧 Troubleshooting
# Sniffer no arranca
```bash
    # Ver logs detallados
    sudo journalctl -u sniffer-ebpf -n 100 --no-pager
    
    # Verificar archivo eBPF
    ls -lh /usr/lib/bpf/sniffer.bpf.o
    
    # Verificar permisos eBPF
    sudo sysctl kernel.unprivileged_bpf_disabled
```
# No captura paquetes
```bash
    # Verificar interfaz
    ip link show
    
    # Probar manualmente con verbosity
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v
    
    # Verificar que XDP se adjunta
    sudo bpftool prog show
```
# Ver qué features se capturan

```bash
    # Nivel 2 es ideal para debugging
    sudo systemctl stop sniffer-ebpf
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -vv | less -R
```

### 📊 Monitoreo y Logs
# Ver estadísticas en tiempo real
```bash
    # Con systemd
    sudo journalctl -u sniffer-ebpf -f | grep "ESTADÍSTICAS"
    
    # Con verbosity básico
    sudo sniffer_ebpf -c /etc/sniffer-ebpf/config.json -v
```

### Casos de Uso

| Nivel | Caso de Uso | Output/Línea | Rendimiento |
|-------|-------------|--------------|-------------|
| 0 (sin -v) | Producción | Solo estadísticas cada 30s | Máximo |
| 1 (-v) | Monitoreo | 1 línea/paquete (~80 chars) | Alto |
| 2 (-vv) | Debugging | ~15-20 líneas/paquete | Medio |
| 3 (-vvv) | Análisis ML | ~200-300 líneas/paquete | Bajo |


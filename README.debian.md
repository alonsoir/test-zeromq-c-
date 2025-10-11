## Configuración de Interfaz de Red

Este paquete está **pre-configurado para el laboratorio Vagrant** con:
- **Interfaz**: `eth2`
- **Perfil**: `lab`
- **Endpoint**: `172.18.0.3:5571`

### Para instalación en bare metal

Si instalas en hardware real, edita `/etc/sniffer-ebpf/config.json`:
```bash
sudo nano /etc/sniffer-ebpf/config.json
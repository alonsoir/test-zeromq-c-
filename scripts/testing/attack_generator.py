#!/usr/bin/env python3
"""
ML Defender - Synthetic Attack Traffic Generator
Genera tráfico malicioso sintético para probar el pipeline completo
"""

import socket
import time
import random
import argparse
import sys
from datetime import datetime

class AttackGenerator:
    """Generador de tráfico de ataque sintético"""

    def __init__(self, target_ip: str = "127.0.0.1", verbose: bool = True):
        self.target_ip = target_ip
        self.verbose = verbose
        self.stats = {
            'ddos_sent': 0,
            'port_scans': 0,
            'suspicious_connections': 0,
            'errors': 0
        }

    def log(self, message: str):
        """Log con timestamp"""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def simulate_ddos_flood(self, duration_seconds: int = 10, packets_per_second: int = 100):
        """
        Simula un ataque DDoS flood
        - Muchos paquetes desde misma IP
        - Alta frecuencia de conexiones
        """
        self.log(f"🚨 [ATTACK] Starting DDoS flood simulation...")
        self.log(f"   Duration: {duration_seconds}s | Rate: {packets_per_second} pps")

        start_time = time.time()
        packets_sent = 0

        # Puertos aleatorios para simular flood
        ports = [80, 443, 8080, 3306, 5432, 27017]

        while time.time() - start_time < duration_seconds:
            try:
                # Crear socket TCP
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)

                # Intentar conexión a puerto aleatorio
                port = random.choice(ports)
                try:
                    sock.connect((self.target_ip, port))
                    sock.send(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
                except (socket.error, socket.timeout):
                    pass  # Ignorar errores de conexión (esperado)

                sock.close()
                packets_sent += 1
                self.stats['ddos_sent'] += 1

                # Control de rate
                time.sleep(1.0 / packets_per_second)

                if packets_sent % 50 == 0:
                    self.log(f"   📊 DDoS packets sent: {packets_sent}")

            except Exception as e:
                self.stats['errors'] += 1
                if self.verbose:
                    self.log(f"   ⚠️  Error: {e}")

        elapsed = time.time() - start_time
        actual_rate = packets_sent / elapsed
        self.log(f"✅ DDoS simulation complete: {packets_sent} packets in {elapsed:.1f}s ({actual_rate:.1f} pps)")

    def simulate_port_scan(self, start_port: int = 1, end_port: int = 1024, delay: float = 0.01):
        """
        Simula un port scan (patrón de ransomware/reconnaissance)
        - Escanea rango de puertos secuencialmente
        - Conexiones rápidas sin datos
        """
        self.log(f"🚨 [ATTACK] Starting port scan simulation...")
        self.log(f"   Port range: {start_port}-{end_port} | Delay: {delay}s")

        scanned = 0
        for port in range(start_port, end_port + 1):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)

                # Intentar conexión
                result = sock.connect_ex((self.target_ip, port))
                sock.close()

                scanned += 1
                self.stats['port_scans'] += 1

                if scanned % 100 == 0:
                    self.log(f"   📊 Ports scanned: {scanned}/{end_port - start_port + 1}")

                time.sleep(delay)

            except Exception as e:
                self.stats['errors'] += 1
                if self.verbose and scanned % 100 == 0:
                    self.log(f"   ⚠️  Error on port {port}: {e}")

        self.log(f"✅ Port scan complete: {scanned} ports scanned")

    def simulate_suspicious_traffic(self, duration_seconds: int = 10, connections_per_second: int = 5):
        """
        Simula tráfico sospechoso genérico
        - Conexiones a puertos no estándar
        - Patrones irregulares
        """
        self.log(f"🚨 [ATTACK] Starting suspicious traffic simulation...")
        self.log(f"   Duration: {duration_seconds}s | Rate: {connections_per_second} cps")

        start_time = time.time()
        connections = 0

        # Puertos no estándar sospechosos
        suspicious_ports = [4444, 5555, 6666, 7777, 8888, 9999, 31337, 12345]

        while time.time() - start_time < duration_seconds:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)

                port = random.choice(suspicious_ports)
                try:
                    sock.connect((self.target_ip, port))
                    # Enviar datos sospechosos
                    sock.send(b"\x00\x01\x02\x03" + b"X" * random.randint(10, 100))
                except (socket.error, socket.timeout):
                    pass

                sock.close()
                connections += 1
                self.stats['suspicious_connections'] += 1

                if connections % 10 == 0:
                    self.log(f"   📊 Suspicious connections: {connections}")

                time.sleep(1.0 / connections_per_second)

            except Exception as e:
                self.stats['errors'] += 1

        self.log(f"✅ Suspicious traffic complete: {connections} connections")

    def simulate_mixed_attack(self, duration_seconds: int = 30):
        """
        Simula un ataque mezclado (más realista)
        - Combina DDoS, port scan y tráfico sospechoso
        - Intensidad variable
        """
        self.log(f"🚨 [ATTACK] Starting MIXED attack simulation...")
        self.log(f"   Duration: {duration_seconds}s")
        self.log(f"   ⚡ This will trigger multiple detection types!")

        start_time = time.time()
        phase = 0

        while time.time() - start_time < duration_seconds:
            elapsed = time.time() - start_time

            # Fase 1: Port scan rápido (primeros 10s)
            if elapsed < 10 and phase == 0:
                self.log(f"\n📍 Phase 1: Port Reconnaissance")
                self.simulate_port_scan(start_port=20, end_port=200, delay=0.005)
                phase = 1

            # Fase 2: DDoS flood (10-20s)
            elif elapsed >= 10 and elapsed < 20 and phase == 1:
                self.log(f"\n📍 Phase 2: DDoS Flood")
                self.simulate_ddos_flood(duration_seconds=10, packets_per_second=50)
                phase = 2

            # Fase 3: Tráfico sospechoso (20-30s)
            elif elapsed >= 20 and phase == 2:
                self.log(f"\n📍 Phase 3: Suspicious Connections")
                self.simulate_suspicious_traffic(duration_seconds=10, connections_per_second=10)
                phase = 3

            time.sleep(0.1)

        self.log(f"\n✅ Mixed attack simulation complete!")
        self.print_stats()

    def print_stats(self):
        """Imprime estadísticas finales"""
        print("\n" + "="*60)
        print("📊 ATTACK SIMULATION STATISTICS")
        print("="*60)
        print(f"DDoS packets sent:         {self.stats['ddos_sent']}")
        print(f"Ports scanned:             {self.stats['port_scans']}")
        print(f"Suspicious connections:    {self.stats['suspicious_connections']}")
        print(f"Errors encountered:        {self.stats['errors']}")
        print(f"Total attack actions:      {sum([v for k, v in self.stats.items() if k != 'errors'])}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="ML Defender - Synthetic Attack Traffic Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DDoS flood attack (10 seconds, 100 pps)
  %(prog)s --attack ddos --duration 10 --rate 100
  
  # Port scan (ports 1-1000)
  %(prog)s --attack portscan --start-port 1 --end-port 1000
  
  # Mixed attack (most realistic)
  %(prog)s --attack mixed --duration 30
  
  # Suspicious traffic
  %(prog)s --attack suspicious --duration 15 --rate 10
        """
    )

    parser.add_argument(
        '--attack', '-a',
        choices=['ddos', 'portscan', 'suspicious', 'mixed'],
        default='mixed',
        help='Type of attack to simulate (default: mixed)'
    )

    parser.add_argument(
        '--target', '-t',
        default='127.0.0.1',
        help='Target IP address (default: 127.0.0.1)'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=30,
        help='Duration in seconds (default: 30)'
    )

    parser.add_argument(
        '--rate', '-r',
        type=int,
        default=50,
        help='Packets/connections per second (default: 50)'
    )

    parser.add_argument(
        '--start-port',
        type=int,
        default=1,
        help='Start port for scan (default: 1)'
    )

    parser.add_argument(
        '--end-port',
        type=int,
        default=1024,
        help='End port for scan (default: 1024)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Banner
    print("\n" + "="*60)
    print("🔥 ML Defender - Attack Traffic Generator")
    print("="*60)
    print(f"Target: {args.target}")
    print(f"Attack Type: {args.attack.upper()}")
    print(f"Duration: {args.duration}s")
    print("="*60 + "\n")

    # Crear generador
    generator = AttackGenerator(
        target_ip=args.target,
        verbose=not args.quiet
    )

    try:
        # Ejecutar ataque según tipo
        if args.attack == 'ddos':
            generator.simulate_ddos_flood(
                duration_seconds=args.duration,
                packets_per_second=args.rate
            )

        elif args.attack == 'portscan':
            generator.simulate_port_scan(
                start_port=args.start_port,
                end_port=args.end_port,
                delay=0.01
            )

        elif args.attack == 'suspicious':
            generator.simulate_suspicious_traffic(
                duration_seconds=args.duration,
                connections_per_second=args.rate
            )

        elif args.attack == 'mixed':
            generator.simulate_mixed_attack(
                duration_seconds=args.duration
            )

        # Estadísticas finales
        if not args.quiet:
            generator.print_stats()

        print("\n✅ Attack simulation completed successfully!")
        print("🔍 Check ML Defender logs for detections and blocks\n")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Attack simulation interrupted by user")
        generator.print_stats()
        return 1

    except Exception as e:
        print(f"\n❌ Error during attack simulation: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
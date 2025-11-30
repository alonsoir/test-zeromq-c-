#!/usr/bin/env python3
import zmq
import sys
sys.path.append('/vagrant/protobuf')
import network_security_pb2 as pb

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect("tcp://localhost:5572")

import time
time.sleep(0.5)  # Esperar conexión

# Crear evento fake con alta confianza
event = pb.NetworkSecurityEvent()
nf = event.network_features
nf.source_ip = "192.168.1.100"
nf.destination_ip = "10.0.0.5"
nf.source_port = 54321
nf.destination_port = 80
nf.protocol_name = "TCP"
nf.flow_packets_per_second = 15000
nf.flow_bytes_per_second = 12000000
nf.flow_duration_microseconds = 1234000

ml = event.ml_analysis
ml.attack_detected_level1 = True
ml.level1_confidence = 0.95  # Alta confianza

event.threat_category = "DDOS"

# Enviar 5 eventos fake
for i in range(5):
    nf.source_ip = f"192.168.1.{100+i}"
    socket.send(event.SerializeToString())
    print(f"✅ Sent fake DDOS detection #{i+1} from {nf.source_ip}")
    time.sleep(0.1)

print("\n🔍 Check /vagrant/logs/blocked/ for JSON and proto files")

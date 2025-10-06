#!/bin/bash
echo "=== Network Status ==="
ip -4 -br addr | grep -v '^lo'
echo ""
echo "=== Active Services ==="
sudo ss -tulpn | grep -E 'LISTEN.*:(5555|2379|2380|3000|5571)' || echo "No ZeroMQ/etcd services running"
echo ""
echo "=== eth2 Test (3 packets) ==="
timeout 3 sudo tcpdump -i eth2 -c 3 -n 2>&1 | grep -v '^tcpdump:'
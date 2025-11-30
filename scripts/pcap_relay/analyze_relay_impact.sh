#!/bin/bash
# analyze_relay_impact.sh

echo "📊 ANÁLISIS POST-RELAY - TEST 1"
echo "Período analizado: 06:56:00 - 06:57:00"

# Estadísticas del detector
echo ""
echo "🤖 DETECTOR - Estadísticas alrededor del relay:"
grep -A5 -B5 "06:56" /vagrant/logs/lab/detector.log | grep -E "(Stats|attack|confidence)"

# Comportamiento del firewall
echo ""
echo "🔥 FIREWALL - Actividad durante relay:"
grep -A3 -B3 "06:56" /vagrant/logs/lab/firewall.log | grep -E "(threat_category|confidence|attack_detected)"

# Sniffer throughput
echo ""
echo "📡 SNIFFER - Throughput durante relay:"
grep "06:56" /vagrant/logs/lab/sniffer.log
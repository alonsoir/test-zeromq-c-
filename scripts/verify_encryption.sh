#!/bin/bash
# Quick verification that encryption/compression is active

echo "🔍 Verifying Pipeline Encryption/Compression Status"
echo "═══════════════════════════════════════════════════"
echo ""

# Check configs
echo "1️⃣  Configuration Files:"
for config in /vagrant/sniffer/config/sniffer.json \
              /vagrant/ml-detector/config/ml_detector_config.json \
              /vagrant/firewall-acl-agent/config/firewall.json; do
    if [ -f "$config" ]; then
        enc=$(jq -r '.encryption_enabled // "not_set"' "$config")
        comp=$(jq -r '.compression_enabled // "not_set"' "$config")
        echo "   $(basename $config):"
        echo "      Encryption: $enc"
        echo "      Compression: $comp"
    else
        echo "   ⚠️  $config not found"
    fi
done

echo ""
echo "2️⃣  Runtime Verification (check logs for 'Encrypted' keyword):"
if [ -f /vagrant/logs/lab/detector.log ]; then
    enc_count=$(grep -c "Encrypted" /vagrant/logs/lab/detector.log 2>/dev/null || echo "0")
    comp_count=$(grep -c "Compressed" /vagrant/logs/lab/detector.log 2>/dev/null || echo "0")
    echo "   Detector log: $enc_count encrypted events, $comp_count compressed events"
else
    echo "   ⚠️  No detector logs yet (start pipeline first)"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "Expected: All configs show 'true', logs show encrypted/compressed events"
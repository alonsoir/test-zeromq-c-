#!/usr/bin/env bash
set -euo pipefail

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# 1) Detect interface
IFACE=$(ip route get 8.8.8.8 2>/dev/null | awk '/dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
if [[ -z "$IFACE" ]]; then
  IFACE=$(ip -o link show | awk -F': ' '/state UP/ && $2!="lo" {print $2; exit}')
fi
if [[ -z "$IFACE" ]]; then
  echo "ERROR: no network interface detected. Available interfaces:"
  ip link show | grep -E '^[0-9]+:' || true
  exit 1
fi

echo "=> Selected interface: $IFACE"

# 2) Update JSON config
CONFIG="sniffer/config/sniffer.json"
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found at $CONFIG"
  exit 1
fi

sed -i -E "s/\"interface\": *\"[^\"]+\"/\"interface\": \"$IFACE\"/" "$CONFIG"
echo "Updated $CONFIG with interface $IFACE"

# 3) Prepare interface
echo "Preparing interface $IFACE..."
sudo ethtool -K "$IFACE" gro off gso off tx-checksumming off rx-checksumming off tso off 2>/dev/null || true
sudo ip link set dev "$IFACE" promisc on || true

# 4) Build and run sniffer
echo "Building sniffer..."
cd sniffer/build || exit 1
make -j4 || exit 1

echo "Launching sniffer..."
exec sudo ./sniffer --config="../config/sniffer.json" --verbose
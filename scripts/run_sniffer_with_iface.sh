#!/usr/bin/env bash
# run_sniffer_with_iface.sh - Enhanced with environment detection
set -euo pipefail

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Function to detect if we're in a virtualized environment
detect_virtualized_environment() {
    local virtualized=false

    echo "🔍 Detecting execution environment..." >&2

    # 1. Check DMI product name
    if [[ -f /sys/class/dmi/id/product_name ]]; then
        product_name=$(cat /sys/class/dmi/id/product_name 2>/dev/null || echo "")
        case "$product_name" in
            "VirtualBox"|"VMware"*|"QEMU"|"KVM"|"Microsoft Corporation")
                echo "   📋 DMI Product: $product_name (virtualized)" >&2
                virtualized=true
                ;;
            *)
                echo "   📋 DMI Product: $product_name" >&2
                ;;
        esac
    fi

    # 2. Check for Vagrant-specific indicators
    if [[ -d /vagrant ]] || [[ "$USER" == "vagrant" ]] || grep -q "vagrant" /etc/passwd 2>/dev/null; then
        echo "   🔧 Vagrant environment detected" >&2
        virtualized=true
    fi

    # 2.1 Check for VirtualBox Guest Additions (common with Vagrant)
    if lsmod 2>/dev/null | grep -q "vboxguest\|vboxsf"; then
        echo "   📦 VirtualBox Guest Additions detected" >&2
        virtualized=true
    fi

    # 3. Check for VirtIO network driver
    if lspci 2>/dev/null | grep -i "virtio network" >/dev/null; then
        echo "   🌐 VirtIO network driver detected (VM)" >&2
        virtualized=true
    fi

    # 4. Check for containerization
    if [[ -f /.dockerenv ]] || grep -q "docker\|lxc" /proc/1/cgroup 2>/dev/null; then
        echo "   🐳 Container environment detected" >&2
        virtualized=true
    fi

    # 5. Check systemd-detect-virt if available
    if command -v systemd-detect-virt >/dev/null 2>&1; then
        virt_type=$(systemd-detect-virt 2>/dev/null || echo "none")
        if [[ "$virt_type" != "none" ]]; then
            echo "   🔍 systemd-detect-virt: $virt_type" >&2
            virtualized=true
        fi
    fi

    echo "$virtualized"
}

# Function to intelligently select network interface
select_interface_intelligently() {
    local virtualized=$1
    local selected_iface=""

    echo "🌐 Selecting optimal network interface..." >&2

    if [[ "$virtualized" == "true" ]]; then
        echo "   🚀 Virtualized environment - checking for VM-compatible interfaces..." >&2

        # First try eth1 (often the second interface in VMs, less likely to have XDP issues)
        if ip link show eth1 >/dev/null 2>&1 && [[ $(ip link show eth1 | grep -c "state UP") -gt 0 ]]; then
            selected_iface="eth1"
            echo "   ✅ Selected eth1 (VM-optimized interface)" >&2
        # Then try enp0s8 (common in VirtualBox)
        elif ip link show enp0s8 >/dev/null 2>&1 && [[ $(ip link show enp0s8 | grep -c "state UP") -gt 0 ]]; then
            selected_iface="enp0s8"
            echo "   ✅ Selected enp0s8 (VirtualBox secondary interface)" >&2
        else
            echo "   ⚠️  No secondary interfaces found, falling back to primary interface detection..." >&2
            selected_iface=""
        fi
    else
        echo "   🖥️  Bare metal environment - using primary interface detection" >&2
        selected_iface=""
    fi

    # If no VM-specific interface found, use original detection logic
    if [[ -z "$selected_iface" ]]; then
        echo "   🔍 Using route-based interface detection..." >&2
        selected_iface=$(ip route get 8.8.8.8 2>/dev/null | awk '/dev/ {for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' || echo "")

        if [[ -z "$selected_iface" ]]; then
            selected_iface=$(ip -o link show | awk -F': ' '/state UP/ && $2!="lo" {print $2; exit}' || echo "")
        fi

        if [[ -z "$selected_iface" ]]; then
            echo "❌ ERROR: No network interface detected. Available interfaces:" >&2
            ip link show | grep -E '^[0-9]+:' || true >&2
            exit 1
        fi

        echo "   ✅ Selected $selected_iface (route-based detection)" >&2
    fi

    echo "$selected_iface"
}

# Function to check XDP compatibility
check_xdp_compatibility() {
    local interface=$1
    local virtualized=$2

    echo "🔧 Checking XDP compatibility for $interface..."

    # In virtualized environments, pre-emptively disable problematic features
    if [[ "$virtualized" == "true" ]]; then
        echo "   ⚙️  Applying VM-specific network optimizations..."

        # Check if interface uses virtio driver
        if [[ -d "/sys/class/net/$interface/device/driver" ]]; then
            driver=$(basename "$(readlink "/sys/class/net/$interface/device/driver" 2>/dev/null)" 2>/dev/null || echo "unknown")
            if [[ "$driver" == *"virtio"* ]]; then
                echo "   🔍 VirtIO driver detected - applying compatibility settings..."

                # More aggressive feature disabling for VirtIO
                sudo ethtool -K "$interface" gro off gso off tx-checksumming off rx-checksumming off tso off \
                    tx-checksum-ip-generic off tx-generic-segmentation off rx-gro off \
                    tx-tcp-segmentation off tx-tcp6-segmentation off 2>/dev/null || true

                echo "   ✅ VirtIO compatibility settings applied"
            fi
        fi
    else
        echo "   🖥️  Bare metal detected - standard interface preparation"
    fi

    return 0
}

# Main execution
main() {
    echo "=== Enhanced Sniffer Interface Detection ==="

    # 1) Detect environment
    virtualized=$(detect_virtualized_environment)

    if [[ "$virtualized" == "true" ]]; then
        echo "🚀 Running in virtualized environment"
    else
        echo "🖥️  Running on bare metal"
    fi

    # 2) Select interface intelligently
    IFACE=$(select_interface_intelligently "$virtualized")
    echo ""
    echo "=> Selected interface: $IFACE"

    # 3) Update JSON config
    CONFIG="sniffer/config/sniffer.json"
    if [[ ! -f "$CONFIG" ]]; then
        echo "❌ ERROR: config file not found at $CONFIG"
        exit 1
    fi

    # Use a more robust sed approach to handle special characters
    if command -v jq >/dev/null 2>&1; then
        # Preferred: use jq for JSON manipulation
        jq --arg iface "$IFACE" '.interface = $iface' "$CONFIG" > "${CONFIG}.tmp" && mv "${CONFIG}.tmp" "$CONFIG"
    else
        # Fallback: more careful sed with proper escaping
        escaped_iface=$(printf '%s\n' "$IFACE" | sed 's/[[\.*^$()+?{|]/\\&/g')
        sed -i.bak -E "s/(\"interface\"[[:space:]]*:[[:space:]]*\")([^\"]*)(\")/\1${escaped_iface}\3/" "$CONFIG"
    fi
    echo "Updated $CONFIG with interface $IFACE"

    # 4) Check XDP compatibility and prepare interface
    check_xdp_compatibility "$IFACE" "$virtualized"

    echo "Preparing interface $IFACE..."
    sudo ethtool -K "$IFACE" gro off gso off tx-checksumming off rx-checksumming off tso off 2>/dev/null || true
    sudo ip link set dev "$IFACE" promisc on 2>/dev/null || true

    # 5) Build and run sniffer
    echo ""
    echo "Building sniffer..."
    cd sniffer/build || exit 1
    make -j4 || exit 1

    echo "Launching sniffer..."
    exec sudo ./sniffer --config="../config/sniffer.json" --verbose
}

# Run main function
main "$@"
#!/bin/bash
#
# ╔═══════════════════════════════════════════════════════════════╗
# ║  Comprehensive Traffic Generator for Ransomware Detection     ║
# ║  Duration: 9-10 hours with variable load patterns            ║
# ╚═══════════════════════════════════════════════════════════════╝
#

set -e

LOG_FILE="/tmp/traffic_generator.log"
START_TIME=$(date +%s)

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# Traffic Generation Functions
# ============================================================================

generate_normal_http() {
    local count=$1
    log "📨 Generating $count normal HTTP requests..."
    for i in $(seq 1 $count); do
        curl -s -m 2 http://example.com > /dev/null 2>&1 &
        curl -s -m 2 http://google.com > /dev/null 2>&1 &
        curl -s -m 2 http://github.com > /dev/null 2>&1 &
        sleep 0.1
    done
}

generate_https_traffic() {
    local count=$1
    log "🔒 Generating $count HTTPS requests..."
    for i in $(seq 1 $count); do
        curl -s -m 2 https://www.cloudflare.com > /dev/null 2>&1 &
        curl -s -m 2 https://www.wikipedia.org > /dev/null 2>&1 &
        sleep 0.15
    done
}

generate_dns_queries() {
    local count=$1
    log "🌐 Generating $count DNS queries..."
    for i in $(seq 1 $count); do
        dig @8.8.8.8 google.com +short > /dev/null 2>&1 &
        dig @8.8.8.8 github.com +short > /dev/null 2>&1 &
        dig @1.1.1.1 cloudflare.com +short > /dev/null 2>&1 &
        sleep 0.05
    done
}

generate_ping_traffic() {
    local count=$1
    log "📡 Generating $count ping packets..."
    for i in $(seq 1 $count); do
        ping -c 1 -W 1 8.8.8.8 > /dev/null 2>&1 &
        ping -c 1 -W 1 1.1.1.1 > /dev/null 2>&1 &
        sleep 0.2
    done
}

generate_ssh_attempts() {
    local count=$1
    log "🔑 Generating $count SSH connection attempts..."
    for i in $(seq 1 $count); do
        # Attempt connection (will fail, but generates traffic)
        timeout 1 nc -z 192.168.1.1 22 2>/dev/null &
        timeout 1 nc -z 10.0.0.1 22 2>/dev/null &
        sleep 0.5
    done
}

generate_high_entropy_traffic() {
    local count=$1
    log "🎲 Generating $count high-entropy (encrypted) packets..."
    for i in $(seq 1 $count); do
        # Generate random data and send via netcat
        dd if=/dev/urandom bs=512 count=1 2>/dev/null | nc -w 1 127.0.0.1 9999 2>/dev/null &
        sleep 0.3
    done
}

simulate_ransomware_c2() {
    local count=$1
    log "🚨 Simulating $count ransomware C2 connections..."
    for i in $(seq 1 $count); do
        # Simulate connections to suspicious domains (will fail but generate traffic)
        timeout 1 nc -z 1.2.3.4 443 2>/dev/null &
        timeout 1 nc -z 5.6.7.8 8080 2>/dev/null &
        curl -s -m 1 http://example.onion 2>/dev/null &
        sleep 1
    done
}

simulate_smb_traffic() {
    local count=$1
    log "📁 Simulating $count SMB-like connections..."
    for i in $(seq 1 $count); do
        timeout 1 nc -z 192.168.1.1 445 2>/dev/null &
        timeout 1 nc -z 192.168.1.2 139 2>/dev/null &
        sleep 0.8
    done
}

stress_burst() {
    local duration=$1
    log "💥 STRESS BURST for ${duration}s - High load!"
    
    end_time=$(($(date +%s) + duration))
    while [ $(date +%s) -lt $end_time ]; do
        # Massive parallel traffic
        for i in {1..50}; do
            curl -s -m 1 http://example.com > /dev/null 2>&1 &
            ping -c 1 -W 1 8.8.8.8 > /dev/null 2>&1 &
            dd if=/dev/urandom bs=512 count=1 2>/dev/null | nc -w 1 127.0.0.1 9999 2>/dev/null &
        done
        sleep 1
    done
    
    log "💥 Stress burst complete. Cooling down..."
    sleep 5
}

# ============================================================================
# Test Phases
# ============================================================================

phase_warmup() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║  PHASE 1: Warm-up (30 minutes)                               ║"
    log "║  Normal traffic patterns, low rate                           ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    
    for min in {1..30}; do
        log "Warm-up minute $min/30"
        generate_normal_http 5
        generate_dns_queries 10
        generate_ping_traffic 3
        sleep 30
    done
}

phase_normal_load() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║  PHASE 2: Normal Load (2 hours)                              ║"
    log "║  Typical mixed traffic, moderate rate                        ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    
    for hour in {1..2}; do
        for min in {1..60}; do
            log "Normal load - Hour $hour, Minute $min"
            generate_normal_http 10
            generate_https_traffic 8
            generate_dns_queries 15
            generate_ping_traffic 5
            generate_ssh_attempts 2
            sleep 30
        done
    done
}

phase_stress_testing() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║  PHASE 3: Stress Testing (1.5 hours)                         ║"
    log "║  High load with periodic bursts                              ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    
    for cycle in {1..18}; do  # 18 cycles of 5 minutes = 90 min
        log "Stress cycle $cycle/18"
        
        # High sustained load (3 minutes)
        for i in {1..6}; do
            generate_normal_http 20
            generate_https_traffic 15
            generate_dns_queries 30
            generate_high_entropy_traffic 10
            simulate_smb_traffic 5
            sleep 30
        done
        
        # Stress burst (1 minute)
        stress_burst 60
        
        # Cool down (1 minute)
        log "Cool down..."
        sleep 60
    done
}

phase_ransomware_simulation() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║  PHASE 4: Ransomware Simulation (1 hour)                     ║"
    log "║  Suspicious patterns, high entropy, C2 traffic               ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    
    for cycle in {1..12}; do  # 12 cycles of 5 minutes
        log "Ransomware simulation cycle $cycle/12"
        
        generate_high_entropy_traffic 20
        simulate_ransomware_c2 10
        simulate_smb_traffic 15
        generate_dns_queries 20
        
        # Mix with normal traffic
        generate_normal_http 5
        generate_https_traffic 3
        
        sleep 30
    done
}

phase_sustained_load() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║  PHASE 5: Sustained Load (3 hours)                           ║"
    log "║  Continuous moderate traffic                                 ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    
    for hour in {1..3}; do
        for min in {1..60}; do
            log "Sustained load - Hour $hour, Minute $min"
            generate_normal_http 12
            generate_https_traffic 10
            generate_dns_queries 20
            generate_ping_traffic 8
            generate_high_entropy_traffic 5
            simulate_smb_traffic 3
            sleep 30
        done
    done
}

phase_cooldown() {
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║  PHASE 6: Cool Down (30 minutes)                             ║"
    log "║  Gradual reduction to minimal traffic                        ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    
    for min in {1..30}; do
        rate=$((30 - min + 1))  # Decreasing rate
        log "Cool down minute $min/30 (rate: $rate)"
        generate_normal_http $((rate / 6))
        generate_dns_queries $((rate / 3))
        sleep 60
    done
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║         COMPREHENSIVE TRAFFIC GENERATOR STARTED               ║"
    log "║         Duration: ~9-10 hours                                 ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    log ""
    log "Test Start Time: $(date)"
    log "Estimated End Time: $(date -d '+10 hours')"
    log ""
    
    # Check dependencies
    command -v curl >/dev/null 2>&1 || { log "❌ curl not found"; exit 1; }
    command -v dig >/dev/null 2>&1 || { log "❌ dig not found"; exit 1; }
    command -v nc >/dev/null 2>&1 || { log "❌ netcat not found"; exit 1; }
    
    log "✅ All dependencies found"
    log ""
    
    # Execute phases
    phase_warmup           # 0.5h
    phase_normal_load      # 2h
    phase_stress_testing   # 1.5h
    phase_ransomware_simulation  # 1h
    phase_sustained_load   # 3h
    phase_cooldown         # 0.5h
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    
    log ""
    log "╔═══════════════════════════════════════════════════════════════╗"
    log "║         TRAFFIC GENERATION COMPLETE                           ║"
    log "╚═══════════════════════════════════════════════════════════════╝"
    log ""
    log "Total Duration: ${HOURS}h ${MINUTES}m"
    log "Test End Time: $(date)"
    log ""
    log "📊 Summary will be in /tmp/traffic_generator.log"
    log "🎯 Check sniffer stats for validation"
}

# Trap Ctrl+C
trap 'log "⚠️  Traffic generation interrupted"; exit 1' INT TERM

# Run
main

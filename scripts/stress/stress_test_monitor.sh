#!/bin/bash
# Resource monitoring for stress test

SNIFFER_PID=$1
ML_DETECTOR_PID=$2
OUTPUT_DIR=$3
INTERVAL=$4

MEMORY_CSV="${OUTPUT_DIR}/memory.csv"
CPU_CSV="${OUTPUT_DIR}/cpu.csv"
LATENCY_CSV="${OUTPUT_DIR}/latency.csv"

# CSV headers
echo "timestamp,sniffer_rss_mb,sniffer_vsz_mb,ml_detector_rss_mb,ml_detector_vsz_mb,total_mb" > "${MEMORY_CSV}"
echo "timestamp,sniffer_cpu_pct,ml_detector_cpu_pct,system_cpu_pct" > "${CPU_CSV}"
echo "timestamp,sniffer_packets,sniffer_avg_latency_us" > "${LATENCY_CSV}"

echo "📊 Resource monitor started (interval: ${INTERVAL}s)"

while true; do
    TIMESTAMP=$(date +%s)

    # Memory usage
    SNIFFER_RSS=$(ps -p ${SNIFFER_PID} -o rss= 2>/dev/null || echo "0")
    SNIFFER_VSZ=$(ps -p ${SNIFFER_PID} -o vsz= 2>/dev/null || echo "0")
    ML_DETECTOR_RSS=$(ps -p ${ML_DETECTOR_PID} -o rss= 2>/dev/null || echo "0")
    ML_DETECTOR_VSZ=$(ps -p ${ML_DETECTOR_PID} -o vsz= 2>/dev/null || echo "0")

    SNIFFER_RSS_MB=$((SNIFFER_RSS / 1024))
    SNIFFER_VSZ_MB=$((SNIFFER_VSZ / 1024))
    ML_DETECTOR_RSS_MB=$((ML_DETECTOR_RSS / 1024))
    ML_DETECTOR_VSZ_MB=$((ML_DETECTOR_VSZ / 1024))
    TOTAL_MB=$((SNIFFER_RSS_MB + ML_DETECTOR_RSS_MB))

    echo "${TIMESTAMP},${SNIFFER_RSS_MB},${SNIFFER_VSZ_MB},${ML_DETECTOR_RSS_MB},${ML_DETECTOR_VSZ_MB},${TOTAL_MB}" >> "${MEMORY_CSV}"

    # CPU usage
    SNIFFER_CPU=$(ps -p ${SNIFFER_PID} -o %cpu= 2>/dev/null || echo "0")
    ML_DETECTOR_CPU=$(ps -p ${ML_DETECTOR_PID} -o %cpu= 2>/dev/null || echo "0")
    SYSTEM_CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

    echo "${TIMESTAMP},${SNIFFER_CPU},${ML_DETECTOR_CPU},${SYSTEM_CPU}" >> "${CPU_CSV}"

    sleep ${INTERVAL}
done
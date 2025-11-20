# ML Defender - 8 Hour Stress Test Report

## Test Information
- **Start Time**:  jue 20 nov 2025 10:10:56 UTC
- **End Time**: jue 20 nov 2025 10:21:22 UTC
- **Planned Duration**: 10 minutes
- **Actual Runtime**: 0h 10m 18s (618 seconds)
- **Traffic Rate**: 75 pps

## Configuration
- **Thresholds**: DDoS=0.85, Ransomware=0.90, Traffic=0.80, Internal=0.85
- **Hardware**:  Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz CPU @ 0.0GHz
- **Memory**:  7,8Gi
- **Kernel**:  6.1.0-41-amd64

## Component Status
- **Sniffer**: ✅ Running (PID: 4765)
- **ML-Detector**: ❌ Stopped

## Statistics

### Sniffer Final Stats
```
No stats found
```

### ML-Detector Final Stats
```
[2025-11-20 10:16:57.918] [ml-detector] [info] 📊 Stats: received=26750, processed=26750, sent=26750, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:17:58.015] [ml-detector] [info] 📊 Stats: received=27846, processed=27846, sent=27846, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:18:58.015] [ml-detector] [info] 📊 Stats: received=30149, processed=30149, sent=30149, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:19:58.085] [ml-detector] [info] 📊 Stats: received=34420, processed=34420, sent=34420, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:20:58.254] [ml-detector] [info] 📊 Stats: received=35387, processed=35387, sent=35387, attacks=0, errors=(deser:0, feat:0, inf:0)
```

## Resource Usage

### Memory (Last 20 samples)
```
timestamp,sniffer_rss_mb,sniffer_vsz_mb,ml_detector_rss_mb,ml_detector_vsz_mb,total_mb
1763633464,4,10,138,519,142
1763633524,4,10,139,519,143
1763633585,4,10,139,519,143
1763633645,4,10,139,519,143
1763633719,4,10,139,519,143
1763633796,4,10,139,519,143
1763633864,4,10,139,519,143
1763633944,4,10,139,519,143
1763634005,4,10,139,519,143
1763634077,4,10,139,519,143
```

### Memory Analysis
- **Initial Memory**: 142 MB
- **Final Memory**: 143 MB
- **Memory Growth**: 1 MB

### CPU (Last 20 samples)
```
timestamp,sniffer_cpu_pct,ml_detector_cpu_pct,system_cpu_pct
1763633464, 0.0,19.3,25,0
1763633524, 0.0, 167,50,0
1763633585, 0.0, 229,0,0
1763633645, 0.0, 224,0,0
1763633719, 0.0, 191,7,1
1763633796, 0.0, 169,80,0
1763633864, 0.0, 160,33,3
1763633944, 0.0, 159,83,3
1763634005, 0.0, 159,0,0
1763634077, 0.0, 156,50,0
```

## Errors and Warnings

### Sniffer Errors
```
[WARNING] Thread count mismatch: calculated=4, configured=2
[WARNING] Failed to set thread priority: Invalid argument
[WARNING] Failed to set thread priority: Invalid argument
```

### ML-Detector Errors
```
[2025-11-20 10:11:57.175] [ml-detector] [info] 📊 Stats: received=3732, processed=3732, sent=3732, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:12:57.175] [ml-detector] [info] 📊 Stats: received=13927, processed=13927, sent=13927, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:13:57.328] [ml-detector] [info] 📊 Stats: received=22340, processed=22340, sent=22340, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:14:57.328] [ml-detector] [info] 📊 Stats: received=23504, processed=23504, sent=23504, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:15:57.917] [ml-detector] [info] 📊 Stats: received=24850, processed=24850, sent=24850, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:16:57.918] [ml-detector] [info] 📊 Stats: received=26750, processed=26750, sent=26750, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:17:58.015] [ml-detector] [info] 📊 Stats: received=27846, processed=27846, sent=27846, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:18:58.015] [ml-detector] [info] 📊 Stats: received=30149, processed=30149, sent=30149, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:19:58.085] [ml-detector] [info] 📊 Stats: received=34420, processed=34420, sent=34420, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:20:58.254] [ml-detector] [info] 📊 Stats: received=35387, processed=35387, sent=35387, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-20 10:21:21.125] [ml-detector] [info]    Errors: deser=0, feat=0, inf=0
```

## Files
- **Logs Directory**: `/vagrant/stress_test_20251120_101056/logs/`
- **Monitoring Directory**: `/vagrant/stress_test_20251120_101056/monitoring/`
- **Compressed Archive**: `stress_test_20251120_101056.tar.gz`

## Analysis

### Memory Leak Detection
✅ **STABLE**: Memory growth 1MB is within normal range

### Stability
❌ One or more components crashed during test

## Next Steps
1. Review error logs for any warnings or failures
2. Analyze memory.csv for linear growth (potential leak)
3. Validate detection rates match expected patterns
4. Calibrate thresholds based on false positive rate
5. If stable, proceed with production deployment

---
Generated: jue 20 nov 2025 10:21:22 UTC

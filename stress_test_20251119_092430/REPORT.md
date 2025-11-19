# ML Defender - 8 Hour Stress Test Report

## Test Information
- **Start Time**:  mié 19 nov 2025 09:24:30 UTC
- **End Time**: mié 19 nov 2025 09:27:01 UTC
- **Planned Duration**: 1 hours
- **Actual Runtime**: 0h 2m 23s (143 seconds)
- **Traffic Rate**: 75 pps

## Configuration
- **Thresholds**: DDoS=0.85, Ransomware=0.90, Traffic=0.80, Internal=0.85
- **Hardware**:  Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz CPU @ 0.0GHz
- **Memory**:  7,8Gi
- **Kernel**:  6.1.0-41-amd64

## Component Status
- **Sniffer**: ✅ Running (PID: 402661)
- **ML-Detector**: ❌ Stopped

## Statistics

### Sniffer Final Stats
```
No stats found
```

### ML-Detector Final Stats
```
[2025-11-19 09:25:31.478] [ml-detector] [info] 📊 Stats: received=15238, processed=15238, sent=15238, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-19 09:26:31.681] [ml-detector] [info] 📊 Stats: received=27263, processed=27263, sent=27263, attacks=0, errors=(deser:0, feat:0, inf:0)
```

## Resource Usage

### Memory (Last 20 samples)
```
timestamp,sniffer_rss_mb,sniffer_vsz_mb,ml_detector_rss_mb,ml_detector_vsz_mb,total_mb
1763544278,4,10,143,519,147
1763544339,4,10,143,519,147
1763544399,4,10,143,519,147
```

### Memory Analysis
- **Initial Memory**: 147 MB
- **Final Memory**: 147 MB
- **Memory Growth**: 0 MB

### CPU (Last 20 samples)
```
timestamp,sniffer_cpu_pct,ml_detector_cpu_pct,system_cpu_pct
1763544278, 0.0,13.8,us,
1763544339, 0.0, 177,66,7
1763544399, 0.0, 175,33,3
```

## Errors and Warnings

### Sniffer Errors
```
[WARNING] Thread count mismatch: calculated=4, configured=2
[WARNING] Failed to set thread priority: Invalid argument
[WARNING] Failed to set thread priority: Invalid argument
[ERROR] ZMQ send falló!
```

### ML-Detector Errors
```
[2025-11-19 09:25:31.478] [ml-detector] [info] 📊 Stats: received=15238, processed=15238, sent=15238, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-19 09:26:31.681] [ml-detector] [info] 📊 Stats: received=27263, processed=27263, sent=27263, attacks=0, errors=(deser:0, feat:0, inf:0)
[2025-11-19 09:26:57.080] [ml-detector] [info]    Errors: deser=0, feat=0, inf=0
```

## Files
- **Logs Directory**: `/vagrant/stress_test_20251119_092430/logs/`
- **Monitoring Directory**: `/vagrant/stress_test_20251119_092430/monitoring/`
- **Compressed Archive**: `stress_test_20251119_092430.tar.gz`

## Analysis

### Memory Leak Detection
✅ **STABLE**: Memory growth 0MB is within normal range

### Stability
❌ One or more components crashed during test

## Next Steps
1. Review error logs for any warnings or failures
2. Analyze memory.csv for linear growth (potential leak)
3. Validate detection rates match expected patterns
4. Calibrate thresholds based on false positive rate
5. If stable, proceed with production deployment

---
Generated: mié 19 nov 2025 09:27:02 UTC

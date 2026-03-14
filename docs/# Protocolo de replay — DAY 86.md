# Protocolo de replay — DAY 86.md
# Experimento 1
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make up
Bringing machine 'defender' up with 'virtualbox' provider...
Bringing machine 'client' up with 'virtualbox' provider...
...
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make pipeline-stop
🛑 Stopping all pipeline components...
✅ Pipeline stopped
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make pipeline-start

...
╔════════════════════════════════════════════════════════════╗
║  ✅ FULL PIPELINE STARTED (Day 74)                        ║
╚════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════╗
║  📊 ML Defender Pipeline Status (via TMUX)                ║
╚════════════════════════════════════════════════════════════╝
  ✅ etcd-server:   RUNNING
  ✅ rag-security:  RUNNING
  ✅ rag-ingester:  RUNNING
  ✅ ml-detector:   RUNNING
  ✅ sniffer:       RUNNING
  ✅ firewall:      RUNNING
╚════════════════════════════════════════════════════════════╝
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % sleep 15
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make test-replay-small
🧪 Replaying CTU-13 smallFlows.pcap...
Test start: 2026-03-14 07:16:23.730783 ...
Actual: 3624 packets (2500833 bytes) sent in 2.00 seconds
Rated: 1249959.0 Bps, 9.99 Mbps, 1811.33 pps
Actual: 8176 packets (5000865 bytes) sent in 4.00 seconds
Rated: 1249941.2 Bps, 9.99 Mbps, 2043.55 pps
Actual: 11931 packets (7501596 bytes) sent in 6.00 seconds
Rated: 1249840.8 Bps, 9.99 Mbps, 1987.82 pps
Test complete: 2026-03-14 07:16:31.104044
Actual: 14261 packets (9216531 bytes) sent in 7.37 seconds
Rated: 1249993.8 Bps, 9.99 Mbps, 1934.15 pps
Flows: 1209 flows, 163.97 fps, 14243 unique flow packets, 18 unique non-flow packets
Statistics for network device: eth1
        Successful packets:        14261
        Failed packets:            0
        Truncated packets:         0
        Retried packets (ENOBUFS): 0
        Retried packets (EAGAIN):  0
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:16:12.434] [ml-detector] [info] 📊 Stats: received=18, processed=18, sent=18, attacks=0, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "cat /vagrant/logs/lab/sniffer.log" > /tmp/sniffer_small.log

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % python3 scripts/calculate_f1_neris.py /tmp/sniffer_small.log --total-events XXXX --day "DAY86_small"

usage: calculate_f1_neris.py [-h] --total-events TOTAL_EVENTS [--day DAY] [--thresholds THRESHOLDS] sniffer_log
calculate_f1_neris.py: error: argument --total-events: invalid int value: 'XXXX'
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:17:12.443] [ml-detector] [info] 📊 Stats: received=1030, processed=1030, sent=1030, attacks=0, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:18:12.461] [ml-detector] [info] 📊 Stats: received=3370, processed=3370, sent=3370, attacks=0, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % echo "Es obvio que hacer el grep lleva tiempo. Esperamos un poco."
Es obvio que hacer el grep lleva tiempo. Esperamos un poco.
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:18:12.461] [ml-detector] [info] 📊 Stats: received=3370, processed=3370, sent=3370, attacks=0, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:19:12.464] [ml-detector] [info] 📊 Stats: received=7435, processed=7435, sent=7435, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:19:12.464] [ml-detector] [info] 📊 Stats: received=7435, processed=7435, sent=7435, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:19:12.464] [ml-detector] [info] 📊 Stats: received=7435, processed=7435, sent=7435, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:20:12.466] [ml-detector] [info] 📊 Stats: received=7441, processed=7441, sent=7441, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:21:12.467] [ml-detector] [info] 📊 Stats: received=7447, processed=7447, sent=7447, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:22:12.467] [ml-detector] [info] 📊 Stats: received=7457, processed=7457, sent=7457, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % wc -l /vagrant/logs/lab/ml-detector.log
wc: /vagrant/logs/lab/ml-detector.log: open: No such file or directory
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % wc -l logs/lab/ml-detector.log 
   55474 logs/lab/ml-detector.log
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % Cuando puedo sacar el número para el script python3 scripts/calculate_f1_neris.py /tmp/sniffer_small.log --total-events XXXX --day "DAY86_small"
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:26:12.470] [ml-detector] [info] 📊 Stats: received=7481, processed=7481, sent=7481, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:26:12.470] [ml-detector] [info] 📊 Stats: received=7481, processed=7481, sent=7481, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:26:12.470] [ml-detector] [info] 📊 Stats: received=7481, processed=7481, sent=7481, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"
[2026-03-14 07:26:12.470] [ml-detector] [info] 📊 Stats: received=7481, processed=7481, sent=7481, attacks=1, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % python3 scripts/calculate_f1_neris.py /tmp/sniffer_small.log --total-events 7481 --day "DAY86_small"

====================================================================
  ML DEFENDER - F1 SCORE CALCULATOR
  Day: DAY86_small  |  Thresholds: 
  Ground truth: CTU-13 Neris (3 malicious IPs)
====================================================================
  Malicious IPs: 147.32.84.165, 147.32.84.191, 147.32.84.192

====================================================================
  FAST DETECTOR — deduplicated alerts vs ground truth
====================================================================
  Raw [FAST ALERT] lines:      10064
  Deduplicated alert events:   916
  Total events (ml-detector):  7481

  TP  (malicious, detected):   766
  FP  (benign, false alarm):   150
  FN  (malicious, missed):     0
  TN  (benign, correct):       6565

  Precision:  0.8362
  Recall:     1.0000
  F1-Score:   0.9108  ← paper metric
  FPR:        0.0223
  Accuracy:   0.9799

  NOTE: FN estimated as 0 — [FAST ALERT] only fires on detected flows. True FN requires per-event IP table. Recall=1.0 is an upper bound, not confirmed.
====================================================================

  DETECTED IPs breakdown:
    147.32.84.165  →  766 flow(s) detected  [MALICIOUS ✓]

  FALSE POSITIVE IPs (sample, max 5):
    src=192.168.3.131  dst=65.55.5.232
    src=72.14.213.147  dst=192.168.3.131
    src=66.235.139.121  dst=192.168.3.131
    src=65.55.239.163  dst=192.168.3.131
    src=192.168.3.131  dst=65.55.239.163

  CSV line for f1_replay_log.csv:
  DAYDAY86_small,,7481,766,150,0,6565,0.9108,0.8362,1.0000,0.0223

# Experimento 2 — neris (el principal)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make pipeline-stop && make logs-lab-clean && make pipeline-start && sleep 15

🛑 Stopping all pipeline components...
✅ Pipeline stopped
🧹 Rotating pipeline logs...
✅ Logs rotated to /vagrant/logs/lab/archive/

...
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make pipeline-status

╔════════════════════════════════════════════════════════════╗
║  📊 ML Defender Pipeline Status (via TMUX)                ║
╚════════════════════════════════════════════════════════════╝
  ✅ etcd-server:   RUNNING
  ✅ rag-security:  RUNNING
  ❌ rag-ingester:  STOPPED
  ✅ ml-detector:   RUNNING
  ✅ sniffer:       RUNNING
  ✅ firewall:      RUNNING
╚════════════════════════════════════════════════════════════╝
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make pipeline-health

╔══════════════════════════════════════════════════════════════╗
║      ML Defender — Pipeline Health Monitor                  ║
╚══════════════════════════════════════════════════════════════╝

  🖥  VM Status:
    defender: running
    client:   running

  📦 Pipeline Components:
    Component              Status     PID        Log Activity
    ---------              ------     ---        ------------
    etcd-server            ✅ UP     2760       ❓ no log
    rag-security           ✅ UP     2817       ❓ no log
    rag-ingester           ❌ DOWN   -          -         
    ml-detector            ✅ UP     2936       ❓ no log
    sniffer                ✅ UP     3170       ❓ no log
    firewall               ✅ UP     3039       ❓ no log

  📊 ML-Detector last stats:
    received=16, processed=16, sent=16, attacks=0, errors=(deser:0, feat:0, inf:0)

  ⚠️  VM 'client' ya está RUNNING — no ejecutar 'vagrant up client'

╚══════════════════════════════════════════════════════════════╝

Me estoy dando cuenta que en el Makefile no tenemos una tarea para arrancar rag-ingester, por eso está parado.

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make rag-ingester-start
🚀 Starting RAG Ingester (Full Context)...
Ejecución desde la raíz del componente para resolver paths relativos del config...

╔════════════════════════════════════════════════════════════╗
║  📊 ML Defender Pipeline Status (via TMUX)                ║
╚════════════════════════════════════════════════════════════╝
  ✅ etcd-server:   RUNNING
  ✅ rag-security:  RUNNING
  ✅ rag-ingester:  RUNNING
  ✅ ml-detector:   RUNNING
  ✅ sniffer:       RUNNING
  ✅ firewall:      RUNNING
╚════════════════════════════════════════════════════════════╝
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make pipeline-health   

╔══════════════════════════════════════════════════════════════╗
║      ML Defender — Pipeline Health Monitor                  ║
╚══════════════════════════════════════════════════════════════╝

  🖥  VM Status:
    defender: running
    client:   running

  📦 Pipeline Components:
    Component              Status     PID        Log Activity
    ---------              ------     ---        ------------
    etcd-server            ✅ UP     2760       ❓ no log
    rag-security           ✅ UP     2817       ❓ no log
    rag-ingester           ✅ UP     3679       ❓ no log
    ml-detector            ✅ UP     2936       ❓ no log
    sniffer                ✅ UP     3170       ❓ no log
    firewall               ✅ UP     3039       ❓ no log

  📊 ML-Detector last stats:
    received=52, processed=52, sent=52, attacks=0, errors=(deser:0, feat:0, inf:0)

  ⚠️  VM 'client' ya está RUNNING — no ejecutar 'vagrant up client'

╚══════════════════════════════════════════════════════════════╝

Por qué ha fallado levantar rag-ingester antes? 

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % make test-replay-neris
🧪 Replaying CTU-13 Neris botnet (492K events)...
...
Test complete: 2026-03-14 07:43:41.718741
Actual: 320524 packets (44200259 bytes) sent in 39.68 seconds
Rated: 1113850.3 Bps, 8.91 Mbps, 8077.23 pps
Flows: 19135 flows, 482.20 fps, 322242 unique flow packets, 906 unique non-flow packets
Statistics for network device: eth1
        Successful packets:        320524
        Failed packets:            2630
        Truncated packets:         0
        Retried packets (ENOBUFS): 0
        Retried packets (EAGAIN):  0
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:45:24.961] [ml-detector] [info] 📊 Stats: received=5941, processed=5941, sent=5941, attacks=0, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:47:24.962] [ml-detector] [info] 📊 Stats: received=11904, processed=11904, sent=11904, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:50:24.972] [ml-detector] [info] 📊 Stats: received=12046, processed=12046, sent=12046, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:50:24.972] [ml-detector] [info] 📊 Stats: received=12046, processed=12046, sent=12046, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:54:24.975] [ml-detector] [info] 📊 Stats: received=12214, processed=12214, sent=12214, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:54:24.975] [ml-detector] [info] 📊 Stats: received=12214, processed=12214, sent=12214, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:54:24.975] [ml-detector] [info] 📊 Stats: received=12214, processed=12214, sent=12214, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:57:25.012] [ml-detector] [info] 📊 Stats: received=12332, processed=12332, sent=12332, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "cat /vagrant/logs/lab/sniffer.log" > /tmp/sniffer_neris.log

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % wc -l /tmp/sniffer_neris.log
   38277 /tmp/sniffer_neris.log
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1" 

[2026-03-14 07:59:25.012] [ml-detector] [info] 📊 Stats: received=12444, processed=12444, sent=12444, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 07:59:25.012] [ml-detector] [info] 📊 Stats: received=12444, processed=12444, sent=12444, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 08:00:25.012] [ml-detector] [info] 📊 Stats: received=12517, processed=12517, sent=12517, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 08:01:25.015] [ml-detector] [info] 📊 Stats: received=12563, processed=12563, sent=12563, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 08:01:25.015] [ml-detector] [info] 📊 Stats: received=12563, processed=12563, sent=12563, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % echo "Es suficiente?"
Es suficiente?
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % python3 scripts/calculate_f1_neris.py /tmp/sniffer_neris.log --total-events 12563 --day "DAY86_neris"


====================================================================
  ML DEFENDER - F1 SCORE CALCULATOR
  Day: DAY86_neris  |  Thresholds: 
  Ground truth: CTU-13 Neris (3 malicious IPs)
====================================================================
  Malicious IPs: 147.32.84.165, 147.32.84.191, 147.32.84.192

====================================================================
  FAST DETECTOR — deduplicated alerts vs ground truth
====================================================================
  Raw [FAST ALERT] lines:      5662
  Deduplicated alert events:   648
  Total events (ml-detector):  12563

  TP  (malicious, detected):   646
  FP  (benign, false alarm):   2
  FN  (malicious, missed):     0
  TN  (benign, correct):       11915

  Precision:  0.9969
  Recall:     1.0000
  F1-Score:   0.9985  ← paper metric
  FPR:        0.0002
  Accuracy:   0.9998

  NOTE: FN estimated as 0 — [FAST ALERT] only fires on detected flows. True FN requires per-event IP table. Recall=1.0 is an upper bound, not confirmed.
====================================================================

  DETECTED IPs breakdown:
    147.32.84.165  →  646 flow(s) detected  [MALICIOUS ✓]

  FALSE POSITIVE IPs (sample, max 5):
    src=192.168.56.1  dst=224.0.0.251
    src=192.168.56.1  dst=192.168.56.255

  CSV line for f1_replay_log.csv:
  DAYDAY86_neris,,12563,646,2,0,11915,0.9985,0.9969,1.0000,0.0002

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"                  

[2026-03-14 08:02:25.023] [ml-detector] [info] 📊 Stats: received=12605, processed=12605, sent=12605, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % python3 scripts/calculate_f1_neris.py /tmp/sniffer_neris.log --total-events 12605 --day "DAY86_neris"


====================================================================
  ML DEFENDER - F1 SCORE CALCULATOR
  Day: DAY86_neris  |  Thresholds: 
  Ground truth: CTU-13 Neris (3 malicious IPs)
====================================================================
  Malicious IPs: 147.32.84.165, 147.32.84.191, 147.32.84.192

====================================================================
  FAST DETECTOR — deduplicated alerts vs ground truth
====================================================================
  Raw [FAST ALERT] lines:      5662
  Deduplicated alert events:   648
  Total events (ml-detector):  12605

  TP  (malicious, detected):   646
  FP  (benign, false alarm):   2
  FN  (malicious, missed):     0
  TN  (benign, correct):       11957

  Precision:  0.9969
  Recall:     1.0000
  F1-Score:   0.9985  ← paper metric
  FPR:        0.0002
  Accuracy:   0.9998

  NOTE: FN estimated as 0 — [FAST ALERT] only fires on detected flows. True FN requires per-event IP table. Recall=1.0 is an upper bound, not confirmed.
====================================================================

  DETECTED IPs breakdown:
    147.32.84.165  →  646 flow(s) detected  [MALICIOUS ✓]

  FALSE POSITIVE IPs (sample, max 5):
    src=192.168.56.1  dst=224.0.0.251
    src=192.168.56.1  dst=192.168.56.255

  CSV line for f1_replay_log.csv:
  DAYDAY86_neris,,12605,646,2,0,11957,0.9985,0.9969,1.0000,0.0002

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"                  

[2026-03-14 08:04:25.069] [ml-detector] [info] 📊 Stats: received=12685, processed=12685, sent=12685, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 08:04:25.069] [ml-detector] [info] 📊 Stats: received=12685, processed=12685, sent=12685, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % vagrant ssh defender -c "grep 'Stats:' /vagrant/logs/lab/ml-detector.log | tail -1"

[2026-03-14 08:05:25.073] [ml-detector] [info] 📊 Stats: received=12723, processed=12723, sent=12723, attacks=12, errors=(deser:0, feat:0, inf:0)
(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % python3 scripts/calculate_f1_neris.py /tmp/sniffer_neris.log --total-events 12723 --day "DAY86_neris"


====================================================================
  ML DEFENDER - F1 SCORE CALCULATOR
  Day: DAY86_neris  |  Thresholds: 
  Ground truth: CTU-13 Neris (3 malicious IPs)
====================================================================
  Malicious IPs: 147.32.84.165, 147.32.84.191, 147.32.84.192

====================================================================
  FAST DETECTOR — deduplicated alerts vs ground truth
====================================================================
  Raw [FAST ALERT] lines:      5662
  Deduplicated alert events:   648
  Total events (ml-detector):  12723

  TP  (malicious, detected):   646
  FP  (benign, false alarm):   2
  FN  (malicious, missed):     0
  TN  (benign, correct):       12075

  Precision:  0.9969
  Recall:     1.0000
  F1-Score:   0.9985  ← paper metric
  FPR:        0.0002
  Accuracy:   0.9998

  NOTE: FN estimated as 0 — [FAST ALERT] only fires on detected flows. True FN requires per-event IP table. Recall=1.0 is an upper bound, not confirmed.
====================================================================

  DETECTED IPs breakdown:
    147.32.84.165  →  646 flow(s) detected  [MALICIOUS ✓]

  FALSE POSITIVE IPs (sample, max 5):
    src=192.168.56.1  dst=224.0.0.251
    src=192.168.56.1  dst=192.168.56.255

  CSV line for f1_replay_log.csv:
  DAYDAY86_neris,,12723,646,2,0,12075,0.9985,0.9969,1.0000,0.0002

(.venv) aironman@MacBook-Pro-de-Alonso test-zeromq-docker % 

Espero que valga, me estoy durmiendo...

# Experimento 3 pendiente de confirmacion

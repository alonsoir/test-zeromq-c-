#!/bin/bash
# ============================================================================
# Day 23 - ML Defender tmux Monitor (5 panels + etcd-server)
# ============================================================================
# Layout:
#   ┌─────────────────────────────────┬─────────────────────────────────┐
#   │  Panel 0: etcd-server logs      │  Panel 1: ML Detector (Dual)    │
#   ├─────────────────────────────────┼─────────────────────────────────┤
#   │  Panel 2: Sniffer Activity      │  Panel 3: Firewall Logs         │
#   ├─────────────────────────────────┴─────────────────────────────────┤
#   │  Panel 4: System Stats & Heartbeat Status                         │
#   └───────────────────────────────────────────────────────────────────┘
# ============================================================================

SESSION_NAME="ml-defender-day23"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Session '$SESSION_NAME' already exists"
    echo "   Options:"
    echo "   1. Attach: tmux attach -t $SESSION_NAME"
    echo "   2. Kill:   tmux kill-session -t $SESSION_NAME && bash $0"
    exit 1
fi

echo "🚀 Creating tmux session: $SESSION_NAME"

# Create session with first panel
tmux new-session -d -s "$SESSION_NAME" -n "ML-Defender"

# ============================================================================
# Panel 0: etcd-server logs (TOP LEFT)
# ============================================================================
tmux send-keys -t "$SESSION_NAME:0.0" "cd /vagrant" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "clear" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo '📡 etcd-server Logs (Config + Heartbeat Supervisor)'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "tail -f /vagrant/logs/etcd-server.log 2>/dev/null || echo '⏳ Waiting for etcd-server logs...'; sleep infinity" C-m

# ============================================================================
# Panel 1: ML Detector Dual-Score logs (TOP RIGHT)
# ============================================================================
tmux split-window -h -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.1" "cd /vagrant" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "clear" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '🤖 ML Detector - Dual-Score Architecture'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "tail -f /vagrant/logs/lab/detector.log 2>/dev/null | grep --line-buffered -E 'DUAL-SCORE|⚠️|✅|Encrypted|Compressed' || echo '⏳ Waiting for detector logs...'; sleep infinity" C-m

# ============================================================================
# Panel 2: Sniffer Activity (MIDDLE LEFT)
# ============================================================================
tmux split-window -v -t "$SESSION_NAME:0.0"
tmux send-keys -t "$SESSION_NAME:0.2" "cd /vagrant" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "clear" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "echo '📡 Sniffer - Packet Capture (eBPF/XDP)'" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0.2" "tail -f /vagrant/logs/lab/sniffer.log 2>/dev/null | grep --line-buffered -E 'Captured|Sent|packets|Encrypted|Compressed' || echo '⏳ Waiting for sniffer logs...'; sleep infinity" C-m

# ============================================================================
# Panel 3: Firewall Logs (MIDDLE RIGHT)
# ============================================================================
tmux split-window -v -t "$SESSION_NAME:0.1"
tmux send-keys -t "$SESSION_NAME:0.3" "cd /vagrant" C-m
tmux send-keys -t "$SESSION_NAME:0.3" "clear" C-m
tmux send-keys -t "$SESSION_NAME:0.3" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.3" "echo '🔥 Firewall ACL Agent - IPTables/IPSet'" C-m
tmux send-keys -t "$SESSION_NAME:0.3" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.3" "echo ''" C-m
tmux send-keys -t "$SESSION_NAME:0.3" "tail -f /vagrant/logs/lab/firewall.log 2>/dev/null | grep --line-buffered -E 'Blocked|Alert|Action|Decrypted|Decompressed' || echo '⏳ Waiting for firewall logs...'; sleep infinity" C-m

# ============================================================================
# Panel 4: System Stats & Heartbeat Status (BOTTOM - FULL WIDTH)
# ============================================================================
tmux split-window -v -t "$SESSION_NAME:0.2"
tmux send-keys -t "$SESSION_NAME:0.4" "cd /vagrant" C-m
tmux send-keys -t "$SESSION_NAME:0.4" "clear" C-m
tmux send-keys -t "$SESSION_NAME:0.4" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.4" "echo '📊 System Stats & Heartbeat Status'" C-m
tmux send-keys -t "$SESSION_NAME:0.4" "echo '═══════════════════════════════════════════════════════════'" C-m
tmux send-keys -t "$SESSION_NAME:0.4" "echo ''" C-m

# Watch command for stats panel
tmux send-keys -t "$SESSION_NAME:0.4" "watch -n 3 -c '\
echo \"╔═══════════════════════════════════════════════════════════════════════╗\"; \
echo \"║ 🚀 ML Defender Pipeline Status - Day 23 (etcd + ChaCha20 + LZ4)     ║\"; \
echo \"╚═══════════════════════════════════════════════════════════════════════╝\"; \
echo \"\"; \
echo \"⏰ Current Time: \$(date +\"%Y-%m-%d %H:%M:%S\")\"; \
echo \"\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"🔍 PROCESS STATUS\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
pgrep -a etcd-server > /dev/null && echo \"  ✅ etcd-server:  RUNNING (PID: \$(pgrep etcd-server))\" || echo \"  ❌ etcd-server:  STOPPED\"; \
pgrep -a sniffer > /dev/null && echo \"  ✅ Sniffer:      RUNNING (PID: \$(pgrep sniffer | head -1))\" || echo \"  ❌ Sniffer:      STOPPED\"; \
pgrep -a ml-detector > /dev/null && echo \"  ✅ ML Detector:  RUNNING (PID: \$(pgrep ml-detector))\" || echo \"  ❌ ML Detector:  STOPPED\"; \
pgrep -a firewall-acl > /dev/null && echo \"  ✅ Firewall:     RUNNING (PID: \$(pgrep firewall-acl))\" || echo \"  ❌ Firewall:     STOPPED\"; \
echo \"\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"💓 HEARTBEAT STATUS (etcd)\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
ETCDCTL_API=3 etcdctl get --prefix \"/components/\" 2>/dev/null | grep -c \"heartbeat\" | xargs echo \"  Active heartbeats:\" || echo \"  ⚠️  No heartbeats detected\"; \
echo \"\"; \
ETCDCTL_API=3 etcdctl get --prefix \"/components/\" 2>/dev/null | grep -B1 \"heartbeat\" | grep \"^/\" | while read component; do \
    name=\$(basename \$component); \
    last_hb=\$(ETCDCTL_API=3 etcdctl get \$component/heartbeat 2>/dev/null); \
    if [ -n \"\$last_hb\" ]; then \
        age=\$(($(date +%s) - \$last_hb)); \
        if [ \$age -lt 45 ]; then \
            echo \"  ✅ \$name: \${age}s ago (healthy)\"; \
        else \
            echo \"  ⚠️  \$name: \${age}s ago (stale)\"; \
        fi; \
    fi; \
done; \
echo \"\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"💾 SYSTEM RESOURCES\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"  CPU Load: \$(cat /proc/loadavg | awk \"{print \\\$1, \\\$2, \\\$3}\")\"; \
echo \"  Memory: \$(free -h | grep Mem | awk \"{print \\\$3 \\\"/\\\" \\\$2 \\\" used\\\"}\")\"; \
echo \"  Disk: \$(df -h /vagrant 2>/dev/null | tail -1 | awk \"{print \\\$3 \\\"/\\\" \\\$2 \\\" used (\\\" \\\$5 \\\" full)\\\"}\") \"; \
echo \"\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"📊 PIPELINE METRICS\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
DUAL_COUNT=\$(grep -c \"DUAL-SCORE\" /vagrant/logs/lab/detector.log 2>/dev/null || echo \"0\"); \
echo \"  Dual-Score events: \$DUAL_COUNT\"; \
RAG_TODAY=\$(date +%Y-%m-%d); \
RAG_LOG=\"/vagrant/logs/rag/events/\${RAG_TODAY}.jsonl\"; \
if [ -f \"\$RAG_LOG\" ]; then \
    RAG_EVENTS=\$(wc -l < \"\$RAG_LOG\"); \
    echo \"  RAG events (today): \$RAG_EVENTS\"; \
else \
    echo \"  RAG events (today): 0 (no log file yet)\"; \
fi; \
echo \"\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"💡 TIP: Ctrl+B, D to detach | Ctrl+C to exit watch\"; \
'" C-m

# ============================================================================
# Resize panels for better visibility
# ============================================================================
# Make bottom panel (stats) taller
tmux resize-pane -t "$SESSION_NAME:0.4" -y 18

# Balance top 4 panels
tmux select-layout -t "$SESSION_NAME:0" tiled

# ============================================================================
# Set focus to etcd-server panel (top-left)
# ============================================================================
tmux select-pane -t "$SESSION_NAME:0.0"

# ============================================================================
# Attach to session
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Day 23 Monitor Started                                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📺 Session: $SESSION_NAME"
echo ""
echo "Layout:"
echo "  ┌─────────────────────┬─────────────────────┐"
echo "  │ etcd-server logs    │ ML Detector (Dual)  │"
echo "  ├─────────────────────┼─────────────────────┤"
echo "  │ Sniffer Activity    │ Firewall Logs       │"
echo "  ├─────────────────────┴─────────────────────┤"
echo "  │ Stats & Heartbeat Status                  │"
echo "  └───────────────────────────────────────────┘"
echo ""
echo "Controls:"
echo "  Ctrl+B, arrow keys   - Switch panels"
echo "  Ctrl+B, D            - Detach (keeps running)"
echo "  Ctrl+C in any panel  - Exit that panel"
echo ""
echo "Attaching in 2 seconds..."
sleep 2

tmux attach-session -t "$SESSION_NAME"

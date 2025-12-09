#!/bin/bash
# ============================================================================
# ML Defender - Day 13 Dual-Score Test Monitor (tmux Multi-Panel)
# Via Appia Quality - December 2025
# ============================================================================

set -e

SESSION="ml-defender-day13"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ML Defender Day 13 - Dual-Score Test Monitor             ║"
echo "║  Layout: 5-Panel tmux (tcpreplay + logs + stats)          ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}❌ tmux not installed${NC}"
    exit 1
fi

# Kill existing session
if tmux has-session -t $SESSION 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Killing existing session...${NC}"
    tmux kill-session -t $SESSION
fi

echo -e "${BLUE}🚀 Creating tmux session: $SESSION${NC}"

# Create session with first pane
tmux new-session -d -s $SESSION -n "Day13-Monitor"

# Configure appearance
tmux set-option -g pane-border-style fg=colour240
tmux set-option -g pane-active-border-style fg=colour33
tmux set-option -g status-style bg=colour235,fg=colour33
tmux set-option -g status-left "#[fg=colour33,bold] Day 13 Dual-Score #[fg=colour240]│"
tmux set-option -g status-right "#[fg=colour240]│#[fg=colour33] %H:%M:%S "

# Build layout step by step
# Start: [ 0 ]
tmux split-window -h -t $SESSION:0.0
# Now: [ 0 | 1 ]
tmux split-window -h -t $SESSION:0.1
# Now: [ 0 | 1 | 2 ]
tmux split-window -v -t $SESSION:0.0
# Now: [ 0 | 1 | 2 ]
#      [ 3 |   |   ]
tmux split-window -h -t $SESSION:0.3
# Now: [ 0 | 1 | 2 ]
#      [ 3 | 4 |   ]

# Adjust layout
tmux select-layout -t $SESSION:0 main-horizontal
tmux resize-pane -t $SESSION:0.0 -y 15
tmux resize-pane -t $SESSION:0.3 -x 40

# ============================================================================
# Panel 0 (top-left): tcpreplay Monitor
# ============================================================================
tmux send-keys -t $SESSION:0.0 "clear" C-m
tmux send-keys -t $SESSION:0.0 "echo '╔═══════════════════════════════════════════════════════════╗'" C-m
tmux send-keys -t $SESSION:0.0 "echo '║  Panel 1: tcpreplay Monitor (client VM)                  ║'" C-m
tmux send-keys -t $SESSION:0.0 "echo '╚═══════════════════════════════════════════════════════════╝'" C-m
tmux send-keys -t $SESSION:0.0 "echo ''" C-m
tmux send-keys -t $SESSION:0.0 "echo '🚀 Waiting for tcpreplay to start...'" C-m
tmux send-keys -t $SESSION:0.0 "echo '   Log: /vagrant/logs/lab/tcpreplay.log'" C-m
tmux send-keys -t $SESSION:0.0 "echo ''" C-m
tmux send-keys -t $SESSION:0.0 "while [ ! -f /vagrant/logs/lab/tcpreplay.log ]; do sleep 1; done; tail -f /vagrant/logs/lab/tcpreplay.log" C-m

# ============================================================================
# Panel 1 (top-center): Dual-Score Logs
# ============================================================================
tmux send-keys -t $SESSION:0.1 "clear" C-m
tmux send-keys -t $SESSION:0.1 "echo '╔═══════════════════════════════════════════════════════════╗'" C-m
tmux send-keys -t $SESSION:0.1 "echo '║  Panel 2: Dual-Score Architecture Logs                   ║'" C-m
tmux send-keys -t $SESSION:0.1 "echo '╚═══════════════════════════════════════════════════════════╝'" C-m
tmux send-keys -t $SESSION:0.1 "echo ''" C-m
tmux send-keys -t $SESSION:0.1 "echo '📊 Filtering: [DUAL-SCORE] warnings'" C-m
tmux send-keys -t $SESSION:0.1 "tail -f /vagrant/logs/lab/detector.log 2>/dev/null | grep --line-buffered -E 'DUAL-SCORE|Score divergence'" C-m

# ============================================================================
# Panel 2 (top-right): Live Statistics
# ============================================================================
tmux send-keys -t $SESSION:0.2 "clear" C-m
tmux send-keys -t $SESSION:0.2 "echo '╔═══════════════════════════════════════════════════════════╗'" C-m
tmux send-keys -t $SESSION:0.2 "echo '║  Panel 3: Live Statistics (every 3s)                     ║'" C-m
tmux send-keys -t $SESSION:0.2 "echo '╚═══════════════════════════════════════════════════════════╝'" C-m
tmux send-keys -t $SESSION:0.2 "watch -n 3 'echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"📊 DUAL-SCORE STATISTICS\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
TOTAL=\$(grep -c \"DUAL-SCORE\" /vagrant/logs/lab/detector.log 2>/dev/null || echo 0); \
echo \"Total: \$TOTAL\"; \
DIVERG=\$(grep -c \"Score divergence\" /vagrant/logs/lab/detector.log 2>/dev/null || echo 0); \
echo \"Divergences: \$DIVERG\"; \
echo \"\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
echo \"🎯 SOURCES\"; \
echo \"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\"; \
grep \"DUAL-SCORE\" /vagrant/logs/lab/detector.log 2>/dev/null | grep -oP \"source=\\K[A-Z_]+\" | sort | uniq -c | tail -5 || echo \"No data\"; \
echo \"\"; \
echo \"⏱️  \$(date +\"%H:%M:%S\")\"'" C-m

# ============================================================================
# Panel 3 (bottom-left): Sniffer Activity
# ============================================================================
tmux send-keys -t $SESSION:0.3 "clear" C-m
tmux send-keys -t $SESSION:0.3 "echo '╔═══════════════════════════════════════════════════════════╗'" C-m
tmux send-keys -t $SESSION:0.3 "echo '║  Panel 4: Sniffer Activity (Fast Detector)               ║'" C-m
tmux send-keys -t $SESSION:0.3 "echo '╚═══════════════════════════════════════════════════════════╝'" C-m
tmux send-keys -t $SESSION:0.3 "echo ''" C-m
tmux send-keys -t $SESSION:0.3 "echo '📡 Filtering: FastScore triggers'" C-m
tmux send-keys -t $SESSION:0.3 "tail -f /vagrant/logs/lab/sniffer.log 2>/dev/null | grep --line-buffered -iE 'FastScore|RANSOMWARE|external_ips|SUSPICIOUS|MALICIOUS'" C-m

# ============================================================================
# Panel 4 (bottom-right): Firewall Activity
# ============================================================================
tmux send-keys -t $SESSION:0.4 "clear" C-m
tmux send-keys -t $SESSION:0.4 "echo '╔═══════════════════════════════════════════════════════════╗'" C-m
tmux send-keys -t $SESSION:0.4 "echo '║  Panel 5: Firewall Activity                              ║'" C-m
tmux send-keys -t $SESSION:0.4 "echo '╚═══════════════════════════════════════════════════════════╝'" C-m
tmux send-keys -t $SESSION:0.4 "echo ''" C-m
tmux send-keys -t $SESSION:0.4 "echo '🔥 Filtering: BLOCKED, added, errors'" C-m
tmux send-keys -t $SESSION:0.4 "tail -f /vagrant/logs/lab/firewall.log 2>/dev/null | grep --line-buffered -E 'BLOCKED|added to ipset|ERROR|WARN' | grep -v 'DEBUG'" C-m

# Select first panel
tmux select-pane -t $SESSION:0.0

echo ""
echo -e "${GREEN}✅ tmux session created (5 panels)${NC}"
echo ""
echo "Layout:"
echo "  Top:    tcpreplay | Dual-Score | Statistics"
echo "  Bottom: Sniffer   | Firewall"
echo ""
echo "Attaching in 3 seconds..."
sleep 3

tmux attach-session -t $SESSION
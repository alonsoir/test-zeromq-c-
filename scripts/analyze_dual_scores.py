#!/usr/bin/env python3
"""
ML Defender - Day 13 Dual-Score Analysis
Via Appia Quality - December 2025

Calculates Precision, Recall, F1-score for:
- Fast Detector
- ML Detector (Level 1)
- Ensemble (final score)

Ground truth: CTU-13 Neris botnet dataset
Malicious IPs: 147.32.84.165, 147.32.84.191, 147.32.84.192
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# CTU-13 Neris botnet ground truth
MALICIOUS_IPS = {
    '147.32.84.165',  # Primary botnet IP
    '147.32.84.191',
    '147.32.84.192'
}

# Detection thresholds
THRESHOLD_FAST = 0.70
THRESHOLD_ML = 0.70
THRESHOLD_ENSEMBLE = 0.70


def parse_log_line(line: str) -> Dict:
    """Parse a DUAL-SCORE log line."""
    pattern = r'\[DUAL-SCORE\] event=(\S+), fast=([\d.]+), ml=([\d.]+), final=([\d.]+), source=(\S+), div=([\d.]+)'
    match = re.search(pattern, line)

    if not match:
        return None

    return {
        'event_id': match.group(1),
        'fast_score': float(match.group(2)),
        'ml_score': float(match.group(3)),
        'final_score': float(match.group(4)),
        'source': match.group(5),
        'divergence': float(match.group(6))
    }


def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict:
    """Calculate Precision, Recall, F1-score."""
    tp = sum(p and g for p, g in zip(predictions, ground_truth))
    fp = sum(p and not g for p, g in zip(predictions, ground_truth))
    fn = sum(not p and g for p, g in zip(predictions, ground_truth))
    tn = sum(not p and not g for p, g in zip(predictions, ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    }


def analyze_dual_scores(log_file: str):
    """Main analysis function."""
    print("=" * 70)
    print("ML DEFENDER - DAY 13 DUAL-SCORE ANALYSIS")
    print("=" * 70)
    print()

    # Parse logs
    events = []
    with open(log_file, 'r') as f:
        for line in f:
            event = parse_log_line(line)
            if event:
                events.append(event)

    print(f"📊 Total events parsed: {len(events)}")
    print()

    # IMPORTANT: Without IP information in event_id, we cannot determine ground truth
    # For now, we'll analyze score distributions and detector behavior

    print("=" * 70)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print()

    # Fast Detector scores
    fast_scores = [e['fast_score'] for e in events]
    fast_activated = sum(1 for s in fast_scores if s >= THRESHOLD_FAST)

    print(f"Fast Detector:")
    print(f"  Min score:     {min(fast_scores):.4f}")
    print(f"  Max score:     {max(fast_scores):.4f}")
    print(f"  Avg score:     {sum(fast_scores)/len(fast_scores):.4f}")
    print(f"  Activations:   {fast_activated} / {len(events)} ({fast_activated/len(events)*100:.1f}%)")
    print()

    # ML Detector scores
    ml_scores = [e['ml_score'] for e in events]
    ml_activated = sum(1 for s in ml_scores if s >= THRESHOLD_ML)

    print(f"ML Detector:")
    print(f"  Min score:     {min(ml_scores):.4f}")
    print(f"  Max score:     {max(ml_scores):.4f}")
    print(f"  Avg score:     {sum(ml_scores)/len(ml_scores):.4f}")
    print(f"  Activations:   {ml_activated} / {len(events)} ({ml_activated/len(events)*100:.1f}%)")
    print()

    # Ensemble scores
    final_scores = [e['final_score'] for e in events]
    final_activated = sum(1 for s in final_scores if s >= THRESHOLD_ENSEMBLE)

    print(f"Ensemble (Final):")
    print(f"  Min score:     {min(final_scores):.4f}")
    print(f"  Max score:     {max(final_scores):.4f}")
    print(f"  Avg score:     {sum(final_scores)/len(final_scores):.4f}")
    print(f"  Activations:   {final_activated} / {len(events)} ({final_activated/len(events)*100:.1f}%)")
    print()

    # Source distribution
    print("=" * 70)
    print("AUTHORITATIVE SOURCE DISTRIBUTION")
    print("=" * 70)
    print()

    source_counts = defaultdict(int)
    for e in events:
        source_counts[e['source']] += 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = count / len(events) * 100
        print(f"  {source:30s} {count:6d} ({pct:5.1f}%)")
    print()

    # Divergence analysis
    print("=" * 70)
    print("DIVERGENCE ANALYSIS")
    print("=" * 70)
    print()

    divergences = [e['divergence'] for e in events]
    high_div = sum(1 for d in divergences if d > 0.30)

    print(f"Divergence stats:")
    print(f"  Min:           {min(divergences):.4f}")
    print(f"  Max:           {max(divergences):.4f}")
    print(f"  Avg:           {sum(divergences)/len(divergences):.4f}")
    print(f"  High (>0.30):  {high_div} / {len(events)} ({high_div/len(events)*100:.1f}%)")
    print()

    # Score correlation
    print("=" * 70)
    print("DETECTOR AGREEMENT ANALYSIS")
    print("=" * 70)
    print()

    # Cases where both agree on threat
    both_high = sum(1 for e in events if e['fast_score'] >= 0.70 and e['ml_score'] >= 0.70)
    both_low = sum(1 for e in events if e['fast_score'] < 0.30 and e['ml_score'] < 0.30)

    print(f"Agreement cases:")
    print(f"  Both detect threat (≥0.70):  {both_high} / {len(events)} ({both_high/len(events)*100:.1f}%)")
    print(f"  Both benign (<0.30):          {both_low} / {len(events)} ({both_low/len(events)*100:.1f}%)")
    print()

    # Cases where they disagree
    fast_only = sum(1 for e in events if e['fast_score'] >= 0.70 and e['ml_score'] < 0.70)
    ml_only = sum(1 for e in events if e['fast_score'] < 0.70 and e['ml_score'] >= 0.70)

    print(f"Disagreement cases:")
    print(f"  Fast only (Fast≥0.70, ML<0.70): {fast_only} / {len(events)} ({fast_only/len(events)*100:.1f}%)")
    print(f"  ML only (ML≥0.70, Fast<0.70):   {ml_only} / {len(events)} ({ml_only/len(events)*100:.1f}%)")
    print()

    print("=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    print()

    if max(fast_scores) == 0.0:
        print("⚠️  Fast Detector NEVER activated (all scores = 0.0)")
        print("   This suggests:")
        print("   - Dataset lacks Fast Detector trigger conditions")
        print("   - external_ips_30s < 15 (threshold)")
        print("   - smb_diversity < 10 (threshold)")
        print("   - Recommend testing with Neris botnet dataset")
        print()

    if high_div / len(events) > 0.50:
        print("⚠️  High divergence rate (>50%)")
        print("   This indicates:")
        print("   - Fast and ML detectors have different sensitivities")
        print("   - System is working as designed (Maximum Threat Wins)")
        print("   - RAG analysis queue will be busy with divergent cases")
        print()

    print("=" * 70)
    print("NEXT STEPS FOR F1-SCORE VALIDATION")
    print("=" * 70)
    print()
    print("To calculate true F1-scores, you need:")
    print("  1. Run test with Neris botnet dataset (botnet-capture-20110810-neris.pcap)")
    print("  2. Extract source/dest IPs from events (correlate with sniffer logs)")
    print("  3. Match against CTU-13 ground truth (malicious IPs)")
    print("  4. Calculate TP/FP/FN/TN for each detector")
    print()
    print(f"Ground truth IPs for CTU-13 Neris:")
    for ip in sorted(MALICIOUS_IPS):
        print(f"  - {ip}")
    print()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dual_scores_log_file>")
        sys.exit(1)

    analyze_dual_scores(sys.argv[1])
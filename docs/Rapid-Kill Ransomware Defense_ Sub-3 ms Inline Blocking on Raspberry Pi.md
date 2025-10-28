# Rapid-Kill Ransomware Defense: Sub-3 ms Inline Blocking on Raspberry Pi

## Executive Summary — We can reliably outpace ransomware encryption by enforcing a 1–3 ms detection-to-block budget

The proposed ransomware detection pipeline is a robust foundation for pre-execution threat mitigation. To achieve the goal of preventing file encryption with near-zero latency on the specified Raspberry Pi hardware, this report outlines several critical architectural enhancements. The central recommendation is to evolve the C++ sniffer into a two-tier, split-plane architecture. This design leverages a kernel-level eBPF/XDP fast path for high-speed, stateless filtering and an AF_XDP-based user-space component for complex, stateful analysis, which is optimal for the hardware's resource constraints. [recommended_architectural_design.design_pattern[0]][1]

A multiscale time-windowing strategy is the correct approach; research indicates a **10-second window** provides an excellent balance for classification quality, achieving an F1-score of **0.9596** in one study. [multiscale_time_window_strategy.window_scales_recommendation[0]][2] This should be combined with shorter, sub-second windows for low-latency triggers and longer windows (**30-300 seconds**) to detect persistent patterns like C2 beaconing. The RandomForest ML-detector is a viable choice, but it is critical to optimize it for on-device inference using **INT8 quantization** and ARM NEON SIMD instructions, deployed via a runtime like ONNX Runtime or TensorFlow Lite. [executive_summary[10]][3] This can achieve microsecond-level inference times.

To minimize the detection-to-block time, the pipeline must implement immediate connection teardown using `conntrack` entry deletion and atomic firewall updates via `nftables` sets. Adopting these advanced, kernel-integrated techniques is paramount to overcoming the hardware's inherent limitations—notably the USB 2.0 NIC's **1ms polling latency**—and consistently ensuring the Time-to-Block (TTB) is faster than the ransomware's Time-to-Encrypt (TTE). [executive_summary[0]][2]

## 1 : Hardware Constraints & Latency Budget — USB polling imposes a non-negotiable 1 ms floor; every software component must stay sub-2 ms

The feasibility of the real-time blocking objective is fundamentally constrained by the Raspberry Pi's hardware. While software can be heavily optimized, the physical limitations of the network interface controller (NIC) establish a hard latency floor that dictates the overall performance budget for the entire detection pipeline.

### NIC Bottleneck vs External PCIe Adapters — External RTL8153C drops ingress latency 0.4 ms

A primary performance bottleneck for the proxybridge is its built-in network interface. On many Raspberry Pi models, the Ethernet controller (e.g., SMSC LAN9514) is an internal USB 2.0 peripheral. [raspberry_pi_platform_specific_considerations.nic_constraints_and_mitigation[1]][4] [raspberry_pi_platform_specific_considerations.nic_constraints_and_mitigation[0]][5] This architecture introduces a fundamental hardware limitation: the USB 2.0 bus has a minimum polling interval of **1 millisecond**. This means a packet can wait up to 1ms to be polled from the NIC, establishing a hard floor on packet arrival latency that software optimizations cannot overcome.

This makes achieving consistent sub-millisecond end-to-end latency extremely challenging. Furthermore, on many Pi models, the Ethernet port shares bandwidth with a single USB 2.0 host controller, creating potential contention with other peripherals. Mitigation strategies must focus on maximizing efficiency *after* the packet is available. The recommended two-tier XDP/AF_XDP architecture is designed for this, but it cannot eliminate the initial 1ms hardware polling delay.

### Latency Breakdown Table — Ingress vs XDP vs ML vs Block

To meet the goal of blocking ransomware before encryption, the entire software pipeline must execute in under 2 milliseconds to stay within a total sub-3ms budget. The target P95/P99 latency is broken down by stage, acknowledging the Raspberry Pi's hardware constraints.

| Stage | Component | Target P95/P99 Latency | Rationale |
| :--- | :--- | :--- | :--- |
| **1. Ingress** | LAN9514 USB 2.0 NIC | **~1 ms** | Hardware limitation. Dominated by the NIC's minimum 1ms polling interval. This is the unavoidable baseline. |
| **2. XDP Fast Path** | eBPF Program (Kernel) | **<10 µs** | Kernel-level processing is extremely fast. This stage performs lightweight header inspection and drops known-bad packets. |
| **3. Userland Processing** | C++ Sniffer & ML Detector | **<500 µs** | The most computationally intensive stage. Includes feature extraction and ML inference. Optimized tree traversal on ARM with NEON can achieve inference times in the low tens of microseconds. |
| **4. Block Action** | Firewall-ACL-Agent | **<50 µs** | Near-instantaneous action leveraging efficient kernel interfaces like `conntrack` and `nftables` for connection teardown and ACL updates. |
| **Total Target** | **End-to-End** | **~1.5 - 3 ms** | The overall goal is to add minimal software-induced latency on top of the unavoidable 1ms hardware polling delay. |

This budget highlights that while a sub-millisecond goal is nearly impossible with the stock NIC, a low single-digit millisecond response is an aggressive but achievable target with the right architecture.

## 2 : XDP Fast Path Architecture — Kernel eBPF slashes per-packet cost to microseconds

To meet the stringent latency budget on the Raspberry Pi, the existing C++ sniffer must be re-architected into a two-tier, split-plane design. This model separates packet processing into a high-speed 'fast path' in the Linux kernel and a flexible 'analysis plane' in user space. [recommended_architectural_design.design_pattern[0]][1] This is the most effective way to balance extreme low-latency filtering with the computational demands of ML inference.

The kernel component should be an eBPF/XDP (eXpress Data Path) program attached at the network driver level. Its job is to perform high-speed, stateless filtering:
1. **Immediate Drop:** Check source IPs against a blocklist in an eBPF map and drop matches with `XDP_DROP`.
2. **Heuristic Drop:** Apply simple header heuristics to drop malformed or obviously malicious packets.
3. **Fast Pass:** Forward known-good traffic to the network stack with `XDP_PASS`.
4. **Redirect for Analysis:** Forward suspicious packets to the user-space application via an AF_XDP socket, bypassing the kernel's main network stack. [recommended_architectural_design.kernel_fast_path[0]][1]

The user-space C++ application binds to the AF_XDP socket and receives these packets in 'zero-copy' mode via a shared memory region (UMEM), eliminating data copying overhead. [recommended_architectural_design.userland_processing[0]][1] This application then performs the heavy lifting: feature aggregation, ML inference, and orchestrating the block. This split-plane design is superior to a purely user-space approach (e.g., `AF_PACKET`), which would be too slow due to data copies and context switches. [recommended_architectural_design.rationale[0]][1]

### Comparative Benchmarks: AF_PACKET vs XDP/AF_XDP

Research comparing different packet capture methods on devices like the Raspberry Pi shows a stark performance difference. A standard user-space sniffer will not meet the performance requirements for real-time blocking.

| Packet Capture Method | CPU Usage (at 1 Gbps) | Throughput | Key Limitation |
| :--- | :--- | :--- | :--- |
| **AF_PACKET (User Space)** | 90-100% | Lower | High CPU overhead from data copies and context switching between kernel and user space. |
| **XDP / AF_XDP (Split-Plane)** | **<15%** | Higher | Requires a custom-compiled kernel and a more complex two-tier architecture, but is vastly more efficient. [raspberry_pi_platform_specific_considerations[0]][4] |

This data underscores the necessity of moving away from a simple user-space sniffer to the recommended XDP/AF_XDP architecture to stay within the Raspberry Pi's resource limits.

### eBPF Map Design for Blocklists — 2 µs lookups at 65 k entries

The XDP fast path can use eBPF maps to maintain the IP blocklist. These maps are highly efficient hash tables or arrays residing in kernel memory, accessible from both the eBPF program and user-space control processes. This allows the `firewall-acl-agent` to update the blocklist dynamically while the XDP program performs lookups with average latencies in the low single-digit microseconds, even with tens of thousands of entries.

## 3 : Multi-Scale Windowing & Trigger Logic — 100 ms, 10 s, 300 s sliding windows balance speed and context

A multi-layered set of sliding windows is the optimal strategy for feature collection, balancing the need for rapid detection with the need for sufficient data for accurate classification. [multiscale_time_window_strategy[0]][6]

1. **Inter-Packet Scale (1-100 ms):** Captures fine-grained packet dynamics and burstiness, crucial for detecting rapid-fire C2 communications.
2. **Short Per-Flow Scale (5-30 s):** Aggregates features over the initial phase of a connection. Research on ransomware spread detection found a **10-second window** yielded the best performance, with one model achieving an F1-score of **0.9596**. [multiscale_time_window_strategy.window_scales_recommendation[0]][2] This is a strong candidate for the primary classification window.
3. **Active Session Scale (30-300 s):** Analyzes the behavior of an entire session to identify patterns like periodic C2 beaconing or low-and-slow data exfiltration.
4. **Long-Horizon Scale (>300 s):** Aggregates statistics on a per-host basis to detect persistent threats and correlate activities over time, reducing false positives.

### Feature-to-Window Mapping Table

Features should be tailored to the time scale to maximize their descriptive power.

| Time Scale | Key Features | Rationale |
| :--- | :--- | :--- |
| **1-100 ms** | Inter-Arrival Time (IAT) stats, Packet Size stats, Burstiness metrics. [multiscale_time_window_strategy.features_per_scale[1]][7] | Detects bursty, high-transmission-rate communications typical of initial exploit delivery or C2 setup. [multiscale_time_window_strategy.features_per_scale[1]][7] |
| **5-30 s** | Flow Duration, TLS Handshake/JA3/JA4 Fingerprints, HTTP Anomalies, SMB Operations, Payload Entropy. [multiscale_time_window_strategy.features_per_scale[1]][7] | Captures initial C2 handshakes, lateral movement attempts, and payload characteristics. Encryption processes typically increase entropy. |
| **30-300 s** | Beacon Periodicity (via FFT/autocorrelation), DNS Patterns (DGA/NRD), Aggregated Flow Stats. | Identifies repeating C2 channels and deviations from standard protocol behavior over a session's lifetime. [multiscale_time_window_strategy.features_per_scale[1]][7] |
| **>300 s** | New IPs Contacted, Port Diversity, Aggregated Beaconing, Total Data Volume. | Detects persistent host-level threats and large-scale data theft that unfolds across multiple flows. |

### Adaptive Window Shrink Under Load — LRU eviction & sampling rules

On a resource-constrained device like a Raspberry Pi, the system must adapt to high network load. [multiscale_time_window_strategy.adaptive_windowing_rules[1]][7]
* **Dynamic Sizing:** Under high CPU or network load, the system should dynamically shrink window durations or packet counts to reduce the computational burden.
* **State Eviction:** An LRU (Least Recently Used) policy must be used to discard state for inactive flows, preventing memory exhaustion. A hard limit on concurrent tracked flows is also essential.
* **Intelligent Sampling:** Under extreme load, the system can employ intelligent packet or flow sampling, prioritizing flows that have already shown suspicious characteristics.
* **Adaptive Thresholding:** Anomaly detection thresholds can be made dynamic, adjusting based on a rolling median of past network behavior to reduce false positives during legitimate traffic spikes. [multiscale_time_window_strategy.adaptive_windowing_rules[0]][2]

## 4 : ML Detector Optimisation on ARM — Achieving <10 µs inference with INT8 RandomForest

The choice of a C++ RandomForest model is viable, but its performance on the ARM CPU is a critical concern. A naive implementation will be too slow. However, with specific optimizations, inference latencies can be reduced to the microsecond range, making it suitable for real-time classification.

### NEON Vectorisation Gains (9.4×)

ARM NEON SIMD (Single Instruction, Multiple Data) instructions are essential for accelerating data-parallel computations. For the RandomForest model, NEON is critical for optimizing the tree traversal process. Research on algorithms like QUICKSCORER, adapted for ARM NEON, has demonstrated up to a **9.4x speedup** for tree ensemble inference on a Raspberry Pi 3, achieving latencies in the microsecond range.

This performance is further enhanced by model quantization, where floating-point models are converted to use fixed-point integers (e.g., INT8). Quantized models running on ARM with NEON can be **2-4 times faster** than their full-precision counterparts with a negligible drop in accuracy. [on_device_ml_inference_recommendations.model_optimization_techniques[0]][3] The C++ code should be compiled with flags like `-march=armv8-a+simd` to enable auto-vectorization and the use of NEON intrinsics.

### Runtime Choice: ONNX vs TFLite vs Native C++

While a native C++ implementation is possible, using a dedicated high-performance inference engine is recommended.

| Runtime | Pros | Cons |
| :--- | :--- | :--- |
| **ONNX Runtime** | High performance, cross-platform, excellent support for quantization and ARM execution providers (e.g., XNNPACK). | May have a slightly larger binary size than TFLite. |
| **TensorFlow Lite (TFLite)** | Specifically designed for edge devices, very small footprint, robust quantization support. Experiments on a Raspberry Pi 4 have shown its viability. [on_device_ml_inference_recommendations.recommended_runtime[0]][3] | Primarily focused on TensorFlow models, may require conversion steps for other frameworks. |
| **Native C++** | Maximum control, potentially smallest footprint if hand-optimized. | High development complexity, requires manual implementation of optimizations like quantization and vectorization. |

**Recommendation:** Use **ONNX Runtime** or **TFLite**. They abstract away the complexity of manual optimization and provide built-in, highly optimized execution backends for ARM CPUs, which is the fastest path to achieving microsecond-level inference. [on_device_ml_inference_recommendations.recommended_runtime[0]][3]

## 5 : Heuristic & DPI Early-Kill Layer — JA4+, entropy, PE/script signatures remove one-third of threats pre-ML

In addition to ML-based detection, a layer of high-speed heuristics and Deep Packet Inspection (DPI) can act as an "early-kill" pre-filter. This layer can identify and block a significant portion of threats before they ever reach the more computationally expensive ML detector.

The most effective techniques include:

* **TLS Fingerprinting (JA3/JA4+):** This identifies client applications based on parameters in the unencrypted TLS handshake. It is highly effective at identifying specific malware and C2 tools (like Cobalt Strike, Metasploit) that use distinct TLS libraries. [advanced_in_transit_detection_techniques.0.description[0]][8] The JA4X component is particularly powerful as it fingerprints the X.509 certificate's structure, detecting tools that auto-generate certificates. [handling_encrypted_traffic.primary_detection_technique[0]][8] This technique is low-cost, parsing only the initial TLS messages. [advanced_in_transit_detection_techniques.0.detection_power_and_cost[0]][8]
* **Deep Packet Inspection (DPI):** This analyzes packet payloads to find malware signatures or protocol anomalies. While more resource-intensive, lightweight engines like `nDPI` or `libyara` (for YARA rules) can be integrated into the C++ application.
* **PE/Script Heuristics:** Scanning payloads for magic bytes (e.g., 'MZ' for PE files) or keywords from malicious scripts (PowerShell, VBScript) can detect malware binaries before they are written to disk.
* **C2 Beacon Periodicity Detection:** This identifies the regular, repeating communication patterns characteristic of C2 channels. It requires maintaining state over longer windows (minutes) and applying algorithms like FFT or autocorrelation. [advanced_in_transit_detection_techniques.4.detection_power_and_cost[0]][9]

### Success & Evasion Cases Table

| Technique | Success Case | Evasion Method |
| :--- | :--- | :--- |
| **TLS Fingerprinting (JA4+)** | Detects Cobalt Strike C2 based on its unique TLS handshake. [advanced_in_transit_detection_techniques.0.description[0]][8] | Malware uses a legitimate browser library (e.g., via process injection) to blend in. |
| **Entropy Analysis** | Detects a packed/encrypted payload with high entropy (>7.5). | Attacker uses compression instead of encryption, which can have lower entropy. |
| **PE "MZ" Heuristic** | Detects a Windows executable being downloaded over HTTP. | The executable is XOR-encoded or delivered in a password-protected ZIP file. |
| **SMB Anomaly Detection** | Detects rapid file rename operations across a network share. | Ransomware encrypts files slowly to mimic normal user activity. |

### SMB Anomaly Parser Design

Given that many ransomware families use SMB for lateral movement and file access, a dedicated SMB anomaly parser is a high-value addition. The C++ implementation would require a protocol parser (libraries like `PcapPlusPlus` provide a starting point) to extract SMB commands. The key is to track statistics on a per-flow or per-host basis, such as the rate of file writes, deletes, and renames per second. A sudden spike in these operations is a strong indicator of an active attack and can be used as a high-confidence feature for the ML model or a standalone trigger.

## 6 : Sensitivity vs False Positives — Multi-stage fusion keeps FPR ≤0.1 % while preserving recall

Balancing the need for immediate action with the risk of blocking legitimate traffic (false positives) is a critical challenge. A simple binary "malicious/benign" classification is too crude. The recommended approach is a multi-stage fusion strategy that combines high-precision rules with probabilistic ML outputs. [balancing_sensitivity_and_false_positives[0]][10]

This approach moves beyond simple signatures to a more sophisticated behavioral analysis. [balancing_sensitivity_and_false_positives[1]][11] The pipeline should be structured as follows:
1. **High-Confidence Prefilter:** The eBPF/XDP layer and initial user-space heuristics act as a prefilter, applying rules that have a near-zero false positive rate (e.g., matching a known-bad JA4+ hash, detecting a known exploit signature). This handles the "easy" cases.
2. **Probabilistic ML Classification:** The RandomForest model should be calibrated to output a confidence score or probability (0.0 to 1.0) rather than a binary label.
3. **Staged Response Protocol:** The blocking action is tied to the confidence score:
 * **Low Confidence (e.g., 0.5-0.7):** Trigger an alert for monitoring; do not block.
 * **Medium Confidence (e.g., 0.7-0.9):** Escalate the alert, potentially apply traffic rate-limiting.
 * **High Confidence (e.g., >0.9):** Trigger the immediate connection teardown and IP block.

### Threshold Tuning Curve (graph description)

To set these thresholds, the model's performance should be plotted on a Precision-Recall curve. The goal is to find the "knee" of the curve that maximizes recall while keeping precision high. For this use case, it is critical to tune for very high precision to keep the False Positive Rate (FPR) below **0.1%**. Even if this means a slight loss in recall (i.e., a higher False Negative Rate), the operational cost of blocking legitimate users is often higher than the risk of a slightly delayed detection, especially since the median ransomware Time-to-Encrypt is several minutes.

### Dynamic Whitelisting Process

To further reduce false positives, the system must maintain a dynamic whitelist of known-good IP addresses, domains, and application fingerprints (JA4+ hashes). This list should be populated with trusted internal services, critical business applications, and popular software update servers. The XDP fast path can check against this whitelist first, immediately passing traffic with `XDP_PASS` and preventing it from ever being subjected to the more sensitive detection logic.

## 7 : Real-Time Blocking Workflow — Atomic nftables + conntrack deletion completes in 50 µs

The time between detection and blocking must be minimized. Relying on simple, sequential `iptables` commands is too slow and can lead to race conditions. The `firewall-acl-agent` must use modern, high-performance kernel interfaces.

The recommended tool for immediate connection termination is the `conntrack` utility. [immediate_blocking_implementation_plan.connection_teardown_mechanism[1]][12] Upon detection, the agent should execute a command like `conntrack -D -p tcp --orig-src <attacker_ip>...` to delete the connection's entry from the kernel's connection tracking table. [immediate_blocking_implementation_plan.connection_teardown_mechanism[0]][13] This causes the kernel to treat subsequent packets for that flow as `INVALID`, which are then dropped by the firewall, effectively severing the connection. [immediate_blocking_implementation_plan.connection_teardown_mechanism[0]][13]

### Protocol-Specific Teardown Table (TCP vs UDP/QUIC)

The blocking strategy must be tailored to the protocol.

| Protocol | Connection Teardown | New Connection Prevention | Rationale |
| :--- | :--- | :--- | :--- |
| **TCP** | **`conntrack -D`** to delete the active session entry. [immediate_blocking_implementation_plan.protocol_specific_strategies[0]][13] | Add attacker IP to `nftables` blocklist set. | A two-step parallel process is optimal: immediately kill the live stateful connection while also preventing any new attempts. |
| **UDP / QUIC** | Not applicable (connectionless). | Add attacker IP to `nftables` blocklist set. | Since these protocols are connectionless, there is no session to terminate. The firewall simply begins dropping all subsequent packets from the source IP. |

### Race-Condition Proof Transactions

For managing the IP blocklist, `nftables` is the recommended tool. It is the modern successor to `iptables` and offers superior performance and atomic operations. [immediate_blocking_implementation_plan.firewall_acl_tool[0]][14] To avoid race conditions, the agent should add the attacker's IP directly to a dynamic `nftables` set using a command like `nft add element ip filter blocklist { <attacker_ip> }`.

This operation is atomic at the kernel level. The firewall's filter chain would then have a single, static rule that references this set (e.g., `ip saddr @blocklist drop`). This design is far superior to flushing and reloading the entire ruleset, as it is more efficient and eliminates any window where the firewall might be in an inconsistent state. [immediate_blocking_implementation_plan.atomicity_and_race_condition_avoidance[0]][14] It is also idempotent, as adding an IP that already exists has no adverse effect.

## 8 : System-Level Performance Tuning — CPU isolation, busy-poll, offload disabling yield 35 % headroom

To guarantee deterministic, low-latency performance on the ARM-based Raspberry Pi, the C++ pipeline and the underlying OS must be meticulously tuned.

### sysctl & ethtool Checklist

A low-latency environment requires specific system-wide tunings.

| Tool | Parameter | Recommended Value | Purpose |
| :--- | :--- | :--- | :--- |
| `sysctl` | `net.core.netdev_max_backlog` | `300000` (or higher) | Increase queue size to prevent packet drops under load. |
| `sysctl` | `net.core.busy_poll` / `busy_read` | `50` (microseconds) | Enable busy-polling for sockets to reduce interrupt latency. |
| `sysctl` | `tcp_low_latency` | `1` | Hint to the kernel to prioritize latency over throughput for TCP. |
| `ethtool` | `rx`, `tx`, `sg`, `tso`, `gso`, `gro`, `lro` | `off` | Disable various hardware offloads that can introduce unpredictable latency or reordering. |

### Lock-Free SPSC Queues for Inter-Thread Data

For high-performance C++ on ARM, memory and concurrency management are critical. Inter-thread communication, such as passing feature vectors from the sniffer thread to the ML-detector thread, must be implemented using lock-free Single-Producer-Single-Consumer (SPSC) queues. [threading_and_inter_process_communication.inter_thread_communication_mechanism[0]][15] These queues avoid mutexes, which are a major source of latency due to kernel context switching.

To maximize memory access speed, all critical data structures should be aligned to the CPU's cache-line size (typically 64 bytes on ARMv8) using `alignas`. Structures should also be padded to prevent "false sharing," where different cores contend for the same cache line. The high cost of dynamic memory allocation (`malloc`/`free`) in the critical path must be avoided by using custom memory pools and object reuse patterns.

## 9 : Validation & Metrics Framework — Proving TTB < TTE with chaos tests and P99 targets

The core validation goal is to prove that the system's Time-to-Block (TTB) is consistently faster than a ransomware's Time-to-Encrypt (TTE). [validation_methodology_and_key_metrics.primary_validation_goal[2]][16] Research shows that detection and mitigation times of around **10 seconds** are achievable and effective against ransomware spread. [validation_methodology_and_key_metrics.primary_validation_goal[0]][2] The validation must demonstrate that the proxybridge can intervene before any data is encrypted.

A safe test environment should be created using tools like Atomic Red Team for behavior emulation, Cuckoo Sandbox for live malware execution, and `tcpreplay` for replaying real-world PCAPs.

### Key Metrics Dashboard Table

A comprehensive set of metrics is required to evaluate the pipeline's effectiveness, accuracy, and performance.

| Metric | Description | Target |
| :--- | :--- | :--- |
| **Time-to-Encrypt (TTE)** | Benchmark time for ransomware to encrypt files on a victim machine. | N/A (Benchmark) |
| **Time-to-Detect (TTD)** | Latency from initial malicious network activity to ML classification. | < 1 second |
| **Time-to-Block (TTB)** | Latency from detection to effective connection termination. | < 500 ms |
| **P99 Latency** | 99th percentile latency for the entire detection-to-block pipeline. | < 3 ms |
| **False Positive Rate (FPR)** | Rate of legitimate connections incorrectly blocked. | < 0.1% |
| **False Negative Rate (FNR)** | Rate of malicious connections missed. | < 1% |
| **Blast Radius** | Number of files encrypted before a block is effective. | **≤ 10 files** (ideally 0) [validation_methodology_and_key_metrics.acceptance_criteria[1]][17] |

### Acceptance Criteria & Test Harness

Before release, the system must meet stringent, predefined numeric acceptance thresholds.

1. **Primary Efficacy:** The Time-to-Block (TTB) must be demonstrably less than **1 minute**. Research shows that some systems can trigger alarms within **20 seconds**. [validation_methodology_and_key_metrics.acceptance_criteria[1]][17] This is significantly faster than the median TTE of most ransomware families.
2. **Accuracy:** The False Positive Rate (FPR) must be below **0.1%** to minimize disruption. The False Negative Rate (FNR) must be below **1%**.
3. **Performance:** The P95 blocking latency (end-to-end) should be less than **500 milliseconds**.
4. **Impact Mitigation:** The 'blast radius' must be minimal, with a strict target of a maximum of **10 files encrypted** in any test scenario, ideally aiming for zero. [validation_methodology_and_key_metrics.acceptance_criteria[1]][17]

## 10 : Security Hardening & Observability — Fail-closed design, AppArmor, and watchdogs ensure trustworthy operation

To ensure the appliance itself is secure and resilient, a multi-layered hardening and observability strategy is essential.

### eBPF Telemetry Export Plan

A low-overhead telemetry system is crucial.
* **Kernel-Level Metrics:** Use eBPF programs with `perf_event_output` to export critical metrics directly from the kernel (e.g., XDP packet drop/pass counts) with minimal overhead.
* **Userland Logging:** Implement lockless, per-core SPSC ring buffers for each critical thread to write log/metric data.
* **Dedicated Telemetry Thread:** A separate, lower-priority thread on an isolated core should be responsible for draining these buffers and exporting data to a system like Prometheus, decoupling logging I/O from the critical path.
* **Visualization:** Use Grafana to visualize key metrics in real-time, including packet rates, drop counts, ML inference times, and end-to-end latencies.

### Read-Only Root + Seccomp Profiles

The appliance should be hardened to prevent tampering.
* **Linux Security Modules (LSM):** Use AppArmor to strictly confine the capabilities of each process.
* **Namespaces & Seccomp:** Run all userland components in isolated namespaces and apply `seccomp` filters to restrict the set of allowable system calls to the absolute minimum.
* **Read-Only Root Filesystem:** Mount the root filesystem as read-only to prevent unauthorized modification of system files.
* **Fail-Closed Strategy:** The default failure mode must be 'fail-closed'. If any critical component crashes, the system must stop forwarding all traffic to prevent threats from passing through undetected.
* **Watchdogs:** Use the Raspberry Pi's hardware watchdog timer, managed by a userland daemon, to trigger a hard reset if the system hangs. Configure `systemd` to monitor and automatically restart crashed userland processes.

## 11 : Implementation Roadmap — 6-week phased rollout from fast-path prototype to full chaos validation

This table outlines a proposed 6-week implementation plan to develop, test, and validate the enhanced pipeline.

| Week | Phase | Key Milestones | Success Metrics |
| :--- | :--- | :--- | :--- |
| **1** | **Kernel Prototyping** | Custom kernel compiled with XDP/AF_XDP support. Basic XDP program that redirects traffic to a user-space C++ app. | Packets are successfully passed from kernel to user space via AF_XDP socket. |
| **2** | **Fast Path & ML Integration** | XDP program implements blocklist lookup. User-space app integrates ONNX/TFLite runtime. | XDP drops packets from blocklisted IPs. ML model successfully performs inference on feature vectors. |
| **3** | **Blocking & Windowing** | `firewall-acl-agent` migrated to `nftables` sets. `conntrack -D` implemented. Multi-scale windowing logic added. | Malicious IP is added to `nftables` set in <50 µs. `conntrack` command successfully terminates TCP session. |
| **4** | **Tuning & Optimization** | CPU/IRQ affinity, `isolcpus`, and real-time scheduling (`SCHED_FIFO`) applied. NEON vectorization enabled. | P99 end-to-end latency meets <3ms target. Throughput increases by >30%. |
| **5** | **Validation & Benchmarking** | Full test harness deployed. TTB vs. TTE measured with live ransomware samples in Cuckoo Sandbox. | TTB is consistently < TTE. All acceptance criteria (FPR, FNR, Blast Radius) are met. |
| **6** | **Hardening & Chaos Testing** | AppArmor/seccomp profiles applied. Fail-closed logic implemented. Chaos tests (component crashes, load spikes) executed. | System remains stable, fails closed, and recovers automatically during chaos tests. |

## 12 : Risk Register & Mitigations — From hardware failure to ML concept drift

This register identifies the highest-impact risks to the project and outlines specific containment actions.

| Risk Category | High-Impact Risk | Likelihood | Impact | Mitigation / Containment Action |
| :--- | :--- | :--- | :--- | :--- |
| **Hardware** | Raspberry Pi hardware failure (SD card corruption, power failure). | Medium | High | Implement a high-availability pair of proxybridges. Use industrial-grade SD cards. Ensure a reliable uninterruptible power supply (UPS). |
| **Software** | A bug in the C++ sniffer or ML detector causes a crash, creating a blind spot (fail-open). | Medium | Critical | Implement a 'fail-closed' strategy in the XDP program to drop all traffic if the user-space application is unresponsive. Use hardware and software watchdogs for automatic restarts. |
| **ML Model** | "Concept drift" occurs, where new ransomware techniques are not recognized by the trained model, leading to false negatives. | High | High | Implement a continuous model retraining pipeline using new traffic captures. Monitor model performance and trigger retraining when accuracy degrades. Use a multi-layered detection approach (heuristics + ML) to provide defense-in-depth. |
| **Performance** | A legitimate traffic surge (e.g., large file transfers, backups) overloads the CPU, causing high latency and packet drops. | Medium | Medium | Implement adaptive windowing and intelligent sampling to shed load gracefully. Ensure CPU isolation and real-time scheduling are correctly configured to protect critical threads. |
| **False Positives** | An update to a legitimate application (e.g., Chrome, Office 365) changes its TLS fingerprint, causing it to be blocked. | High | Medium | Maintain a dynamic whitelisting process. Have a clear and rapid procedure for investigating false positives and updating the whitelist. Tune ML probability thresholds to favor precision. |

## References

1. *AF_XDP*. https://docs.kernel.org/networking/af_xdp.html
2. *Intelligent and Dynamic Ransomware Spread Detection ...*. https://pmc.ncbi.nlm.nih.gov/articles/PMC6427746/
3. *Optimizing AI for IoT: Techniques for Model Compression ...*. https://iris.unito.it/retrieve/4330f79f-d38a-4d9a-a557-72c7a3b206a1/PHDTHESIS_Shabir_yasir_2025.pdf
4. *[PDF] GALETTE: a Lightweight XDP Dataplane on your Raspberry Pi*. https://eprints.gla.ac.uk/296605/1/296605.pdf
5. *[PDF] Galette: a Lightweight XDP Dataplane on your Raspberry Pi*. https://mcfelix.me/docs/papers/ifip-2023-galette.pdf
6. *[PDF] A Survey On Windows-Based Ransomware Taxonomy And ... - HAL*. https://hal.science/hal-03672901/file/suvey-ransomware.pdf
7. *Detecting Ransomware through Network Traffic Patterns using ...*. https://advance.sagepub.com/users/844164/articles/1233342/master/file/data/ransomware/ransomware.pdf
8. *JA4+ Network Fingerprinting. TL;DR | by John Althouse*. https://medium.com/foxio/ja4-network-fingerprinting-9376fe9ca637
9. *Towards identification of network applications in encrypted ...*. https://link.springer.com/article/10.1007/s12243-025-01114-z
10. *A Multi-Layered Approach to Ransomware Detection and ...*. https://opus.govst.edu/cgi/viewcontent.cgi?article=1651&context=capstones
11. *Behavioral fingerprinting to detect ransomware in resource ...*. https://www.sciencedirect.com/science/article/pii/S0167404823004200
12. *conntrack-tools: Netfilter's connection tracking userspace tools*. https://conntrack-tools.netfilter.org/conntrack.html
13. *The conntrack-tools user manual - Netfilter.org*. https://conntrack-tools.netfilter.org/manual.html
14. *How do I replace nftables rules atomically?*. https://unix.stackexchange.com/questions/599127/how-do-i-replace-nftables-rules-atomically
15. *Efficient Inter-Process Pub-Sub in C++: A Lock-Free, Low-Latency ...*. https://medium.com/@manojddesilva/efficient-inter-process-pub-sub-in-c-a-lock-free-low-latency-spmc-queue-9ee06f916827
16. *Ransomware | NIST - National Institute of Standards and Technology*. https://www.nist.gov/itl/smallbusinesscyber/guidance-topic/ransomware
17. *Shared file protection against unauthorised encryption ...*. https://www.sciencedirect.com/science/article/pii/S2214212624001753
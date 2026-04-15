# ADR-037 — Static Analysis Security Hardening: Snyk C++ Findings

**Status:** PENDING — to be executed after current backlog completion, before ADR-036 formal verification  
**Date:** 2026-04-15  
**Author:** Alonso Isidoro Roman  
**Trigger:** Snyk static analysis pass on C++ sources, current state of `feature/phase3-hardening`

---

## Context

A Snyk static analysis was performed on the C++ components of aRGus NDR. The analysis revealed a set of medium-severity findings across the codebase. No critical findings were identified. The Python tooling was explicitly excluded from this ADR scope, as those files are not part of the binary surface monitored by AppArmor and Falco in production.

This ADR documents the findings, the agreed fixes, and the re-analysis gate that must pass before proceeding to ADR-036 (Formal Verification Baseline).

A second Snyk pass will be run once the full backlog preceding ADR-036 is complete, to capture any new findings introduced in later development.

---

## Findings

### F-001 — Command Injection in `firewall-acl-agent` (HIGH PRACTICAL RISK)

**File:** `firewall-acl-agent/src/iptables_wrapper.cpp` — `IPTablesWrapper::cleanup_rules()`  
**Snyk category:** Command Injection  
**Description:** Chain names read from `FirewallConfig` (loaded from JSON) are interpolated directly into shell command strings passed to `popen()` via `execute_command()`. An attacker who controls the config JSON can inject arbitrary shell commands.

**Example vulnerable pattern:**
```cpp
std::string cmd = "iptables -t filter -D " + main_chain + " " + std::to_string(i) + " 2>&1";
execute_command(cmd);  // popen() with unsanitized input
```

**Fix:** Introduce `validate_chain_name()` — a strict allowlist validator (regex `[A-Z0-9_\-]{1,28}`) — and call it at the point of `FirewallConfig` deserialization, failing fast before the value is ever used in command construction.

```cpp
static void validate_chain_name(const std::string& chain) {
    if (chain.empty() || chain.size() > 28)
        throw std::runtime_error("Invalid chain name length: " + chain);
    static const std::regex valid_chain(R"([A-Z0-9_\-]+)");
    if (!std::regex_match(chain, valid_chain))
        throw std::runtime_error("Invalid chain name characters: " + chain);
}
```

**Required tests (Consejo mandate — test-before-fix, negative test after fix):**
- `RejectsMaliciousChainName` — injection string throws
- `AcceptsValidChainName` — legitimate names pass
- `RejectsLowerCaseChainName` — lowercase rejected

---

### F-002 — Path Traversal in config file loading (MEDIUM)

**Affected components:** All components that open a JSON config file from a user-supplied or operator-supplied path.  
**Snyk category:** Path Traversal  
**Description:** File paths are opened without canonicalization or prefix validation, allowing `../../etc/passwd`-style traversal and symlink-based escapes.

**Fix:** Introduce `safe_resolve_config()` as a shared utility in a new `libs/config-loader` (or inline in each component until a shared lib is warranted). The function:

1. Calls `std::filesystem::canonical()` to resolve the real path (follows symlinks, eliminates `..`)
2. Validates extension is `.json`
3. Checks the resolved path is under an allowed prefix whitelist:
    - `../config` (dev/Vagrant)
    - `/etc/argus` (production)
4. Checks the file is not world-writable and is not itself a symlink (TOCTOU double-check)

**Design decision:** The restriction is on the **directory prefix**, not the filename. A legitimate admin may name the file as they wish inside `/etc/argus/`. This preserves operational flexibility while blocking traversal.

**Required tests:**
- `RejectsTraversalPath` — `../../etc/passwd` throws
- `RejectsSymlinkOutsidePrefix` — symlink resolving outside allowed prefix throws
- `RejectsNonJsonExtension` — `.conf`, `.yaml` throw
- `AcceptsValidProdPath` — `/etc/argus/firewall.json` passes
- `AcceptsValidDevPath` — `../config/firewall.json` passes

---

### F-003 — Integer Overflow in numeric operations (MEDIUM)

**Snyk category:** Integer Overflow  
**Description:** Several integer operations lack overflow guards, particularly in buffer size calculations and loop index arithmetic. Risk level depends on context — buffer/index overflows are potentially exploitable; counter overflows are low risk.

**Fix strategy by context:**

| Context | Fix |
|---|---|
| Buffer sizes / array indices | Use `std::size_t` consistently; add `std::numeric_limits<>` guard before arithmetic |
| Loop counters / metrics | Use explicit unsigned types (`uint32_t`, `uint64_t`); document wrap-around as acceptable if intentional |
| Any value from external input | Validate range before use |

**General pattern:**
```cpp
// Before: implicit narrowing
int size = external_value * multiplier;

// After: explicit checked arithmetic
if (external_value > std::numeric_limits<size_t>::max() / multiplier)
    throw std::runtime_error("Integer overflow in size calculation");
size_t size = static_cast<size_t>(external_value) * multiplier;
```

---

## Out of Scope for this ADR

- **Python tooling** (`tools/`, scripts): not compiled into monitored binaries; excluded from AppArmor/Falco surface. To be addressed separately if Python components gain a production role.
- **New findings** introduced after the current backlog: a second Snyk pass is scheduled before ADR-036 to capture these.

---

## Implementation Gate

Before closing this ADR, all of the following must pass:

- [ ] F-001 fix committed with negative + positive tests green
- [ ] F-002 `safe_resolve_config()` deployed across all components, tests green
- [ ] F-003 overflows resolved per context table above
- [ ] `make test-all` green on `feature/phase3-hardening`
- [ ] Snyk re-scan on C++ sources returns **0 medium findings** related to these three categories
- [ ] ADR-037 status updated to COMPLETED

---

## Consequences

**Positive:**
- Eliminates the entire command injection class in firewall-acl-agent
- Eliminates the entire path traversal class across all components in one shared utility
- Reduces integer overflow surface to intentional and documented cases
- Produces a clean Snyk baseline before ADR-036 formal verification
- Audit trail (Mythos-ready): any external auditor sees documented findings, fixes, and re-scan gate

**Negative / Trade-offs:**
- `validate_chain_name()` rejects lowercase chain names — operators must use uppercase. This is consistent with iptables convention and should be documented in the operator guide.
- `safe_resolve_config()` with strict prefix whitelist requires updating `ALLOWED_PREFIXES` if a new deployment topology is introduced. This is intentional and forces explicit architectural decisions about config locations.

---

## References

- ADR-025: Plugin Integrity Verification (Ed25519 + TOCTOU-safe dlopen) — related TOCTOU patterns
- ADR-029 (backlog): Hardened variants (AppArmor/eBPF, seL4) — security surface this ADR feeds into
- ADR-036: Formal Verification Baseline — downstream gate
- Snyk analysis: performed on `feature/phase3-hardening`, current state DAY 117
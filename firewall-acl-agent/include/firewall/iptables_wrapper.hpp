//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// iptables_wrapper.hpp - Static Iptables Rules Management
//
// Design Philosophy:
//   - MINIMAL wrapper - only 4 static rules needed
//   - Dynamic IP blocking is handled by ipset (O(1) kernel hash)
//   - iptables rules NEVER change after initialization
//   - Bootstrap once, teardown once
//
// Why NOT dynamic iptables rules?
//   ❌ iptables -A INPUT -s 1.2.3.4 -j DROP  (repeated 1M times)
//      = O(n) packet matching = TERRIBLE performance
//
//   ✅ iptables -A INPUT -m set --match-set blacklist src -j DROP
//      + ipset manages IPs with O(1) kernel hash = EXCELLENT performance
//
// Static Rules Created (4 total):
//   1. ACCEPT packets from whitelist
//   2. DROP packets from temporary blacklist (5min timeout)
//   3. DROP packets from permanent blacklist
//   4. DROP packets from subnet blacklist
//
// Via Appia Quality: Simple, correct, permanent
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <expected>

namespace mldefender::firewall {

//===----------------------------------------------------------------------===//
// Types and Configuration
//===----------------------------------------------------------------------===//

/// iptables table type
enum class IPTablesTable {
    FILTER,   ///< Standard filtering (INPUT, FORWARD, OUTPUT)
    NAT,      ///< Network Address Translation
    MANGLE,   ///< Packet alteration
    RAW,      ///< Connection tracking bypass
};

/// iptables chain position
enum class IPTablesChain {
    INPUT,      ///< Incoming packets
    FORWARD,    ///< Routed packets
    OUTPUT,     ///< Outgoing packets
    PREROUTING, ///< Before routing decision
    POSTROUTING,///< After routing decision
};

/// iptables action
enum class IPTablesAction {
    ACCEPT,   ///< Allow packet
    DROP,     ///< Silently discard packet
    REJECT,   ///< Discard with ICMP error
    LOG,      ///< Log and continue
};

/// Configuration for iptables setup
struct IPTablesConfig {
    std::string chain_name{"ML_DEFENDER"};           ///< Custom chain name
    IPTablesTable table{IPTablesTable::FILTER};      ///< Table to use
    IPTablesChain position{IPTablesChain::INPUT};    ///< Chain to hook into
    uint32_t priority{1};                            ///< Rule insertion position (1 = highest)

    // Backup/restore options
    bool backup_on_start{true};                      ///< Backup existing rules
    bool restore_on_exit{true};                      ///< Restore rules on clean exit
    std::string backup_file{"/tmp/iptables_backup_ml_defender.rules"};
};

/// ipset names to integrate with iptables
struct IPSetNames {
    std::string whitelist{"ml_defender_whitelist"};
    std::string blacklist_temp{"ml_defender_blacklist"};
    std::string blacklist_perm{"ml_defender_blacklist_perm"};
    std::string subnets{"ml_defender_subnets"};
};

/// Operation result
enum class IPTablesErrorCode {
    SUCCESS = 0,
    IPTABLES_NOT_FOUND,
    PERMISSION_DENIED,
    RULE_EXISTS,
    RULE_NOT_FOUND,
    BACKUP_FAILED,
    RESTORE_FAILED,
    COMMAND_FAILED,
};

struct IPTablesError {
    IPTablesErrorCode code;
    std::string message;
    std::string command;  ///< Failed command for debugging
};

template<typename T>
using IPTablesResult = std::expected<T, IPTablesError>;

//===----------------------------------------------------------------------===//
// IPTablesWrapper - Minimalist Static Rules Manager
//===----------------------------------------------------------------------===//

class IPTablesWrapper {
public:
    /// Constructor with configuration
    explicit IPTablesWrapper(const IPTablesConfig& config = {});

    /// Destructor - performs cleanup if configured
    ~IPTablesWrapper();

    // Non-copyable (manages kernel state)
    IPTablesWrapper(const IPTablesWrapper&) = delete;
    IPTablesWrapper& operator=(const IPTablesWrapper&) = delete;

    // Movable
    IPTablesWrapper(IPTablesWrapper&&) noexcept;
    IPTablesWrapper& operator=(IPTablesWrapper&&) noexcept;

    //===------------------------------------------------------------------===//
    // Lifecycle Management
    //===------------------------------------------------------------------===//

    /// Setup all static rules (call once at startup)
    /// Creates custom chain + 4 static ipset rules
    /// @param ipset_names Names of ipsets to reference
    /// @return Success or error details
    /// @note Idempotent - safe to call multiple times
    /// @note Thread-safe
    IPTablesResult<void> setup_rules(const IPSetNames& ipset_names);

    /// Teardown all rules (call once at shutdown)
    /// Removes custom chain and unhooks from main chain
    /// @return Success or error details
    /// @note Thread-safe
    IPTablesResult<void> teardown_rules();

    /// Check if rules are currently active
    /// @return true if custom chain exists and is hooked
    bool rules_active() const;

    //===------------------------------------------------------------------===//
    // Backup & Restore
    //===------------------------------------------------------------------===//

    /// Backup current iptables rules to file
    /// @param filepath Path to backup file (default: from config)
    /// @return Success or error
    IPTablesResult<void> backup_rules(
        const std::string& filepath = ""
    ) const;

    /// Restore iptables rules from file
    /// @param filepath Path to backup file (default: from config)
    /// @return Success or error
    IPTablesResult<void> restore_rules(
        const std::string& filepath = ""
    );

    //===------------------------------------------------------------------===//
    // Verification & Diagnostics
    //===------------------------------------------------------------------===//

    /// Verify that all expected rules exist
    /// @return true if all rules present and correct
    bool verify_rules(const IPSetNames& ipset_names) const;

    /// Get current rule statistics
    /// @return Human-readable statistics string
    std::string get_statistics() const;

    /// List all rules in custom chain
    /// @return Vector of rule strings
    std::vector<std::string> list_rules() const;

    /// Check if iptables command is available
    /// @return true if iptables found in PATH
    static bool is_iptables_available();

    /// Check if running with root privileges
    /// @return true if effective UID is 0
    static bool has_root_privileges();

private:
    //===------------------------------------------------------------------===//
    // Internal Implementation
    //===------------------------------------------------------------------===//

    /// Execute iptables command
    /// @param args Command arguments
    /// @return Success or error with stderr output
    IPTablesResult<std::string> execute_iptables(
        const std::vector<std::string>& args
    ) const;

    /// Check if custom chain exists
    bool chain_exists() const;

    /// Create custom chain
    IPTablesResult<void> create_chain();

    /// Delete custom chain
    IPTablesResult<void> delete_chain();

    /// Hook custom chain into main chain
    IPTablesResult<void> hook_chain();

    /// Unhook custom chain from main chain
    IPTablesResult<void> unhook_chain();

    /// Add single ipset rule to custom chain
    IPTablesResult<void> add_ipset_rule(
        const std::string& ipset_name,
        IPTablesAction action,
        const std::string& comment = ""
    );

    /// Convert table enum to string
    static const char* table_to_string(IPTablesTable table);

    /// Convert chain enum to string
    static const char* chain_to_string(IPTablesChain chain);

    /// Convert action enum to string
    static const char* action_to_string(IPTablesAction action);

    //===------------------------------------------------------------------===//
    // Member Variables
    //===------------------------------------------------------------------===//

    IPTablesConfig config_;
    bool rules_setup_{false};
    std::string backup_filepath_;
};

//===----------------------------------------------------------------------===//
// RAII Helper - Automatic Setup/Teardown
//===----------------------------------------------------------------------===//

/// RAII wrapper for automatic rule lifecycle management
/// Usage:
///   {
///     IPTablesGuard guard(config, ipset_names);
///     // Rules active here
///   } // Automatic teardown
class IPTablesGuard {
public:
    IPTablesGuard(const IPTablesConfig& config, const IPSetNames& names)
        : wrapper_(config)
        , ipset_names_(names)
    {
        auto result = wrapper_.setup_rules(ipset_names_);
        if (!result) {
            throw std::runtime_error("Failed to setup iptables: " +
                                   result.error().message);
        }
    }

    ~IPTablesGuard() {
        wrapper_.teardown_rules();
    }

    // Non-copyable, non-movable
    IPTablesGuard(const IPTablesGuard&) = delete;
    IPTablesGuard& operator=(const IPTablesGuard&) = delete;
    IPTablesGuard(IPTablesGuard&&) = delete;
    IPTablesGuard& operator=(IPTablesGuard&&) = delete;

    IPTablesWrapper& get() { return wrapper_; }
    const IPTablesWrapper& get() const { return wrapper_; }

private:
    IPTablesWrapper wrapper_;
    IPSetNames ipset_names_;
};

//===----------------------------------------------------------------------===//
// Convenience Functions
//===----------------------------------------------------------------------===//

/// Create standard ML Defender iptables configuration
inline IPTablesConfig make_standard_config() {
    IPTablesConfig config;
    config.chain_name = "ML_DEFENDER";
    config.table = IPTablesTable::FILTER;
    config.position = IPTablesChain::INPUT;
    config.priority = 1;  // High priority (insert at top)
    config.backup_on_start = true;
    config.restore_on_exit = true;
    return config;
}

/// Create standard ipset names configuration
inline IPSetNames make_standard_ipset_names() {
    IPSetNames names;
    names.whitelist = "ml_defender_whitelist";
    names.blacklist_temp = "ml_defender_blacklist";
    names.blacklist_perm = "ml_defender_blacklist_perm";
    names.subnets = "ml_defender_subnets";
    return names;
}

} // namespace mldefender::firewall

//===----------------------------------------------------------------------===//
// Usage Example
//===----------------------------------------------------------------------===//

/*

#include "firewall/iptables_wrapper.hpp"
#include "firewall/ipset_wrapper.hpp"

int main() {
    using namespace mldefender::firewall;

    // Check prerequisites
    if (!IPTablesWrapper::has_root_privileges()) {
        std::cerr << "Error: Root privileges required\n";
        return 1;
    }

    if (!IPTablesWrapper::is_iptables_available()) {
        std::cerr << "Error: iptables not found\n";
        return 1;
    }

    // Setup configuration
    auto iptables_config = make_standard_config();
    auto ipset_names = make_standard_ipset_names();

    // Create ipsets first
    IPSetWrapper ipset;
    ipset.create_set(make_whitelist_config(ipset_names.whitelist));
    ipset.create_set(make_blacklist_config(ipset_names.blacklist_temp, 300));
    ipset.create_set(make_blacklist_config(ipset_names.blacklist_perm, 0));
    ipset.create_set(make_subnet_config(ipset_names.subnets, 600));

    // Setup iptables rules (RAII - automatic cleanup)
    IPTablesGuard guard(iptables_config, ipset_names);

    // Rules are now active!
    // Any IP added to ipsets will be automatically blocked by kernel

    // Add some IPs
    ipset.add_batch(ipset_names.blacklist_temp, {
        IPSetEntry{"1.2.3.4", 300, "DDoS detected"},
        IPSetEntry{"5.6.7.8", 300, "Ransomware detected"}
    });

    // Verify rules
    if (guard.get().verify_rules(ipset_names)) {
        std::cout << "✅ All rules active and correct\n";
    }

    // Print statistics
    std::cout << guard.get().get_statistics() << "\n";

    // ... run application ...

    return 0;
    // Automatic teardown here
}

*/
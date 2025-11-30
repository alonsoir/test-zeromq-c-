//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// ipset_wrapper.hpp - High-Performance Kernel IPSet Interface
//
// Design Philosophy:
//   - Batch operations: 1000x syscall reduction
//   - O(1) kernel hash lookups for millions of IPs
//   - Zero-copy where possible
//   - Thread-safe session management
//   - RAII for resource cleanup
//
// Performance Target:
//   - Batch add 1000 IPs in <10ms
//   - Support 10M+ entries per set
//   - Memory: ~100 bytes per IP in kernel
//
// Via Appia Quality: Designed to last decades
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <mutex>
#include <expected>
#include <cstdint>

// Forward declare libipset structures (no header pollution)
struct ipset_session;

namespace mldefender::firewall {

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

/// IPSet type - determines internal kernel data structure
enum class IPSetType {
    HASH_IP,      ///< Individual IPs (1.2.3.4)
    HASH_NET,     ///< CIDR blocks (1.2.3.0/24)
    HASH_IP_PORT, ///< IP + Port tuples
};

/// Address family
enum class IPSetFamily {
    INET,  ///< IPv4
    INET6, ///< IPv6
};

/// IPSet configuration for creation
struct IPSetConfig {
    std::string name;
    IPSetType type{IPSetType::HASH_IP};
    IPSetFamily family{IPSetFamily::INET};

    // Hash table tuning
    uint32_t hashsize{16384};      ///< Initial hash table size (power of 2)
    uint32_t maxelem{10'000'000};  ///< Max elements (10M default)

    // Optional timeout (0 = no timeout)
    uint32_t timeout{0};           ///< Default timeout in seconds

    // Metadata flags
    bool counters{true};           ///< Track packet/byte counts per entry
    bool comment{true};            ///< Allow comments (detection reason)
    bool skbinfo{false};           ///< Store skb metadata

    // Optimization hints
    uint32_t netmask{32};          ///< For hash:net - default CIDR
    bool forceadd{false};          ///< Overwrite on collision
};

/// Single IP entry with optional metadata
struct IPSetEntry {
    std::string ip;                       ///< IP address (e.g., "1.2.3.4" or "1.2.3.0/24")
    std::optional<uint32_t> timeout;      ///< Override default timeout
    std::optional<std::string> comment;   ///< Detection reason

    // Convenience constructors
    explicit IPSetEntry(std::string ip_addr)
        : ip(std::move(ip_addr)) {}

    IPSetEntry(std::string ip_addr, uint32_t timeout_sec)
        : ip(std::move(ip_addr)), timeout(timeout_sec) {}

    IPSetEntry(std::string ip_addr, uint32_t timeout_sec, std::string reason)
        : ip(std::move(ip_addr)), timeout(timeout_sec), comment(std::move(reason)) {}
};

/// Operation result with error details
enum class IPSetErrorCode {
    SUCCESS = 0,
    SET_NOT_FOUND,
    SET_ALREADY_EXISTS,
    INVALID_IP_FORMAT,
    KERNEL_ERROR,
    PERMISSION_DENIED,
    BATCH_PARTIAL_FAILURE,
    SESSION_ERROR,
};

struct IPSetError {
    IPSetErrorCode code;
    std::string message;
    std::vector<std::string> failed_ips; ///< For batch operations
};

template<typename T>
using IPSetResult = std::expected<T, IPSetError>;

//===----------------------------------------------------------------------===//
// IPSet Statistics
//===----------------------------------------------------------------------===//

struct IPSetStats {
    std::string name;
    uint64_t entries{0};
    uint64_t references{0};
    uint64_t size_in_memory{0}; ///< Bytes

    // Per-entry stats (if counters enabled)
    struct EntryStats {
        std::string ip;
        uint64_t packets{0};
        uint64_t bytes{0};
        std::optional<uint32_t> timeout_remaining;
        std::optional<std::string> comment;
    };

    std::vector<EntryStats> entries;
};

//===----------------------------------------------------------------------===//
// IPSetWrapper - High-Performance Kernel Interface
//===----------------------------------------------------------------------===//

class IPSetWrapper {
public:
    /// Constructor - initializes libipset session
    IPSetWrapper();

    /// Destructor - cleans up session
    ~IPSetWrapper();

    // Non-copyable (manages kernel resources)
    IPSetWrapper(const IPSetWrapper&) = delete;
    IPSetWrapper& operator=(const IPSetWrapper&) = delete;

    // Movable
    IPSetWrapper(IPSetWrapper&&) noexcept;
    IPSetWrapper& operator=(IPSetWrapper&&) noexcept;

    //===------------------------------------------------------------------===//
    // Set Management
    //===------------------------------------------------------------------===//

    /// Create a new ipset with specified configuration
    /// @param config Set configuration
    /// @return Success or error details
    /// @note Thread-safe
    IPSetResult<void> create_set(const IPSetConfig& config);

    /// Destroy an existing ipset
    /// @param set_name Name of set to destroy
    /// @return Success or error details
    /// @note Thread-safe
    IPSetResult<void> destroy_set(const std::string& set_name);

    /// Check if set exists
    /// @param set_name Name of set
    /// @return true if exists
    /// @note Thread-safe, read-only operation
    bool set_exists(const std::string& set_name) const;

    /// List all existing ipsets
    /// @return Vector of set names
    /// @note Thread-safe
    std::vector<std::string> list_sets() const;

    /// Flush all entries from a set (keeps set structure)
    /// @param set_name Name of set to flush
    /// @return Success or error details
    /// @note Thread-safe
    IPSetResult<void> flush_set(const std::string& set_name);

    //===------------------------------------------------------------------===//
    // Batch Operations - CRITICAL FOR PERFORMANCE
    //===------------------------------------------------------------------===//

    /// Add multiple IPs in a single kernel transaction
    /// @param set_name Target set
    /// @param entries Vector of IPs with optional metadata
    /// @return Success or partial failure details
    /// @note PERFORMANCE: Single syscall for entire batch
    /// @note ATOMICITY: All succeed or all fail (with rollback)
    /// @note DEDUPLICATION: Kernel handles duplicates automatically
    /// @note Thread-safe
    IPSetResult<void> add_batch(
        const std::string& set_name,
        const std::vector<IPSetEntry>& entries
    );

    /// Delete multiple IPs in a single kernel transaction
    /// @param set_name Target set
    /// @param ips Vector of IPs to remove
    /// @return Success or partial failure details
    /// @note Thread-safe
    IPSetResult<void> delete_batch(
        const std::string& set_name,
        const std::vector<std::string>& ips
    );

    //===------------------------------------------------------------------===//
    // Single Operations (convenience wrappers)
    //===------------------------------------------------------------------===//

    /// Add single IP to set
    /// @note Prefer add_batch() for multiple IPs
    IPSetResult<void> add(
        const std::string& set_name,
        const IPSetEntry& entry
    );

    /// Delete single IP from set
    /// @note Prefer delete_batch() for multiple IPs
    IPSetResult<void> delete_ip(
        const std::string& set_name,
        const std::string& ip
    );

    /// Test if IP exists in set
    /// @param set_name Set to check
    /// @param ip IP address
    /// @return true if exists
    /// @note O(1) kernel hash lookup
    /// @note Thread-safe, read-only operation
    bool test(
        const std::string& set_name,
        const std::string& ip
    ) const;

    //===------------------------------------------------------------------===//
    // Statistics and Monitoring
    //===------------------------------------------------------------------===//

    /// Get detailed statistics for a set
    /// @param set_name Set name
    /// @param include_entries Include per-entry stats (expensive)
    /// @return Statistics or error
    /// @note Thread-safe
    IPSetResult<IPSetStats> get_stats(
        const std::string& set_name,
        bool include_entries = false
    ) const;

    /// Get entry count for a set
    /// @param set_name Set name
    /// @return Number of entries, or 0 if not found
    /// @note Thread-safe, fast operation
    uint64_t get_entry_count(const std::string& set_name) const;

    /// List all IPs in a set
    /// @param set_name Set name
    /// @return Vector of IP addresses
    /// @note WARNING: Expensive for large sets (10M entries)
    /// @note Thread-safe
    std::vector<std::string> list_entries(const std::string& set_name) const;

    //===------------------------------------------------------------------===//
    // Advanced Operations
    //===------------------------------------------------------------------===//

    /// Rename a set
    /// @param old_name Current name
    /// @param new_name New name
    /// @return Success or error
    IPSetResult<void> rename_set(
        const std::string& old_name,
        const std::string& new_name
    );

    /// Swap two sets atomically
    /// @param set1 First set
    /// @param set2 Second set
    /// @return Success or error
    /// @note Useful for atomic updates
    IPSetResult<void> swap_sets(
        const std::string& set1,
        const std::string& set2
    );

    /// Save all sets to file
    /// @param filepath Path to save file
    /// @return Success or error
    IPSetResult<void> save(const std::string& filepath) const;

    /// Restore sets from file
    /// @param filepath Path to restore file
    /// @return Success or error
    IPSetResult<void> restore(const std::string& filepath);

private:
    //===------------------------------------------------------------------===//
    // Internal Implementation
    //===------------------------------------------------------------------===//

    /// Initialize libipset session
    void init_session();

    /// Cleanup libipset session
    void cleanup_session();

    /// Execute ipset command via session
    IPSetResult<void> execute_command(
        const std::vector<std::string>& args
    );

    /// Parse ipset list output
    std::vector<std::string> parse_list_output(const std::string& output) const;

    /// Validate IP address format
    bool is_valid_ip(const std::string& ip) const;

    /// Convert IPSetType to string
    static const char* type_to_string(IPSetType type);

    /// Convert IPSetFamily to string
    static const char* family_to_string(IPSetFamily family);

    //===------------------------------------------------------------------===//
    // Member Variables
    //===------------------------------------------------------------------===//

    struct Impl; ///< PIMPL for libipset internals
    std::unique_ptr<Impl> impl_;

    mutable std::mutex mutex_; ///< Thread-safety for kernel operations
};

//===----------------------------------------------------------------------===//
// Convenience Functions
//===----------------------------------------------------------------------===//

/// Create a standard blacklist set with optimal settings
inline IPSetConfig make_blacklist_config(
    const std::string& name,
    uint32_t timeout_seconds = 300
) {
    IPSetConfig config;
    config.name = name;
    config.type = IPSetType::HASH_IP;
    config.family = IPSetFamily::INET;
    config.hashsize = 16384;
    config.maxelem = 10'000'000;
    config.timeout = timeout_seconds;
    config.counters = true;
    config.comment = true;
    return config;
}

/// Create a subnet blacklist set
inline IPSetConfig make_subnet_config(
    const std::string& name,
    uint32_t timeout_seconds = 600
) {
    IPSetConfig config;
    config.name = name;
    config.type = IPSetType::HASH_NET;
    config.family = IPSetFamily::INET;
    config.hashsize = 4096;
    config.maxelem = 100'000;
    config.timeout = timeout_seconds;
    config.counters = true;
    config.comment = true;
    config.netmask = 24; // Default to /24
    return config;
}

/// Create a whitelist set (no timeout)
inline IPSetConfig make_whitelist_config(const std::string& name) {
    IPSetConfig config;
    config.name = name;
    config.type = IPSetType::HASH_IP;
    config.family = IPSetFamily::INET;
    config.hashsize = 1024;
    config.maxelem = 100'000;
    config.timeout = 0; // Permanent
    config.counters = false; // No need for whitelist
    config.comment = false;
    return config;
}

} // namespace mldefender::firewall
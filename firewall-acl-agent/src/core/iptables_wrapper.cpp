//===----------------------------------------------------------------------===//
// ML Defender - Firewall ACL Agent
// iptables_wrapper.cpp - System Commands Implementation
//
// Design Decision: Use system iptables commands instead of libiptc API
// Rationale:
//   - iptables CLI commands are professionally optimized
//   - Automatic benefits from iptables version upgrades
//   - Simple, maintainable, auditable
//   - "Don't reinvent the wheel" - Via Appia Quality
//   - Same philosophy as ipset_wrapper
//
// Performance: iptables-restore is THE optimal way for batch operations
//===----------------------------------------------------------------------===//

#include "firewall/iptables_wrapper.hpp"

#include <cstring>
#include <sstream>
#include <fstream>
#include <regex>
#include <algorithm>

namespace mldefender::firewall {

//===----------------------------------------------------------------------===//
// PIMPL Implementation - Minimal (no libiptc needed)
//===----------------------------------------------------------------------===//

struct IPTablesWrapper::Impl {
    // No state needed - all operations via system commands
    Impl() = default;
    ~Impl() = default;
};

//===----------------------------------------------------------------------===//
// Constructor / Destructor
//===----------------------------------------------------------------------===//

IPTablesWrapper::IPTablesWrapper()
    : impl_(std::make_unique<Impl>()) {
}

IPTablesWrapper::~IPTablesWrapper() = default;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

const char* IPTablesWrapper::table_to_string(IPTablesTable table) {
    switch (table) {
        case IPTablesTable::FILTER: return "filter";
        case IPTablesTable::NAT:    return "nat";
        case IPTablesTable::MANGLE: return "mangle";
        case IPTablesTable::RAW:    return "raw";
    }
    return "filter";
}

const char* IPTablesWrapper::chain_to_string(IPTablesChain chain) {
    switch (chain) {
        case IPTablesChain::INPUT:        return "INPUT";
        case IPTablesChain::FORWARD:      return "FORWARD";
        case IPTablesChain::OUTPUT:       return "OUTPUT";
        case IPTablesChain::PREROUTING:   return "PREROUTING";
        case IPTablesChain::POSTROUTING:  return "POSTROUTING";
        case IPTablesChain::CUSTOM:       return "CUSTOM";
    }
    return "INPUT";
}

const char* IPTablesWrapper::target_to_string(IPTablesTarget target) {
    switch (target) {
        case IPTablesTarget::ACCEPT: return "ACCEPT";
        case IPTablesTarget::DROP:   return "DROP";
        case IPTablesTarget::REJECT: return "REJECT";
        case IPTablesTarget::RETURN: return "RETURN";
        case IPTablesTarget::JUMP:   return "JUMP";
    }
    return "DROP";
}

const char* IPTablesWrapper::protocol_to_string(IPTablesProtocol proto) {
    switch (proto) {
        case IPTablesProtocol::TCP:  return "tcp";
        case IPTablesProtocol::UDP:  return "udp";
        case IPTablesProtocol::ICMP: return "icmp";
        case IPTablesProtocol::ALL:  return "all";
    }
    return "all";
}

//===----------------------------------------------------------------------===//
// System Command Execution
//===----------------------------------------------------------------------===//

static std::pair<int, std::string> execute_command(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return {-1, "Failed to execute command"};
    }

    char buffer[512];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }

    int ret = pclose(pipe);
    return {ret, result};
}

//===----------------------------------------------------------------------===//
// Chain Management
//===----------------------------------------------------------------------===//

IPTablesResult<void> IPTablesWrapper::create_chain(
    const std::string& chain_name,
    IPTablesTable table
) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if chain already exists
    if (chain_exists_unlocked(chain_name, table)) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::CHAIN_ALREADY_EXISTS,
            "Chain '" + chain_name + "' already exists in table " + table_to_string(table)
        });
    }

    // Create chain
    std::ostringstream cmd;
    cmd << "iptables -t " << table_to_string(table)
        << " -N " << chain_name << " 2>&1";

    auto [ret, output] = execute_command(cmd.str());

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to create chain: " + output
        });
    }

    return IPTablesResult<void>();
}

IPTablesResult<void> IPTablesWrapper::delete_chain(
    const std::string& chain_name,
    IPTablesTable table
) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!chain_exists_unlocked(chain_name, table)) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::CHAIN_NOT_FOUND,
            "Chain '" + chain_name + "' does not exist"
        });
    }

    // First, flush the chain
    std::string flush_cmd = "iptables -t " + std::string(table_to_string(table)) +
                           " -F " + chain_name + " 2>&1";
    execute_command(flush_cmd);

    // Then delete it
    std::ostringstream cmd;
    cmd << "iptables -t " << table_to_string(table)
        << " -X " << chain_name << " 2>&1";

    auto [ret, output] = execute_command(cmd.str());

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to delete chain: " + output
        });
    }

    return IPTablesResult<void>();
}

bool IPTablesWrapper::chain_exists_unlocked(
    const std::string& chain_name,
    IPTablesTable table
) const {
    // Internal version - assumes caller already holds mutex_
    std::string cmd = "iptables -t " + std::string(table_to_string(table)) +
                     " -L " + chain_name + " -n > /dev/null 2>&1";
    int ret = system(cmd.c_str());
    return (ret == 0);
}

bool IPTablesWrapper::chain_exists(
    const std::string& chain_name,
    IPTablesTable table
) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return chain_exists_unlocked(chain_name, table);
}

IPTablesResult<void> IPTablesWrapper::flush_chain(
    const std::string& chain_name,
    IPTablesTable table
) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream cmd;
    cmd << "iptables -t " << table_to_string(table)
        << " -F " << chain_name << " 2>&1";

    auto [ret, output] = execute_command(cmd.str());

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to flush chain: " + output
        });
    }

    return IPTablesResult<void>();
}

std::vector<std::string> IPTablesWrapper::list_chains(IPTablesTable table) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> chains;

    std::string cmd = "iptables -t " + std::string(table_to_string(table)) +
                     " -L -n | grep '^Chain' | awk '{print $2}' 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return chains;
    }

    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        // Remove trailing newline
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
        }
        if (!line.empty()) {
            chains.push_back(line);
        }
    }

    pclose(pipe);

    return chains;
}

//===----------------------------------------------------------------------===//
// Rule Management
//===----------------------------------------------------------------------===//

IPTablesResult<void> IPTablesWrapper::add_rule(const IPTablesRule& rule) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Verify chain exists if it's a custom chain
    if (rule.chain != IPTablesChain::INPUT &&
        rule.chain != IPTablesChain::FORWARD &&
        rule.chain != IPTablesChain::OUTPUT &&
        rule.chain != IPTablesChain::PREROUTING &&
        rule.chain != IPTablesChain::POSTROUTING) {

        if (!rule.custom_chain_name.empty()) {
            if (!chain_exists_unlocked(rule.custom_chain_name, rule.table)) {
                return IPTablesResult<void>(IPTablesError{
                    IPTablesErrorCode::CHAIN_NOT_FOUND,
                    "Chain '" + rule.custom_chain_name + "' does not exist"
                });
            }
        }
    }

    // Build iptables command
    std::ostringstream cmd;
    cmd << "iptables -t " << table_to_string(rule.table);

    // Append or insert
    if (rule.position == 0) {
        cmd << " -A ";  // Append
    } else {
        cmd << " -I ";  // Insert at position
    }

    // Chain
    if (!rule.custom_chain_name.empty()) {
        cmd << rule.custom_chain_name;
    } else {
        cmd << chain_to_string(rule.chain);
    }

    if (rule.position > 0) {
        cmd << " " << rule.position;
    }

    // Protocol
    if (rule.protocol != IPTablesProtocol::ALL) {
        cmd << " -p " << protocol_to_string(rule.protocol);
    }

    // Source
    if (!rule.source.empty()) {
        cmd << " -s " << rule.source;
    }

    // Destination
    if (!rule.destination.empty()) {
        cmd << " -d " << rule.destination;
    }

    // Interface
    if (!rule.in_interface.empty()) {
        cmd << " -i " << rule.in_interface;
    }
    if (!rule.out_interface.empty()) {
        cmd << " -o " << rule.out_interface;
    }

    // Ports (requires protocol to be set)
    if (rule.source_port > 0) {
        cmd << " --sport " << rule.source_port;
    }
    if (rule.dest_port > 0) {
        cmd << " --dport " << rule.dest_port;
    }

    // Match extensions (ipset, conntrack, etc.)
    if (!rule.match_set.empty()) {
        cmd << " -m set --match-set " << rule.match_set << " src";
    }

    if (!rule.match_extensions.empty()) {
        cmd << " " << rule.match_extensions;
    }

    // Target
    // Target - use jump_target if specified, otherwise use enum target
    if (!rule.jump_target.empty()) {
        cmd << " -j " << rule.jump_target;
    } else {
        cmd << " -j " << target_to_string(rule.target);
    }


    // Custom target options
    if (!rule.target_options.empty()) {
        cmd << " " << rule.target_options;
    }

    // Comment
    if (!rule.comment.empty()) {
        cmd << " -m comment --comment \"" << rule.comment << "\"";
    }

    cmd << " 2>&1";

    auto [ret, output] = execute_command(cmd.str());

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to add rule: " + output
        });
    }

    return IPTablesResult<void>();
}

IPTablesResult<void> IPTablesWrapper::delete_rule(
    const std::string& chain_name,
    int position,
    IPTablesTable table
) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream cmd;
    cmd << "iptables -t " << table_to_string(table)
        << " -D " << chain_name << " " << position << " 2>&1";

    auto [ret, output] = execute_command(cmd.str());

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to delete rule: " + output
        });
    }

    return IPTablesResult<void>();
}

std::vector<std::string> IPTablesWrapper::list_rules(
    const std::string& chain_name,
    IPTablesTable table
) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> rules;

    std::string cmd = "iptables -t " + std::string(table_to_string(table)) +
                     " -S " + chain_name + " 2>/dev/null";

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return rules;
    }

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        // Remove trailing newline
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
        }
        // Skip chain policy lines
        if (line.find("-P ") != 0 && !line.empty()) {
            rules.push_back(line);
        }
    }

    pclose(pipe);

    return rules;
}

//===----------------------------------------------------------------------===//
// High-Level Setup
//===----------------------------------------------------------------------===//

IPTablesResult<void> IPTablesWrapper::setup_base_rules(const FirewallConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Create custom chains for ML Defender
    std::vector<std::string> custom_chains = {
        config.blacklist_chain,
        config.whitelist_chain,
        config.ratelimit_chain
    };

    for (const auto& chain : custom_chains) {
        if (!chain.empty() && !chain_exists_unlocked(chain, IPTablesTable::FILTER)) {
            std::string cmd = "iptables -t filter -N " + chain + " 2>&1";
            auto [ret, output] = execute_command(cmd);
            if (ret != 0) {
                return IPTablesResult<void>(IPTablesError{
                    IPTablesErrorCode::KERNEL_ERROR,
                    "Failed to create chain " + chain + ": " + output
                });
            }
        }
    }

    // Setup whitelist rule (first - highest priority)
    if (!config.whitelist_chain.empty() && !config.whitelist_ipset.empty()) {
        IPTablesRule whitelist_rule;
        whitelist_rule.table = IPTablesTable::FILTER;
        whitelist_rule.chain = IPTablesChain::INPUT;
        whitelist_rule.match_set = config.whitelist_ipset;
        whitelist_rule.target = IPTablesTarget::ACCEPT;
        whitelist_rule.comment = "ML Defender: Whitelist";
        whitelist_rule.position = 1;  // Insert at top

        // Unlock before calling add_rule (which will lock again)
        mutex_.unlock();
        auto result = add_rule(whitelist_rule);
        mutex_.lock();

        if (!result) {
            return result;
        }
    }

    // Setup blacklist rule (second - block malicious IPs)
    if (!config.blacklist_chain.empty() && !config.blacklist_ipset.empty()) {
        IPTablesRule blacklist_rule;
        blacklist_rule.table = IPTablesTable::FILTER;
        blacklist_rule.chain = IPTablesChain::INPUT;
        blacklist_rule.match_set = config.blacklist_ipset;
        blacklist_rule.target = IPTablesTarget::DROP;
        blacklist_rule.comment = "ML Defender: Blacklist";
        blacklist_rule.position = 2;  // After whitelist

        mutex_.unlock();
        auto result = add_rule(blacklist_rule);
        mutex_.lock();

        if (!result) {
            return result;
        }
    }

    // Setup rate limiting if configured
    if (!config.ratelimit_chain.empty()) {
        // Jump to rate limit chain
        IPTablesRule jump_rule;
        jump_rule.table = IPTablesTable::FILTER;
        jump_rule.chain = IPTablesChain::INPUT;
        jump_rule.target = IPTablesTarget::JUMP;
        jump_rule.jump_target = config.ratelimit_chain;
        jump_rule.comment = "ML Defender: Rate Limiting";
        jump_rule.position = 3;  // After whitelist and blacklist

        mutex_.unlock();
        auto result = add_rule(jump_rule);
        mutex_.lock();

        if (!result) {
            return result;
        }

        // Inside rate limit chain: limit new connections
        IPTablesRule limit_rule;
        limit_rule.table = IPTablesTable::FILTER;
        limit_rule.custom_chain_name = config.ratelimit_chain;
        limit_rule.protocol = IPTablesProtocol::TCP;
        limit_rule.match_extensions = "-m conntrack --ctstate NEW -m recent --set";
        limit_rule.target = IPTablesTarget::ACCEPT;
        limit_rule.comment = "Mark new connections";

        mutex_.unlock();
        result = add_rule(limit_rule);
        mutex_.lock();

        if (!result) {
            return result;
        }

        // Drop if too many connections from same IP
        IPTablesRule drop_rule;
        drop_rule.table = IPTablesTable::FILTER;
        drop_rule.custom_chain_name = config.ratelimit_chain;
        drop_rule.protocol = IPTablesProtocol::TCP;
        drop_rule.match_extensions = "-m conntrack --ctstate NEW -m recent --update "
                                    "--seconds 60 --hitcount 100";
        drop_rule.target = IPTablesTarget::DROP;
        drop_rule.comment = "Rate limit: Drop excessive connections";

        mutex_.unlock();
        result = add_rule(drop_rule);
        mutex_.lock();

        if (!result) {
            return result;
        }

        // Return to main chain for other traffic
        IPTablesRule return_rule;
        return_rule.table = IPTablesTable::FILTER;
        return_rule.custom_chain_name = config.ratelimit_chain;
        return_rule.target = IPTablesTarget::RETURN;
        return_rule.comment = "Return to INPUT";

        mutex_.unlock();
        result = add_rule(return_rule);
        mutex_.lock();

        if (!result) {
            return result;
        }
    }

    return IPTablesResult<void>();
}

IPTablesResult<void> IPTablesWrapper::cleanup_rules(const FirewallConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Remove rules that reference our ipsets/chains
    // This is a simplified cleanup - in production you'd want more robust cleanup

    std::vector<std::string> chains_to_remove = {
        config.blacklist_chain,
        config.whitelist_chain,
        config.ratelimit_chain
    };

    for (const auto& chain : chains_to_remove) {
        if (!chain.empty() && chain_exists_unlocked(chain, IPTablesTable::FILTER)) {
            // First remove rules that jump to this chain
            auto all_chains = list_chains(IPTablesTable::FILTER);
            for (const auto& main_chain : all_chains) {
                if (main_chain == chain) continue;

                auto rules = list_rules(main_chain, IPTablesTable::FILTER);
                for (size_t i = rules.size(); i > 0; --i) {
                    if (rules[i-1].find(chain) != std::string::npos) {
                        std::string cmd = "iptables -t filter -D " + main_chain +
                                        " " + std::to_string(i) + " 2>&1";
                        execute_command(cmd);
                    }
                }
            }

            // Now flush and delete the chain
            std::string flush_cmd = "iptables -t filter -F " + chain + " 2>&1";
            execute_command(flush_cmd);

            std::string delete_cmd = "iptables -t filter -X " + chain + " 2>&1";
            execute_command(delete_cmd);
        }
    }

    return IPTablesResult<void>();
}

//===----------------------------------------------------------------------===//
// Save/Restore
//===----------------------------------------------------------------------===//

IPTablesResult<void> IPTablesWrapper::save(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string cmd = "iptables-save > " + filepath + " 2>&1";
    auto [ret, output] = execute_command(cmd);

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to save iptables: " + output
        });
    }

    return IPTablesResult<void>();
}

IPTablesResult<void> IPTablesWrapper::restore(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string cmd = "iptables-restore < " + filepath + " 2>&1";
    auto [ret, output] = execute_command(cmd);

    if (ret != 0) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::KERNEL_ERROR,
            "Failed to restore iptables: " + output
        });
    }

    return IPTablesResult<void>();
}

} // namespace mldefender::firewall
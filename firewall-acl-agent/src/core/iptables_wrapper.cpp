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
#include "safe_exec.hpp"   // CWE-78: execv() sin shell (Consejo 8/8 DAY 128)

#include <cstring>
#include <sstream>
#include <fstream>
#include <regex>
#include <algorithm>
#include <iostream>

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

// execute_command() ELIMINADO — CWE-78 (DEBT-IPTABLES-INJECTION-001).
// Sustituido por safe_exec() / safe_exec_with_output() / safe_exec_with_file_{out,in}()
// Ver safe_exec.hpp. Consejo 8/8 DAY 128: execv() sin shell, sin excepcion.

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

    // Create chain — CWE-78 fix
    {
        const std::string tbl = table_to_string(table);
        if (!validate_table_name(tbl) || !validate_chain_name(chain_name)) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::INVALID_RULE, "Invalid table or chain name"});
        }
        if (m_dry_run) {
            std::cout << "[DRY-RUN] iptables -t " << tbl << " -N " << chain_name << std::endl;
            return IPTablesResult<void>();
        }
        auto [ret, output] = safe_exec_with_output(
            {"/usr/sbin/iptables", "-t", tbl, "-N", chain_name});
        if (ret != 0) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::KERNEL_ERROR,
                "Failed to create chain: " + output
            });
        }
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

    // CWE-78 fix: delete_chain
    {
        const std::string tbl = table_to_string(table);
        if (!validate_table_name(tbl) || !validate_chain_name(chain_name)) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::INVALID_RULE, "Invalid table or chain name"});
        }
        safe_exec({"/usr/sbin/iptables", "-t", tbl, "-F", chain_name});
        if (m_dry_run) {
            std::cout << "[DRY-RUN] iptables -t " << tbl << " -X " << chain_name << std::endl;
            return IPTablesResult<void>();
        }
        auto [ret, output] = safe_exec_with_output(
            {"/usr/sbin/iptables", "-t", tbl, "-X", chain_name});
        if (ret != 0) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::KERNEL_ERROR,
                "Failed to delete chain: " + output
            });
        }
    }

    return IPTablesResult<void>();
}

bool IPTablesWrapper::chain_exists_unlocked(
    const std::string& chain_name,
    IPTablesTable table
) const {
    // Internal version - assumes caller already holds mutex_
    // CWE-78 fix: safe_exec() con execv() sin shell.
    const std::string table_str = table_to_string(table);
    if (!validate_table_name(table_str) || !validate_chain_name(chain_name)) {
        return false;
    }
    int ret = safe_exec({"/usr/sbin/iptables", "-t", table_str,
                         "-L", chain_name, "-n"});
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
    // CWE-78 fix: flush_chain
    {
        const std::string tbl = table_to_string(table);
        if (!validate_table_name(tbl) || !validate_chain_name(chain_name)) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::INVALID_RULE, "Invalid table or chain name"});
        }
        if (m_dry_run) {
            std::cout << "[DRY-RUN] iptables -t " << tbl << " -F " << chain_name << std::endl;
            return IPTablesResult<void>();
        }
        auto [ret, output] = safe_exec_with_output(
            {"/usr/sbin/iptables", "-t", tbl, "-F", chain_name});
        if (ret != 0) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::KERNEL_ERROR,
                "Failed to flush chain: " + output
            });
        }
    }

    return IPTablesResult<void>();
}

std::vector<std::string> IPTablesWrapper::list_chains(IPTablesTable table) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> chains;
    // CWE-78 fix: safe_exec_with_output() + parsing C++ (sin grep/awk via shell).
    const std::string table_str_lc = table_to_string(table);
    if (!validate_table_name(table_str_lc)) return chains;
    auto [ret_lc, output_lc] = safe_exec_with_output(
        {"/usr/sbin/iptables", "-t", table_str_lc, "-L", "-n"});
    std::istringstream iss_lc(output_lc);
    std::string line_lc;
    while (std::getline(iss_lc, line_lc)) {
        if (line_lc.rfind("Chain ", 0) == 0) {
            auto first_space = line_lc.find(' ');
            if (first_space == std::string::npos) continue;
            auto second_space = line_lc.find(' ', first_space + 1);
            std::string name = (second_space == std::string::npos)
                ? line_lc.substr(first_space + 1)
                : line_lc.substr(first_space + 1, second_space - first_space - 1);
            if (!name.empty()) chains.push_back(name);
        }
    }

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

    // Build iptables argv — CWE-78 fix: vector<string> sin concatenacion shell.
    std::vector<std::string> ipt_args = {"/usr/sbin/iptables", "-t",
                                          std::string(table_to_string(rule.table))};
    // Append or insert
    if (rule.position == 0) {
        ipt_args.push_back("-A");
    } else {
        ipt_args.push_back("-I");
    }
    // Chain
    if (!rule.custom_chain_name.empty()) {
        ipt_args.push_back(rule.custom_chain_name);
    } else {
        ipt_args.push_back(chain_to_string(rule.chain));
    }
    if (rule.position > 0) {
        ipt_args.push_back(std::to_string(rule.position));
    }

    // Protocol
    if (rule.protocol != IPTablesProtocol::ALL) {
        ipt_args.push_back("-p");
        ipt_args.push_back(protocol_to_string(rule.protocol));
    }
    // Source
    if (!rule.source.empty()) {
        ipt_args.push_back("-s"); ipt_args.push_back(rule.source);
    }
    // Destination
    if (!rule.destination.empty()) {
        ipt_args.push_back("-d"); ipt_args.push_back(rule.destination);
    }
    // Interface
    if (!rule.in_interface.empty()) {
        ipt_args.push_back("-i"); ipt_args.push_back(rule.in_interface);
    }
    if (!rule.out_interface.empty()) {
        ipt_args.push_back("-o"); ipt_args.push_back(rule.out_interface);
    }
    // Ports
    if (rule.source_port > 0) {
        ipt_args.push_back("--sport");
        ipt_args.push_back(std::to_string(rule.source_port));
    }
    if (rule.dest_port > 0) {
        ipt_args.push_back("--dport");
        ipt_args.push_back(std::to_string(rule.dest_port));
    }
    // Match extensions (ipset, conntrack, etc.)
    if (!rule.match_set.empty()) {
        ipt_args.push_back("-m"); ipt_args.push_back("set");
        ipt_args.push_back("--match-set"); ipt_args.push_back(rule.match_set);
        ipt_args.push_back("src");
    }
    if (!rule.match_extensions.empty()) {
        std::istringstream iss_me(rule.match_extensions);
        std::string tok;
        while (iss_me >> tok) ipt_args.push_back(tok);
    }
    // Target — use jump_target if specified, otherwise use enum target
    ipt_args.push_back("-j");
    if (!rule.jump_target.empty()) {
        ipt_args.push_back(rule.jump_target);
    } else {
        ipt_args.push_back(target_to_string(rule.target));
    }
    // Custom target options
    if (!rule.target_options.empty()) {
        std::istringstream iss_to(rule.target_options);
        std::string tok;
        while (iss_to >> tok) ipt_args.push_back(tok);
    }
    // Comment — argumento separado, sin comillas de shell
    if (!rule.comment.empty()) {
        ipt_args.push_back("-m"); ipt_args.push_back("comment");
        ipt_args.push_back("--comment"); ipt_args.push_back(rule.comment);
    }
    if (m_dry_run) {
        std::ostringstream dry;
        for (const auto& a : ipt_args) dry << a << " ";
        std::cout << "[DRY-RUN] Would execute: " << dry.str() << std::endl;
        return IPTablesResult<void>();
    }
    auto [ret, output] = safe_exec_with_output(ipt_args);

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

    // CWE-78 fix: delete_rule
    {
        const std::string tbl = table_to_string(table);
        if (!validate_table_name(tbl) || !validate_chain_name(chain_name)) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::INVALID_RULE, "Invalid table or chain name"});
        }
        if (m_dry_run) {
            std::cout << "[DRY-RUN] iptables -t " << tbl << " -D "
                      << chain_name << " " << position << std::endl;
            return IPTablesResult<void>();
        }
        auto [ret, output] = safe_exec_with_output(
            {"/usr/sbin/iptables", "-t", tbl, "-D",
             chain_name, std::to_string(position)});
        if (ret != 0) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::KERNEL_ERROR,
                "Failed to delete rule: " + output
            });
        }
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

    // CWE-78 fix: safe_exec_with_output() sin shell.
    const std::string table_str_lr = table_to_string(table);
    if (!validate_table_name(table_str_lr) || !validate_chain_name(chain_name)) {
        return rules;
    }
    auto [ret_lr, output_lr] = safe_exec_with_output(
        {"/usr/sbin/iptables", "-t", table_str_lr, "-S", chain_name});
    std::istringstream iss_lr(output_lr);
    std::string line_lr;
    while (std::getline(iss_lr, line_lr)) {
        if (!line_lr.empty() && line_lr.rfind("-P ", 0) != 0) {
            rules.push_back(line_lr);
        }
    }

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
            // CWE-78 fix: setup_base_rules
            if (!validate_chain_name(chain)) {
                return IPTablesResult<void>(IPTablesError{
                    IPTablesErrorCode::INVALID_RULE,
                    "Invalid chain name: " + chain});
            }
            if (m_dry_run) {
                std::cout << "[DRY-RUN] iptables -t filter -N " << chain << std::endl;
                return IPTablesResult<void>();
            }
            auto [ret, output] = safe_exec_with_output(
                {"/usr/sbin/iptables", "-t", "filter", "-N", chain});
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

    // Early return in dry-run mode
    if (m_dry_run) {
        std::cout << "[DRY-RUN] Would cleanup ML Defender chains" << std::endl;
        return IPTablesResult<void>();
    }


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
                        // CWE-78 fix
                        if (validate_chain_name(main_chain)) {
                            safe_exec({"/usr/sbin/iptables", "-t", "filter",
                                       "-D", main_chain, std::to_string(i)});
                        }
                    }
                }
            }

            // Now flush and delete the chain
            // CWE-78 fix: flush + delete
            if (validate_chain_name(chain)) {
                safe_exec({"/usr/sbin/iptables", "-t", "filter", "-F", chain});
                safe_exec({"/usr/sbin/iptables", "-t", "filter", "-X", chain});
            }
        }
    }

    return IPTablesResult<void>();
}

//===----------------------------------------------------------------------===//
// Save/Restore
//===----------------------------------------------------------------------===//

IPTablesResult<void> IPTablesWrapper::save(const std::string& filepath) const {
    std::lock_guard<std::mutex> lock(mutex_);

    // CWE-78 fix: save via safe_exec_with_file_out
    if (!validate_filepath(filepath)) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::INVALID_RULE, "Invalid filepath for save"});
    }
    if (m_dry_run) {
        std::cout << "[DRY-RUN] iptables-save > " << filepath << std::endl;
        return IPTablesResult<void>();
    }
    {
        int ret = safe_exec_with_file_out({"/usr/sbin/iptables-save"}, filepath);
        if (ret != 0) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::KERNEL_ERROR,
                "Failed to save iptables (exit " + std::to_string(ret) + ")"
            });
        }
    }

    return IPTablesResult<void>();
}

IPTablesResult<void> IPTablesWrapper::restore(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);

    // CWE-78 fix: restore via safe_exec_with_file_in
    if (!validate_filepath(filepath)) {
        return IPTablesResult<void>(IPTablesError{
            IPTablesErrorCode::INVALID_RULE, "Invalid filepath for restore"});
    }
    if (m_dry_run) {
        std::cout << "[DRY-RUN] iptables-restore < " << filepath << std::endl;
        return IPTablesResult<void>();
    }
    {
        int ret = safe_exec_with_file_in({"/usr/sbin/iptables-restore"}, filepath);
        if (ret != 0) {
            return IPTablesResult<void>(IPTablesError{
                IPTablesErrorCode::KERNEL_ERROR,
                "Failed to restore iptables (exit " + std::to_string(ret) + ")"
            });
    }
        }

    return IPTablesResult<void>();
}

} // namespace mldefender::firewall
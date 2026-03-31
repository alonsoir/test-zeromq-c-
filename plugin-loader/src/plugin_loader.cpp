// ============================================================================
// plugin_loader.cpp — ML Defender Plugin Loader Implementation
// ============================================================================
// dlopen/dlsym lazy loading. Sin crypto. Sin seed-client (PHASE 1).
// ============================================================================

#include "plugin_loader/plugin_loader.hpp"
#include <dlfcn.h>
#include <chrono>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>

// Minimal JSON parsing — evitamos dependencia de nlohmann aquí.
// El componente host ya tiene nlohmann; usamos una lectura simple
// basada en buscar la clave "plugins" en el JSON raw.
// Para PHASE 2 con seed-client se migrará a nlohmann completo.

namespace ml_defender {

// ----------------------------------------------------------------------------
// LoadedPlugin — internal representation of a loaded plugin
// ----------------------------------------------------------------------------
struct PluginLoader::LoadedPlugin {
    void*         handle   = nullptr;

    // Resolved symbols
    const char*  (*fn_name)()            = nullptr;
    const char*  (*fn_version)()         = nullptr;
    int          (*fn_api_version)()     = nullptr;
    PluginResult (*fn_init)(const PluginConfig*) = nullptr;
    PluginResult (*fn_process)(PacketContext*)    = nullptr;
    void         (*fn_shutdown)()        = nullptr;

    std::string   name;

    ~LoadedPlugin() {
        if (handle) {
            dlclose(handle);
            handle = nullptr;
        }
    }
};

// ----------------------------------------------------------------------------
// Simple helpers
// ----------------------------------------------------------------------------
static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Minimal extraction of a string value from flat JSON.
// Only used to read plugins.directory and plugins.budget_us.
// Full JSON parsing delegated to component host.
static std::string json_string_value(const std::string& json, const std::string& key) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = json.find("\"", pos);
    if (pos == std::string::npos) return "";
    auto end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";
    return json.substr(pos + 1, end - pos - 1);
}

static uint32_t json_uint_value(const std::string& json, const std::string& key, uint32_t def) {
    auto pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return def;
    // skip whitespace
    while (pos < json.size() && (json[pos] == ':' || json[pos] == ' ')) ++pos;
    try { return static_cast<uint32_t>(std::stoul(json.substr(pos))); }
    catch (...) { return def; }
}

// Extract the "plugins" JSON block as a string
static std::string extract_plugins_block(const std::string& json) {
    auto pos = json.find("\"plugins\"");
    if (pos == std::string::npos) return "";
    pos = json.find("{", pos);
    if (pos == std::string::npos) return "";
    int depth = 0;
    size_t start = pos;
    for (size_t i = pos; i < json.size(); ++i) {
        if (json[i] == '{') ++depth;
        else if (json[i] == '}') { --depth; if (depth == 0) return json.substr(start, i - start + 1); }
    }
    return "";
}

// Extract enabled plugin descriptors from "enabled": [{name, path, active, ...}]
// Returns pairs of {name, path} for entries where active==true only.
static std::vector<std::pair<std::string,std::string>>
extract_enabled_objects(const std::string& plugins_block) {
    std::vector<std::pair<std::string,std::string>> result;
    auto pos = plugins_block.find("\"enabled\"");
    if (pos == std::string::npos) return result;
    pos = plugins_block.find("[", pos);
    if (pos == std::string::npos) return result;

    // Walk each object {...} inside the array
    for (size_t i = pos + 1; i < plugins_block.size(); ++i) {
        if (plugins_block[i] == ']') break;
        if (plugins_block[i] != '{') continue;

        // Extract the object substring
        int depth = 0; size_t obj_start = i;
        std::string obj;
        for (size_t j = i; j < plugins_block.size(); ++j) {
            if (plugins_block[j] == '{') ++depth;
            else if (plugins_block[j] == '}') {
                --depth;
                if (depth == 0) { obj = plugins_block.substr(obj_start, j - obj_start + 1); i = j; break; }
            }
        }
        if (obj.empty()) continue;

        // active must be true
        auto act_pos = obj.find("\"active\"");
        if (act_pos == std::string::npos) continue;
        auto colon = obj.find(":", act_pos);
        if (colon == std::string::npos) continue;
        size_t vstart = colon + 1;
        while (vstart < obj.size() && obj[vstart] == ' ') ++vstart;
        if (obj.substr(vstart, 4) != "true") continue;

        // Extract name
        std::string name = json_string_value(obj, "name");
        if (name.empty()) continue;

        // Extract path (explicit .so path from JSON)
        std::string so_path = json_string_value(obj, "path");
        if (so_path.empty()) continue;

        result.push_back({name, so_path});
    }
    return result;
}

// ----------------------------------------------------------------------------
// PluginLoader implementation
// ----------------------------------------------------------------------------
PluginLoader::PluginLoader(const std::string& config_json_path)
    : config_path_(config_json_path) {}

PluginLoader::~PluginLoader() {
    if (!shutdown_called_) {
        shutdown();
    }
}

void PluginLoader::load_plugins() {
    std::string raw = read_file(config_path_);
    if (raw.empty()) {
        std::cerr << "[plugin-loader] WARNING: cannot read config: " << config_path_ << "\n";
        return;
    }

    std::string plugins_block = extract_plugins_block(raw);
    if (plugins_block.empty()) {
        std::cerr << "[plugin-loader] INFO: no 'plugins' section in " << config_path_ << " — no plugins loaded\n";
        return;
    }

    std::string directory = json_string_value(plugins_block, "directory");
    if (directory.empty()) directory = "/usr/lib/ml-defender/plugins";

    budget_us_ = json_uint_value(plugins_block, "budget_us", 100);

    auto enabled = extract_enabled_objects(plugins_block);
    if (enabled.empty()) {
        std::cerr << "[plugin-loader] INFO: plugins.enabled is empty — no plugins loaded\n";
        return;
    }

    for (const auto& [plugin_name, so_path] : enabled) {

        if (!std::filesystem::exists(so_path)) {
            std::cerr << "[plugin-loader] WARNING: plugin '" << plugin_name
                      << "' not found at " << so_path << " — skipping\n";
            continue;
        }

        void* handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            std::cerr << "[plugin-loader] WARNING: dlopen failed for '" << plugin_name
                      << "': " << dlerror() << " — skipping\n";
            continue;
        }

        auto plugin = std::make_shared<LoadedPlugin>();
        plugin->handle = handle;

        // Resolve mandatory symbols
        #define RESOLVE(sym, type, field) \
            plugin->field = reinterpret_cast<type>(dlsym(handle, #sym)); \
            if (!plugin->field) { \
                std::cerr << "[plugin-loader] WARNING: symbol '" #sym "' not found in '" \
                          << plugin_name << "' — skipping\n"; \
                goto skip_plugin; \
            }

        RESOLVE(plugin_name,           const char*(*)(void),          fn_name)
        RESOLVE(plugin_version,        const char*(*)(void),          fn_version)
        RESOLVE(plugin_api_version,    int(*)(void),                  fn_api_version)
        RESOLVE(plugin_init,           PluginResult(*)(const PluginConfig*), fn_init)
        RESOLVE(plugin_process_packet, PluginResult(*)(PacketContext*),      fn_process)
        RESOLVE(plugin_shutdown,       void(*)(void),                 fn_shutdown)
        #undef RESOLVE

        // API version check
        if (plugin->fn_api_version() != PLUGIN_API_VERSION) {
            std::cerr << "[plugin-loader] WARNING: plugin '" << plugin_name
                      << "' API version mismatch (got " << plugin->fn_api_version()
                      << ", expected " << PLUGIN_API_VERSION << ") — skipping\n";
            goto skip_plugin;
        }

        // Call init
        {
            // Extract plugin-specific config JSON section
            std::string plugin_cfg = "{}";
            auto cfg_pos = plugins_block.find("\"" + plugin_name + "\"");
            if (cfg_pos != std::string::npos) {
                auto brace = plugins_block.find("{", cfg_pos);
                if (brace != std::string::npos) {
                    int depth = 0; size_t start = brace;
                    for (size_t i = brace; i < plugins_block.size(); ++i) {
                        if (plugins_block[i] == '{') ++depth;
                        else if (plugins_block[i] == '}') {
                            --depth;
                            if (depth == 0) { plugin_cfg = plugins_block.substr(start, i - start + 1); break; }
                        }
                    }
                }
            }

            PluginConfig cfg;
            cfg.name        = plugin_name.c_str();
            cfg.config_json = plugin_cfg.c_str();

            PluginResult r = plugin->fn_init(&cfg);
            if (r != PLUGIN_OK) {
                std::cerr << "[plugin-loader] WARNING: plugin_init() failed for '"
                          << plugin_name << "' (result=" << r << ") — skipping\n";
                goto skip_plugin;
            }
        }

        plugin->name = plugin->fn_name();
        plugins_.push_back(plugin);
        stats_.push_back(PluginStats{plugin->name, 0, 0, 0});
        std::cerr << "[plugin-loader] INFO: loaded plugin '" << plugin->name
                  << "' v" << plugin->fn_version() << "\n";
        continue;

        skip_plugin:
        dlclose(handle);
        plugin->handle = nullptr;
    }
}

void PluginLoader::invoke_all(PacketContext& ctx) {
    for (size_t i = 0; i < plugins_.size(); ++i) {
        auto& p = plugins_[i];
        auto  t0 = std::chrono::steady_clock::now();

        PluginResult r = p->fn_process(&ctx);

        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();

        stats_[i].invocations++;

        if (r == PLUGIN_ERROR) {
            stats_[i].errors++;
            std::cerr << "[plugin-loader] WARNING: plugin '" << p->name
                      << "' returned PLUGIN_ERROR\n";
        }

        if (static_cast<uint32_t>(elapsed_us) > budget_us_) {
            stats_[i].budget_overruns++;
            std::cerr << "[plugin-loader] WARNING: plugin '" << p->name
                      << "' overrun: " << elapsed_us << "us > " << budget_us_ << "us budget\n";
        }
    }
}

void PluginLoader::shutdown() {
    for (size_t i = 0; i < plugins_.size(); ++i) {
        auto& p = plugins_[i];
        if (p && p->fn_shutdown) {
            p->fn_shutdown();
        }
        std::cerr << "[plugin-loader] INFO: shutdown plugin '" << p->name
                  << "' — invocations=" << stats_[i].invocations
                  << " overruns=" << stats_[i].budget_overruns
                  << " errors=" << stats_[i].errors << "\n";
    }
    plugins_.clear();
    shutdown_called_ = true;
}

const std::vector<PluginStats>& PluginLoader::stats() const { return stats_; }
size_t PluginLoader::loaded_count() const { return plugins_.size(); }

}  // namespace ml_defender

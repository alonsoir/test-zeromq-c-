#pragma once
#include <string>
#include <nlohmann/json.hpp>

// Estructuras para configuración
struct RagConfig {
    std::string host = "localhost";
    int port = 8080;
    std::string model_name = "unknown";
    int embedding_dimension = 512;
};

struct EtcdConfig {
    std::string host = "localhost";
    int port = 2379;
};

namespace Rag {

class ConfigManager {
public:
    // ✅ ELIMINAR declaración duplicada del constructor
    // ConfigManager(); // <- QUITAR ESTA LÍNEA

    ~ConfigManager();

    // Método Singleton
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }

    // Eliminar copia y asignación
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    bool loadFromFile(const std::string& config_path = "../config/rag-config.json");
    bool saveToFile(const std::string& config_path = "");

    // Acceso a configuraciones
    std::string getModelPath() const;
    int getSecurityLevel() const;
    bool useLLM() const;
    std::string getEtcdEndpoint() const;
    std::string getLogLevel() const;

    // ✅ NUEVOS MÉTODOS para estructura actual
    RagConfig getRagConfig() const;
    EtcdConfig getEtcdConfig() const;

    // Modificación de configuraciones
    bool updateSetting(const std::string& path, const std::string& value);
    bool updateSetting(const std::string& path, int value);
    bool updateSetting(const std::string& path, bool value);

    // Para registro en etcd
    nlohmann::json getConfigForEtcd() const;
    std::string getComponentId() const;

private:
    ConfigManager() = default; // ✅ Solo el constructor privado del Singleton
    nlohmann::json config_;
    std::string current_config_path_;

    bool validateConfig() const;
};

} // namespace Rag
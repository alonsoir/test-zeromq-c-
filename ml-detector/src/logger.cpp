#include "logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>

namespace ml_detector {

std::shared_ptr<spdlog::logger> Logger::create(
    const std::string& name,
    const std::string& log_file,
    spdlog::level::level_enum level
) {
    try {
        // Verificar si ya existe
        auto existing = spdlog::get(name);
        if (existing) {
            return existing;
        }
        
        std::vector<spdlog::sink_ptr> sinks;
        
        // Console sink (coloreado)
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(level);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%^%l%$] %v");
        sinks.push_back(console_sink);
        
        // File sink (si se especifica)
        if (!log_file.empty()) {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_file, 
                1024 * 1024 * 10,  // 10MB por archivo
                3                   // 3 archivos rotados
            );
            file_sink->set_level(spdlog::level::debug);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] [%t] %v");
            sinks.push_back(file_sink);
        }
        
        // Crear logger
        auto logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());
        logger->set_level(level);
        logger->flush_on(spdlog::level::warn);
        
        // Registrar en spdlog
        spdlog::register_logger(logger);
        
        return logger;
        
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        throw;
    }
}

void Logger::set_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
}

void Logger::shutdown() {
    spdlog::shutdown();
}

} // namespace ml_detector

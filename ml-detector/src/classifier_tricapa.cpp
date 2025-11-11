// En classifier_tricapa.cpp
#include "ml_defender/ransomware_detector.hpp"

class ClassifierTricapa {
private:
    // Añadir detector embebido
    std::unique_ptr<ml_defender::RansomwareDetector> m_ransomware_detector;

public:
    void initialize() {
        // Inicializar detector embebido
        m_ransomware_detector = std::make_unique<ml_defender::RansomwareDetector>();

        // ... resto de inicialización
    }

    void classify_level2(const FlowFeatures& features) {
        // Convertir features al formato del detector
        ml_defender::RansomwareDetector::Features rf_features{
            .io_intensity = features.io_intensity,
            .entropy = features.entropy,
            .resource_usage = features.resource_usage,
            // ... mapear resto de features
        };

        // Predicción <100μs
        auto result = m_ransomware_detector->predict(rf_features);

        // Aplicar threshold del config (0.75)
        if (result.ransomware_prob >= 0.75f) {
            // RANSOMWARE DETECTADO!
            handle_ransomware_detection(result);
        }
    }
};
// test_auto_isolate.cpp — ADR-042 IRP should_auto_isolate() unit tests
// DAY 143 — lógica de decisión pura, sin fork(), sin nftables, sin root
#include <gtest/gtest.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <thread>
#include "firewall/batch_processor.hpp"
#include "firewall/config_loader.hpp"
#include "firewall/ipset_wrapper.hpp"

using namespace mldefender::firewall;

namespace {

IrpConfig make_irp(bool auto_isolate = true,
                   double threshold = 0.95,
                   std::vector<std::string> types = {"ransomware","lateral_movement","c2_beacon"}) {
    IrpConfig cfg;
    cfg.auto_isolate             = auto_isolate;
    cfg.threat_score_threshold   = threshold;
    cfg.auto_isolate_event_types = std::move(types);
    cfg.isolate_interface        = "eth1";
    cfg.isolate_binary_path      = "/usr/local/bin/argus-network-isolate";
    cfg.isolate_config_path      = "/etc/ml-defender/firewall-acl-agent/isolate.json";
    return cfg;
}

protobuf::Detection make_detection(protobuf::DetectionType type, float confidence) {
    protobuf::Detection d;
    d.set_type(type);
    d.set_confidence(confidence);
    d.set_src_ip("192.168.1.100");
    return d;
}

class AutoIsolateTest : public ::testing::Test {
protected:
    void SetUp() override {
        BatchProcessorConfig cfg;
        cfg.confidence_threshold = 0.5f;
        processor_ = std::make_unique<BatchProcessor>(ipset_, cfg);
    }
    IPSetWrapper ipset_;
    std::unique_ptr<BatchProcessor> processor_;
};

// ── TEST 1: dispara con score >= threshold y tipo correcto ────────────────
TEST_F(AutoIsolateTest, TriggersOnRansomwareHighScore) {
    processor_->set_irp_config(make_irp());
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.97f);
    EXPECT_TRUE(processor_->should_auto_isolate(d));
}

TEST_F(AutoIsolateTest, TriggersOnInternalThreatHighScore) {
    processor_->set_irp_config(make_irp());
    auto d = make_detection(protobuf::DetectionType::DETECTION_INTERNAL_THREAT, 0.96f);
    EXPECT_TRUE(processor_->should_auto_isolate(d));
}

TEST_F(AutoIsolateTest, TriggersOnSuspiciousTrafficHighScore) {
    processor_->set_irp_config(make_irp());
    auto d = make_detection(protobuf::DetectionType::DETECTION_SUSPICIOUS_TRAFFIC, 0.95f);
    EXPECT_TRUE(processor_->should_auto_isolate(d));
}

// ── TEST 2: NO dispara con score < threshold ──────────────────────────────
TEST_F(AutoIsolateTest, NoTriggerOnLowScore) {
    processor_->set_irp_config(make_irp());
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.94f);
    EXPECT_FALSE(processor_->should_auto_isolate(d));
}

TEST_F(AutoIsolateTest, NoTriggerOnScoreExactlyBelowThreshold) {
    processor_->set_irp_config(make_irp(true, 0.95));
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.9499f);
    EXPECT_FALSE(processor_->should_auto_isolate(d));
}

// ── TEST 3: NO dispara con tipo no mapeado ────────────────────────────────
TEST_F(AutoIsolateTest, NoTriggerOnDDoS) {
    processor_->set_irp_config(make_irp());
    auto d = make_detection(protobuf::DetectionType::DETECTION_DDOS, 0.99f);
    EXPECT_FALSE(processor_->should_auto_isolate(d));
}

TEST_F(AutoIsolateTest, NoTriggerOnUnknown) {
    processor_->set_irp_config(make_irp());
    auto d = make_detection(protobuf::DetectionType::DETECTION_UNKNOWN, 0.99f);
    EXPECT_FALSE(processor_->should_auto_isolate(d));
}

// ── TEST 4: NO dispara cuando auto_isolate=false ──────────────────────────
TEST_F(AutoIsolateTest, NoTriggerWhenDisabled) {
    processor_->set_irp_config(make_irp(false));
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.99f);
    EXPECT_FALSE(processor_->should_auto_isolate(d));
}

// ── TEST 5: NO dispara cuando lista de tipos vacía ────────────────────────
TEST_F(AutoIsolateTest, NoTriggerOnEmptyEventTypes) {
    processor_->set_irp_config(make_irp(true, 0.95, {}));
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.99f);
    EXPECT_FALSE(processor_->should_auto_isolate(d));
}

// ── TEST 6: score exactamente en el umbral dispara ────────────────────────
TEST_F(AutoIsolateTest, TriggersOnExactThreshold) {
    processor_->set_irp_config(make_irp(true, 0.95));
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.95f);
    EXPECT_TRUE(processor_->should_auto_isolate(d));
}

// ── TEST INTEGRACIÓN: fork()+execv() se lanza con evento sintético ──────────
// Usa /bin/true como binario sintético — verifica que el hijo se lanzó
// y terminó con exit 0. No ejecuta argus-network-isolate real.
TEST_F(AutoIsolateTest, IntegrationForkExecOnSyntheticEvent) {
    IrpConfig irp = make_irp();
    irp.isolate_binary_path = "/bin/true";  // binario sintético seguro
    irp.isolate_config_path = "/tmp/test_isolate_synthetic.json";
    processor_->set_irp_config(irp);

    // Evento sintético: score >= 0.95 + ransomware
    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.97f);

    // Registrar PIDs de hijos antes del disparo
    pid_t before = getpid();

    // Disparar — debe hacer fork()+execv(/bin/true)
    processor_->check_auto_isolate(d);

    // Esperar hasta 2 segundos a que el hijo termine
    bool child_exited = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline) {
        int status = 0;
        pid_t result = waitpid(-1, &status, WNOHANG);
        if (result > 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            child_exited = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    EXPECT_TRUE(child_exited) << "fork()+execv(/bin/true) no lanzó hijo o no terminó con exit 0";
    (void)before;
}

// ── TEST INTEGRACIÓN: NO fork cuando score bajo ───────────────────────────
TEST_F(AutoIsolateTest, IntegrationNoForkOnLowScore) {
    IrpConfig irp = make_irp();
    irp.isolate_binary_path = "/bin/true";
    processor_->set_irp_config(irp);

    auto d = make_detection(protobuf::DetectionType::DETECTION_RANSOMWARE, 0.80f);
    processor_->check_auto_isolate(d);

    // Esperar 200ms — no debe haber hijo
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    int status = 0;
    pid_t result = waitpid(-1, &status, WNOHANG);
    EXPECT_LE(result, 0) << "fork() no debería haberse llamado con score bajo";
}

} // namespace

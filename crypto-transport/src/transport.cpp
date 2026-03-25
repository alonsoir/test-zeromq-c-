// crypto-transport/src/transport.cpp
#include "crypto_transport/transport.hpp"
#include <sodium.h>
#include <stdexcept>
#include <cstring>

namespace crypto_transport {

// ============================================================================
// Constructor / Destructor
// ============================================================================

CryptoTransport::CryptoTransport(const ml_defender::SeedClient& seed_client,
                                 const std::string& context) {
    if (!seed_client.is_loaded()) {
        throw std::runtime_error(
            "CryptoTransport: SeedClient not loaded — call seed_client.load() first");
    }
    if (sodium_init() < 0) {
        throw std::runtime_error("CryptoTransport: libsodium initialization failed");
    }
    derive_key(seed_client.seed(), context);
}

CryptoTransport::~CryptoTransport() {
    sodium_memzero(session_key_.data(), session_key_.size());
}

// ============================================================================
// Move semantics
// ============================================================================

CryptoTransport::CryptoTransport(CryptoTransport&& other) noexcept
    : session_key_(other.session_key_),
      nonce_counter_(other.nonce_counter_.load(std::memory_order_relaxed))
{
    sodium_memzero(other.session_key_.data(), other.session_key_.size());
    other.nonce_counter_.store(0, std::memory_order_relaxed);
}

CryptoTransport& CryptoTransport::operator=(CryptoTransport&& other) noexcept {
    if (this != &other) {
        sodium_memzero(session_key_.data(), session_key_.size());
        session_key_ = other.session_key_;
        nonce_counter_.store(
            other.nonce_counter_.load(std::memory_order_relaxed),
            std::memory_order_relaxed);
        sodium_memzero(other.session_key_.data(), other.session_key_.size());
        other.nonce_counter_.store(0, std::memory_order_relaxed);
    }
    return *this;
}

// ============================================================================
// HKDF-SHA256 — libsodium 1.0.19 native API (ADR-013 PHASE 2)
//
// Extract: PRK = HMAC-SHA256(salt=zeros32, IKM=seed)
// Expand:  OKM = HKDF-Expand(PRK, info, 32)
// Salt: 32 zero bytes — RFC 5869 default when no salt is provided
// ============================================================================

void CryptoTransport::derive_key(const std::array<uint8_t, 32>& ikm,
                                  const std::string& context) {
    // Extract: PRK = HMAC-SHA256(salt=zeros, IKM=seed)
    // extract_final() writes PRK into prk[] buffer
    const uint8_t salt[crypto_kdf_hkdf_sha256_KEYBYTES] = {};
    uint8_t prk[crypto_kdf_hkdf_sha256_KEYBYTES];
    crypto_kdf_hkdf_sha256_state st;

    if (crypto_kdf_hkdf_sha256_extract_init(&st, salt, sizeof(salt)) != 0 ||
        crypto_kdf_hkdf_sha256_extract_update(&st, ikm.data(), ikm.size()) != 0 ||
        crypto_kdf_hkdf_sha256_extract_final(&st, prk) != 0) {
        throw std::runtime_error("CryptoTransport: HKDF extract failed");
    }

    // Expand: OKM = HKDF-Expand(PRK, info, 32)
    // expand() takes prk[] directly, not the state
    if (crypto_kdf_hkdf_sha256_expand(
            session_key_.data(),
            session_key_.size(),
            context.c_str(),
            context.size(),
            prk) != 0) {
        sodium_memzero(prk, sizeof(prk));
        throw std::runtime_error("CryptoTransport: HKDF expand failed");
    }
    sodium_memzero(prk, sizeof(prk));  // PRK es secreto intermedio — limpiar
}

// ============================================================================
// Nonce — 96-bit monotonic counter
// Layout: [0x00 0x00 0x00 0x00 | counter_uint64_little_endian]
// ============================================================================

std::array<uint8_t, CryptoTransport::NONCE_SIZE> CryptoTransport::next_nonce() {
    uint64_t counter = nonce_counter_.fetch_add(1, std::memory_order_relaxed);
    if (counter == UINT64_MAX) {
        throw std::runtime_error(
            "CryptoTransport: nonce counter overflow — ordered pipeline restart required");
    }
    std::array<uint8_t, NONCE_SIZE> nonce{};
    std::memcpy(nonce.data() + 4, &counter, sizeof(counter));
    return nonce;
}

// ============================================================================
// Encrypt
// ============================================================================

std::vector<uint8_t> CryptoTransport::encrypt(const std::vector<uint8_t>& plaintext) {
    if (plaintext.empty()) return {};

    auto nonce = next_nonce();
    std::vector<uint8_t> result(NONCE_SIZE + plaintext.size() + MAC_SIZE);
    std::memcpy(result.data(), nonce.data(), NONCE_SIZE);

    unsigned long long ciphertext_len = 0;
    int ret = crypto_aead_chacha20poly1305_ietf_encrypt(
        result.data() + NONCE_SIZE, &ciphertext_len,
        plaintext.data(), plaintext.size(),
        nullptr, 0, nullptr,
        nonce.data(), session_key_.data()
    );

    if (ret != 0) {
        throw std::runtime_error("CryptoTransport::encrypt: ChaCha20-Poly1305 IETF failed");
    }
    result.resize(NONCE_SIZE + ciphertext_len);
    return result;
}

// ============================================================================
// Decrypt
// ============================================================================

std::vector<uint8_t> CryptoTransport::decrypt(const std::vector<uint8_t>& ciphertext) {
    if (ciphertext.empty()) return {};

    const size_t min_size = NONCE_SIZE + MAC_SIZE;
    if (ciphertext.size() < min_size) {
        throw std::runtime_error(
            "CryptoTransport::decrypt: buffer too short — expected at least " +
            std::to_string(min_size) + " bytes, got " +
            std::to_string(ciphertext.size()));
    }

    const uint8_t* nonce  = ciphertext.data();
    const uint8_t* ct     = ciphertext.data() + NONCE_SIZE;
    const size_t   ct_len = ciphertext.size() - NONCE_SIZE;

    std::vector<uint8_t> plaintext(ct_len - MAC_SIZE);
    unsigned long long plaintext_len = 0;

    int ret = crypto_aead_chacha20poly1305_ietf_decrypt(
        plaintext.data(), &plaintext_len,
        nullptr,
        ct, ct_len,
        nullptr, 0,
        nonce, session_key_.data()
    );

    if (ret != 0) {
        throw std::runtime_error(
            "CryptoTransport::decrypt: MAC verification failed "
            "(wrong key, corrupted data, or nonce mismatch)");
    }
    plaintext.resize(plaintext_len);
    return plaintext;
}

// ============================================================================
// Diagnostics
// ============================================================================

uint64_t CryptoTransport::nonce_count() const noexcept {
    return nonce_counter_.load(std::memory_order_relaxed);
}

} // namespace crypto_transport

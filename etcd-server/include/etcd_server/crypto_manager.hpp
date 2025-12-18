#pragma once

#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <cryptopp/osrng.h>
#include <cryptopp/hex.h>
#include <cryptopp/filters.h>
#include <cryptopp/secblock.h>
#include <cryptopp/sha.h>

class CryptoManager {
private:
    CryptoPP::SecByteBlock encryption_key_;
    std::string seed_;
    std::chrono::system_clock::time_point key_generation_time_;

    // Constantes para ChaCha20-Poly1305
    static constexpr size_t KEY_SIZE = 32;    // 256 bits (crypto_secretbox_KEYBYTES)
    static constexpr size_t IV_SIZE = 12;     // ← ELIMINAR esta línea (no se usa)
    static constexpr size_t TAG_SIZE = 16;    // ← ELIMINAR esta línea (no se usa)

public:
    CryptoManager();

    // Gestión de claves
    bool generate_key_from_seed(const std::string& seed);
    std::string get_current_seed() const;
    std::string get_encryption_key() const;
    bool should_rotate_key() const;
    void rotate_key();

    // Cifrado/Descifrado
    std::string encrypt(const std::string& plaintext);
    std::string decrypt(const std::string& ciphertext);

    // Utilidades
    std::string generate_random_seed();
    bool validate_ciphertext(const std::string& ciphertext);

private:
    void generate_random_bytes(CryptoPP::SecByteBlock& buffer, size_t size);
    std::string bytes_to_hex(const CryptoPP::SecByteBlock& bytes) const;
    CryptoPP::SecByteBlock hex_to_bytes(const std::string& hex);
};
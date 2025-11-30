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
    CryptoPP::SecByteBlock iv_;
    std::string seed_;
    std::chrono::system_clock::time_point key_generation_time_;

    // Constantes para AES-GCM
    static constexpr size_t KEY_SIZE = 32;    // 256 bits para AES-256
    static constexpr size_t IV_SIZE = 12;     // 96 bits recomendado para GCM
    static constexpr size_t TAG_SIZE = 16;    // 128 bits para autenticación

public:
    CryptoManager();

    // Gestión de claves
    bool generate_key_from_seed(const std::string& seed);
    std::string get_current_seed() const;
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
    std::string bytes_to_hex(const CryptoPP::SecByteBlock& bytes);
    CryptoPP::SecByteBlock hex_to_bytes(const std::string& hex);
};
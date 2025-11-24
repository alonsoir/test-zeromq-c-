#include "etcd_server/crypto_manager.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

// Headers para AES-CBC
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/filters.h>
#include <cryptopp/sha.h>
#include <cryptopp/hkdf.h>

CryptoManager::CryptoManager() {
    // Generar una seed inicial automáticamente
    seed_ = generate_random_seed();
    if (!generate_key_from_seed(seed_)) {
        std::cerr << "[CRYPTO] ❌ Error inicializando sistema de cifrado" << std::endl;
        throw std::runtime_error("No se pudo inicializar el sistema de cifrado");
    }
    std::cout << "[CRYPTO] ✅ Sistema de cifrado inicializado con seed: " << seed_ << std::endl;
}

bool CryptoManager::generate_key_from_seed(const std::string& seed) {
    try {
        // Usar HKDF para derivación segura de clave (mejor que SHA256 directo)
        CryptoPP::SecByteBlock derived_key(KEY_SIZE + IV_SIZE);

        CryptoPP::HKDF<CryptoPP::SHA256> hkdf;
        hkdf.DeriveKey(derived_key, derived_key.size(),
            (const CryptoPP::byte*)seed.data(), seed.size(),
            (const CryptoPP::byte*)"etcd-server-salt", 16,  // salt
            (const CryptoPP::byte*)"AES-256-CBC", 11);      // info

        // Generar IV aleatorio único
        CryptoPP::AutoSeededRandomPool rng;
        rng.GenerateBlock(iv_, iv_.size());

        encryption_key_.Assign(derived_key, KEY_SIZE);
        key_generation_time_ = std::chrono::system_clock::now();
        seed_ = seed;

        std::cout << "[CRYPTO] 🔑 Clave derivada con HKDF desde seed" << std::endl;
        std::cout << "[CRYPTO]   Key: " << bytes_to_hex(encryption_key_) << std::endl;
        std::cout << "[CRYPTO]   IV: " << bytes_to_hex(iv_) << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[CRYPTO] ❌ Error generando clave: " << e.what() << std::endl;
        return false;
    }
}

std::string CryptoManager::get_current_seed() const {
    return seed_;
}

bool CryptoManager::should_rotate_key() const {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::hours>(now - key_generation_time_);

    // Rotar cada 24 horas (en producción podría ser más frecuente)
    return duration.count() >= 24;
}

void CryptoManager::rotate_key() {
    std::cout << "[CRYPTO] 🔄 Rotando clave de cifrado..." << std::endl;
    std::string new_seed = generate_random_seed();
    if (generate_key_from_seed(new_seed)) {
        std::cout << "[CRYPTO] ✅ Nueva clave generada: " << new_seed << std::endl;
    }
}

std::string CryptoManager::encrypt(const std::string& plaintext) {
    try {
        CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption encryptor;
        encryptor.SetKeyWithIV(encryption_key_, encryption_key_.size(), iv_, iv_.size());

        std::string ciphertext;
        CryptoPP::StringSource ss(plaintext, true,
            new CryptoPP::StreamTransformationFilter(encryptor,
                new CryptoPP::StringSink(ciphertext)
            )
        );

        // Codificar en hexadecimal para facilitar el transporte
        std::string encoded;
        CryptoPP::StringSource ss2(ciphertext, true,
            new CryptoPP::HexEncoder(
                new CryptoPP::StringSink(encoded)
            )
        );

        std::cout << "[CRYPTO] 🔒 Cifrado CBC: " << plaintext.length()
                  << " bytes -> " << encoded.length() << " bytes hex" << std::endl;
        return encoded;

    } catch (const std::exception& e) {
        std::cerr << "[CRYPTO] ❌ Error cifrando: " << e.what() << std::endl;
        throw;
    }
}

std::string CryptoManager::decrypt(const std::string& ciphertext_hex) {
    try {
        // Decodificar desde hexadecimal
        std::string ciphertext;
        CryptoPP::StringSource ss(ciphertext_hex, true,
            new CryptoPP::HexDecoder(
                new CryptoPP::StringSink(ciphertext)
            )
        );

        CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption decryptor;
        decryptor.SetKeyWithIV(encryption_key_, encryption_key_.size(), iv_, iv_.size());

        std::string plaintext;
        CryptoPP::StringSource ss2(ciphertext, true,
            new CryptoPP::StreamTransformationFilter(decryptor,
                new CryptoPP::StringSink(plaintext)
            )
        );

        std::cout << "[CRYPTO] 🔓 Descifrado CBC: " << ciphertext_hex.length()
                  << " bytes hex -> " << plaintext.length() << " bytes" << std::endl;
        return plaintext;

    } catch (const std::exception& e) {
        std::cerr << "[CRYPTO] ❌ Error descifrando: " << e.what() << std::endl;
        throw;
    }
}

std::string CryptoManager::generate_random_seed() {
    try {
        CryptoPP::AutoSeededRandomPool rng;
        CryptoPP::SecByteBlock seed(32); // 256 bits

        rng.GenerateBlock(seed, seed.size());

        // Convertir a hexadecimal legible
        std::string hex_seed;
        CryptoPP::StringSource ss(seed, seed.size(), true,
            new CryptoPP::HexEncoder(
                new CryptoPP::StringSink(hex_seed)
            )
        );

        return hex_seed;

    } catch (const std::exception& e) {
        std::cerr << "[CRYPTO] ❌ Error generando seed aleatoria: " << e.what() << std::endl;

        // Fallback: usar timestamp + random
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        std::stringstream ss;
        ss << "fallback_seed_" << timestamp << "_" << std::rand();
        return ss.str();
    }
}

bool CryptoManager::validate_ciphertext(const std::string& ciphertext) {
    // Validación básica: debe ser hexadecimal y tener longitud mínima
    if (ciphertext.empty() || ciphertext.length() < 2) {
        return false;
    }

    // Verificar que todos los caracteres son hexadecimales
    for (char c : ciphertext) {
        if (!std::isxdigit(c)) {
            return false;
        }
    }

    return true;
}

void CryptoManager::generate_random_bytes(CryptoPP::SecByteBlock& buffer, size_t size) {
    CryptoPP::AutoSeededRandomPool rng;
    rng.GenerateBlock(buffer, size);
}

std::string CryptoManager::bytes_to_hex(const CryptoPP::SecByteBlock& bytes) {
    std::string hex;
    CryptoPP::StringSource ss(bytes, bytes.size(), true,
        new CryptoPP::HexEncoder(
            new CryptoPP::StringSink(hex)
        )
    );
    return hex;
}

CryptoPP::SecByteBlock CryptoManager::hex_to_bytes(const std::string& hex) {
    CryptoPP::SecByteBlock bytes(hex.length() / 2);
    CryptoPP::StringSource ss(hex, true,
        new CryptoPP::HexDecoder(
            new CryptoPP::ArraySink(bytes, bytes.size())
        )
    );
    return bytes;
}
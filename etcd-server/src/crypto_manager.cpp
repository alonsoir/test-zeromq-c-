#include "etcd_server/crypto_manager.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <sodium.h>

// Headers para HKDF (mantener CryptoPP solo para derivación de clave)
#include <cryptopp/hkdf.h>
#include <cryptopp/sha.h>

CryptoManager::CryptoManager() {
    // Initialize libsodium
    if (sodium_init() < 0) {
        throw std::runtime_error("Failed to initialize libsodium");
    }
    
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
        // Usar HKDF para derivación segura de clave
        CryptoPP::SecByteBlock derived_key(KEY_SIZE);

        CryptoPP::HKDF<CryptoPP::SHA256> hkdf;
        hkdf.DeriveKey(derived_key, derived_key.size(),
            (const CryptoPP::byte*)seed.data(), seed.size(),
            (const CryptoPP::byte*)"etcd-server-salt", 16,  // salt
            (const CryptoPP::byte*)"ChaCha20-Poly1305", 17); // info

        encryption_key_.Assign(derived_key, KEY_SIZE);
        key_generation_time_ = std::chrono::system_clock::now();
        seed_ = seed;

        std::cout << "[CRYPTO] 🔑 Clave derivada con HKDF desde seed" << std::endl;
        std::cout << "[CRYPTO]   Key: " << bytes_to_hex(encryption_key_) << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[CRYPTO] ❌ Error generando clave: " << e.what() << std::endl;
        return false;
    }
}

std::string CryptoManager::get_current_seed() const {
    return seed_;
}

std::string CryptoManager::get_encryption_key() const {
    return bytes_to_hex(encryption_key_);
}

bool CryptoManager::should_rotate_key() const {
    auto now = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::hours>(now - key_generation_time_);
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
        if (plaintext.empty()) {
            return "";
        }
        
        // Verify key size
        if (encryption_key_.size() != crypto_secretbox_KEYBYTES) {
            throw std::runtime_error("Invalid key size for ChaCha20");
        }
        
        // Generate random nonce
        std::vector<unsigned char> nonce(crypto_secretbox_NONCEBYTES);
        randombytes_buf(nonce.data(), nonce.size());
        
        // Allocate buffer for ciphertext (plaintext + MAC)
        std::vector<unsigned char> ciphertext(plaintext.size() + crypto_secretbox_MACBYTES);
        
        // Encrypt
        int ret = crypto_secretbox_easy(
            ciphertext.data(),
            reinterpret_cast<const unsigned char*>(plaintext.data()),
            plaintext.size(),
            nonce.data(),
            encryption_key_.data()
        );
        
        if (ret != 0) {
            throw std::runtime_error("ChaCha20 encryption failed");
        }
        
        // Return: nonce + ciphertext (as raw bytes, not hex)
        std::string result;
        result.reserve(nonce.size() + ciphertext.size());
        result.append(reinterpret_cast<const char*>(nonce.data()), nonce.size());
        result.append(reinterpret_cast<const char*>(ciphertext.data()), ciphertext.size());
        
        std::cout << "[CRYPTO] 🔒 Cifrado ChaCha20: " << plaintext.length()
                  << " bytes -> " << result.length() << " bytes" << std::endl;
        
        return result;

    } catch (const std::exception& e) {
        std::cerr << "[CRYPTO] ❌ Error cifrando: " << e.what() << std::endl;
        throw;
    }
}

std::string CryptoManager::decrypt(const std::string& encrypted_data) {
    try {
        if (encrypted_data.empty()) {
            return "";
        }
        
        // Verify key size
        if (encryption_key_.size() != crypto_secretbox_KEYBYTES) {
            throw std::runtime_error("Invalid key size for ChaCha20");
        }
        
        // Minimum size: nonce + MAC
        size_t min_size = crypto_secretbox_NONCEBYTES + crypto_secretbox_MACBYTES;
        if (encrypted_data.size() < min_size) {
            throw std::runtime_error("Encrypted data too short (corrupted?)");
        }
        
        // Extract nonce
        const unsigned char* nonce = reinterpret_cast<const unsigned char*>(
            encrypted_data.data()
        );
        
        // Extract ciphertext
        const unsigned char* ciphertext = reinterpret_cast<const unsigned char*>(
            encrypted_data.data() + crypto_secretbox_NONCEBYTES
        );
        size_t ciphertext_len = encrypted_data.size() - crypto_secretbox_NONCEBYTES;
        
        // Allocate buffer for plaintext
        std::vector<unsigned char> plaintext(ciphertext_len - crypto_secretbox_MACBYTES);
        
        // Decrypt
        int ret = crypto_secretbox_open_easy(
            plaintext.data(),
            ciphertext,
            ciphertext_len,
            nonce,
            encryption_key_.data()
        );
        
        if (ret != 0) {
            throw std::runtime_error("ChaCha20 decryption failed (wrong key or corrupted data)");
        }
        
        std::string result(reinterpret_cast<const char*>(plaintext.data()), plaintext.size());
        
        std::cout << "[CRYPTO] 🔓 Descifrado ChaCha20: " << encrypted_data.length()
                  << " bytes -> " << result.length() << " bytes" << std::endl;
        
        return result;

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
    // ChaCha20 validation: must have minimum size (nonce + MAC)
    size_t min_size = crypto_secretbox_NONCEBYTES + crypto_secretbox_MACBYTES;
    return ciphertext.size() >= min_size;
}

void CryptoManager::generate_random_bytes(CryptoPP::SecByteBlock& buffer, size_t size) {
    CryptoPP::AutoSeededRandomPool rng;
    rng.GenerateBlock(buffer, size);
}

std::string CryptoManager::bytes_to_hex(const CryptoPP::SecByteBlock& bytes) const {
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

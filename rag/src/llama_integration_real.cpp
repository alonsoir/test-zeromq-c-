#include "rag/llama_integration.hpp"
#include <llama.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

class LlamaIntegration::Impl {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    bool model_loaded = false;

public:
    bool loadModel(const std::string& model_path) {
        std::cout << "🔄 Cargando modelo REAL: " << model_path << std::endl;

        try {
            // Inicializar backend
            llama_backend_init();

            // Parámetros del modelo
            llama_model_params model_params = llama_model_default_params();
            model = llama_model_load_from_file(model_path.c_str(), model_params);

            if (!model) {
                std::cerr << "❌ ERROR: No se pudo cargar el modelo desde: " << model_path << std::endl;
                std::cerr << "   Verifica que el archivo existe y es un modelo GGUF válido" << std::endl;
                return false;
            }

            std::cout << "✅ Modelo cargado exitosamente: " << model_path << std::endl;

            // Parámetros del contexto
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = 1024;
            ctx_params.n_threads = 2;

            ctx = llama_init_from_model(model, ctx_params);
            if (!ctx) {
                std::cerr << "❌ ERROR: No se pudo crear el contexto" << std::endl;
                llama_model_free(model);
                model = nullptr;
                return false;
            }

            model_loaded = true;
            std::cout << "🚀 Sistema LLM REAL listo para generar respuestas" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "❌ EXCEPCIÓN durante carga del modelo: " << e.what() << std::endl;
            return false;
        }
    }

    std::string generateResponse(const std::string& prompt, int max_tokens = 128) {
    if (!model_loaded) {
        return "❌ Error: Modelo no cargado correctamente";
    }

    std::cout << "🎯 Generando respuesta REAL para: \"" << prompt << "\"" << std::endl;

    try {
        const llama_vocab* vocab = llama_model_get_vocab(model);

        // MEJORAR EL PROMPT PARA CONTEXTO DE SEGURIDAD
        std::string enhanced_prompt =
            "<|system|>\n"
            "Eres un asistente especializado en seguridad informática y análisis de comandos de Linux. "
            "Responde de manera concisa y profesional. Analiza comandos peligrosos y proporciona recomendaciones de seguridad.\n"
            "<|user|>\n" + prompt + "\n"
            "<|assistant|>\n";

        // Tokenizar el prompt mejorado
        std::vector<llama_token> tokens;
        tokens.resize(enhanced_prompt.size() + 16);

        int n_tokens = llama_tokenize(vocab, enhanced_prompt.c_str(), enhanced_prompt.length(),
                                     tokens.data(), tokens.size(), true, false);

        if (n_tokens <= 0) {
            return "❌ Error: No se pudo tokenizar el prompt";
        }

        tokens.resize(n_tokens);
        std::cout << "📊 Tokens generados: " << tokens.size() << std::endl;

        // Configurar batch
        llama_batch batch = llama_batch_init(512, 0, 1);
        batch.n_tokens = tokens.size();

        for (int i = 0; i < n_tokens; i++) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.seq_id[i][0] = 0;
            batch.n_seq_id[i] = 1;
            batch.logits[i] = (i == n_tokens - 1);
        }

        // Evaluar prompt inicial
        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            return "❌ Error en decodificación inicial: " + std::to_string(ret);
        }

        std::stringstream response;
        int tokens_generated = 0;
        const int n_vocab = llama_vocab_n_tokens(vocab);

        // Generación de tokens
        for (int i = 0; i < max_tokens; i++) {
            float* logits = llama_get_logits_ith(ctx, -1);

            // Sampling greedy
            llama_token new_token = 0;
            float max_logit = logits[0];
            for (int j = 1; j < n_vocab; j++) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    new_token = j;
                }
            }

            // Verificar token EOS
            if (new_token == llama_vocab_eos(vocab)) {
                std::cout << "🔚 Token EOS detectado" << std::endl;
                break;
            }

            // Convertir token a texto
            char piece[32];
            int n_chars = llama_token_to_piece(vocab, new_token, piece, sizeof(piece), 0, false);
            if (n_chars < 0) break;

            if (n_chars > 0) {
                response << std::string(piece, n_chars);
            }

            // Preparar siguiente token
            llama_batch next_batch = llama_batch_init(1, 0, 1);
            next_batch.n_tokens = 1;
            next_batch.token[0] = new_token;
            next_batch.pos[0] = n_tokens + i;
            next_batch.seq_id[0][0] = 0;
            next_batch.n_seq_id[0] = 1;
            next_batch.logits[0] = true;

            ret = llama_decode(ctx, next_batch);
            llama_batch_free(next_batch);

            if (ret != 0) break;

            tokens_generated++;
        }

        std::string result = response.str();
        std::cout << "✅ Generación REAL completada: " << tokens_generated << " tokens" << std::endl;
        return result.empty() ? "🤖 [El modelo no generó respuesta]" : result;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error durante generación: " << e.what() << std::endl;
        return "⚠️  Error en generación";
    }
}

    ~Impl() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
        llama_backend_free();
    }
};

// Implementaciones wrapper (igual que antes)
LlamaIntegration::LlamaIntegration() : pImpl(std::make_unique<Impl>()) {}
LlamaIntegration::~LlamaIntegration() = default;

bool LlamaIntegration::loadModel(const std::string& model_path) {
    return pImpl->loadModel(model_path);
}

std::string LlamaIntegration::generateResponse(const std::string& prompt) {
    return pImpl->generateResponse(prompt);
}
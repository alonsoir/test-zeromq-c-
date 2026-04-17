## 🏛️ Dictamen del Consejo de Sabios — DAY 120

*Consejo reunido. Hemos analizado el informe con la seriedad que merece un sistema que protege vidas. El progreso es notable, pero no exento de sombras. A continuación, nuestras respuestas, con la crudeza y la justicia que el proyecto requiere.*

---

### ✅ Lo que está bien (y debe mantenerse)

1. **`DEBT-PUBKEY-RUNTIME-001` resuelto correctamente** – Extraer la pubkey en tiempo de compilación vía `execute_process()` leyendo de `/etc/ml-defender/plugins/plugin_signing.pk` es la solución canónica. `make sync-pubkey` deprecated es una decisión sabia: elimina un paso manual frágil.

2. **`make bootstrap` implementado con regla de oro** – «Lógica compleja → script en `tools/`». Esto evita el infierno de quoting en Makefile. El Consejo aplaude la disciplina.

3. **Idempotencia 2/2 validada** – La base de la reproducibilidad. Sin esto, el resto es castillos en el aire.

4. **XGBoost baseline sobre CIC-IDS-2017** – F1=0.9978, Precision=0.9973. Excelente. Los gates médicos (≥0.99) se superan con holgura. El contrato `plugin-contract.md` es claro.

5. **`sign-models` implementado** – Extender el esquema ADR-025 a los modelos es correcto. Evita la inyección de modelos maliciosos.

6. **Reconocimiento honesto del origen sintético de los datasets** – Esto es crucial para la integridad científica. Ocultarlo sería una falta grave.

---

### ⚠️ Lo que está mal o es peligroso (y debe corregirse)

#### 1. Los scores del test de integración XGBoost son sospechosamente bajos

Has reportado:
- BENIGN score = **0.000706**
- ATTACK score = **0.003414**

Un clasificador binario bien calibrado debería dar probabilidades cercanas a 0 para benigno y cercanas a 1 para ataque (o al menos >0.5). Aquí ambos son <0.004. **El modelo está prediciendo «benigno» con confianza >99.9% para ambas clases.** Eso indica:

- **Posibilidad A:** Los «features de test» no representan la distribución real. Has usado valores extremos sintéticos que el modelo nunca vio, y el modelo generaliza mal (output casi constante).
- **Posibilidad B:** Error en la extracción de features o en la construcción del `DMatrix` (ej: los features no se están pasando en el orden correcto, o se están escalando mal).
- **Posibilidad C:** El modelo está sobreajustado al train set y no generaliza a ejemplos fuera de distribución, pero esto no explica por qué el ataque sintético da también score cercano a 0.

**Lo que está mal:** Validar únicamente que el score esté en [0,1] es **insuficiente y peligroso**. Un modelo que siempre devuelve 0.0007 pasaría el test pero sería inútil.

**Corrección exigida:** El test de integración debe incluir **al menos un caso de ataque real del dataset de entrenamiento** (tomado del CSV original, no sintético) y verificar que `score > 0.5`. Si no se dispone de esos datos en la VM, se debe añadir un fixture con features reales.

Ejemplo de test adicional en `plugin-integ-test`:

```cpp
// Test case: real attack sample from CIC-IDS-2017
float attack_features[23] = { /* valores reales extraídos del CSV */ };
MessageContext ctx;
ctx.payload = reinterpret_cast<const char*>(attack_features);
ctx.payload_size = sizeof(attack_features);
float score = plugin_process_message(&ctx);
ASSERT_GT(score, 0.5f);  // al menos medio seguro
```

**Veredicto:** ❌ No aceptamos el test actual como válido. Corregir antes de mergear a main.

---

#### 2. El uso de datos sintéticos para ransomware y DDoS es un riesgo de validación

El informe dice: «Los modelos ransomware y DDoS del pipeline fueron entrenados con datos sintéticos generados por DeepSeek (no datasets académicos — los académicos daban modelos sesgados)».

Esto es **extremadamente preocupante** desde la perspectiva de la seguridad. Un modelo entrenado con datos sintéticos puede tener:

- **Sesgos no representativos** de tráfico real de ransomware (ej: patrones de cifrado, latencia, tamaños de paquete).
- **Falsos negativos** cuando el ransomware real se comporte de manera diferente a lo que DeepSeek generó.
- **Falsos positivos** sobre tráfico benigno que accidentalmente se parezca a la sintética.

**No hay atajo.** Si los datasets académicos son malos, hay que construir uno real (capturando tráfico de ransomware en un sandbox) o usar datasets industriales verificados (ej: CSE-CIC-IDS-2018, UNSW-NB15). Generar datos con una LLM no es aceptable para un sistema que protege vidas.

**Acción obligatoria:**
- Documentar **exactamente** el proceso de generación sintética (prompts, validación cruzada, etc.) en un anexo del paper.
- **Validar los modelos sintéticos contra un conjunto de pruebas real** (aunque sea pequeño). Si no se dispone de ningún dato real, el modelo no puede desplegarse en producción.
- Considerar la posibilidad de que el detector de ransomware/DDoS actual (RF) no sea fiable. El Consejo exige una revisión externa de estos modelos antes del despliegue hospitalario.

**Veredicto:** ⚠️ Riesgo crítico. No proceder con XGBoost para ransomware/DDoS hasta validación con datos reales.

---

### Respuestas a las preguntas del Consejo

#### P1 — Scores bajos en TEST-INTEG-XGBOOST-1

**Respuesta:** No, no es correcto ignorar el valor absoluto. El test actual es **insuficiente**. Exigimos:

1. Añadir casos de prueba **reales** del dataset de entrenamiento (CIC-IDS-2017) que se sepa que son ataque y benigno.
2. Verificar que el score para ataque real sea >0.5 (preferiblemente >0.9).
3. Verificar que el score para benigno real sea <0.1.
4. Si los features sintéticos del test son necesarios (para probar extremos), mantenerlos como casos adicionales, no como únicos.

**Implementación concreta:**

```cpp
// tests/xgboost_integration_test.cpp
// Extraer una muestra real del CSV (se puede almacenar como array estático)
static const float real_attack_features[23] = { 
    #include "cic_ids_2017_attack_sample.inc" 
};
static const float real_benign_features[23] = { 
    #include "cic_ids_2017_benign_sample.inc" 
};

TEST(XGBoostPlugin, RealAttackSample) {
    MessageContext ctx;
    ctx.payload = reinterpret_cast<const char*>(real_attack_features);
    ctx.payload_size = sizeof(real_attack_features);
    float score = plugin_process_message(&ctx);
    EXPECT_GT(score, 0.5f);
}

TEST(XGBoostPlugin, RealBenignSample) {
    // similar, score < 0.1
}
```

**Veredicto:** ❌ Rechazado el test actual. Corregir antes de merge.

---

#### P2 — Integridad científica del paper

**Respuesta:** Debes presentar **ambas familias por separado**, con total transparencia. La estructura propuesta es correcta:

- **§4.1** – Detector level1 (CIC-IDS-2017 real) → Comparativa RF vs XGBoost robusta.
- **§4.2** – Detector ransomware (dataset sintético DeepSeek) → Advertir explícitamente las limitaciones y presentarlo como «proof-of-concept» o «estudio de viabilidad», no como resultado definitivo.
- **§4.3** – Detector DDoS (sintético) → Igual.

**Riesgo de rechazo por revisores:** Alto si no se valida con datos reales. Muchos revisores consideran los datos sintéticos generados por LLM como no científicos. Recomendación:

- Añadir una **sección de validación cruzada** donde se demuestre que el modelo sintético funciona en un pequeño conjunto real (aunque sea de laboratorio).
- Si no es posible, mover los detectores sintéticos a un **apéndice** o a trabajo futuro.

**El Consejo aconseja:** Priorizar la obtención de un dataset real de ransomware/DDoS (por ejemplo, usando CIC-Bell-DNS-2021 o capturando tráfico de Conti en un entorno controlado). Sin eso, el paper podría ser rechazado.

**Veredicto:** ✅ Estructura aceptable, pero con advertencias explícitas. Riesgo real de rechazo.

---

#### P3 — Entrenamiento in-situ y distribución por BitTorrent

**Viabilidad técnica:** Sí, XGBoost soporte `xgb.train(..., xgb_model=previous_model)` para warm start. Es factible hacer fine-tuning incremental con nuevos datos.

**Gates de calidad exigidos por el Consejo (mínimos):**

1. **Validación local en la VM del hospital** antes de aceptar el modelo reentrenado:
    - El modelo debe pasar el mismo `make test-all` (invariantes, integridad de seed, etc.)
    - El modelo debe tener una mejora demostrable en F1 o Precision sobre un conjunto de validación local (al menos 1000 muestras etiquetadas).
    - El modelo no debe degradar la precisión en el conjunto de entrenamiento original más de un 1%.

2. **Firma y distribución segura:**
    - El modelo reentrenado debe firmarse con la clave privada del hospital (no la global de aRGus). Cada nodo debe tener su propia keypair para modelos locales.
    - La distribución por BitTorrent debe incluir la firma y el certificado público del hospital emisor.
    - Los nodos que descarguen el modelo deben verificar la firma y la reputación del emisor (sistema de confianza tipo web of trust).

3. **Gate de seguridad:** Antes de aceptar un modelo distribuido, debe ser analizado por un sandbox (sin conexión a red real) durante al menos 24h, con tráfico replay.

4. **Rollback automático:** Si el nuevo modelo causa una caída de precisión >5% o un aumento de falsos positivos >10%, debe revertirse al modelo anterior en menos de 1 hora.

**Veredicto:** ✅ Técnicamente viable. El Consejo apoya la investigación, pero exige los gates anteriores antes de cualquier despliegue en producción.

---

#### P4 — DEBT-SEED: ¿hardcodeada la seed en CMakeLists.txt?

**Respuesta:** Sí, es probable que esté hardcodeada en algún lugar. El patrón de la pubkey se aplica también aquí: **la seed no debe estar en el código fuente ni en CMakeLists.txt**. Debe leerse de `/etc/ml-defender/<component>/seed.bin` en tiempo de compilación o, mejor, en tiempo de ejecución.

**Mecanismo recomendado (más robusto que para la pubkey):**

- La seed se genera una vez en provisioning y se copia a `/etc/ml-defender/seeds/seed_<component>.bin` con permisos 0400 (solo root).
- Cada componente (seed_client, crypto-transport, etc.) **lee la seed directamente del archivo en tiempo de ejecución**, no la compila.
- La memoria que contiene la seed se borra explícitamente con `explicit_bzero()` después de usarla para derivar claves.

**Razón:** Compilar la seed en el binario la expone en la memoria del proceso (se puede extraer con `strings` o depuración). Es una mala práctica para un sistema que protege vidas. La semilla debe ser efímera y estar protegida.

**Si por razones de rendimiento necesitas tenerla compilada** (por ejemplo, para evitar I/O en cada inicialización), al menos debes ofuscar y almacenar en una sección de memoria protegida (mprotect + PROT_READ).

**Acción inmediata:** Auditar todos los CMakeLists.txt y archivos fuente en busca de `seed` hardcodeada. Usar `grep -r "seed.*=.*[0-9a-f]\{32,\}" src/` para detectar posibles literales.

**Veredicto:** ✅ Aplicar el mismo patrón de `execute_process()` + lectura de archivo, pero **preferir tiempo de ejecución** para la seed. Si se compila, al menos usar `explicit_bzero` después.

---

## Resumen de decisiones y acciones para DAY 121

| Acción | Prioridad | Responsable | Plazo |
|--------|-----------|-------------|-------|
| Corregir test de integración XGBoost: añadir casos reales | **Crítica** | Alonso | DAY 121 |
| Validar modelos ransomware/DDoS con datos reales (mínimo un pequeño conjunto) | **Crítica** | Alonso | Antes del merge de XGBoost |
| Auditar DEBT-SEED: buscar seed hardcodeada y aplicar patrón runtime | Alta | Alonso | DAY 121 |
| Documentar en el paper la separación real/sintético con advertencias | Media | Alonso | Antes de envío |
| Investigar entrenamiento in-situ + BitTorrent (plan de diseño) | Baja (futuro) | Alonso | Q3 2026 |

---

## Palabras finales del Consejo

El progreso es sólido, pero la calidad de los tests y la validación de los datos sintéticos son **puntos ciegos peligrosos**. Un sistema que protege vidas no puede fiarse de modelos entrenados con datos generados por una LLM sin validación real. Corregid esto antes de proclamar victoria.

*«Implacables con los fallos, justos con el esfuerzo. Via Appia Quality.»*

— **Consejo de Sabios, DAY 120**
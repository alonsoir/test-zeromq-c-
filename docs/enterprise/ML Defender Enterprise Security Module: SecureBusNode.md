# ML Defender Enterprise Security Module: SecureBusNode

## Plan de Migración y Arquitectura del Módulo Plug-and-Play de Cifrado E2E por Rol

**Proyecto:** ML Defender (aegisIDS)  
**Autor principal:** Alonso (UEX)  
**Co-autores IA:** Claude (Anthropic), ChatGPT (OpenAI)  
**Estado:** Documentación de diseño — versión enterprise futura  
**Fecha de inicio de documentación:** Febrero 2026  
**Clasificación:** Roadmap post-publicación académica  
**Versión del documento:** 2.1

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Análisis del Estado Actual y Problemas](#2-análisis-del-estado-actual-y-problemas)
3. [Alternativas Evaluadas](#3-alternativas-evaluadas)
4. [Arquitectura del Módulo SecureBusNode](#4-arquitectura-del-módulo-securebusnode)
5. [El Problema del Bootstrap: Resolviendo el Huevo Criptográfico](#5-el-problema-del-bootstrap-resolviendo-el-huevo-criptográfico)
6. [Sistema de Identidad Criptográfica Auto-Provisionada](#6-sistema-de-identidad-criptográfica-auto-provisionada)
7. [Protocolo de Cifrado de Mensajes del Pipeline](#7-protocolo-de-cifrado-de-mensajes-del-pipeline)
8. [Gestión de la Clave Raíz: Arquitectura Extensible por Niveles](#8-gestión-de-la-clave-raíz-arquitectura-extensible-por-niveles)
9. [Respuesta ante Compromiso de la Clave Raíz](#9-respuesta-ante-compromiso-de-la-clave-raíz)
10. [Estrategias de Backup por Nivel](#10-estrategias-de-backup-por-nivel)
11. [Rol Residual de etcd-server](#11-rol-residual-de-etcd-server)
12. [Licenciamiento y Mensajería al Usuario](#12-licenciamiento-y-mensajería-al-usuario)
13. [Plan de Migración Paso a Paso](#13-plan-de-migración-paso-a-paso)
14. [Consideraciones de Seguridad Avanzadas](#14-consideraciones-de-seguridad-avanzadas)
15. [Relación con la Publicación Académica](#15-relación-con-la-publicación-académica)
16. [Dependencias Técnicas](#16-dependencias-técnicas)
17. [Riesgos y Mitigaciones](#17-riesgos-y-mitigaciones)
18. [Métricas de Éxito](#18-métricas-de-éxito)
19. [Timeline Estimado](#19-timeline-estimado)
20. [Compromiso Open Source y Modelo Open-Core](#20-compromiso-open-source-y-modelo-open-core)
21. [Apéndices](#apéndices)

---

## 1. Resumen Ejecutivo

Este documento define la arquitectura, diseño e implementación del módulo de seguridad enterprise de ML Defender, denominado **SecureBusNode**. Este módulo sustituye el cifrado centralizado actual basado en distribución de semillas ChaCha20 y certificados HMAC a través de etcd-server, por un sistema de cifrado extremo a extremo (E2E) autenticado por rol, con generación automática de identidad criptográfica en cada componente.

El módulo opera como una capa **plug-and-play**: si está presente, los componentes del pipeline activan automáticamente cifrado de grado militar moderno sin dependencia de etcd-server como autoridad criptográfica. Si no está presente, el sistema opera en modo demo con el cifrado centralizado actual y emite advertencias claras de que esa configuración no es apta para producción.

La gestión de la clave raíz de confianza se implementa mediante una **arquitectura extensible de proveedores** (`RootKeyProvider`), que permite escalar el nivel de protección desde un USB cifrado para pymes hasta HSM con ceremonia Shamir para infraestructura crítica, sin modificar el código del módulo ni de los componentes del pipeline.

### Requisitos fundamentales no negociables

1. **Latencia sub-milisegundo:** Ninguna operación criptográfica puede comprometer el rendimiento del pipeline eBPF/XDP.
2. **Eliminación del vector MITM:** etcd-server deja de ser autoridad de confianza criptográfica; comprometerlo no permite suplantación ni interceptación.
3. **Plug-and-play:** Instalación del módulo sin reconfiguración manual de componentes existentes.
4. **Auto-provisión de identidad:** Cada componente genera su propio par de claves al detectar el módulo, con aprobación anclada a una raíz de confianza de despliegue.
5. **Extensibilidad de custodia:** La protección de la clave raíz se adapta al perfil de riesgo del cliente sin cambios en el módulo.

---

## 2. Análisis del Estado Actual y Problemas

### 2.1 Arquitectura criptográfica actual (modo demo)

| Componente | Función criptográfica actual |
|---|---|
| **etcd-server** | Genera y distribuye semillas ChaCha20 a todos los componentes. Genera y distribuye certificados HMAC. Publica endpoints de discovery. Sirve JSON de configuración desde RAG. |
| **etcd-client** | Descifra mensajes de configuración con ChaCha20. Verifica HMAC de mensajes recibidos. Aplica compresión LZ4 a transmisiones JSON. |
| **Componentes del pipeline** | Reciben semilla compartida de etcd-server. Cifran/descifran mensajes inter-componente con semilla compartida. Verifican HMAC con certificado compartido. |

### 2.2 Vulnerabilidades identificadas

**V-001: Punto único de compromiso criptográfico (CRÍTICO)**  
etcd-server posee y distribuye todos los secretos. Un atacante que comprometa etcd-server obtiene capacidad completa de descifrado y suplantación de cualquier componente del pipeline.

**V-002: Ausencia de Perfect Forward Secrecy (ALTO)**  
La semilla ChaCha20 es estática y compartida. El compromiso de una semilla expone todo el historial de comunicaciones pasadas y futuras hasta la rotación.

**V-003: Sin aislamiento criptográfico por componente (ALTO)**  
Todos los componentes comparten la misma semilla. Un componente comprometido puede leer y falsificar mensajes de cualquier otro componente.

**V-004: Watcher de endpoints como vector de manipulación (MEDIO)**  
El mecanismo de actualización automática de endpoints permite inyección de rutas falsificadas si se compromete etcd o la conexión al mismo.

**V-005: Sin protección anti-replay (MEDIO)**  
El esquema actual no implementa contadores ni ventanas temporales que impidan la re-inyección de mensajes capturados previamente.

**V-006: Bootstrap circular de confianza (MEDIO)**  
Los componentes confían en etcd para recibir las claves con las que verificarán la autenticidad de etcd. Esto constituye un problema de confianza circular sin anclaje externo.

---

## 3. Alternativas Evaluadas

### 3.1 Cifrado Homomórfico Completo (FHE)

**Esquemas evaluados:** TFHE, BGV, CKKS (Microsoft SEAL, OpenFHE).

**Ventaja teórica:** Permite operar sobre datos cifrados sin descifrarlos, eliminando la exposición de datos en tránsito y en procesamiento.

**Motivo de descarte:** Penalización de rendimiento entre 10.000x y 1.000.000x respecto a operaciones en claro. Incompatible con el requisito de latencia sub-milisegundo del pipeline eBPF/XDP. Una sola operación FHE puede consumir milisegundos a segundos, frente a los microsegundos del pipeline actual.

**Conclusión:** FHE es adecuado para análisis offline de datos sensibles (médicos, financieros), no para detección de intrusiones en tiempo real.

### 3.2 Cifrado Homomórfico Parcial (PHE)

**Ventaja:** Órdenes de magnitud más rápido que FHE. Permite operaciones limitadas (solo sumas O solo multiplicaciones) sobre datos cifrados.

**Caso de uso potencial futuro:** Agregación de contadores de alertas cifrados entre nodos distribuidos sin exposición de métricas individuales. No como base del pipeline, sino como optimización puntual para telemetría sensible en despliegues multi-sitio.

**Conclusión:** No descartado, pero relegado a funcionalidad opcional futura del módulo enterprise. No resuelve el problema central de cifrado de mensajes del pipeline.

### 3.3 Cifrado E2E autenticado por rol con criptografía de curva elíptica (SELECCIONADO)

**Esquema:** Ed25519 (firma/identidad) + X25519 (intercambio efímero) + AEAD ChaCha20-Poly1305 (cifrado autenticado).

**Ventajas:**
- Rendimiento: operaciones de firma Ed25519 en ~50-70μs, intercambio X25519 en ~120μs, cifrado ChaCha20-Poly1305 en nanosegundos por byte. Compatible con latencia sub-milisegundo.
- Perfect Forward Secrecy: claves efímeras por sesión o por ventana temporal.
- Aislamiento por rol: cada componente tiene identidad propia, el compromiso de uno no afecta a otros.
- Sin punto central de confianza: etcd no participa en el cifrado.
- Fundamentación probada: Noise Framework (base de WireGuard), Signal Protocol, TLS 1.3.

**Conclusión:** Seleccionado como base del módulo SecureBusNode.

### 3.4 mTLS (Mutual TLS) estándar

**Ventaja:** Ampliamente adoptado, librerías maduras.

**Motivo de descarte como solución principal:** Overhead del handshake TLS (1-3 RTT) excesivo para mensajes de alta frecuencia entre componentes del pipeline. Complejidad de gestión de CA y certificados X.509. Diseñado para conexiones de larga duración, no para mensajes individuales de alta frecuencia en un bus interno.

**Nota:** mTLS puede complementar SecureBusNode para la comunicación con etcd-server (discovery/JSON), pero no como mecanismo de cifrado del bus inter-componente.

---

## 4. Arquitectura del Módulo SecureBusNode

### 4.1 Principio de diseño: Plug-and-Play

```
┌─────────────────────────────────────────────────────────────┐
│                    ML Defender Pipeline                       │
│                                                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ Componente A │   │ Componente B │   │ Componente C │        │
│  │             │   │             │   │             │        │
│  │ ┌─────────┐ │   │ ┌─────────┐ │   │ ┌─────────┐ │        │
│  │ │SecureBus│ │   │ │SecureBus│ │   │ │SecureBus│ │        │
│  │ │  Node   │ │   │ │  Node   │ │   │ │  Node   │ │        │
│  │ │(plug-in)│ │   │ │(plug-in)│ │   │ │(plug-in)│ │        │
│  │ └────┬────┘ │   │ └────┬────┘ │   │ └────┬────┘ │        │
│  └──────┼──────┘   └──────┼──────┘   └──────┼──────┘        │
│         │                 │                 │                │
│         └────────── E2E Cifrado ────────────┘                │
│                   (sin etcd en el path)                       │
│                                                               │
│  ┌──────────────────────────────────┐                        │
│  │          etcd-server              │                        │
│  │  (solo discovery + JSON pasivo)   │                        │
│  │  Sin secretos · Sin autoridad     │                        │
│  └──────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Detección y activación automática

El mecanismo de detección sigue un patrón de descubrimiento por presencia de fichero, similar a cómo los sistemas Unix detectan módulos del kernel:

```
Arranque del componente
        │
        ▼
¿Existe /opt/mldefender/enterprise/libsecurebusnode.so
  + /opt/mldefender/enterprise/LICENSE.key válida?
        │
    ┌───┴───┐
    │       │
   SÍ      NO
    │       │
    ▼       ▼
Cargar    Modo DEMO
módulo    ┌──────────────────────────────────────┐
E2E       │ ⚠ ADVERTENCIA en log (WARNING):      │
    │     │ "SecureBusNode enterprise module      │
    │     │  not detected. Running in DEMO        │
    │     │  mode with centralized etcd           │
    │     │  encryption. This configuration       │
    │     │  is NOT certified for production      │
    │     │  use. Please acquire an enterprise    │
    │     │  license for production-grade         │
    │     │  security, or use current mode        │
    │     │  for demonstration purposes only."    │
    │     └──────────────────────────────────────┘
    ▼
Detectar RootKeyProvider
disponible (USB/Shamir/HSM)
    │
    ▼
Auto-generar identidad
criptográfica del componente
(si no existe previamente)
    │
    ▼
Solicitar aprobación de
identidad via RootKeyProvider
    │
    ▼
Registrar certificado aprobado
en peer registry local
    │
    ▼
Iniciar handshake con
peers aprobados
    │
    ▼
Pipeline operativo con
cifrado E2E por rol
```

### 4.3 Interfaz del módulo

El módulo expone una interfaz C++20 que los componentes del pipeline invocan de forma transparente:

```cpp
// Interfaz pública de SecureBusNode (concepto)
namespace mldefender::enterprise {

class ISecureBusNode {
public:
    virtual ~ISecureBusNode() = default;

    // Auto-provisión: genera o carga identidad del componente
    // Usa el RootKeyProvider configurado para aprobación
    virtual Status initialize(const RoleIdentity& role) = 0;

    // Cifrar mensaje hacia un rol destino
    virtual EncryptedPayload encrypt(
        std::span<const std::byte> plaintext,
        RoleId destination
    ) = 0;

    // Descifrar mensaje recibido (solo si somos el destino)
    virtual DecryptResult decrypt(
        const EncryptedPayload& ciphertext
    ) = 0;

    // Verificar firma de un peer
    virtual bool verify(
        const SignedMessage& message,
        const PeerPublicKey& peer
    ) = 0;

    // Negociar claves efímeras con un peer
    virtual SessionKey handshake(RoleId peer) = 0;

    // Estado del módulo
    virtual ModuleStatus status() const = 0;

    // Acceso al proveedor de clave raíz (para enrollment)
    virtual IRootKeyProvider& rootKeyProvider() = 0;
};

// Factory: carga dinámica del módulo
std::unique_ptr<ISecureBusNode> loadEnterpriseModule();
// Retorna nullptr si el módulo no está presente → modo DEMO

} // namespace mldefender::enterprise
```

### 4.4 Fallback pattern (modo DEMO)

```cpp
// En cada componente del pipeline (concepto)
auto secureBus = mldefender::enterprise::loadEnterpriseModule();

if (secureBus) {
    secureBus->initialize(myRole);
    // Pipeline con cifrado E2E
    logger.info("Enterprise SecureBusNode active — E2E encryption enabled");
} else {
    // Modo DEMO: cifrado centralizado vía etcd
    logger.warn(
        "SecureBusNode enterprise module not detected. "
        "Running in DEMO mode. NOT certified for production use. "
        "Acquire enterprise license or use for demonstration only."
    );
    useLegacyEtcdEncryption();
}
```

---

## 5. El Problema del Bootstrap: Resolviendo el Huevo Criptográfico

### 5.1 El problema fundamental

En criptografía descentralizada, la confianza inicial no puede crearse de la nada a través del mismo canal que se intenta proteger. Este es un problema lógico, no técnico: la criptografía garantiza la integridad de las comunicaciones *después* de establecer confianza mutua, pero no puede generar esa confianza inicial sin un **canal fuera de banda** (out-of-band channel).

Una autofirma Ed25519 por sí sola no resuelve el problema: cualquier atacante puede generar un par Ed25519 y autofirmarse como `detector-rf-01`. La autofirma solo demuestra posesión de la clave privada, no legitimidad del componente.

### 5.2 Modelos de bootstrap evaluados

| Modelo | Mecanismo | Fortaleza | Debilidad | Adecuado para ML Defender |
|---|---|---|---|---|
| **TOFU puro** (SSH) | Confiar en la primera clave vista | Simple, sin infraestructura | Vulnerable en el primer contacto | Solo como fallback degradado |
| **Pre-distribución manual de fingerprints** | Admin verifica visualmente cada fingerprint | Muy seguro | No escala, error humano | Solo para despliegues pequeños |
| **CA centralizada** (PKI tradicional) | Autoridad de certificación firma identidades | Estándar, maduro | Reintroduce punto central de fallo — exactamente lo que queremos eliminar | No |
| **Hardware root of trust** (TPM/HSM) | Clave de endorsement firmada en fábrica | Muy fuerte, sin dependencia de red | Requiere hardware específico, confianza en fabricante | Como nivel avanzado |
| **Ancla de despliegue embebida** | Clave pública raíz en el módulo, aprobación offline | Sin punto central runtime, air-gapped, auditable | Requiere ceremonia humana inicial | **SÍ — solución primaria** |

### 5.3 Solución seleccionada: Ancla de Despliegue con Aprobación Offline

El "huevo" criptográfico de ML Defender es la **ceremonia de despliegue**: el momento físico en que un humano autorizado, con acceso legítimo al servidor y posesión de la clave raíz offline, aprueba la identidad de cada nodo.

**Principio fundamental:** La confianza criptográfica NO se establece en tiempo de ejecución a través de la red. Se establece en tiempo de despliegue, mediante un acto humano verificable y auditable, usando un secreto que nunca toca la red.

```
Fabricación del módulo enterprise
        │
        ▼
Clave pública raíz de la organización
embebida en libsecurebusnode.so en
tiempo de compilación del módulo
        │
        ▼
Módulo se distribuye al cliente
(la clave embebida es PÚBLICA — no es un secreto)
        │
        ▼
Despliegue físico en servidor autorizado  ← ESTE ES EL "HUEVO"
(acto humano, acceso físico/SSH verificado)
        │
        ▼
Componente genera par Ed25519 local
+ certificado autofirmado
+ solicitud de aprobación (enrollment request)
        │
        ▼
Administrador ejecuta herramienta de enrollment
con acceso a la clave raíz privada
(vía RootKeyProvider: USB / Shamir / HSM)
        │
        ▼
RootKeyProvider firma la aprobación del certificado
sin exponer la clave raíz privada al sistema
        │
        ▼
Certificado aprobado (firmado por raíz) se deposita
en el componente → peers lo aceptan porque la cadena
de firma llega al ancla pública embebida
        │
        ▼
Handshake X25519 entre peers aprobados
→ Canal seguro E2E establecido
```

### 5.4 Por qué NO se requiere compilar en un entorno autorizado

Una distinción crítica: el módulo `libsecurebusnode.so` se compila **una vez** con la clave pública raíz embebida. Esta clave es pública — no es un secreto. Se puede distribuir el módulo por canales normales (descarga, paquete, etc.) sin comprometer la seguridad.

Lo que se requiere es que el **despliegue y la ceremonia de enrollment** ocurran en un entorno autorizado. Esto es un requisito operativo normal: ya se hace cada vez que se instala software en un servidor de producción. No es un requisito extraordinario.

La seguridad del sistema depende de:
1. Que la clave raíz **privada** esté custodiada adecuadamente (ver sección 8).
2. Que la ceremonia de enrollment se realice con acceso legítimo al nodo.
3. Que el módulo no haya sido modificado (verificable por hash/firma del paquete de distribución).

### 5.5 Trust-On-First-Use (TOFU) como fallback degradado

Para escenarios donde la ceremonia de enrollment formal no es viable (laboratorios, desarrollo, pruebas), el módulo puede operar en modo TOFU:

- La primera vez que un componente ve la clave pública de un peer, la acepta y la almacena (pin).
- Las conexiones subsiguientes verifican que la clave pública coincida con la almacenada.
- Si la clave cambia inesperadamente, se rechaza la conexión y se genera alerta CRITICAL.

**Este modo se marca explícitamente en los logs como inferior al modo con enrollment completo:**

```
[WARN] SecureBusNode operating in TOFU mode — enrollment ceremony not completed
[WARN] First-contact trust window is vulnerable to MITM
[WARN] Complete enrollment ceremony for production-grade trust anchoring
```

---

## 6. Sistema de Identidad Criptográfica Auto-Provisionada

### 6.1 Proceso de auto-provisión con enrollment

Cada componente, al detectar el módulo SecureBusNode y arrancar por primera vez, ejecuta el siguiente proceso:

**Paso 1 — Generación de par de claves Ed25519:**
- Se genera un par de claves Ed25519 (clave privada de 32 bytes, clave pública de 32 bytes) usando un CSPRNG del sistema operativo (`/dev/urandom` o `getrandom(2)`).
- La clave privada se almacena en un fichero local protegido con permisos `0600`, idealmente cifrado con una clave derivada de un secreto local del nodo (TPM si está disponible, o passphrase de despliegue).
- La clave pública se asocia al identificador de rol del componente.

**Paso 2 — Generación de solicitud de enrollment:**
- El componente genera un enrollment request que contiene:
    - `role_id`: identificador del componente (ej: `capture-node-01`, `detector-rf-01`, `logger-csv-01`)
    - `public_key`: clave pública Ed25519 (32 bytes)
    - `created_at`: timestamp de creación (UTC)
    - `challenge`: nonce aleatorio de 32 bytes para prevenir replay del enrollment
    - `self_signature`: firma Ed25519 del enrollment request (demuestra posesión de la clave privada)
    - `module_version`: versión del módulo SecureBusNode
    - `pipeline_version`: versión de ML Defender
    - `system_fingerprint`: hash del hardware/OS para auditoría (opcional)

**Paso 3 — Aprobación por el administrador (ceremonia de enrollment):**
- El administrador ejecuta la herramienta de enrollment: `mldefender-enroll --provider <usb|shamir|hsm>`
- La herramienta carga el `RootKeyProvider` correspondiente.
- El proveedor firma la solicitud de enrollment sin exponer la clave raíz al sistema.
- El resultado es un **certificado de rol aprobado**:
    - Todo el contenido del enrollment request
    - `expires_at`: timestamp de expiración (configurable, recomendado 90 días)
    - `root_signature`: firma de la clave raíz sobre el certificado completo
    - `approval_timestamp`: timestamp de la ceremonia de enrollment
    - `approver_fingerprint`: fingerprint de la clave raíz usada (para auditoría)

**Paso 4 — Registro en peer registry:**
- El certificado aprobado se deposita en `/opt/mldefender/enterprise/peers/`.
- Los peers descubren nuevos certificados por presencia en este directorio.
- Cada peer verifica que el certificado esté firmado por el ancla raíz embebida en su módulo.
- Opcionalmente, etcd-server puede servir como mirror del directorio de certificados públicos.

**Paso 5 — Primer handshake con peers:**
- Al descubrir un peer con certificado aprobado, se ejecuta un handshake Noise Protocol (patrón **IK** o **XX**):
    - Verificación de que el certificado del peer tiene `root_signature` válida.
    - Intercambio de claves efímeras X25519.
    - Derivación de clave de sesión AEAD (ChaCha20-Poly1305).
    - Verificación mutua de identidad Ed25519.
- El resultado es una clave de sesión efímera única para cada par de componentes.

### 6.2 Diagrama de auto-provisión con enrollment

```
Componente arranca con SecureBusNode
        │
        ▼
┌──────────────────────────────────┐
│ ¿Existe identidad local aprobada?│
│ (/opt/.../identity/ con          │
│  certificado + root_signature)   │
└──────────┬───────────────────────┘
       ┌───┴───┐
       │       │
      SÍ      NO
       │       │
       │       ▼
       │   Generar Ed25519 keypair (CSPRNG)
       │       │
       │       ▼
       │   Almacenar clave privada
       │   (permisos 0600, cifrada)
       │       │
       │       ▼
       │   Generar enrollment request
       │   (con challenge anti-replay)
       │       │
       │       ▼
       │   ┌──────────────────────────────┐
       │   │ ESPERAR CEREMONIA DE         │
       │   │ ENROLLMENT                   │
       │   │                              │
       │   │ Admin ejecuta:               │
       │   │ mldefender-enroll            │
       │   │   --role detector-rf-01      │
       │   │   --provider usb|shamir|hsm  │
       │   │                              │
       │   │ RootKeyProvider firma        │
       │   │ el enrollment request        │
       │   └──────────┬───────────────────┘
       │              │
       │              ▼
       │   Certificado aprobado generado
       │   (con root_signature)
       │       │
       ▼       ▼
  Cargar identidad + certificado aprobado
        │
        ▼
  Verificar que certificado no ha expirado
        │
        ▼
  Publicar certificado aprobado en peer registry
        │
        ▼
  Descubrir peers con certificados aprobados
  (verificar root_signature de cada peer)
        │
        ▼
  Handshake Noise IK/XX con cada peer aprobado
  (X25519 efímero + verificación Ed25519)
        │
        ▼
  Claves de sesión AEAD establecidas
  → Pipeline operativo con E2E anclado
```

### 6.3 Rotación de claves y ciclo de vida

| Evento | Acción | Impacto en pipeline |
|---|---|---|
| Expiración de certificado (90 días) | Auto-regeneración de par Ed25519. Nueva solicitud de enrollment. Requiere nueva ceremonia de aprobación. | Pipeline continúa con la sesión existente hasta que la nueva aprobación se complete. |
| Compromiso sospechado de un nodo | Revocación: eliminar certificado del peer registry + añadir a CRL. Regenerar identidad del nodo comprometido con nueva ceremonia. | Peers rechazan el nodo revocado inmediatamente. |
| Rotación de clave de sesión | Automática por ventana temporal (configurable, cada 1 hora o cada N mensajes). | Nuevo handshake X25519. PFS garantizado. |
| Actualización del módulo | Nueva versión del .so reemplaza la anterior. Identidades y certificados aprobados se mantienen si la clave raíz no ha cambiado. | Reinicio de handshakes con negociación de nueva versión de protocolo. |
| Rotación de clave raíz | Nueva clave raíz. Todos los certificados deben ser re-aprobados. | Requiere ceremonia completa de re-enrollment de todos los nodos. Planificar con ventana de mantenimiento. |
| Desinstalación del módulo | Componente detecta ausencia → fallback a modo DEMO con advertencia. | Alerta CRITICAL si `enforce_enterprise=true`. |

---

## 7. Protocolo de Cifrado de Mensajes del Pipeline

### 7.1 Formato de mensaje cifrado

```
┌──────────────────────────────────────────────────────────┐
│                  SecureBus Message v1                      │
├──────────────────────────────────────────────────────────┤
│ Header (no cifrado, firmado)                              │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ version      : uint8   (0x01)                        │ │
│ │ sender_role  : uint16  (ID del rol emisor)           │ │
│ │ dest_role    : uint16  (ID del rol destino)          │ │
│ │ timestamp    : uint64  (nanosegundos desde epoch)    │ │
│ │ nonce        : byte[24] (nonce único por mensaje)    │ │
│ │ session_id   : byte[8]  (ID de sesión efímera)       │ │
│ │ sequence     : uint64  (contador monotónico)         │ │
│ │ payload_len  : uint32  (longitud del payload cifrado)│ │
│ │ header_sig   : byte[64] (firma Ed25519 del header)   │ │
│ └──────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────┤
│ Payload (cifrado con AEAD ChaCha20-Poly1305)              │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ ciphertext   : byte[payload_len - 16]                │ │
│ │ auth_tag     : byte[16] (Poly1305 MAC)               │ │
│ └──────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

Tamaño total del overhead: 137 bytes fijos por mensaje
```

### 7.2 Protección anti-replay

El esquema combina tres mecanismos complementarios para robustez en un sistema distribuido con procesamiento paralelo:

**Mecanismo 1 — Contador monotónico por sesión:**
- Cada sesión entre un par de peers mantiene un contador `sequence` que se incrementa con cada mensaje enviado.
- El receptor mantiene el valor más alto recibido y rechaza mensajes con `sequence` menor o igual.

**Mecanismo 2 — Ventana temporal con tolerancia configurable:**
- El `timestamp` del header se verifica contra el reloj local del receptor.
- Tolerancia configurable (recomendado: ±500ms misma red, ±2s nodos distribuidos).
- Mensajes fuera de la ventana temporal se rechazan, incluso si el `sequence` es válido.

**Mecanismo 3 — Bitmap de nonces recientes:**
- Bitmap de los últimos N nonces recibidos (configurable, recomendado N=1024).
- Nonces repetidos se rechazan inmediatamente.

### 7.3 Benchmarks estimados de rendimiento

| Operación | Latencia estimada | Fuente |
|---|---|---|
| Firma Ed25519 del header | ~50-70 μs | libsodium / BoringSSL benchmarks |
| Verificación Ed25519 | ~120-180 μs | libsodium / BoringSSL benchmarks |
| Cifrado ChaCha20-Poly1305 (1 KB payload) | ~0.5-1 μs | Hardware moderno con SIMD |
| Cifrado ChaCha20-Poly1305 (64 B payload típico) | ~0.1-0.2 μs | Hardware moderno con SIMD |
| Handshake X25519 (una vez por sesión) | ~120-150 μs | libsodium benchmarks |
| **Overhead total por mensaje típico** | **~180-260 μs** | Firma + cifrado + verificación |

**Optimizaciones disponibles:**
- Firma del header en paralelo con preparación del payload.
- Verificación asíncrona (verificar post-procesamiento con rollback si falla).
- Soporte AVX2/AVX-512 para ChaCha20 acelerado por hardware.
- Modo "batch signature" con firma Merkle sobre grupo de mensajes consecutivos.

---

## 8. Gestión de la Clave Raíz: Arquitectura Extensible por Niveles

### 8.1 El problema irreducible

La cadena de seguridad criptográfica siempre termina en un secreto que alguien tiene que custodiar físicamente. No existe solución puramente matemática a este problema. La clave raíz privada es el activo más valioso del sistema: quien la posee puede aprobar nodos falsos.

La estrategia de ML Defender no es hacer imposible el robo de la clave raíz, sino hacerlo **detectable, contenible y recuperable**, con niveles de protección escalables según el perfil de riesgo del cliente.

### 8.2 Interfaz abstracta: IRootKeyProvider

El módulo SecureBusNode no sabe ni necesita saber dónde ni cómo se almacena la clave raíz privada. Toda interacción con la clave raíz se realiza a través de una interfaz abstracta que permite extensibilidad sin cambiar el código del módulo ni del pipeline:

```cpp
namespace mldefender::enterprise {

// Interfaz abstracta — el módulo solo conoce esta interfaz
class IRootKeyProvider {
public:
    virtual ~IRootKeyProvider() = default;

    // Identificación del nivel de protección
    virtual ProviderLevel level() const = 0;
    // Enum: USB_ENCRYPTED, SHAMIR_SPLIT, HSM_DEVICE, HSM_SHAMIR

    // Nombre legible del proveedor
    virtual std::string_view name() const = 0;

    // ¿Está el proveedor listo para firmar?
    // (USB insertado, quorum Shamir alcanzado, HSM conectado)
    virtual bool isReady() const = 0;

    // Firmar un enrollment request
    // La clave privada NUNCA sale del proveedor
    virtual SignatureResult signEnrollment(
        const EnrollmentRequest& request
    ) = 0;

    // Firmar una revocación
    virtual SignatureResult signRevocation(
        const RevocationRequest& request
    ) = 0;

    // Verificar una firma (usa la clave pública — siempre disponible)
    virtual bool verifySignature(
        std::span<const std::byte> data,
        const Signature& signature
    ) const = 0;

    // Obtener la clave pública raíz (para verificación por peers)
    virtual PublicKey rootPublicKey() const = 0;

    // Estado del proveedor para diagnóstico
    virtual ProviderStatus status() const = 0;

    // Capacidades del proveedor
    virtual ProviderCapabilities capabilities() const = 0;
};

// Capabilities flags
struct ProviderCapabilities {
    bool supports_backup;        // ¿Puede exportar backup cifrado?
    bool supports_rotation;      // ¿Puede rotar la clave raíz?
    bool supports_audit_log;     // ¿Registra operaciones?
    bool key_never_in_memory;    // ¿La clave privada nunca toca RAM del host?
    bool anti_tamper;            // ¿Tiene protección física anti-manipulación?
    uint8_t min_custodians;      // Mínimo de custodios requeridos (1 para USB/HSM)
    uint8_t quorum_threshold;    // Quorum K-de-N (0 si no aplica)
};

// Factory: detecta automáticamente el proveedor disponible
// Busca en orden de seguridad: HSM_SHAMIR > HSM_DEVICE > SHAMIR_SPLIT > USB_ENCRYPTED
std::unique_ptr<IRootKeyProvider> detectRootKeyProvider(
    const ProviderConfig& config
);

} // namespace mldefender::enterprise
```

### 8.3 Nivel 1 — USB Cifrado con Passphrase (Pymes, hospitales, escuelas)

**Perfil de cliente:** Pequeñas organizaciones con presupuesto limitado. Tu público objetivo principal.

**Mecanismo:**
- La clave raíz Ed25519 se almacena en un fichero dentro de un USB.
- El fichero está cifrado con AEAD ChaCha20-Poly1305 usando una clave derivada por **Argon2id** de una passphrase fuerte.
- Para firmar un enrollment, se necesita: (1) el USB físico insertado, y (2) la passphrase en la cabeza del administrador.
- Un atacante necesita **ambas cosas simultáneamente** para usar la clave.

**Implementación:**

```cpp
class UsbEncryptedProvider : public IRootKeyProvider {
    // La clave se descifra en memoria solo durante la operación de firma
    // y se borra (sodium_memzero) inmediatamente después.
    // El USB se monta como read-only.
    // El fichero cifrado usa Argon2id con parámetros:
    //   - m_cost: 256 MB (resistente a GPU)
    //   - t_cost: 4 iteraciones
    //   - parallelism: 2
    // La passphrase nunca se almacena — se solicita interactivamente.

    ProviderLevel level() const override {
        return ProviderLevel::USB_ENCRYPTED;
    }

    ProviderCapabilities capabilities() const override {
        return {
            .supports_backup = true,
            .supports_rotation = true,
            .supports_audit_log = true,    // Log local
            .key_never_in_memory = false,  // Breve exposición durante firma
            .anti_tamper = false,
            .min_custodians = 1,
            .quorum_threshold = 0
        };
    }
};
```

**Ceremonia de enrollment con USB:**

```bash
# Administrador inserta el USB cifrado
$ mldefender-enroll \
    --role detector-rf-01 \
    --provider usb \
    --usb-path /media/admin/mldefender-root/root.key.enc

Enter root key passphrase: ********
[OK] Root key decrypted successfully
[OK] Enrollment request for 'detector-rf-01' verified
[OK] Certificate signed with root key
[OK] Approved certificate written to /opt/mldefender/enterprise/peers/detector-rf-01.cert
[OK] Root key erased from memory

# Administrador retira el USB y lo guarda
```

**Coste:** ~10€ (USB) + 0€ (software). Accesible para cualquier organización.

### 8.4 Nivel 2 — Shamir's Secret Sharing (Organizaciones medianas, cooperativas)

**Perfil de cliente:** Organizaciones donde ningún individuo debe tener control total. Cooperativas, administraciones públicas pequeñas, equipos con múltiples administradores.

**Mecanismo:**
- La clave raíz se fragmenta matemáticamente en **N partes** usando el esquema de Shamir, de las cuales se necesitan **K** para reconstruirla (ej: 3-de-5).
- Cada fragmento lo custodia una persona diferente de la organización, cifrado individualmente con su propia passphrase.
- Para firmar un enrollment, K custodios deben proporcionar sus fragmentos simultáneamente.
- Un atacante tendría que comprometer a K personas independientes.

**Propiedades matemáticas de Shamir's Secret Sharing:**
- Con K-1 fragmentos, no se obtiene NINGUNA información sobre la clave (seguridad de teoría de información, no computacional).
- Cualquier subconjunto de K fragmentos reconstruye la clave idénticamente.
- Los fragmentos pueden renovarse sin cambiar la clave raíz (proactive secret sharing).

**Implementación:**

```cpp
class ShamirSplitProvider : public IRootKeyProvider {
    // Parámetros configurables:
    //   total_shares (N): número total de fragmentos
    //   threshold (K): número mínimo para reconstruir
    //   Recomendación: 3-de-5 para organizaciones medianas
    //                  5-de-9 para organizaciones grandes
    //
    // Cada fragmento se almacena cifrado con Argon2id
    // individual por custodio (puede ser USB, fichero, o papel impreso).
    //
    // La reconstrucción ocurre en memoria protegida,
    // la clave se usa para firmar y se borra inmediatamente.

    ProviderLevel level() const override {
        return ProviderLevel::SHAMIR_SPLIT;
    }

    ProviderCapabilities capabilities() const override {
        return {
            .supports_backup = true,      // Fragmentos adicionales
            .supports_rotation = true,    // Proactive resharing
            .supports_audit_log = true,
            .key_never_in_memory = false, // Breve exposición durante reconstrucción
            .anti_tamper = false,
            .min_custodians = threshold_k,
            .quorum_threshold = threshold_k
        };
    }
};
```

**Ceremonia de enrollment con Shamir (3-de-5):**

```bash
$ mldefender-enroll \
    --role detector-rf-01 \
    --provider shamir \
    --threshold 3

Shamir enrollment ceremony — 3 of 5 custodians required

Custodian 1 — insert share media or enter share path:
  Path: /media/custodian1/share.enc
  Passphrase: ********
  [OK] Share 1/3 accepted

Custodian 2 — insert share media or enter share path:
  Path: /media/custodian2/share.enc
  Passphrase: ********
  [OK] Share 2/3 accepted

Custodian 3 — insert share media or enter share path:
  Path: /media/custodian3/share.enc
  Passphrase: ********
  [OK] Share 3/3 accepted — quorum reached

[OK] Root key reconstructed in protected memory
[OK] Enrollment request for 'detector-rf-01' verified
[OK] Certificate signed with root key
[OK] Approved certificate written to /opt/mldefender/enterprise/peers/detector-rf-01.cert
[OK] Root key and all shares erased from memory

Ceremony complete. All custodians may remove their media.
```

**Coste:** ~30-50€ (múltiples USBs) + 0€ (software). Requiere coordinación humana.

### 8.5 Nivel 3 — HSM (Hardware Security Module) (Infraestructura crítica)

**Perfil de cliente:** Hospitales grandes, infraestructura energética, redes de transporte, administraciones públicas.

**Mecanismo:**
- La clave raíz Ed25519 se genera **dentro** del HSM y nunca sale de él.
- Todas las operaciones de firma ocurren dentro del chip del HSM.
- No hay fichero que copiar, no hay clave en memoria del host.
- Los HSM tienen protecciones anti-tampering: si detectan manipulación física (apertura, voltaje anómalo, temperatura extrema), borran las claves automáticamente.
- Un atacante tendría que llevarse el dispositivo físico entero, y aun así las protecciones anti-tamper pueden destruir las claves.

**Hardware recomendado:**
- **YubiHSM 2:** ~650€. Factor de forma USB. Almacena claves Ed25519. API PKCS#11 y propia. Suficiente para la mayoría de despliegues.
- **Nitrokey HSM 2:** ~100-200€. Alternativa open-source. Soporte PKCS#11.
- **HSM de rack (Thales Luna, AWS CloudHSM):** Miles de euros. Para despliegues de muy alta disponibilidad.

**Implementación:**

```cpp
class HsmDeviceProvider : public IRootKeyProvider {
    // Interactúa con el HSM via PKCS#11 o API nativa del fabricante.
    // La clave privada NUNCA es accesible fuera del HSM.
    // La operación de firma envía datos al HSM y recibe la firma.
    //
    // Protección adicional: PIN de acceso al HSM requerido
    // para cada operación de firma.
    //
    // Respaldo: clave puede replicarse a un segundo HSM
    // usando el protocolo de backup seguro del fabricante
    // (wrapped key export — la clave nunca se expone en claro).

    ProviderLevel level() const override {
        return ProviderLevel::HSM_DEVICE;
    }

    ProviderCapabilities capabilities() const override {
        return {
            .supports_backup = true,      // Wrapped export a segundo HSM
            .supports_rotation = true,
            .supports_audit_log = true,   // HSM tiene log interno tamper-proof
            .key_never_in_memory = true,  // NUNCA toca RAM del host
            .anti_tamper = true,          // Protección física del fabricante
            .min_custodians = 1,
            .quorum_threshold = 0
        };
    }
};
```

**Ceremonia de enrollment con HSM:**

```bash
$ mldefender-enroll \
    --role detector-rf-01 \
    --provider hsm \
    --hsm-slot 0

HSM enrollment — device detected: YubiHSM 2 (serial: XXXX)
Enter HSM access PIN: ********

[OK] HSM authenticated
[OK] Enrollment request for 'detector-rf-01' sent to HSM for signing
[OK] HSM returned signature (key never left device)
[OK] Certificate signed with root key (HSM-protected)
[OK] Approved certificate written to /opt/mldefender/enterprise/peers/detector-rf-01.cert

HSM session closed.
```

**Coste:** 100-650€ (hardware) + 0€ (software).

### 8.6 Nivel 4 — HSM + Shamir (Infraestructura crítica de alta seguridad)

**Perfil de cliente:** Infraestructura de defensa, sistemas financieros, infraestructura nacional. Máximo nivel de protección.

**Mecanismo:**
- Combinación del HSM (clave nunca en memoria) con Shamir (ningún individuo tiene control).
- El HSM requiere la presencia física simultánea de K custodios para desbloquearse.
- Cada custodio posee un token de activación (puede ser otro hardware token, smart card, o fragmento de PIN).
- Es el modelo que usa ICANN para firmar la raíz de DNSSEC: 7 custodios de los cuales se necesitan 5, con ceremonia presencial documentada y registrada en vídeo.

**Implementación:**

```cpp
class HsmShamirProvider : public IRootKeyProvider {
    // Combina HsmDeviceProvider con quorum de activación.
    // El HSM no ejecuta operaciones de firma hasta que K de N
    // custodios han proporcionado su factor de autenticación.
    //
    // Factores posibles por custodio:
    //   - PIN parcial (fragmento Shamir del PIN del HSM)
    //   - Smart card personal
    //   - Token hardware (YubiKey como segundo factor)
    //   - Combinación de los anteriores

    ProviderLevel level() const override {
        return ProviderLevel::HSM_SHAMIR;
    }

    ProviderCapabilities capabilities() const override {
        return {
            .supports_backup = true,
            .supports_rotation = true,
            .supports_audit_log = true,
            .key_never_in_memory = true,
            .anti_tamper = true,
            .min_custodians = threshold_k,
            .quorum_threshold = threshold_k
        };
    }
};
```

**Coste:** 650-5000€ (HSM + tokens) + coordinación organizacional significativa.

### 8.7 Detección automática y selección de proveedor

El módulo SecureBusNode detecta automáticamente el proveedor disponible al arrancar, priorizando siempre el nivel más alto detectado:

```cpp
// Orden de detección (mayor a menor seguridad):
// 1. HSM + Shamir  → si HSM presente + config de quorum
// 2. HSM solo      → si HSM presente sin quorum
// 3. Shamir split  → si config de shares presente
// 4. USB cifrado   → si path de USB configurado
// 5. (ninguno)     → enrollment pendiente, nodo no puede operar en E2E

auto provider = mldefender::enterprise::detectRootKeyProvider(config);
if (!provider) {
    logger.error("No RootKeyProvider detected — enrollment cannot proceed");
    logger.error("Configure provider in /opt/mldefender/enterprise/provider.conf");
}
```

### 8.8 Tabla comparativa de niveles

| Aspecto | Nivel 1: USB | Nivel 2: Shamir | Nivel 3: HSM | Nivel 4: HSM+Shamir |
|---|---|---|---|---|
| **Clave en memoria del host** | Breve (durante firma) | Breve (durante reconstrucción) | Nunca | Nunca |
| **Protección anti-tampering** | No | No | Sí (hardware) | Sí (hardware) |
| **Personas necesarias** | 1 | K de N | 1 | K de N |
| **Robo del medio físico** | Necesita passphrase | Necesita K medios + K passphrases | PIN del HSM | K tokens + PIN HSM |
| **Copia no autorizada** | Posible (fichero) | Parcial (un fragmento es inútil) | Imposible (hardware) | Imposible (hardware) |
| **Coste aproximado** | ~10€ | ~30-50€ | ~100-650€ | ~650-5000€ |
| **Complejidad operativa** | Baja | Media | Baja-Media | Alta |
| **Auditoría** | Log local | Log local | Log tamper-proof del HSM | Log tamper-proof + registro de ceremonia |
| **Cliente objetivo** | Pymes, escuelas | Cooperativas, admins. públicas | Hospitales, energía | Defensa, finanzas |
| **Compatibilidad air-gapped** | Sí | Sí | Sí | Sí |

---

## 9. Respuesta ante Compromiso de la Clave Raíz

### 9.1 Principio: el compromiso de la clave raíz NO es el fin del mundo

Un buen diseño de seguridad asume que cualquier secreto puede ser comprometido. La calidad del sistema se mide por lo que ocurre **después** del compromiso, no por la improbabilidad del mismo.

Si un atacante obtiene la clave raíz, puede firmar certificados de nodos falsos. Pero las siguientes mitigaciones limitan severamente el impacto:

### 9.2 Mitigaciones arquitectónicas (activas antes del compromiso)

**M-001: Los peers legítimos ya están pineados**
Los nodos que ya completaron el enrollment tienen los certificados de sus peers almacenados y pineados. Un nodo falso firmado con la clave raíz robada no entra automáticamente en las sesiones existentes — los peers establecidos no aceptan nuevos handshakes no solicitados salvo que se configure explícitamente.

**M-002: Registro de enrollment (append-only log)**
Cada certificado aprobado se registra en un log append-only (inmutable). Si aparece un certificado que el administrador no aprobó, es detectable por auditoría. El log incluye: timestamp, role_id, fingerprint del certificado, fingerprint de la clave raíz usada, IP de origen de la solicitud.

**M-003: Alerta de nodos no reconocidos**
Cuando un peer recibe un handshake de un nodo con certificado válido (firmado por raíz) pero que no está en su lista de peers esperados, genera una alerta de nivel WARNING. Si el nodo no fue anunciado por el administrador, esto es una señal temprana de compromiso.

**M-004: Ventana temporal del certificado**
Los certificados aprobados tienen `expires_at`. Incluso con la clave raíz robada, los certificados falsos expiran. Si se detecta el compromiso y se rota la clave raíz antes de la expiración, los certificados falsos quedan inutilizados.

**M-005: Firma del enrollment incluye system_fingerprint**
El enrollment request incluye un hash del hardware/OS del nodo. Un certificado firmado para un nodo con fingerprint diferente al hardware real genera alerta. No es infalible (el fingerprint puede falsificarse), pero añade una capa de detección.

### 9.3 Procedimiento de respuesta ante compromiso confirmado

```
COMPROMISO DE CLAVE RAÍZ DETECTADO
        │
        ▼
  1. CONTENCIÓN INMEDIATA (minutos)
  ┌─────────────────────────────────────────┐
  │ • Desconectar el medio comprometido     │
  │   (USB, HSM, shares afectados)          │
  │ • Alertar a todos los custodios         │
  │ • Activar modo "paranoid" en todos      │
  │   los nodos: rechazar CUALQUIER nuevo   │
  │   handshake, solo mantener sesiones     │
  │   existentes verificadas                │
  └────────────────┬────────────────────────┘
                   │
                   ▼
  2. EVALUACIÓN (horas)
  ┌─────────────────────────────────────────┐
  │ • Revisar el enrollment log:            │
  │   ¿hay certificados no autorizados?     │
  │ • Identificar ventana de exposición     │
  │ • Determinar si algún nodo falso fue    │
  │   aceptado por peers                    │
  │ • Auditar tráfico del pipeline          │
  └────────────────┬────────────────────────┘
                   │
                   ▼
  3. REVOCACIÓN (horas)
  ┌─────────────────────────────────────────┐
  │ • Emitir CRL (Certificate Revocation    │
  │   List) con todos los certificados      │
  │   firmados desde la ventana de          │
  │   exposición sospechada                 │
  │ • Distribuir CRL a todos los nodos      │
  │ • Los nodos rechazan peers cuyo         │
  │   certificado está en la CRL            │
  └────────────────┬────────────────────────┘
                   │
                   ▼
  4. ROTACIÓN DE CLAVE RAÍZ (horas-días)
  ┌─────────────────────────────────────────┐
  │ • Generar nueva clave raíz Ed25519      │
  │   en nuevo medio (nuevo USB/HSM/shares) │
  │ • Compilar nuevo módulo con nueva clave │
  │   pública embebida, o actualizar config │
  │   de clave pública raíz en los nodos    │
  │ • Planificar ventana de mantenimiento   │
  └────────────────┬────────────────────────┘
                   │
                   ▼
  5. RE-ENROLLMENT COMPLETO (días)
  ┌─────────────────────────────────────────┐
  │ • Cada nodo legítimo genera nueva       │
  │   identidad Ed25519                     │
  │ • Nueva ceremonia de enrollment con     │
  │   la nueva clave raíz                   │
  │ • Nuevos handshakes entre todos los     │
  │   peers re-aprobados                    │
  │ • Pipeline restaurado con nueva cadena  │
  │   de confianza completa                 │
  └────────────────┬────────────────────────┘
                   │
                   ▼
  6. POST-MORTEM (semanas)
  ┌─────────────────────────────────────────┐
  │ • Análisis forense del compromiso       │
  │ • Identificar cómo se robó la clave     │
  │ • Implementar medidas correctivas       │
  │ • Considerar escalar el nivel de        │
  │   protección (ej: USB → HSM)            │
  │ • Documentar y comunicar según          │
  │   requerimientos legales/regulatorios   │
  └─────────────────────────────────────────┘
```

### 9.4 Impacto en el pipeline durante la respuesta

| Fase | Estado del pipeline | Latencia de detección | Impacto en seguridad |
|---|---|---|---|
| Contención | Sesiones existentes operan normalmente. No se aceptan nuevos nodos. | Inmediato | Pipeline protegido contra nodos falsos nuevos |
| Revocación | Nodos con certificados revocados desconectados. Pipeline puede perder nodos si el atacante comprometió nodos existentes. | Horas | Nodos falsos eliminados |
| Re-enrollment | Pipeline en mantenimiento parcial o total durante la ceremonia. | Horas-días | Cadena de confianza restaurada |
| Post-mortem | Pipeline operativo con nueva clave raíz | — | Seguridad restaurada completamente |

---

## 10. Estrategias de Backup por Nivel

### 10.1 Principio fundamental de backups criptográficos

**La clave y la passphrase NUNCA se almacenan juntas.** Esto aplica a todos los niveles. Un backup de una clave cifrada sin la passphrase es, a efectos prácticos, un bloque de bytes inútil. Argon2id con parámetros adecuados hace computacionalmente inviable el ataque por fuerza bruta incluso con hardware dedicado.

### 10.2 Backup por nivel

**Nivel 1 — USB cifrado:**

| Aspecto | Estrategia |
|---|---|
| **Copias del USB** | Mínimo 2 copias en ubicaciones físicas diferentes (ej: oficina + caja de seguridad bancaria). |
| **Cifrado de cada copia** | Cada copia puede usar la misma passphrase o passphrases diferentes (mayor seguridad). |
| **Passphrase** | Memorizada por el administrador. Opcionalmente, escrita en papel en sobre sellado en caja de seguridad bancaria (separada del USB). |
| **Verificación periódica** | Cada 6 meses: verificar que el USB es legible y la passphrase funciona. |
| **Cloud backup** | Aceptable: el fichero cifrado con Argon2id puede almacenarse en cloud. Sin la passphrase, es irrompible. La passphrase NUNCA toca el cloud. |

**Nivel 2 — Shamir:**

| Aspecto | Estrategia |
|---|---|
| **Fragmentos de repuesto** | Generar N > K fragmentos. Los fragmentos extra son backups inherentes (ej: 5 fragmentos para quorum de 3 → tolerancia de 2 pérdidas). |
| **Almacenamiento de fragmentos** | Cada custodio almacena su fragmento cifrado en su propio medio (USB personal, caja fuerte, papel impreso con QR). |
| **Regeneración de fragmentos** | Si un custodio pierde su fragmento, se puede regenerar un nuevo set de fragmentos (proactive resharing) con K custodios restantes sin cambiar la clave raíz. |
| **Redundancia geográfica** | Fragmentos distribuidos en diferentes ubicaciones geográficas. |
| **Verificación periódica** | Cada 6 meses: verificar que al menos K custodios pueden proporcionar su fragmento. |

**Nivel 3 — HSM:**

| Aspecto | Estrategia |
|---|---|
| **HSM de backup** | Segundo HSM sincronizado. La clave se exporta en formato wrapped (cifrada con clave del HSM receptor) y se importa al segundo HSM. En ningún momento la clave existe en claro fuera de un HSM. |
| **Wrapped key backup** | Fichero de backup wrapped generado por el HSM. Solo puede ser importado en un HSM del mismo modelo/fabricante (o compatible). Almacenable en cloud — inútil sin un HSM para descifrarlo. |
| **PIN de backup** | El PIN del HSM se almacena en sobre sellado en caja de seguridad, separado del HSM. |
| **Disaster recovery** | HSM de backup en ubicación geográfica diferente. Hot standby o cold standby según criticidad. |

**Nivel 4 — HSM + Shamir:**

| Aspecto | Estrategia |
|---|---|
| **Todo lo del nivel 3** | Más... |
| **Tokens de custodio de backup** | Cada custodio tiene un token de backup almacenado en ubicación diferente a su token primario. |
| **Ceremonia de backup anual** | Ceremonia documentada donde se verifica que todos los componentes de backup funcionan y K custodios pueden activar el HSM de backup. |

### 10.3 Matriz de escenarios de pérdida y recuperación

| Escenario | Nivel 1 (USB) | Nivel 2 (Shamir) | Nivel 3 (HSM) | Nivel 4 (HSM+Shamir) |
|---|---|---|---|---|
| Pérdida de 1 medio | Usar copia de backup | Fragmento extra cubre | Usar HSM de backup | HSM backup + tokens backup |
| Olvido de passphrase/PIN | Sobre sellado en caja fuerte | K custodios pueden reshare sin el afectado | PIN en sobre sellado | PIN en sobre sellado + quorum restante |
| Destrucción física (incendio) | Copia en otra ubicación | Fragmentos geográficamente distribuidos | HSM de backup en otra ubicación | HSM backup + tokens distribuidos |
| Muerte/incapacidad del admin | Passphrase en sobre sellado accesible a sucesor designado | Quorum de K restantes suficiente | PIN en sobre sellado + procedimiento de sucesión | Quorum de K restantes + PIN en sobre sellado |
| Compromiso de 1 custodio | N/A (1 custodio) | Insuficiente sin K-1 más | PIN sin HSM físico es inútil | Token + PIN sin HSM + K-1 tokens más |

---

## 11. Rol Residual de etcd-server

### 11.1 Funciones que mantiene

| Función | Descripción | Riesgo si se compromete etcd |
|---|---|---|
| Discovery de endpoints | Publica `role → endpoint:port` para que los componentes se descubran. | Un atacante podría inyectar endpoints falsos, pero no podría completar el handshake criptográfico con los componentes legítimos (no tiene certificado aprobado por raíz). |
| Distribución de JSON de configuración | El RAG sube JSON de configuración que los componentes consumen. | Un atacante podría inyectar configuración maliciosa. Mitigación: firmar JSON con clave del RAG. |
| Directorio de certificados públicos (opcional) | Puede servir como mirror de los certificados públicos aprobados. | Sin impacto: las claves públicas son, por definición, públicas. No otorgan capacidad de suplantación. |

### 11.2 Funciones que pierde

- Generación y distribución de semillas ChaCha20 → **ELIMINADO**
- Generación y distribución de certificados HMAC → **ELIMINADO**
- Autoridad de confianza criptográfica → **ELIMINADO**
- Watcher de actualización automática de endpoints → **DESACTIVADO en producción** (solo modo pull manual o controlado)

### 11.3 Hardening de etcd-server residual

1. **Acceso solo lectura para componentes del pipeline:** Solo el RAG y el administrador pueden escribir.
2. **mTLS para la conexión a etcd:** Protege la integridad del canal de discovery (no los mensajes del pipeline).
3. **Firma de JSON de configuración:** Todo JSON subido por el RAG firmado con clave Ed25519 del RAG. Los componentes verifican antes de aplicar.
4. **Rate limiting y auditoría:** Limitar actualizaciones de endpoints por ventana temporal. Registrar todos los cambios en log inmutable.

---

## 12. Licenciamiento y Mensajería al Usuario

### 12.1 Validación de licencia enterprise

```
/opt/mldefender/enterprise/
├── libsecurebusnode.so          # Módulo compilado
├── LICENSE.key                   # Fichero de licencia firmado
├── provider.conf                 # Configuración del RootKeyProvider
├── peers/                        # Directorio de certificados públicos aprobados
│   ├── capture-node-01.cert
│   ├── detector-rf-01.cert
│   └── ...
├── identity/                     # Identidad local del componente
│   ├── private.key               # Clave privada Ed25519 (0600)
│   └── certificate.cert          # Certificado aprobado (con root_signature)
├── enrollment/                   # Solicitudes pendientes de enrollment
│   └── pending/
├── revocation/                   # Listas de revocación
│   └── crl.json
└── audit/                        # Log de auditoría
    └── enrollment.log            # Append-only log de ceremonias
```

La licencia se valida mediante firma digital verificada contra la clave pública de ML Defender embebida en el módulo. Campos: `organization`, `valid_from`, `valid_until`, `max_nodes`, `features_enabled`, `provider_levels_enabled`. Sin conexión a servidor de licencias externo (air-gapped compatible).

### 12.2 Mensajes al usuario según estado

**Estado: Módulo enterprise activo con enrollment completo**
```
[INFO] ML Defender Enterprise Security Module v1.x active
[INFO] SecureBusNode E2E encryption enabled — Ed25519 + X25519 + ChaCha20-Poly1305
[INFO] RootKeyProvider: HSM_DEVICE (YubiHSM 2, serial XXXX)
[INFO] Identity: detector-rf-01 (certificate valid until 2026-05-19)
[INFO] Root trust anchor: verified (fingerprint: a3b7...c9f2)
[INFO] Peers discovered: 4/4 — All handshakes complete (all root-approved)
[INFO] Pipeline security: PRODUCTION GRADE
```

**Estado: Módulo activo en modo TOFU (sin enrollment completo)**
```
[WARN] ML Defender Enterprise Security Module v1.x active — TOFU MODE
[WARN] SecureBusNode E2E encryption enabled but enrollment ceremony NOT completed
[WARN] Trust model: Trust-On-First-Use (vulnerable to MITM on first contact)
[WARN] Complete enrollment ceremony for production-grade trust anchoring
[WARN] Run: mldefender-enroll --role <role_id> --provider <usb|shamir|hsm>
[WARN] Pipeline security: ENHANCED (not production-grade without enrollment)
```

**Estado: Módulo enterprise no presente**
```
[WARN] ════════════════════════════════════════════════════════════════
[WARN] ║  SecureBusNode enterprise module NOT DETECTED                ║
[WARN] ║                                                              ║
[WARN] ║  Running in DEMO MODE with centralized etcd encryption.      ║
[WARN] ║  This configuration is NOT certified for production use.     ║
[WARN] ║                                                              ║
[WARN] ║  Options:                                                    ║
[WARN] ║  • Acquire an enterprise license for production-grade        ║
[WARN] ║    E2E encryption: https://mldefender.io/enterprise          ║
[WARN] ║  • Continue in demo mode for evaluation and development      ║
[WARN] ║    purposes only.                                            ║
[WARN] ║                                                              ║
[WARN] ║  The current etcd-based encryption is suitable for           ║
[WARN] ║  demonstrating pipeline concepts but does not provide        ║
[WARN] ║  the security guarantees required for production             ║
[WARN] ║  deployment protecting real infrastructure.                  ║
[WARN] ════════════════════════════════════════════════════════════════
```

**Estado: Módulo presente pero licencia inválida/expirada**
```
[ERROR] SecureBusNode enterprise module detected but LICENSE INVALID
[ERROR] Reason: License expired on 2026-01-15
[ERROR] Falling back to DEMO MODE — NOT certified for production use
[ERROR] Contact: enterprise@mldefender.io to renew
```

---

## 13. Plan de Migración Paso a Paso

### Fase 0: Preparación (pre-implementación)

1. Inventariar todos los componentes que reciben semilla/HMAC de etcd-server.
2. Documentar los flujos de datos inter-componente y sus requisitos de latencia.
3. Definir el mapa de roles y sus relaciones de comunicación.
4. Seleccionar librería criptográfica: **libsodium** (recomendada) o **BoringSSL**.
5. Definir la estructura de directorios del módulo enterprise.
6. Diseñar e implementar la interfaz `IRootKeyProvider` con los cuatro niveles.

### Fase 1: Implementación del módulo SecureBusNode

1. Implementar `IRootKeyProvider` y los cuatro proveedores concretos.
2. Implementar la herramienta CLI `mldefender-enroll`.
3. Implementar generación de identidad Ed25519 auto-provisionada.
4. Implementar formato de certificado ligero de rol con `root_signature`.
5. Implementar handshake Noise Protocol Framework (patrón IK o XX) con verificación de raíz.
6. Implementar cifrado/descifrado AEAD ChaCha20-Poly1305 por mensaje.
7. Implementar firma y verificación de headers Ed25519.
8. Implementar protección anti-replay (contador + timestamp + bitmap de nonces).
9. Implementar peer discovery local (directorio de ficheros + opcional multicast).
10. Implementar rotación automática de claves de sesión.
11. Implementar expiración, renovación de certificados y CRL.
12. Implementar enrollment log append-only.
13. Empaquetar como `libsecurebusnode.so` con interfaz `ISecureBusNode`.

### Fase 2: Integración con componentes del pipeline

1. Implementar patrón de carga dinámica (`dlopen`/`dlsym`) del módulo.
2. Implementar detección automática de `RootKeyProvider` disponible.
3. Implementar fallback a modo DEMO con mensajes de advertencia.
4. Implementar fallback a modo TOFU si enrollment no completado.
5. Modificar cada componente para usar `ISecureBusNode` si está disponible.
6. Asegurar que el pipeline funciona en los tres modos (demo, TOFU, enrollment completo).

### Fase 3: Actualización de etcd-server y etcd-client

1. Eliminar generación de semillas ChaCha20 de etcd-server (modo enterprise).
2. Eliminar distribución de certificados HMAC de etcd-server (modo enterprise).
3. Mantener discovery de endpoints y distribución de JSON.
4. Actualizar etcd-client: mantener compresión LZ4, quitar cripto legacy en modo enterprise.
5. Implementar firma de JSON de configuración por parte del RAG.
6. Desactivar watcher automático en producción.

### Fase 4: Validación y testing

1. **Tests unitarios:** Cada `RootKeyProvider`, generación de identidad, enrollment, cifrado/descifrado, anti-replay, CRL.
2. **Tests de integración:** Pipeline completo con cada nivel de proveedor.
3. **Test de degradación:** Desinstalar módulo → fallback limpio a demo. Quitar enrollment → fallback a TOFU.
4. **Test de resiliencia:** Apagar etcd → pipeline E2E continúa.
5. **Test de rendimiento:** Comparativa demo vs TOFU vs enrollment completo. Objetivo: overhead < 300μs.
6. **Test de seguridad:**
    - Enrollment con clave raíz incorrecta → debe ser rechazado.
    - Nonce repetido, timestamp fuera de ventana, certificado falso → rechazados.
    - Compromiso de etcd → no permite descifrar mensajes.
    - Replay → rechazado por tres mecanismos.
    - Nodo con certificado revocado (CRL) → rechazado.
    - Nodo con certificado expirado → rechazado.
7. **Test de compromiso simulado:** Rotación completa de clave raíz y re-enrollment.
8. **Stress testing:** Datasets CTU-13 y MAWI con módulo enterprise activo.

### Fase 5: Despliegue gradual

1. Activar módulo en staging con tráfico sintético.
2. Monitorear métricas: latencia p50/p95/p99, throughput, CPU, memoria.
3. Pre-producción con tráfico real duplicado (mirror).
4. Producción con rollback preparado.
5. Documentar diferencias de rendimiento.

---

## 14. Consideraciones de Seguridad Avanzadas

### 14.1 Protección de la clave privada del nodo en reposo

Esta es la clave Ed25519 de cada componente individual (NO la clave raíz, que se trata en la sección 8).

- **Mínimo:** Permisos `0600`, propiedad del usuario del servicio.
- **Recomendado:** Cifrado con AEAD usando clave derivada por Argon2id de passphrase o TPM.
- **Ideal:** Almacenamiento en TPM local. La clave nunca toca el sistema de ficheros.

### 14.2 Revocación de peers comprometidos

- Eliminación del certificado del peer del directorio `peers/`.
- Distribución de CRL firmada por la clave raíz a todos los nodos.
- Los nodos verifican la CRL antes de aceptar handshakes.
- Timeout configurable: sesión expira si peer no visto en N horas.

### 14.3 Protección contra downgrade

- Si el módulo enterprise ha estado activo previamente y desaparece:
    - Genera alerta CRITICAL.
    - Con `enforce_enterprise=true`, rehúsa arrancar en modo demo.
    - Previene que un atacante fuerce downgrade a cifrado centralizado.

### 14.4 Aislamiento de roles

- Cada par de componentes tiene clave de sesión propia.
- Componente solo descifra mensajes dirigidos a él.
- Compromiso de A no permite leer mensajes entre B y C.
- Claves efímeras aseguran PFS.

### 14.5 Protección contra manipulación del módulo

- El módulo `libsecurebusnode.so` se distribuye con hash SHA-256 firmado.
- Antes de cargarlo con `dlopen`, el componente verifica el hash contra la firma del distribuidor.
- Un módulo manipulado (ej: con clave pública raíz falsa embebida) es rechazado.

---

## 15. Relación con la Publicación Académica

### 15.1 Encuadre en los papers

> "The current implementation employs centralized key distribution via etcd-server as a proof-of-concept mechanism that validates the detection pipeline architecture and demonstrates the viability of the integrated security approach. This design prioritizes clarity of the core detection concepts — eBPF/XDP packet capture, embedded RandomForest classification, and multi-stage alert correlation — over production-grade cryptographic isolation.
>
> For real-world deployment protecting critical infrastructure, the architecture requires migration to per-role authenticated encryption with ephemeral key exchange (Ed25519 + X25519 + AEAD ChaCha20-Poly1305), eliminating the centralized trust dependency and providing Perfect Forward Secrecy. This enhancement, planned for the enterprise release, operates as a plug-and-play module that transparently upgrades the security posture of the pipeline without modifying the detection logic. The trust anchor is established through an offline deployment ceremony with extensible key custody levels ranging from encrypted USB storage to hardware security modules with multi-party quorum.
>
> The centralized etcd-based encryption described in this paper is suitable for demonstration, evaluation, and academic reproducibility. Production deployments should utilize the enterprise security module to meet the threat model requirements of real infrastructure protection."

### 15.2 Lo que se publica vs lo que se reserva

| Aspecto | Paper (público) | Enterprise (reservado) |
|---|---|---|
| Arquitectura del pipeline de detección | ✅ Completo | — |
| Modelo eBPF/XDP + RandomForest | ✅ Completo | — |
| Cifrado centralizado vía etcd | ✅ Descrito como PoC | — |
| Vulnerabilidades del cifrado centralizado | ✅ Documentadas como limitaciones | — |
| Arquitectura SecureBusNode | ✅ Alto nivel como trabajo futuro | Implementación completa |
| Protocolo de handshake y formato de mensaje | ❌ No publicado | ✅ Completo |
| Problema del bootstrap y solución del ancla | ✅ Mencionado conceptualmente | ✅ Implementación completa |
| Auto-provisión + enrollment | ❌ No publicado | ✅ Completo |
| Arquitectura IRootKeyProvider extensible | ❌ No publicado | ✅ Completo |
| Niveles de custodia de clave raíz | ❌ No publicado | ✅ Completo |
| Procedimiento de respuesta a compromiso | ❌ No publicado | ✅ Completo |
| Mecanismo de licenciamiento | ❌ No publicado | ✅ Completo |
| Benchmarks de rendimiento E2E | ✅ Estimaciones teóricas | ✅ Mediciones reales |

---

## 16. Dependencias Técnicas

| Dependencia | Propósito | Licencia | Notas |
|---|---|---|---|
| **libsodium** (recomendada) | Ed25519, X25519, ChaCha20-Poly1305, Argon2id, Shamir | ISC | Auditada, incluye `crypto_secretsharing` para Shamir |
| **BoringSSL** (alternativa) | Mismo stack + TLS para mTLS con etcd | OpenSSL compatible | Mantenida por Google |
| **Noise Protocol Framework** | Patrón de handshake IK/XX | Dominio público | Implementación propia basada en especificación |
| **PKCS#11** (para nivel 3/4) | Interfaz estándar con HSMs | Estándar OASIS | Para YubiHSM, Nitrokey, Thales Luna |
| **LZ4** (existente) | Compresión de JSON en etcd-client | BSD | Ya integrado |
| **C++20** | Lenguaje de implementación | — | Consistente con el pipeline |
| **dlopen/dlsym** | Carga dinámica del módulo .so | POSIX | Estándar en Linux |

---

## 17. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| Overhead criptográfico excede 1ms | Baja | Alto | Batch signatures, verificación asíncrona, SIMD. Benchmarking exhaustivo. |
| Error en implementación criptográfica | Media | Crítico | Usar libsodium para todas las primitivas. Auditoría externa pre-release. |
| Desincronización de relojes | Media | Medio | Ventana temporal configurable. NTP obligatorio. Fallback a solo contador. |
| Pérdida de clave privada de nodo | Baja | Medio | Auto-regeneración + nueva ceremonia de enrollment. |
| Compromiso de clave raíz | Baja | Alto | Procedimiento de respuesta documentado (sección 9). Rotación + re-enrollment. Niveles de protección escalables. |
| Pérdida de acceso a clave raíz | Baja | Alto | Estrategias de backup por nivel (sección 10). Múltiples copias/fragmentos/HSMs. |
| Ataque durante ventana TOFU | Baja | Alto | TOFU solo como modo degradado. Enrollment completo para producción. |
| Complejidad de ceremonia Shamir/HSM | Media | Medio | CLI guiada paso a paso. Documentación detallada. Modo simulación para práctica. |
| Fallo del HSM | Baja | Alto | HSM de backup. Wrapped key export. Procedimiento de disaster recovery. |
| Manipulación del módulo .so | Baja | Crítico | Verificación de hash/firma antes de dlopen. Distribución por canal seguro. |

---

## 18. Métricas de Éxito

| Métrica | Objetivo | Método de medición |
|---|---|---|
| Overhead de latencia por mensaje | < 300 μs p99 | Instrumentación de timestamps por etapa criptográfica |
| Tiempo de auto-provisión | < 5 segundos (primer arranque, excluyendo ceremonia) | Medición desde detección del módulo hasta enrollment request generado |
| Tiempo de enrollment por nodo | < 30 segundos (interacción del admin) | Medición de la ceremonia completa |
| Tiempo de handshake por peer | < 1 ms | Latencia del intercambio X25519 completo |
| Throughput del pipeline | ≥ 95% del throughput sin módulo enterprise | Comparativa con datasets CTU-13 y MAWI |
| Resiliencia ante caída de etcd | 100% continuidad E2E | Test: apagar etcd → verificar flujo entre peers establecidos |
| Protección anti-replay | 100% rechazo | Test: replay de N mensajes → todos rechazados |
| Protección anti-suplantación | 100% rechazo | Test: handshake con certificado no aprobado → fallo |
| Detección de nodo no autorizado | 100% alerta | Test: nodo con certificado raíz robada → alerta WARNING |
| Tiempo de rotación de clave raíz | < 4 horas para 10 nodos | Medición de ceremonia completa de re-enrollment |

---

## 19. Timeline Estimado

| Fase | Duración estimada | Dependencia |
|---|---|---|
| Fase 0: Preparación + diseño IRootKeyProvider | 2-3 semanas | Post-publicación de papers |
| Fase 1: Implementación SecureBusNode + 4 proveedores + CLI enrollment | 6-8 semanas | Fase 0 completa |
| Fase 2: Integración con pipeline + 3 modos (demo/TOFU/enrollment) | 2-3 semanas | Fase 1 completa |
| Fase 3: Actualización etcd | 1-2 semanas | Fase 2 completa |
| Fase 4: Validación, testing y test de compromiso simulado | 4-5 semanas | Fase 3 completa |
| Fase 5: Despliegue gradual | 2-3 semanas | Fase 4 completa |
| **Total estimado** | **17-24 semanas** | Post-publicación |

---

## 20. Compromiso Open Source y Modelo Open-Core

### 20.1 Principio fundamental

Toda la tecnología criptográfica utilizada en el módulo SecureBusNode enterprise es 100% open source. No existe ningún componente propietario ni de código cerrado en la cadena de seguridad. Esto no es una concesión — es una decisión de diseño deliberada y alineada con la filosofía fundacional de ML Defender: democratizar la seguridad de grado enterprise para organizaciones que no pueden permitirse soluciones comerciales cerradas.

La seguridad por oscuridad no es seguridad. Un sistema criptográfico cuya fortaleza dependa de que el código sea secreto es un sistema fundamentalmente débil. La fortaleza de SecureBusNode reside en las propiedades matemáticas de los algoritmos y en la corrección de la implementación, ambas verificables públicamente.

### 20.2 Auditoría completa de dependencias

| Dependencia | Función en SecureBusNode | Licencia | Tipo | Verificable |
|---|---|---|---|---|
| **libsodium** | Ed25519, X25519, ChaCha20-Poly1305, Argon2id, Shamir Secret Sharing | ISC (permisiva) | Código abierto, auditada públicamente | [github.com/jedisct1/libsodium](https://github.com/jedisct1/libsodium) |
| **Noise Protocol Framework** | Especificación del handshake IK/XX | Dominio público | Especificación abierta | [noiseprotocol.org](https://noiseprotocol.org) |
| **PKCS#11 (estándar)** | Interfaz con HSMs | Estándar abierto OASIS | Especificación pública | [docs.oasis-open.org](https://docs.oasis-open.org/pkcs11/) |
| **YubiHSM 2 SDK** | Integración con YubiHSM | BSD | Código abierto | [github.com/Yubico/yubihsm-shell](https://github.com/Yubico/yubihsm-shell) |
| **Nitrokey HSM** | Alternativa HSM (hardware + software) | CERN OHL (hw) + GPL (sw) | Open source completo incluyendo hardware | [github.com/Nitrokey](https://github.com/Nitrokey) |
| **OpenSC** | Capa PKCS#11 genérica | LGPL 2.1 | Código abierto | [github.com/OpenSC/OpenSC](https://github.com/OpenSC/OpenSC) |
| **LZ4** | Compresión de JSON en etcd-client | BSD | Código abierto | [github.com/lz4/lz4](https://github.com/lz4/lz4) |
| **GCC / Clang** | Compilador C++20 | GPL / Apache 2.0 | Código abierto | — |
| **Linux (kernel)** | Sistema operativo base | GPL v2 | Código abierto | — |
| **dlopen/dlsym** | Carga dinámica de módulos | POSIX (glibc: LGPL) | Estándar abierto | — |

**Algoritmos criptográficos subyacentes:** Ed25519, X25519, ChaCha20, Poly1305 y Curve25519 fueron diseñados por Daniel J. Bernstein y colaboradores, y están explícitamente liberados al dominio público. Son matemáticas públicas, no propiedad intelectual de nadie.

### 20.3 Modelo de licencia: Open-Core

ML Defender sigue el modelo **open-core**, el mismo modelo que emplean con éxito proyectos como Red Hat Enterprise Linux, Canonical Ubuntu Pro, Elastic (Elasticsearch), GitLab, y Grafana:

```
┌─────────────────────────────────────────────────────────┐
│                    ML Defender                            │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │            Núcleo Open Source (GPLv3)                │ │
│  │                                                     │ │
│  │  • Pipeline de detección eBPF/XDP + RandomForest    │ │
│  │  • Sistema de captura y clasificación               │ │
│  │  • etcd-server/client con cifrado demo              │ │
│  │  • RAG con TinyLlama                                │ │
│  │  • Logger CSV con HMAC                              │ │
│  │  • Toda la lógica de detección                      │ │
│  │  • Documentación y papers académicos                │ │
│  │                                                     │ │
│  │  100% funcional · 100% auditable · 100% libre       │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │          Módulo Enterprise (licencia comercial)      │ │
│  │                                                     │ │
│  │  • SecureBusNode (cifrado E2E por rol)              │ │
│  │  • IRootKeyProvider (4 niveles de custodia)         │ │
│  │  • Herramienta de enrollment (mldefender-enroll)    │ │
│  │  • Procedimientos de respuesta a incidentes         │ │
│  │  • Tests de seguridad validados y certificados      │ │
│  │  • Soporte profesional y SLA                        │ │
│  │                                                     │ │
│  │  Construido 100% con tecnología open source         │ │
│  │  El valor es: integración + validación + soporte    │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 20.4 Qué paga el cliente enterprise

El cliente no paga por tecnología cerrada. Paga por:

1. **Integración validada:** Las piezas open source, ensambladas, testeadas y verificadas como sistema cohesivo.
2. **Auditoría de seguridad:** El módulo ha sido auditado por expertos antes del release. El cliente recibe garantía de que la implementación es correcta.
3. **Certificación:** El módulo ha pasado los tests de seguridad documentados en la Fase 4. El cliente recibe evidencia de validación.
4. **Procedimientos operativos:** Ceremonias de enrollment, respuesta a compromiso, rotación de claves — documentados, probados y con soporte.
5. **Soporte profesional:** Asistencia en despliegue, respuesta a incidentes, y actualizaciones de seguridad con SLA definido.
6. **Tranquilidad:** La diferencia entre "podrías ensamblarlo tú" y "alguien competente lo ha validado y responde si algo falla".

### 20.5 Derecho a inspección

Cualquier cliente enterprise tiene derecho a inspeccionar el código fuente completo del módulo SecureBusNode. La licencia comercial protege la redistribución, no la transparencia. Un cliente que quiera auditar independientemente el módulo que está desplegando en su infraestructura crítica tiene todo el derecho a hacerlo, y será facilitado.

Esto es consistente con el principio de Kerckhoffs: la seguridad de un sistema criptográfico debe residir en la clave, no en el secreto del algoritmo ni de la implementación.

### 20.6 Compromiso con la comunidad

Las mejoras y correcciones de seguridad descubiertas durante el desarrollo del módulo enterprise que afecten a las librerías open source subyacentes (libsodium, OpenSC, etc.) serán contribuidas de vuelta a sus proyectos upstream. ML Defender se beneficia del ecosistema open source y contribuye a él.

---

## Apéndice A: Glosario

| Término | Definición |
|---|---|
| **AEAD** | Authenticated Encryption with Associated Data. Cifrado que garantiza confidencialidad e integridad simultáneamente. |
| **Argon2id** | Función de derivación de claves resistente a ataques por GPU y ASIC. Ganadora de la Password Hashing Competition (2015). |
| **ChaCha20-Poly1305** | Algoritmo AEAD: ChaCha20 para confidencialidad, Poly1305 para autenticación. Diseñado por Daniel J. Bernstein. |
| **CRL** | Certificate Revocation List. Lista de certificados revocados que los nodos consultan antes de aceptar peers. |
| **CSPRNG** | Cryptographically Secure Pseudo-Random Number Generator. |
| **Ed25519** | Esquema de firma digital sobre curva de Edwards. Claves de 32 bytes, firmas de 64 bytes. |
| **Enrollment** | Ceremonia de aprobación de la identidad de un nodo mediante firma de la clave raíz. |
| **FHE** | Fully Homomorphic Encryption. Cifrado que permite computación sobre datos cifrados. |
| **HSM** | Hardware Security Module. Dispositivo físico dedicado al almacenamiento y operación de claves criptográficas. |
| **IRootKeyProvider** | Interfaz abstracta del módulo enterprise que abstrae la custodia de la clave raíz. |
| **MITM** | Man-In-The-Middle. Ataque donde un adversario intercepta y modifica comunicaciones. |
| **Noise Protocol Framework** | Framework para protocolos criptográficos basados en Diffie-Hellman. Base de WireGuard. |
| **PFS** | Perfect Forward Secrecy. El compromiso de claves a largo plazo no expone sesiones pasadas. |
| **PHE** | Partially Homomorphic Encryption. Operaciones limitadas sobre datos cifrados. |
| **PKCS#11** | Estándar de interfaz para dispositivos criptográficos (HSMs, smart cards). |
| **RootKeyProvider** | Implementación concreta de IRootKeyProvider para un nivel de custodia específico. |
| **Shamir's Secret Sharing** | Esquema matemático que divide un secreto en N partes, reconstruible con K de ellas. Seguridad de teoría de información. |
| **TOFU** | Trust-On-First-Use. Modelo de confianza donde la primera conexión establece la identidad. Usado por SSH. |
| **X25519** | Función de intercambio de claves Diffie-Hellman sobre Curve25519. |

---

## Apéndice B: Configuración de ejemplo del proveedor

```ini
# /opt/mldefender/enterprise/provider.conf

# Nivel de proveedor: usb | shamir | hsm | hsm_shamir
provider_type = usb

# === Configuración USB (Nivel 1) ===
[usb]
key_path = /media/admin/mldefender-root/root.key.enc
argon2_m_cost = 262144    # 256 MB
argon2_t_cost = 4
argon2_parallelism = 2

# === Configuración Shamir (Nivel 2) ===
[shamir]
total_shares = 5
threshold = 3
share_paths = /media/custodian{N}/share.enc
argon2_m_cost = 262144
argon2_t_cost = 4

# === Configuración HSM (Nivel 3) ===
[hsm]
pkcs11_module = /usr/lib/pkcs11/yubihsm_pkcs11.so
slot_id = 0
key_label = mldefender-root

# === Configuración HSM + Shamir (Nivel 4) ===
[hsm_shamir]
pkcs11_module = /usr/lib/pkcs11/yubihsm_pkcs11.so
slot_id = 0
key_label = mldefender-root
activation_threshold = 3
activation_total = 5
```

---

*Documento vivo — se actualizará conforme avance la implementación post-publicación académica.*  
*v2.0 — Añadida arquitectura extensible de custodia de clave raíz (IRootKeyProvider), solución al problema del bootstrap, procedimiento de respuesta a compromiso, y estrategias de backup por nivel.*  
*v2.1 — Añadida sección de compromiso open source, modelo open-core, auditoría completa de dependencias y derecho a inspección del cliente enterprise.*
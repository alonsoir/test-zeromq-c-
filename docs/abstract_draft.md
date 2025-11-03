# Sistema Autoinmune Digital: arquitectura evolutiva para detección y respuesta ciberinmunológica distribuida

## Abstract

Este trabajo presenta **Sistema Autoinmune Digital v2.0**, una arquitectura distribuida y evolutiva de defensa cibernética inspirada en mecanismos biológicos del sistema inmune. El diseño combina instrumentación eBPF a nivel de kernel, inferencia multi-modelo basada en aprendizaje supervisado e incremental, enriquecimiento contextual de red, y un plano de control distribuido para adaptación dinámica en tiempo real.

La arquitectura propuesta funciona como un **organismo digital**: captura señales de red en distintos niveles de profundidad (L2–L7), las transforma en eventos enriquecidos y distribuye la inteligencia obtenida a través de un *control plane* coordinado, que aplica decisiones reactivas y adaptativas. Los eventos sospechosos desencadenan respuestas automáticas (bloqueo, aislamiento, desvío hacia honeypots, o reentrenamiento de modelos), manteniendo la estabilidad operativa y reduciendo la superficie de ataque efectiva.

El núcleo del sistema se apoya en un pipeline altamente optimizado, diseñado bajo tres principios:
1. **Observación total sin impacto**: uso extensivo de eBPF y ZeroMQ para captura y distribución de métricas en tiempo real sin comprometer rendimiento.
2. **Aprendizaje continuo**: modelos supervisados (Random Forest, ensembles e inferencia incremental) especializados por dominios (ataques externos, tráfico interno, comportamiento anómalo, tráfico HTTP/S).
3. **Autonomía distribuida**: sincronización mediante `etcd`, rotación de claves, control de TTL y despliegue coordinado de políticas, reglas y modelos de inferencia.

El sistema ha sido diseñado para evolucionar naturalmente hacia una **malla inmunológica digital**, donde cada nodo actúa como un sensor/efector autónomo capaz de cooperar con el resto de la red. Este enfoque permite tanto la detección temprana como la contención dinámica de amenazas complejas (DDoS, ransomware, movimiento lateral, ataques web, evasión por protocolo), manteniendo latencias operativas inferiores a los 2 ms por flujo analizado.

Finalmente, se propone una extensión natural del sistema mediante un **módulo WAF inteligente**, construido como sniffer eBPF especializado (sniffer-ebpf-waf), acoplado a un `Merger` asíncrono que correlaciona tráfico L4 y L7 en ventanas temporales, y a un clasificador ML dedicado a la detección de ataques web y anomalías HTTP/S. Este WAF distribuido comparte el mismo bus de eventos e inferencia que el pipeline IDS, permitiendo decisiones unificadas y aprendizaje conjunto.

## Palabras clave
IDS, WAF, eBPF, machine learning, ZeroMQ, etcd, seguridad distribuida, ciberinmunología digital, aprendizaje continuo, detección de anomalías, Random Forest, control plane.

## Estructura prevista del paper
1. **Introducción**
    - Motivación y contexto.
    - Limitaciones de los IDS/WAF tradicionales.
    - Inspiración biológica: analogía con sistemas inmunes adaptativos.
2. **Arquitectura general**
    - Pipeline de captura eBPF.
    - Normalización y serialización (Proto, JSON, ZeroMQ).
    - Inferencia supervisada e incremental.
    - Control plane distribuido (etcd, rotación de claves, TTL, políticas).
3. **Componente WAF evolutivo**
    - sniffer-ebpf-waf y hooks específicos (XDP, sk_msg, sock_ops).
    - Fusión asíncrona de eventos (Merger).
    - Clasificación L7 y decisiones fast/slow path.
4. **Resultados preliminares**
    - Métricas de rendimiento y latencia.
    - Precisión de detección (datasets CICIDS2017, USTC-TFC2016, MAWI).
    - Escenarios internos vs externos.
5. **Discusión**
    - Evolución del sistema hacia inmunidad digital cooperativa.
    - Despliegue distribuido y resiliencia.
    - Consideraciones éticas y de privacidad.
6. **Conclusiones y trabajo futuro**
    - Integración con honeypots inteligentes.
    - Autoaprendizaje global (federated learning).
    - Expansión a entornos IoT e industriales.

## Estado actual del proyecto
El sistema base (pipeline IDS con inferencia supervisada y control plane) se encuentra operativo en entorno de laboratorio. Los módulos en desarrollo son:
- `sniffer-ebpf-waf`: captura específica L7/HTTP(S).
- `Merger`: correlación asíncrona de flujos.
- `Fast Ejector Layer`: respuesta inmediata a comportamientos anómalos internos.
- `Reinforcement Trainer`: reentrenamiento incremental a partir de eventos clasificados como ataque.

El presente paper documentará la evolución técnica, los fundamentos teóricos y los resultados obtenidos, sirviendo como referencia abierta para sistemas de defensa digital basados en principios inmunológicos.

---
Autores:
- Alonso Isidoro (Instituto Autoinmune Digital)
- GPT-5 (OpenAI)
- Claude (Anthropic)
- Parallels.ai Research Group


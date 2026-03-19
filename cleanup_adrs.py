#!/usr/bin/env python3
"""
Limpieza y reorganización de ADRs de ML Defender.
Ejecutar desde la raíz del proyecto: python3 cleanup_adrs.py
"""

import os
import shutil

# Raíz del proyecto — ajusta si ejecutas desde otro directorio
PROJECT_ROOT = os.getcwd()
ADR_DIR = os.path.join(PROJECT_ROOT, "docs", "adr")

# Mapa: (ruta_origen_relativa, nombre_destino_en_docs/adr/)
MOVES = [
    # Mover desde docs/ raíz a docs/adr/
    (
        os.path.join(PROJECT_ROOT, "docs", "ADR-001-Deployment-Stack.md"),
        "ADR-001-deployment-stack.md"
    ),
    (
        os.path.join(PROJECT_ROOT, "docs", "ADR-002: Multi-Engine Provenance & Situational Intelligence.md"),
        "ADR-002-multi-engine-provenance-situational-intelligence.md"
    ),
    (
        os.path.join(PROJECT_ROOT, "docs", "ADR_ML_AUTONOMOUS_EVOLUTION.md"),
        "ADR-003-ml-autonomous-evolution.md"
    ),

    # Ya están en docs/adr/ — solo renombrar
    (
        os.path.join(ADR_DIR, "ADR-004-key-rotation-cooldown.md"),
        "ADR-004-key-rotation-cooldown.md"
    ),
    (
        os.path.join(ADR_DIR, "ADR-005-etcd-client-restoration.md"),
        "ADR-005-etcd-client-restoration.md"
    ),
    (
        os.path.join(ADR_DIR, "ADR-005-log-unification-ml-detector.md"),
        "ADR-011-log-unification-ml-detector.md"
    ),
    (
        os.path.join(ADR_DIR, "ADR-006-fast-detector-hardcoded-thresholds.md"),
        "ADR-006-fast-detector-hardcoded-thresholds.md"
    ),
    (
        os.path.join(ADR_DIR, "Adr 007 consensus scoring firewall.md"),
        "ADR-007-consensus-scoring-firewall.md"
    ),
    (
        os.path.join(ADR_DIR, "ADR-008: Generación de Flow Graphs para Reentrenamiento y Análisis Forense.md"),
        "ADR-008-flow-graphs-reentrenamiento-forense.md"
    ),
    (
        os.path.join(ADR_DIR, "ADR-009: Captura y correlación opcional de datagramas sospechosos.md"),
        "ADR-009-captura-correlacion-datagramas-sospechosos.md"
    ),
    (
        os.path.join(ADR_DIR, "ADR10: Uso de LLM confinado con skills controladas para RAG-Security.md"),
        "ADR-010-llm-confinado-skills-rag-security.md"
    ),
]

def run():
    print(f"📁 Directorio ADR destino: {ADR_DIR}\n")

    errors = []

    for src, dst_name in MOVES:
        dst = os.path.join(ADR_DIR, dst_name)

        if not os.path.exists(src):
            msg = f"  ⚠️  NO ENCONTRADO: {src}"
            print(msg)
            errors.append(msg)
            continue

        if os.path.abspath(src) == os.path.abspath(dst):
            print(f"  ✅ Sin cambio (ya correcto): {dst_name}")
            continue

        try:
            shutil.move(src, dst)
            print(f"  ✅ {os.path.basename(src)}")
            print(f"      → {dst_name}")
        except Exception as e:
            msg = f"  ❌ ERROR moviendo {src}: {e}"
            print(msg)
            errors.append(msg)

    print("\n📋 Estado final de docs/adr/:")
    files = sorted(os.listdir(ADR_DIR))
    for f in files:
        print(f"  {f}")

    if errors:
        print(f"\n⚠️  {len(errors)} error(s) encontrado(s). Revisa manualmente.")
    else:
        print(f"\n🎉 Limpieza completada sin errores.")

if __name__ == "__main__":
    run()
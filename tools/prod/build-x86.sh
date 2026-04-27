#!/usr/bin/env bash
# tools/prod/build-x86.sh
# Compila todos los componentes del pipeline con flags de producción.
# Se ejecuta DENTRO de la dev VM via vagrant ssh -c.
# ADR-039 (BSR): este script NUNCA corre en la hardened VM.
#
# DAY 133 — aRGus NDR — ADR-030 Variant A
set -euo pipefail

DIST=/vagrant/dist/x86/bin
VAGRANT=/vagrant

CXX_FLAGS="-std=c++20 -Wall -Wextra -Wpedantic -O3 -march=native -DNDEBUG -flto -fno-omit-frame-pointer"
C_FLAGS="-std=c11 -O3 -march=native -DNDEBUG -flto -fno-omit-frame-pointer"
CMAKE_PROD="-DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_FLAGS=${CXX_FLAGS} \
            -DCMAKE_C_FLAGS=${C_FLAGS}"

mkdir -p "${DIST}"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  aRGus NDR — Production Build (x86-64)                   ║"
echo "║  Flags: -O3 -march=native -DNDEBUG -flto                 ║"
echo "╚════════════════════════════════════════════════════════════╝"

# ── Función genérica de build ─────────────────────────────────────────────────
build_component() {
    local name=$1
    local src_dir=$2
    local binary=$3

    echo ""
    echo "── Building ${name} ──"
    local build_dir="${src_dir}/build-prod"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"
    cmake ${CMAKE_PROD} .. 2>&1 | tail -5
    make -j"$(nproc)" "${binary}" 2>&1 | tail -3
    cp "${binary}" "${DIST}/${binary}"
    echo "  ✅ ${binary} → dist/x86/bin/${binary}"
    cd "${VAGRANT}"
}

# ── etcd-server ──────────────────────────────────────────────────────────────
build_component "etcd-server"       "${VAGRANT}/etcd-server"         "etcd-server"

# ── sniffer ──────────────────────────────────────────────────────────────────
build_component "sniffer"           "${VAGRANT}/sniffer"             "sniffer"

# ── ml-detector ──────────────────────────────────────────────────────────────
build_component "ml-detector"       "${VAGRANT}/ml-detector"         "ml-detector"

# ── firewall-acl-agent ───────────────────────────────────────────────────────
build_component "firewall-acl-agent" "${VAGRANT}/firewall-acl-agent" "firewall-acl-agent"

# ── rag-security ─────────────────────────────────────────────────────────────
build_component "rag-security"      "${VAGRANT}/rag"                 "rag-security"

# ── rag-ingester ─────────────────────────────────────────────────────────────
build_component "rag-ingester"      "${VAGRANT}/rag-ingester"        "rag-ingester"

# ── plugins ──────────────────────────────────────────────────────────────────
echo ""
echo "── Building plugins ──"
mkdir -p /vagrant/dist/x86/plugins

cd "${VAGRANT}/plugins/xgboost"
rm -rf build-prod && mkdir -p build-prod && cd build-prod
cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tail -3
make -j"$(nproc)" 2>&1 | tail -2
cp libplugin_xgboost.so /vagrant/dist/x86/plugins/
echo "  ✅ libplugin_xgboost.so → dist/x86/plugins/"

cd "${VAGRANT}"

echo ""
echo "── Build summary ──"
ls -lh "${DIST}/"
echo ""
echo "✅ prod-build-x86 completado"
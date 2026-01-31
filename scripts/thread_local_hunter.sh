#!/bin/bash
# thread_local_hunter.sh
# Detecta código obsoleto en tests

echo "🏛️ Iniciando Thread_local Hunter..."
echo ""

echo "--- Tests con referencias a thread_local o FlowManager manual ---"
grep -rn "thread_local\|FlowManager [a-zA-Z_]*;" /vagrant/sniffer/tests/ \
    --exclude="test_sharded_flow_*" \
    --exclude="test_ring_consumer_protobuf.cpp" \
    --color=always

echo ""
echo "--- Tests huérfanos (no en Makefile) ---"
cd /vagrant/sniffer
ls tests/test_*.cpp | xargs -n 1 basename | sed 's/\.cpp//' | sort > /tmp/all_tests.txt
grep -oE "test_[a-zA-Z0-9_]+" Makefile CMakeLists.txt 2>/dev/null | sort | uniq > /tmp/makefile_targets.txt
comm -23 /tmp/all_tests.txt /tmp/makefile_targets.txt

echo ""
echo "✅ Hunter completado. Revisa la salida arriba."
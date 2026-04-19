#!/bin/bash
# Check xgboost version — called from make check-system-deps
EXPECTED="3.2.0"
ACTUAL=$(python3 -c 'import xgboost; print(xgboost.__version__)' 2>/dev/null)
if [ "$ACTUAL" = "$EXPECTED" ]; then
    exit 0
else
    echo "❌ xgboost $EXPECTED missing (found: $ACTUAL)" >&2
    exit 1
fi

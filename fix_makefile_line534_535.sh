#!/bin/bash
# fix_makefile_line534_535.sh - Fix tab/space issue in Makefile
# The issue: Lines 534-535 have spaces instead of tabs
# Make REQUIRES tabs before recipe commands

set -e

MAKEFILE="/vagrant/Makefile"

echo "🔧 Fixing Makefile lines 534-535 (tabs issue)..."
echo ""

# Check if file exists
if [ ! -f "$MAKEFILE" ]; then
    echo "❌ Makefile not found: $MAKEFILE"
    exit 1
fi

# Show problematic lines BEFORE fix
echo "📋 BEFORE (lines 533-536):"
sed -n '533,536p' "$MAKEFILE" | cat -A
echo ""

# Backup original
BACKUP="${MAKEFILE}.backup_$(date +%Y%m%d_%H%M%S)"
cp "$MAKEFILE" "$BACKUP"
echo "💾 Backup saved: $BACKUP"
echo ""

# Fix the issue: Replace leading spaces with a TAB on lines 534-535
# The pattern matches 4 or more spaces at start of line
perl -i -pe 's/^    /\t/ if 534..535 and $. >= 534 and $. <= 535' "$MAKEFILE"

echo "✅ Fixed!"
echo ""

# Show lines AFTER fix
echo "📋 AFTER (lines 533-536):"
sed -n '533,536p' "$MAKEFILE" | cat -A
echo ""

# Verify the fix worked
if grep -q "^	@echo" <(sed -n '534p' "$MAKEFILE"); then
    echo "✅ Line 534: TAB detected (correct)"
else
    echo "⚠️  Line 534: Still has spaces (may need manual fix)"
fi

if grep -q "^	@vagrant" <(sed -n '535p' "$MAKEFILE"); then
    echo "✅ Line 535: TAB detected (correct)"
else
    echo "⚠️  Line 535: Still has spaces (may need manual fix)"
fi

echo ""
echo "🎯 Ready to compile!"
echo "   Now run: ./tsan_investigation.sh"
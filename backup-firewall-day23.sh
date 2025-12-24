#!/bin/bash
# Backup script for Day 23 firewall modifications
# Via Appia Quality - Always backup before changes

BACKUP_DIR="/vagrant/firewall-acl-agent/backups/day23-$(date +%Y%m%d-%H%M%S)"

echo "🔒 Creating backups in: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup CMakeLists.txt
echo "📦 Backing up CMakeLists.txt..."
cp /vagrant/firewall-acl-agent/CMakeLists.txt "$BACKUP_DIR/CMakeLists.txt.backup"

# Backup config
echo "📦 Backing up firewall.json..."
cp /vagrant/firewall-acl-agent/config/firewall.json "$BACKUP_DIR/firewall.json.backup"

# Backup all source files
echo "📦 Backing up source files..."
mkdir -p "$BACKUP_DIR/src"
cp -r /vagrant/firewall-acl-agent/src/* "$BACKUP_DIR/src/"

# Backup main.cpp specifically
if [ -f "/vagrant/firewall-acl-agent/src/main.cpp" ]; then
    cp /vagrant/firewall-acl-agent/src/main.cpp "$BACKUP_DIR/main.cpp.backup"
fi

# Create restoration script
cat > "$BACKUP_DIR/RESTORE.sh" << 'EOF'
#!/bin/bash
# Restoration script - use if things go wrong

echo "⚠️  RESTORING FROM BACKUP..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cp "$SCRIPT_DIR/CMakeLists.txt.backup" /vagrant/firewall-acl-agent/CMakeLists.txt
cp "$SCRIPT_DIR/firewall.json.backup" /vagrant/firewall-acl-agent/config/firewall.json
cp -r "$SCRIPT_DIR/src/"* /vagrant/firewall-acl-agent/src/

echo "✅ Restoration complete"
echo "💡 Run 'cd /vagrant/firewall-acl-agent/build && rm -rf * && cmake .. && make' to rebuild"
EOF

chmod +x "$BACKUP_DIR/RESTORE.sh"

echo ""
echo "✅ Backup complete!"
echo "📂 Location: $BACKUP_DIR"
echo "💡 To restore: $BACKUP_DIR/RESTORE.sh"
echo ""
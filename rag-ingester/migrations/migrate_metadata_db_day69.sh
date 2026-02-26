#!/usr/bin/env bash
# migrate_metadata_db_day69.sh
# Idempotent migration — safe to run multiple times
# Usage: ./migrate_metadata_db_day69.sh /vagrant/shared/indices/metadata.db

set -euo pipefail

DB="${1:-/vagrant/shared/indices/metadata.db}"

if [[ ! -f "$DB" ]]; then
    echo "ERROR: DB not found at $DB"
    exit 1
fi

echo "[Day69 Migration] Target: $DB"

add_column_if_missing() {
    local col="$1"
    local typedef="$2"
    local exists
    exists=$(sqlite3 "$DB" "SELECT COUNT(*) FROM pragma_table_info('events') WHERE name='$col';")
    if [[ "$exists" -eq 0 ]]; then
        sqlite3 "$DB" "ALTER TABLE events ADD COLUMN $col $typedef;"
        echo "  + Added column: $col $typedef"
    else
        echo "  ~ Skipped (exists): $col"
    fi
}

add_column_if_missing "trace_id"           "TEXT"
add_column_if_missing "source_ip"          "TEXT"
add_column_if_missing "dest_ip"            "TEXT"
add_column_if_missing "timestamp_ms"       "INTEGER"
add_column_if_missing "pb_artifact_path"   "TEXT"
add_column_if_missing "firewall_action"    "TEXT"
add_column_if_missing "firewall_timestamp" "INTEGER"
add_column_if_missing "firewall_score"     "REAL"

sqlite3 "$DB" "CREATE INDEX IF NOT EXISTS idx_trace_id  ON events(trace_id);"
sqlite3 "$DB" "CREATE INDEX IF NOT EXISTS idx_src_ip_ts ON events(source_ip, timestamp_ms);"

echo "[Day69 Migration] Done."
sqlite3 "$DB" "PRAGMA table_info(events);" | awk -F'|' '{printf "  col %s: %s\n", $2, $3}'
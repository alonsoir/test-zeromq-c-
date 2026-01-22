#include "metadata_db.hpp"
#include <iostream>
#include <cstring>

namespace rag {

MetadataDB::MetadataDB(const std::string& db_path)
    : db_(nullptr)
{
    int rc = sqlite3_open(db_path.c_str(), &db_);

    if (rc != SQLITE_OK) {
        std::string error = "Cannot open metadata database: ";
        error += db_path;
        error += " (";
        error += sqlite3_errmsg(db_);
        error += ")";

        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }

        throw std::runtime_error(error);
    }

    std::cout << "📊 MetadataDB opened: " << db_path << std::endl;

    // Create schema
    create_schema();

    // Enable WAL mode for better concurrent access
    exec_sql("PRAGMA journal_mode=WAL");

    // Enable foreign keys
    exec_sql("PRAGMA foreign_keys=ON");
}

MetadataDB::~MetadataDB() {
    if (db_) {
        // Checkpoint WAL before closing
        sqlite3_exec(db_, "PRAGMA wal_checkpoint(TRUNCATE)", nullptr, nullptr, nullptr);

        sqlite3_close(db_);
        db_ = nullptr;
    }
}

void MetadataDB::create_schema() {
    const char* schema = R"(
        CREATE TABLE IF NOT EXISTS events (
            faiss_idx INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            classification TEXT NOT NULL,
            discrepancy_score REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );

        CREATE INDEX IF NOT EXISTS idx_event_id
            ON events(event_id);

        CREATE INDEX IF NOT EXISTS idx_classification
            ON events(classification);

        CREATE INDEX IF NOT EXISTS idx_discrepancy
            ON events(discrepancy_score DESC);

        CREATE INDEX IF NOT EXISTS idx_timestamp
            ON events(timestamp DESC);
    )";

    exec_sql(schema);

    std::cout << "✅ Metadata schema ready" << std::endl;
}

void MetadataDB::insert_event(
    size_t faiss_idx,
    const std::string& event_id,
    const std::string& classification,
    float discrepancy_score
) {
    const char* sql = R"(
        INSERT OR REPLACE INTO events
        (faiss_idx, event_id, classification, discrepancy_score, timestamp)
        VALUES (?, ?, ?, ?, ?)
    )";

    sqlite3_stmt* stmt = nullptr;
    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("Failed to prepare insert: ") + sqlite3_errmsg(db_)
        );
    }

    // Bind parameters
    sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>(faiss_idx));
    sqlite3_bind_text(stmt, 2, event_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, classification.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 4, discrepancy_score);
    sqlite3_bind_int64(stmt, 5, std::time(nullptr));

    // Execute
    rc = sqlite3_step(stmt);

    if (rc != SQLITE_DONE) {
        std::string error = sqlite3_errmsg(db_);
        sqlite3_finalize(stmt);
        throw std::runtime_error("Failed to insert event: " + error);
    }

    sqlite3_finalize(stmt);
}

size_t MetadataDB::count() const {
    const char* sql = "SELECT COUNT(*) FROM events";

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);

    size_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);
    return count;
}

void MetadataDB::flush() {
    // Force WAL checkpoint
    exec_sql("PRAGMA wal_checkpoint(PASSIVE)");
}

void MetadataDB::exec_sql(const char* sql) {
    char* error_msg = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &error_msg);

    if (rc != SQLITE_OK) {
        std::string error = "SQL execution failed: ";
        if (error_msg) {
            error += error_msg;
            sqlite3_free(error_msg);
        }
        throw std::runtime_error(error);
    }
}

} // namespace rag
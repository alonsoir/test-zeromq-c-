# ML Defender - Day 60 Continuation (Incomplete)

## Day 60 Summary - PARCIALMENTE COMPLETADO ⏳

**Date:** 2026-02-17 (Tuesday)
**Status:** Major bug fixed, CSV logging path issue pending

### Achievements ✅

**Priority 1: PUT Config Decryption Bug - FIXED**

**Root Cause:** Refactorización incompleta en etcd-client
- `put_config()` llamaba a `process_outgoing_data()`
- Pero `process_outgoing_data()` NO EXISTÍA
- Cliente enviaba JSON plano marcado como `application/octet-stream`
- Servidor intentaba descifrar JSON plano → ChaCha20 error

**Solution Implemented:**
1. ✅ Implementado `process_outgoing_data()` en `etcd_client.cpp`
2. ✅ Conversión hex→binary de encryption key (64 chars → 32 bytes)
3. ✅ Compresión con `compress_with_size()` (header de 4 bytes)
4. ✅ Cifrado con ChaCha20-Poly1305
5. ✅ Content-Type dinámico según config

**Test Results:**
```
✅ Compresión: 8258 → 4213 bytes (49% reducción)
✅ Cifrado: 4213 → 4253 bytes (+40 bytes nonce+MAC)
✅ Servidor descifra correctamente
✅ Servidor descomprime correctamente
✅ Config llega intacta
```

**Files Modified:**
```
etcd-client/CMakeLists.txt                    - Fixed CRYPTO_TRANSPORT_INCLUDE path
etcd-client/src/etcd_client.cpp               - Added process_outgoing_data()
firewall-acl-agent/src/core/etcd_client.cpp   - Cleaned up error messages
firewall-acl-agent/src/core/config_loader.cpp - Read csv_batch_logger from root JSON
```

---

### Pending Issues ⏳

**Priority 1: CSV Logger Path Not Applied Correctly**

**Symptom:**
- Config loader READS correct path: `[CONFIG] CSV logger output_dir: /vagrant/logs/firewall_logs`
- But FirewallLogger likely initializes with WRONG path
- CSV logs probably writing to `/vagrant/firewall-acl-agent/build/logs` (default)

**Next Steps (Day 61):**
1. Debug where `config.operation.log_directory` flows to `ZMQSubscriber::Config`
2. Add logs in `ZMQSubscriber` constructor to verify path received
3. Verify `FirewallLogger` initialization path
4. Send malicious traffic to trigger CSV writes
5. Verify `.csv` + `.csv.hmac` files appear in correct directory

**Hypothesis:**
- `config_loader.cpp` sets `config.operation.log_directory` correctly
- But `main.cpp` or intermediate code overwrites it with default
- OR config flows through different path that doesn't read `operation.log_directory`

---

## System State (Current)

**etcd-client:**
- ✅ `process_outgoing_data()` works (compress + encrypt)
- ✅ Hex→binary conversion works
- ✅ PUT config end-to-end validated
- ⚠️  Needs cleanup: remove DEBUG logs

**firewall-acl-agent:**
- ✅ Registers with etcd (encrypted+compressed config)
- ✅ Obtains HMAC key via service discovery
- ✅ Obtains crypto seed
- ✅ `append_csv_log()` exists and is called
- ❌ CSV logs likely writing to wrong directory
- ⏳ Needs verification with actual malicious traffic

**etcd-server:**
- ✅ Receives encrypted+compressed configs
- ✅ Decrypts correctly
- ✅ Decompresses correctly
- ✅ Stores configs
- ✅ Provides HMAC keys via `/secrets/{component}`

---

## Quick Start Commands (Day 61)
```bash
# 1. Debug logger path
cd /vagrant/firewall-acl-agent/src/api
grep -A 10 "logger_ = std::make_unique" zmq_subscriber.cpp

# 2. Add debug log in ZMQSubscriber constructor
# Show what log_directory value is received

# 3. Check where CSV actually writes
# Send test traffic, find where .csv appears

# 4. Verify flow: config_loader → main → ZMQSubscriber
cd /vagrant/firewall-acl-agent/src
grep -r "log_directory" main.cpp

# Remember: Piano piano 🏛️
```

---

## Lessons Learned

**Lesson 1: Shared Library Hell**
- Tests used old `/usr/local/lib/libetcd_client.so`
- New code in `/vagrant/etcd-client/build/libetcd_client.so` ignored
- **Solution:** Always use `LD_LIBRARY_PATH` in tests or reinstall via root Makefile

**Lesson 2: Root JSON vs Sections**
- `csv_batch_logger` at root level, not in `operation`
- `parse_operation()` only sees `operation` section
- **Solution:** Read root-level configs in `load_from_file()`, not section parsers

**Lesson 3: Config Flow Complexity**
- Config loader → FirewallAgentConfig → main.cpp → multiple initializers
- Easy to lose values in translations
- **Solution:** Add debug logs at each handoff point

**Lesson 4: Incremental Compilation Traps**
- Changed code, but `.o` didn't recompile
- **Solution:** `make clean` or delete specific `.o` when in doubt

---

## Architecture Decisions

**Why compress_with_size() with 4-byte header?**
- Server needs exact original size for LZ4 decompression
- Header provides size without guessing
- Only 4 bytes overhead (negligible)
- Via Appia Quality: explicit > implicit

**Why hex→binary conversion in client?**
- Server sends keys as hex strings (JSON-safe)
- Crypto functions need binary bytes
- Conversion at point of use (client)
- Server stays simple (just stores hex)

---

Last Updated: 2026-02-17 22:00 (Day 60 Incomplete)
Co-authored-by: Claude (Anthropic)
Co-authored-by: Alonso

Piano piano 🏛️
```

---



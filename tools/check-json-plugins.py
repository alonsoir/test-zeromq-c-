import json, os, sys

PLUGIN_DIR = "/usr/lib/ml-defender/plugins"
CONFIG_DIR = "/etc/ml-defender"

fail = 0
checked = 0

for root, dirs, files in os.walk(CONFIG_DIR):
    # Ignorar directorios de backup
    dirs[:] = [d for d in dirs if '.bak.' not in d]
    for fname in files:
        if not fname.endswith(".json") or fname == "provision_meta.json":
            continue
        path = os.path.join(root, fname)
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        plugins_block = d.get("plugins", {})
        # Estructura: {"enabled": [...]} donde cada item es string o dict
        if isinstance(plugins_block, dict):
            enabled = plugins_block.get("enabled", [])
        elif isinstance(plugins_block, list):
            enabled = plugins_block
        else:
            continue
        for item in enabled:
            if isinstance(item, dict):
                lib = item.get("library", item.get("name", ""))
                active = item.get("active", True)
                if not active:
                    continue
            elif isinstance(item, str):
                lib = item
            else:
                continue
            if not lib:
                continue
            base = os.path.basename(lib)
            if base.endswith(".so"):
                base = base[:-3]
            checked += 1
            so_path  = os.path.join(PLUGIN_DIR, base + ".so")
            sig_path = os.path.join(PLUGIN_DIR, base + ".so.sig")
            if not os.path.exists(so_path):
                print(f"  ❌ {fname}: {base}.so ausente en {PLUGIN_DIR}")
                fail += 1
            if not os.path.exists(sig_path):
                print(f"  ❌ {fname}: {base}.so.sig ausente en {PLUGIN_DIR}")
                fail += 1

if fail == 0:
    if checked == 0:
        print(f"  ✅ Sin plugins activos en configs de producción (correcto)")
    else:
        print(f"  ✅ {checked} plugin(s) activos verificados — todos tienen .so y .sig")
sys.exit(fail)

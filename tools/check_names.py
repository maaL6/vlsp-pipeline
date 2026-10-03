"""Static check: every global name referenced in a module (incl. inside functions/classes) exists in
the module namespace after import, or in builtins. Catches names lost when splitting files."""
import builtins, importlib, pkgutil, symtable, sys
from pathlib import Path

def globals_used(table, out):
    for s in table.get_symbols():
        if s.is_referenced() and (s.is_global() or (table.get_type() == "module" and not s.is_assigned() and not s.is_imported())):
            out.add(s.get_name())
    for c in table.get_children():
        globals_used(c, out)
    return out

def main(pkg_name):
    pkg = importlib.import_module(pkg_name)
    bad = 0
    for m in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
        if m.name.endswith("__main__"):
            continue
        mod = importlib.import_module(m.name)
        src = Path(mod.__file__).read_text(encoding="utf-8")
        names = globals_used(symtable.symtable(src, mod.__file__, "exec"), set())
        missing = sorted(n for n in names if not hasattr(mod, n) and not hasattr(builtins, n) and n != "__file__")
        if missing:
            bad += 1
            print(m.name, missing)
    print("names ok" if not bad else f"{bad} modules with missing names")
    return bad

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docmt"))

import os
import sys
import atexit
import builtins
from pathlib import Path
import subprocess

# Nur aktiv wenn eingeschaltet
if os.getenv("TRACE_ALL_FILES") != "1":
    raise SystemExit

project_root = Path(os.getenv("TRACE_PROJECT_ROOT", "/app")).resolve()
out_file = Path(os.getenv("TRACE_OUT", "/app/used_files_all_runtime.txt")).resolve()
log_subprocess = os.getenv("TRACE_SUBPROCESS") == "1"

seen = set()

def _is_in_project(p: Path) -> bool:
    try:
        p.resolve().relative_to(project_root)
        return True
    except Exception:
        return False

def record(path_like):
    try:
        p = Path(path_like).resolve()
    except Exception:
        return
    if _is_in_project(p):
        seen.add(str(p))

# 1) Hook für open()
_real_open = builtins.open

def traced_open(file, mode="r", *args, **kwargs):
    try:
        record(file)
    except Exception:
        pass
    return _real_open(file, mode, *args, **kwargs)

builtins.open = traced_open

# 2) Hook für Path.open()
_real_path_open = Path.open

def traced_path_open(self, mode="r", *args, **kwargs):
    try:
        record(self)
    except Exception:
        pass
    return _real_path_open(self, mode, *args, **kwargs)

Path.open = traced_path_open

# 3) Optional subprocess Logging
_real_run = subprocess.run
_real_popen = subprocess.Popen

def _record_cmd(cmd):
    try:
        if isinstance(cmd, (list, tuple)):
            c = " ".join(str(x) for x in cmd)
        else:
            c = str(cmd)
        seen.add(f"[subprocess] {c}")
    except Exception:
        pass

def traced_run(*args, **kwargs):
    if log_subprocess and args:
        _record_cmd(args[0])
    return _real_run(*args, **kwargs)

def traced_popen(*args, **kwargs):
    if log_subprocess and args:
        _record_cmd(args[0])
    return _real_popen(*args, **kwargs)

subprocess.run = traced_run
subprocess.Popen = traced_popen

# 4) Auch Python-Dateien, die wirklich ausgeführt werden
def tracer(frame, event, arg):
    if event == "call":
        fn = frame.f_code.co_filename
        if fn and fn != "<string>":
            record(fn)
    return tracer

sys.settrace(tracer)

@atexit.register
def dump():
    # imports mitnehmen
    for m in list(sys.modules.values()):
        fp = getattr(m, "__file__", None)
        if fp:
            record(fp)

    # bestehende Datei einlesen und vereinigen
    existing = set()
    if out_file.exists():
        try:
            existing = set(
                x.strip()
                for x in out_file.read_text(encoding="utf-8").splitlines()
                if x.strip()
            )
        except Exception:
            existing = set()

    all_lines = sorted(existing | seen)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")

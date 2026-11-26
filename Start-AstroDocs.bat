@echo off
setlocal DisableDelayedExpansion
title AstroDocs Launcher
chcp 65001 >nul
if not exist "%~dp0backend\.venv\Scripts\python.exe" (
  echo Place this file in the astrodocs-rag project folder,
  echo next to the backend and frontend folders.
  echo The backend virtual environment must already be installed.
  pause
  exit /b 1
)
"%~dp0backend\.venv\Scripts\python.exe" -u -c "import pathlib,sys; p=pathlib.Path(sys.argv[1]); s=p.read_text(encoding='utf-8-sig').split('\n# ASTRODOCS_PYTHON\n',1)[1]; exec(compile(s,str(p),'exec'), {'__name__':'__main__','__file__':str(p)})" "%~f0"
exit /b %errorlevel%
# ASTRODOCS_PYTHON
"""One-window Windows launcher for the existing AstroDocs project."""

import ctypes
import html
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
import webbrowser


BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://127.0.0.1:5173"
LOCAL_HTTP = build_opener(ProxyHandler({}))
DWORD = ctypes.c_uint32
HANDLE = ctypes.c_void_p
SIZE_T = ctypes.c_size_t


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
        ("LimitFlags", DWORD), ("MinWorkingSet", SIZE_T),
        ("MaxWorkingSet", SIZE_T), ("ActiveProcessLimit", DWORD),
        ("Affinity", SIZE_T), ("PriorityClass", DWORD),
        ("SchedulingClass", DWORD),
    ]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperations", "WriteOperations", "OtherOperations",
        "ReadBytes", "WriteBytes", "OtherBytes",
    )]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("Basic", BasicLimits), ("IO", IOCounters),
        ("ProcessMemory", SIZE_T), ("JobMemory", SIZE_T),
        ("PeakProcessMemory", SIZE_T), ("PeakJobMemory", SIZE_T),
    ]


class WindowsRuntime:
    """Track only services created by this launcher."""

    def __init__(self):
        self.processes = {}
        self.job = None
        self.mutex = None
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        declarations = {
            "CreateMutexW": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p], HANDLE),
            "CreateJobObjectW": ([ctypes.c_void_p, ctypes.c_wchar_p], HANDLE),
            "SetInformationJobObject": ([HANDLE, ctypes.c_int, ctypes.c_void_p, DWORD], ctypes.c_int),
            "OpenProcess": ([DWORD, ctypes.c_int, DWORD], HANDLE),
            "AssignProcessToJobObject": ([HANDLE, HANDLE], ctypes.c_int),
            "CloseHandle": ([HANDLE], ctypes.c_int),
        }
        for name, (arguments, result) in declarations.items():
            function = getattr(self.kernel, name)
            function.argtypes = arguments
            function.restype = result

        ctypes.set_last_error(0)
        self.mutex = self.kernel.CreateMutexW(
            None, False, "Local\\AstroDocsLauncher_8000_5173"
        )
        if not self.mutex:
            raise ctypes.WinError(ctypes.get_last_error())
        self.already_running = ctypes.get_last_error() == 183

    def ensure_job(self):
        if self.job:
            return
        self.job = self.kernel.CreateJobObjectW(None, None)
        if not self.job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.Basic.LimitFlags = 0x00002000
        if not self.kernel.SetInformationJobObject(
            self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.kernel.CloseHandle(self.job)
            self.job = None
            raise error

    def start(self, name, command, cwd):
        self.ensure_job()
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["NO_COLOR"] = "1"
        process = subprocess.Popen(
            command, cwd=str(cwd), env=environment,
            stdin=subprocess.DEVNULL, shell=False,
        )
        process_handle = self.kernel.OpenProcess(0x0101, False, process.pid)
        try:
            if not process_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.AssignProcessToJobObject(self.job, process_handle):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            process.terminate()
            process.wait(timeout=5)
            raise
        finally:
            if process_handle:
                self.kernel.CloseHandle(process_handle)
        self.processes[name] = process

    def check(self):
        for name, process in self.processes.items():
            code = process.poll()
            if code is not None:
                raise RuntimeError(
                    f"{name} stopped with exit code {code}. "
                    "Read its error message above."
                )

    def close(self):
        if self.job:
            self.kernel.CloseHandle(self.job)
            self.job = None
        for process in self.processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
        if self.mutex:
            self.kernel.CloseHandle(self.mutex)
            self.mutex = None


def read_url(url):
    request = Request(url, headers={"User-Agent": "AstroDocs-Launcher/1.0"})
    try:
        with LOCAL_HTTP.open(request, timeout=3) as response:
            return response.read(250000).decode("utf-8", errors="replace")
    except (URLError, OSError, TimeoutError, ValueError):
        return None


def read_json(url):
    text = read_url(url)
    try:
        data = json.loads(text) if text else None
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def backend_identity():
    data = read_json(f"{BACKEND_URL}/")
    return bool(data and data.get("name") == "AstroDocs RAG API")


def backend_ready():
    data = read_json(f"{BACKEND_URL}/health")
    return bool(data and data.get("status") == "ok")


def frontend_ready():
    text = read_url(FRONTEND_URL)
    if not text:
        return False
    title = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
    root = re.search(r'''\bid\s*=\s*["']root["']''', text, re.I)
    return bool(title and root and "astrodocs" in html.unescape(title.group(1)).casefold())


def wait_until(label, check, runtime, timeout):
    deadline = time.monotonic() + timeout
    next_message = time.monotonic() + 10
    while time.monotonic() < deadline:
        runtime.check()
        if check():
            return
        now = time.monotonic()
        if now >= next_message:
            print(f"Still waiting for {label}...", flush=True)
            next_message = now + 10
        time.sleep(0.75)
    raise RuntimeError(f"{label} did not become ready within {timeout} seconds.")


def preflight(root):
    required = (
        "backend/.venv/Scripts/python.exe", "backend/app/main.py",
        "backend/data/index/rag_index.faiss", "backend/data/index/chunks.json",
        "backend/data/index/index_config.json", "frontend/package.json",
        "frontend/node_modules/vite/bin/vite.js",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(
            "Place Start-AstroDocs.bat in the project root, next to backend and frontend.\n"
            "The existing project setup must be complete.\nMissing:\n  "
            + "\n  ".join(missing)
        )
    node = shutil.which("node.exe") or shutil.which("node")
    if not node:
        raise RuntimeError("Node.js was not found. Install Node.js before launching the project.")
    return node


def start_project(root, runtime, node):
    if port_open(8000):
        if not backend_identity():
            raise RuntimeError(
                "Port 8000 is occupied by another application. "
                "Close that application or its server, then try again."
            )
        print("Using the AstroDocs backend already running on port 8000.", flush=True)
    else:
        print("Starting the backend. Loading the local models may take a moment...", flush=True)
        runtime.start("Backend", [
            str(root / "backend/.venv/Scripts/python.exe"),
            "-u", "-m", "uvicorn", "app.main:app",
            "--app-dir", str(root / "backend"),
            "--host", "127.0.0.1", "--port", "8000", "--workers", "1",
        ], root)

    wait_until("the backend", backend_ready, runtime, 300)

    if port_open(5173):
        if not frontend_ready():
            raise RuntimeError(
                "Port 5173 is occupied by another application. "
                "Close that application or its server, then try again."
            )
        print("Using the AstroDocs frontend already running on port 5173.", flush=True)
    else:
        print("Starting the frontend...", flush=True)
        runtime.start("Frontend", [
            node, str(root / "frontend/node_modules/vite/bin/vite.js"),
            "--host", "127.0.0.1", "--port", "5173", "--strictPort",
        ], root / "frontend")

    wait_until("the frontend", frontend_ready, runtime, 60)
    print(f"\nAstroDocs is ready: {FRONTEND_URL}", flush=True)
    webbrowser.open(FRONTEND_URL, new=2)


def wait_for_stop(runtime):
    import msvcrt
    print("\nKeep this window open while using AstroDocs.", flush=True)
    print("Press ENTER here to stop services started by this launcher.", flush=True)
    print("Services that were already running will remain open.\n", flush=True)
    while True:
        runtime.check()
        if msvcrt.kbhit() and msvcrt.getwch() in ("\r", "\n"):
            return
        time.sleep(0.15)


def main():
    if os.name != "nt":
        print("This launcher is for Windows only.")
        return 1
    root = Path(__file__).resolve().parent
    runtime = None
    error_message = None
    try:
        print("AstroDocs - Project Launcher\n", flush=True)
        node = preflight(root)
        runtime = WindowsRuntime()
        if runtime.already_running:
            print("An AstroDocs launcher is already open in this Windows session.", flush=True)
            runtime.close()
            runtime = None
            if backend_ready() and frontend_ready():
                webbrowser.open(FRONTEND_URL, new=2)
            input("Press ENTER to close this extra launcher window.")
            return 0
        start_project(root, runtime, node)
        wait_for_stop(runtime)
    except KeyboardInterrupt:
        print("\nStopping AstroDocs...", flush=True)
    except Exception as error:
        error_message = str(error)
    finally:
        if runtime is not None:
            runtime.close()
    if error_message:
        print(f"\nERROR: {error_message}\n", flush=True)
        input("Press ENTER to close this window.")
        return 1
    print("Launcher closed. Services started here have been stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

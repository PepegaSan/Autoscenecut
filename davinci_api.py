#!/usr/bin/env python3
# Vendored unchanged from Hail Mary (bridges/vendor/scene_cutter/davinci_api.py).
"""Reference implementation for reliably connecting to DaVinci Resolve Studio.

Drop this file into a new project and call :func:`connect_resolve` — it
enforces every rule that tends to cause silent Resolve-API failures on
Windows. Ships with project-side helpers (:func:`list_render_presets`,
:func:`apply_project_timeline_settings`, :func:`render_with_preset`) so a
downstream pipeline can stay small.

Pattern summary (why each piece exists)
=======================================

1. **Purge stale env vars before import**
   Resolve sets ``RESOLVE_SCRIPT_API`` / ``RESOLVE_SCRIPT_LIB`` globally.
   If a previous run of a different Resolve edition (Free vs Studio vs Beta)
   left them pointing at a dead DLL, the import fails with a useless
   ``ImportError("Could not locate module dependencies")``. Always pop the
   old values first, then set fresh ones from a multi-edition candidate
   list.

2. **Prefer the edition of the running ``Resolve.exe``**
   With multiple Resolve editions installed side by side the candidate
   list alone can bind the wrong ``fusionscript.dll``. Query the running
   process's full path via PowerShell, derive the matching DLL + modules
   dir from it, fall back to the static list only if no Resolve is up.

3. **Search every install layout, not just Free**
   Blackmagic's stock ``DaVinciResolveScript.py`` hardcodes the Free path.
   Studio / Studio 21 Beta / 21 Beta live elsewhere. ``_first_existing``
   picks whichever one actually exists on disk.

4. **``time.sleep(2)`` before importing ``DaVinciResolveScript``**
   Right after Resolve starts, fusionscript.dll can still be file-locked;
   the import silently fails without the short delay.

5. **Forward slashes for every path handed to the API**
   Backslashes silently break ``ImportMedia`` / render target paths on
   Windows. Convert once at the boundary (:func:`to_forward`).

6. **Initialise COM on every worker thread**
   Resolve's scripting C-module uses COM. Threads that haven't called
   ``CoInitializeEx`` get a silent ``None`` from ``scriptapp('Resolve')``
   even though the main thread works fine — a classic "works in script,
   broken in GUI" trap. Wrap worker-thread interaction in the
   :func:`scripting_thread` context manager.

7. **Connect strategy — try, launch only if needed, poll**
   - Call ``scriptapp("Resolve")`` immediately. On a running Resolve this
     returns in < 1s, so the no-op case stays fast.
   - If it returns ``None`` **and** ``Resolve.exe`` is not in the task list,
     launch Resolve.exe. NEVER Popen a second time on an already-running
     Resolve — it wobbles the scripting socket and breaks the session.
   - Poll ``scriptapp`` every 2s for up to 90s. Each call blocks ~5s during
     boot, so a healthy cold start completes in 2-3 attempts (~10-15s).
   - Emit an actionable "External scripting = Local" hint after the second
     failed poll (~4s) rather than burying it in a 90s timeout.

8. **Scratch-project fallback**
   Cold-launched Resolve lands on the Project Manager screen where
   ``GetCurrentProject()`` returns ``None`` forever. Automation must create
   a scratch project so it can proceed unattended.

Common failure modes the connect loop surfaces (in order of frequency):
    1. ``Preferences → System → General → External scripting using`` is not
       set to ``Local``. Saving alone isn't enough; Resolve must restart.
    2. Worker thread never called CoInitializeEx (see :func:`scripting_thread`).
    3. Privilege mismatch — Resolve started as admin while Python runs as
       user, or vice versa. Windows isolates the scripting socket per
       privilege level; both must run at the same elevation.
    4. Modal dialog open inside Resolve (unsaved-changes, render progress,
       auto-save prompt). These block the scripting server from answering.
    5. DaVinci Resolve **Free** — external scripting is Studio-only. Note:
       on Resolve 21 the ``ProductName`` version resource reports the same
       string ("DaVinci Resolve") for both editions, so we deliberately do
       NOT hard-gate on that. We log it for info and let ``scriptapp`` fail.
    6. No project open — Project Manager screen. Handled automatically by
       the scratch-project fallback when you allow it.

Usage
-----

Single-threaded (CLI script)::

    from davinci_api import connect_resolve, to_forward

    resolve, project, media_pool, root_folder = connect_resolve(
        status_callback=print,   # wire to your UI status bar
        auto_launch=True,
    )
    media_pool.ImportMedia([to_forward(r"C:\\clips\\take01.mp4")])

GUI worker thread (CustomTkinter / PySide / Qt threads)::

    from davinci_api import connect_resolve, scripting_thread

    def worker():
        with scripting_thread():            # COM init / uninit for this thread
            resolve, project, *_ = connect_resolve()
            print(project.GetName())

    threading.Thread(target=worker, daemon=True).start()

Full pipeline example (render preset picker + matching timeline)::

    from davinci_api import (
        connect_resolve, scripting_thread, to_forward,
        cleanup_timelines, list_render_presets,
        apply_project_timeline_settings, render_with_preset,
    )

    with scripting_thread():
        resolve, project, media_pool, root = connect_resolve()

        clips = media_pool.ImportMedia([to_forward(video_path)])
        clip = clips[0]

        # Purge our OWN leftover timelines so Resolve no longer locks
        # the project frame rate. Without this step, SetSetting below
        # silently no-ops and the new timeline inherits the old rate.
        cleanup_timelines(project, media_pool, name_prefix="AutoRun_")

        fps = clip.GetClipProperty("FPS") or "25"
        res = clip.GetClipProperty("Resolution") or "1920x1080"
        apply_project_timeline_settings(project, fps, res)  # pass raw string

        timeline = media_pool.CreateEmptyTimeline(f"AutoRun_{int(time.time())}")
        project.SetCurrentTimeline(timeline)
        media_pool.AppendToTimeline([{"mediaPoolItem": clip}])

        # Render preset dropdown UX: if Resolve is already up, use the
        # live list; otherwise the user can still type the preset name
        # by hand into their combobox (the render helper below validates
        # via its fallback chain).
        presets = list_render_presets(project)
        render_with_preset(
            project,
            output_dir=r"C:\\output",
            output_name="take01_autoedit",
            preset_name=presets[0] if presets else None,
            status_callback=print,
        )

Run this file directly to see a live demo::

    python davinci_api.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Install-location candidate lists
# ---------------------------------------------------------------------------

# Where ``DaVinciResolveScript.py`` can live, across editions.
_RESOLVE_MODULE_DIRS: Tuple[str, ...] = (
    r"C:\ProgramData\Blackmagic Design\DaVinci Resolve Studio\Support\Developer\Scripting\Modules",
    r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
    r"C:\ProgramData\Blackmagic Design\DaVinci Resolve Studio 21 Beta\Support\Developer\Scripting\Modules",
    r"C:\ProgramData\Blackmagic Design\DaVinci Resolve 21 Beta\Support\Developer\Scripting\Modules",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve Studio\Support\Developer\Scripting\Modules",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
)

# Where ``fusionscript.dll`` can live. The stock loader only falls back to
# the Free path; we explicitly try every edition.
_RESOLVE_LIB_CANDIDATES: Tuple[str, ...] = (
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve Studio\fusionscript.dll",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve Studio 21 Beta\fusionscript.dll",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve 21 Beta\fusionscript.dll",
)

# Where ``Resolve.exe`` can live.
_RESOLVE_EXE_CANDIDATES: Tuple[str, ...] = (
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve Studio\Resolve.exe",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve\Resolve.exe",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve Studio 21 Beta\Resolve.exe",
    r"C:\Program Files\Blackmagic Design\DaVinci Resolve 21 Beta\Resolve.exe",
)

# Optional extra candidates from the host app (non-default / portable installs).
_EXTRA_EXE: List[str] = []
_EXTRA_MODULES: List[str] = []
_EXTRA_LIBS: List[str] = []


def set_resolve_path_overrides(
    *,
    resolve_exe: Optional[str] = None,
    modules_dir: Optional[str] = None,
    fusionscript_dll: Optional[str] = None,
) -> None:
    """Register extra Resolve paths. Call before :func:`connect_resolve`.

    Clears the cached ``DaVinciResolveScript`` import and env vars so the next
    :func:`bootstrap_resolve_api` run binds the requested layout.
    """
    global _DAVINCI_MODULE, _EXTRA_EXE, _EXTRA_MODULES, _EXTRA_LIBS
    _DAVINCI_MODULE = None
    sys.modules.pop("DaVinciResolveScript", None)
    for key in ("RESOLVE_SCRIPT_API", "RESOLVE_SCRIPT_LIB", "PYTHONPATH"):
        os.environ.pop(key, None)

    exe = (resolve_exe or "").strip()
    _EXTRA_EXE = [exe] if exe and os.path.isfile(exe) else []
    mod = (modules_dir or "").strip()
    _EXTRA_MODULES = [mod] if mod and os.path.isdir(mod) else []
    dll = (fusionscript_dll or "").strip()
    _EXTRA_LIBS = [dll] if dll and os.path.isfile(dll) else []


def _exe_candidates() -> Tuple[str, ...]:
    return tuple(_EXTRA_EXE) + _RESOLVE_EXE_CANDIDATES


def _module_dir_candidates() -> Tuple[str, ...]:
    return tuple(_EXTRA_MODULES) + _RESOLVE_MODULE_DIRS


def _lib_candidates() -> Tuple[str, ...]:
    return tuple(_EXTRA_LIBS) + _RESOLVE_LIB_CANDIDATES


# Tuning.
RESOLVE_STARTUP_TIMEOUT_S = 90.0
RESOLVE_POLL_INTERVAL_S = 2.0
RESOLVE_DIAG_AFTER_S = 18.0

# Default preset fallback chain used by :func:`render_with_preset`. Override
# per-call by passing ``fallback_presets``.
DEFAULT_RENDER_PRESETS: Tuple[str, ...] = ("YouTube - 1080p", "H.264 Master")

_DAVINCI_MODULE: Any = None  # cached across calls


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------


class ResolveError(RuntimeError):
    """Raised when the Resolve API cannot be reached or returns something
    unexpected. Always carries a human-readable hint pointing at the likely
    root cause so errors surface usefully in logs / UI dialogs."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _first_existing(paths: Tuple[str, ...]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p) or os.path.isdir(p):
            return p
    return None


def to_forward(path: Any) -> str:
    """Convert any path to forward-slash form for the Resolve API.

    Use for every path handed to ``ImportMedia``, ``SetRenderSettings``,
    ``TargetDir`` etc. — backslashes silently break those calls on Windows.
    """
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Process / environment discovery
# ---------------------------------------------------------------------------


def is_resolve_process_running() -> bool:
    """Return ``True`` if ``Resolve.exe`` currently appears in the task list.

    Use *only* to decide whether a ``Popen`` should be skipped — a second
    launch on an already-running Resolve wobbles the scripting socket and is
    the most common cause of mid-session breakage.

    Reads raw bytes from ``tasklist`` because non-English Windows uses OEM
    codepages that Python's implicit cp1252 decoder rejects.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        raw = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq Resolve.exe", "/NH"],
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if not raw:
        return False
    return "Resolve.exe" in raw.decode("utf-8", errors="replace")


def running_resolve_exe() -> Optional[str]:
    """Return the full path to the currently-running ``Resolve.exe``, or None."""
    if not sys.platform.startswith("win"):
        return None
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Process -Name Resolve -ErrorAction SilentlyContinue | "
                "Select-Object -First 1).Path",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    path = out.decode("utf-8", errors="replace").strip()
    if path and os.path.isfile(path):
        return path
    return None


def running_resolve_dir() -> Optional[str]:
    """Return the directory of the currently-running ``Resolve.exe``, or None.

    Lets callers prefer the exact edition's ``fusionscript.dll`` over the
    static candidate list — critical when multiple Resolve editions are
    installed side by side. Version mismatch between DLL and running
    Resolve breaks scripting silently.
    """
    exe = running_resolve_exe()
    return os.path.dirname(exe) if exe else None


def resolve_product_name(exe_path: Optional[str] = None) -> Optional[str]:
    """Return the PE ``ProductName`` string of ``Resolve.exe`` for logging.

    **Informational only — do NOT gate on this.** On Resolve 21 both the
    Free and Studio editions report the same ``ProductName`` ("DaVinci
    Resolve"), so using this to decide if the user is on Studio is a
    false-positive trap. Log it alongside the exe path for diagnostic
    clarity and let ``scriptapp()`` fail cleanly instead.
    """
    if exe_path is None:
        exe_path = running_resolve_exe()
    if not sys.platform.startswith("win") or not exe_path:
        return None
    try:
        ps_literal = exe_path.replace("'", "''")
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-Item '{ps_literal}').VersionInfo.ProductName",
            ],
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    product = out.decode("utf-8", errors="replace").strip()
    return product or None


def is_python_elevated() -> bool:
    """Return ``True`` if this Python process is running with admin rights.

    Privilege level isolates Windows' per-session scripting socket.
    Resolve + Python must run at the same level or ``scriptapp`` returns
    ``None`` forever regardless of preferences and paths.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # pragma: no cover - never crash on the check
        return False


def launch_resolve() -> bool:
    """Spawn ``Resolve.exe`` detached from the current process.

    Returns ``True`` if one of the candidate executables was found and
    launched, ``False`` otherwise. Caller should have checked
    :func:`is_resolve_process_running` first and skipped this call on a
    hit — launching a second instance destabilises the scripting socket.
    """
    for exe in _exe_candidates():
        if not os.path.isfile(exe):
            continue
        creation = 0
        if hasattr(subprocess, "DETACHED_PROCESS"):
            creation = subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        try:
            subprocess.Popen(
                [exe],
                close_fds=True,
                creationflags=creation,
                cwd=os.path.dirname(exe),
            )
            return True
        except OSError:
            continue
    return False


# ---------------------------------------------------------------------------
# Thread-safety helpers
# ---------------------------------------------------------------------------


@contextmanager
def scripting_thread() -> Iterator[None]:
    """Context manager that properly initialises COM for the current thread.

    DaVinci Resolve's ``fusionscript.dll`` uses COM internally. On Windows
    every thread that touches a COM object must first call
    ``CoInitializeEx`` — without it, ``scriptapp('Resolve')`` silently
    returns ``None`` from a worker thread even though it works fine from
    the main thread of a standalone script (classic "works in CLI, fails
    in GUI" trap).

    We pick MTA (``COINIT_MULTITHREADED`` / 0x0) because workers here
    don't run a Windows message pump. Matching ``CoUninitialize`` at exit
    keeps the thread clean. Safe to nest; only the outermost init pairs
    actually touch COM.

    Usage::

        def worker():
            with scripting_thread():
                resolve, project, *_ = connect_resolve()
                project.GetName()

    No-op on non-Windows platforms.
    """
    initialised = False
    if sys.platform.startswith("win"):
        try:
            import ctypes

            # CoInitializeEx returns S_OK(0) or S_FALSE(1) when already
            # initialised — both are fine. Negative HRESULT means a real
            # failure; we silently skip in that case so a diagnostic helper
            # can't wedge the pipeline.
            hr = ctypes.windll.ole32.CoInitializeEx(None, 0x0)
            if hr >= 0:
                initialised = True
        except Exception:
            pass
    try:
        yield
    finally:
        if initialised:
            try:
                import ctypes

                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Bootstrap: import DaVinciResolveScript
# ---------------------------------------------------------------------------


def bootstrap_resolve_api() -> Any:
    """Import Blackmagic's ``DaVinciResolveScript`` module.

    Enforces DaVinci API rules 1 and 2:
      * purge stale env vars that survive across Resolve reinstalls / edition
        switches, then set them freshly from the first install location that
        actually exists on disk (preferring the edition of the currently
        running ``Resolve.exe``);
      * ``time.sleep(2)`` before importing so fusionscript.dll's file lock
        has been released after a fresh Resolve launch.

    The result is cached on module level — subsequent calls are no-ops.
    """
    global _DAVINCI_MODULE
    if _DAVINCI_MODULE is not None:
        return _DAVINCI_MODULE

    for key in ("RESOLVE_SCRIPT_API", "RESOLVE_SCRIPT_LIB", "PYTHONPATH"):
        os.environ.pop(key, None)

    lib_path: Optional[str] = None
    modules_dir: Optional[str] = None

    # Prefer files from the edition whose Resolve.exe is actually running.
    running_dir = running_resolve_dir()
    if running_dir:
        dll_candidate = os.path.join(running_dir, "fusionscript.dll")
        if os.path.isfile(dll_candidate):
            lib_path = dll_candidate
        edition = os.path.basename(running_dir)
        mirrored_modules = os.path.join(
            r"C:\ProgramData\Blackmagic Design",
            edition,
            r"Support\Developer\Scripting\Modules",
        )
        if os.path.isdir(mirrored_modules):
            modules_dir = mirrored_modules

    if modules_dir is None:
        modules_dir = _first_existing(_module_dir_candidates())
    if lib_path is None:
        lib_path = _first_existing(_lib_candidates())
    if modules_dir is None or lib_path is None:
        raise ResolveError(
            "Could not locate the DaVinci Resolve scripting files. Tried "
            "Modules dirs: "
            + ", ".join(_module_dir_candidates())
            + " | fusionscript.dll: "
            + ", ".join(_lib_candidates())
        )

    os.environ["RESOLVE_SCRIPT_API"] = os.path.dirname(modules_dir)
    os.environ["RESOLVE_SCRIPT_LIB"] = lib_path

    time.sleep(2)

    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)

    # NOTE: Resolve 21's DaVinciResolveScript.py ends with
    # ``sys.modules[__name__] = script_module``. If the DLL fails to load
    # the whole import raises ImportError — there is no silent-``None``
    # state to guard against after this call.
    import DaVinciResolveScript as dvr_script  # type: ignore  # noqa: E402
    _DAVINCI_MODULE = dvr_script
    return dvr_script


def _poll_for_scriptapp(
    dvr_script: Any,
    log: Callable[[str], None],
) -> Any:
    """Poll ``scriptapp("Resolve")`` until it returns a live object or times
    out. Returns the object or ``None``."""
    start = time.monotonic()
    deadline = start + RESOLVE_STARTUP_TIMEOUT_S
    last_heartbeat = start
    attempt = 0
    diag_printed = False
    preference_hint_logged = False
    while time.monotonic() < deadline:
        attempt += 1
        time.sleep(RESOLVE_POLL_INTERVAL_S)
        resolve = dvr_script.scriptapp("Resolve")
        if resolve is not None:
            log(
                f"Resolve is up after ~{time.monotonic() - start:.0f}s "
                f"(attempt {attempt})."
            )
            return resolve

        # Early actionable hint: Resolve's process is clearly up but the
        # scripting socket never answers. Telling the user after ~4s is far
        # more useful than waiting 18s for the generic diag dump.
        if (
            not preference_hint_logged
            and attempt >= 2
            and is_resolve_process_running()
        ):
            log(
                "Resolve is running but not answering scripting. "
                "Fix: Preferences → System → General → 'External "
                "scripting using' = Local, THEN restart Resolve."
            )
            preference_hint_logged = True

        now = time.monotonic()
        if not diag_printed and now - start >= RESOLVE_DIAG_AFTER_S:
            log(
                "Still no scripting response after "
                f"{RESOLVE_DIAG_AFTER_S:.0f}s — check: "
                "(1) External scripting = Local AND Resolve restarted since "
                "you toggled it, (2) your worker thread wrapped the call in "
                "scripting_thread() so COM is initialised, (3) Resolve and "
                "Python run at the same privilege level (both admin or both "
                "user), (4) a project is loaded (not Project Manager)."
            )
            diag_printed = True
        if now - last_heartbeat >= 8.0:
            log(
                "Waiting for Resolve scripting server… "
                f"({max(0.0, deadline - now):.0f}s left)"
            )
            last_heartbeat = now
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def connect_resolve(
    *,
    status_callback: Optional[Callable[[str], None]] = None,
    auto_launch: bool = True,
    create_scratch_project_name: Optional[str] = "AutoPipeline",
) -> Tuple[Any, Any, Any, Any]:
    """Connect to a Resolve instance and return ``(resolve, project, media_pool, root_folder)``.

    Parameters
    ----------
    status_callback:
        Receives human-readable progress strings ("Launching…", "Waiting for
        scripting server… 72s left"). Wire this to your UI status bar.
    auto_launch:
        When ``True`` and Resolve is not running, ``Resolve.exe`` is started
        automatically. When ``False`` a missing Resolve raises immediately.
    create_scratch_project_name:
        When non-empty and no project is open after connecting, a new project
        with this base name (plus a timestamp) is created. Set to ``None`` to
        fail hard instead — useful for tools that must never touch the
        Project Manager.

    Must be called from a thread where COM is initialised (use the
    :func:`scripting_thread` context manager on GUI worker threads).

    Raises
    ------
    ResolveError
        When scripting files can't be located, when Resolve can't be launched
        / reached within the timeout, or when no project is open and no
        scratch name was supplied.
    """
    def _log(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    dvr_script = bootstrap_resolve_api()

    # Surface exactly what bound — invaluable when tracking down silent
    # connection failures. Same info the pipeline's error message shows
    # below on timeout, logged upfront so the user sees it at startup too.
    import platform

    lib_env = os.environ.get("RESOLVE_SCRIPT_LIB", "?")
    api_env = os.environ.get("RESOLVE_SCRIPT_API", "?")
    running_exe = running_resolve_exe()
    product = resolve_product_name(running_exe) if running_exe else None
    py_admin = "admin" if is_python_elevated() else "user"

    _log(
        f"Python: {sys.version.split()[0]} "
        f"({platform.architecture()[0]}) [{py_admin}]"
    )
    _log(f"Scripting lib: {lib_env}")
    _log(f"Scripting API: {api_env}")
    if running_exe:
        _log(
            f"Running exe:   {running_exe}"
            + (f"  ({product})" if product else "")
        )

    resolve = dvr_script.scriptapp("Resolve")
    if resolve is None:
        if auto_launch and not is_resolve_process_running():
            _log("DaVinci Resolve not running — launching…")
            if not launch_resolve():
                raise ResolveError(
                    "Could not find Resolve.exe under any of the default "
                    "install paths: " + ", ".join(_RESOLVE_EXE_CANDIDATES)
                )
        elif is_resolve_process_running():
            _log("Resolve is starting — waiting for scripting server…")
        elif not auto_launch:
            raise ResolveError(
                "DaVinci Resolve is not running. Start Resolve Studio and "
                "open a project first."
            )

        resolve = _poll_for_scriptapp(dvr_script, _log)
        if resolve is None:
            running_exe_now = running_resolve_exe() or "(Resolve.exe not detected)"
            raise ResolveError(
                f"Could not connect to Resolve within "
                f"{RESOLVE_STARTUP_TIMEOUT_S:.0f}s.\n\n"
                f"Python:        {sys.version.split()[0]} "
                f"({platform.architecture()[0]}) [{py_admin}]\n"
                f"Running exe:   {running_exe_now}\n"
                f"Scripting lib: {lib_env}\n"
                f"Scripting API: {api_env}\n\n"
                "Remaining causes when paths look right: "
                "(1) 'External scripting using' isn't 'Local' in "
                "Preferences → System → General (toggle requires a Resolve "
                "restart, Save alone is not enough); "
                "(2) the calling thread did not initialise COM — wrap the "
                "call in `with scripting_thread(): ...`; "
                "(3) Resolve was started as admin while Python runs as "
                f"'{py_admin}' — run both at the same privilege level; "
                "(4) a modal dialog is blocking Resolve's UI thread "
                "(click through any 'Welcome' / 'Quick Setup' wizard); "
                "(5) Resolve is on the Project Manager screen — double-"
                "click a project so scriptapp() can bind."
            )

    project_manager = resolve.GetProjectManager()
    if project_manager is None:
        raise ResolveError("Could not access the Project Manager.")

    project = project_manager.GetCurrentProject()
    if project is None:
        if not create_scratch_project_name:
            raise ResolveError(
                "No project is open in Resolve. Open or create a project "
                "before running this tool."
            )
        fallback_name = f"{create_scratch_project_name}_{int(time.time())}"
        _log(f"No project open — creating scratch project {fallback_name!r}…")
        project = project_manager.CreateProject(fallback_name)
        if project is None:
            raise ResolveError(
                "No project is open and a fallback project could not be "
                "created. Open a project in Resolve and retry."
            )

    media_pool = project.GetMediaPool()
    if media_pool is None:
        raise ResolveError("Could not access the Media Pool.")
    root_folder = media_pool.GetRootFolder()

    return resolve, project, media_pool, root_folder


# ---------------------------------------------------------------------------
# Project-side helpers (timeline settings + render presets)
# ---------------------------------------------------------------------------


def cleanup_timelines(
    project: Any,
    media_pool: Any,
    *,
    name_prefix: Optional[str] = None,
) -> int:
    """Delete existing timelines so the project frame rate unlocks.

    Resolve silently refuses ``SetSetting('timelineFrameRate', …)`` as
    long as *any* timeline exists in the project at a different rate.
    That's the #1 reason "setting the FPS doesn't stick" — a leftover
    timeline from a prior run locks it.

    Pass ``name_prefix`` to delete only timelines whose name starts with
    that string (e.g. ``"AutoRun_"``) so user-created timelines stay
    untouched. Pass ``None`` to wipe them all — only safe in a
    freshly-created scratch project.

    Returns the number of timelines actually removed.
    """
    try:
        count = int(project.GetTimelineCount() or 0)
    except Exception:
        return 0
    if count <= 0:
        return 0

    victims: List[Any] = []
    for i in range(1, count + 1):
        try:
            tl = project.GetTimelineByIndex(i)
        except Exception:
            continue
        if not tl:
            continue
        if name_prefix is not None:
            try:
                if not str(tl.GetName() or "").startswith(name_prefix):
                    continue
            except Exception:
                continue
        victims.append(tl)
    if not victims:
        return 0
    try:
        media_pool.DeleteTimelines(victims)
    except Exception:
        return 0
    return len(victims)


def delete_all_timelines(project: Any, media_pool: Any) -> None:
    """Remove every timeline so Master timelineFrameRate can change (Resolve locks it otherwise)."""
    try:
        count = int(project.GetTimelineCount() or 0)
    except Exception:
        return
    if count <= 0:
        return
    timelines: List[Any] = []
    for idx in range(1, count + 1):
        try:
            tl = project.GetTimelineByIndex(idx)
            if tl:
                timelines.append(tl)
        except Exception:
            continue
    if not timelines:
        return
    try:
        media_pool.DeleteTimelines(timelines)
    except Exception:
        pass


def try_create_export_project(project_manager: Any) -> Tuple[Optional[Any], Optional[str]]:
    """Fresh project with empty pool — same pattern as Oxco AutoCut_Export_*."""
    for _ in range(12):
        name = (
            f"CutterExport_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{random.randint(100000, 999999)}"
        )
        try:
            proj = project_manager.CreateProject(name)
        except Exception:
            proj = None
        if proj:
            return proj, name
    return None, None


def restore_resolve_project(
    project_manager: Any, orig_name: Optional[str], switched: bool
) -> None:
    if not switched or not orig_name:
        return
    try:
        project_manager.LoadProject(orig_name)
    except Exception:
        pass


def read_timeline_fps_settings(project: Any, timeline: Any) -> Tuple[str, str]:
    try:
        gp = str(project.GetSetting("timelineFrameRate") or "").strip() or "?"
    except Exception:
        gp = "?"
    try:
        gt = str(timeline.GetSetting("timelineFrameRate") or "").strip() or "?"
    except Exception:
        gt = "?"
    return gp, gt


def parse_fps_setting(value: str) -> Optional[float]:
    try:
        f = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def snap_standard_fps(fps: float) -> float:
    for anchor in _STANDARD_CFR_ANCHORS:
        if abs(float(fps) - anchor) < 0.04:
            return anchor
    return float(fps)


def fps_setting_matches(analysis_fps: float, setting: str, tol: float = 0.06) -> bool:
    got = parse_fps_setting(setting)
    if got is None:
        return False
    return abs(got - float(analysis_fps)) <= tol


def reconcile_export_fps(
    probe_fps: float,
    project_setting: str,
    timeline_setting: str,
) -> Tuple[float, str]:
    """
    Pick the FPS used for frame math + render.

    Returns (export_fps, mode) where mode is:
    - ``ok`` — probe matches Resolve
    - ``use_timeline`` — project + timeline agree on a finer rate (e.g. 29.97 vs probe 29)
    - ``locked_wrong`` — Resolve rate clearly conflicts with probe (e.g. 24 vs 30)
    """
    probe = float(probe_fps)
    tl = parse_fps_setting(timeline_setting)
    proj = parse_fps_setting(project_setting)
    if tl is None:
        return probe, "ok"
    if proj is not None and abs(tl - proj) > 0.05:
        return probe, "locked_wrong"
    if abs(tl - probe) <= 0.06:
        return probe, "ok"
    if abs(tl - probe) >= 1.0:
        return probe, "locked_wrong"
    if proj is not None and abs(tl - proj) <= 0.02:
        return snap_standard_fps(tl), "use_timeline"
    return probe, "ok"


def scene_source_frames(
    start_sec: float,
    end_sec: float,
    analysis_fps: float,
    clip_frames: int = 0,
) -> Tuple[int, int]:
    """Half-open [start, end) source frames — matches Oxco / FFmpeg trim."""
    start_f = int(round(float(start_sec) * analysis_fps))
    end_f = int(round(float(end_sec) * analysis_fps))
    if end_f <= start_f:
        end_f = start_f + 1
    if clip_frames > 0:
        start_f = max(0, min(start_f, clip_frames - 1))
        end_f = max(start_f + 1, min(end_f, clip_frames))
    return start_f, end_f


# Clips per AppendToTimeline call (same limit as marker_autocut).
APPEND_BATCH = 48


def append_scenes_sequential(
    media_pool: Any,
    timeline: Any,
    clip: Any,
    scenes_sec: List[Tuple[float, float]],
    analysis_fps: float,
    clip_frames: int = 0,
) -> int:
    """
    Append trimmed segments back-to-back: batched AppendToTimeline calls without
    recordFrame, so Resolve places each clip directly after the previous one.

    (Placing each clip at GetEnd()+1 left one empty frame per cut: GetEnd() is
    already the first frame after the item. Same approach as clip_joiner /
    marker_autocut.)
    """
    infos = []
    for start_sec, end_sec in scenes_sec:
        start_f, end_f = scene_source_frames(start_sec, end_sec, analysis_fps, clip_frames)
        infos.append({"mediaPoolItem": clip, "startFrame": start_f, "endFrame": end_f})
    count = 0
    for i in range(0, len(infos), APPEND_BATCH):
        batch = infos[i : i + APPEND_BATCH]
        items = media_pool.AppendToTimeline(batch) or []
        if len(items) != len(batch):
            first = scenes_sec[i + len(items)] if i + len(items) < len(scenes_sec) else scenes_sec[i]
            raise ResolveError(
                f"AppendToTimeline failed for scene {first[0]:.3f}s–{first[1]:.3f}s "
                f"({count + len(items)} of {len(infos)} scenes placed)."
            )
        count += len(items)
    return count


def open_deliver_page(resolve: Any) -> None:
    try:
        resolve.OpenPage("deliver")
        time.sleep(0.5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Source FPS / resolution (ffprobe — Resolve clip metadata is often wrong)
# ---------------------------------------------------------------------------

_STANDARD_CFR_ANCHORS: Tuple[float, ...] = (
    23.976023976023978,
    24.0,
    25.0,
    29.97002997002997,
    30.0,
    48.0,
    50.0,
    59.94005994005994,
    60.0,
)


def ffprobe_executable() -> Optional[str]:
    """ffprobe next to this module, else on PATH."""
    local = Path(__file__).resolve().parent / "ffprobe.exe"
    if local.is_file():
        return str(local)
    return shutil.which("ffprobe")


def _eval_fraction_fps(raw: Any) -> Optional[float]:
    s = str(raw).strip()
    if not s or s == "0/0":
        return None
    if "/" in s:
        num_s, den_s = s.split("/", 1)
        try:
            num, den = float(num_s), float(den_s)
            if den == 0:
                return None
            return num / den
        except (TypeError, ValueError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _near_standard_cfr(fps: Optional[float], tol: float = 0.06) -> bool:
    if fps is None or fps <= 0:
        return False
    for anchor in _STANDARD_CFR_ANCHORS:
        if abs(float(fps) - anchor) < tol:
            return True
    return False


def _ffprobe_json(args: List[str]) -> Optional[dict[str, Any]]:
    exe = ffprobe_executable()
    if not exe:
        return None
    kw: dict[str, Any] = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]
    try:
        cp = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            **kw,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if cp.returncode != 0:
        return None
    try:
        return json.loads(cp.stdout or "{}")
    except json.JSONDecodeError:
        return None


def probe_video_stream_fps_rates(video_path: str | Path) -> Tuple[Optional[float], Optional[float]]:
    """Return (r_frame_rate, avg_frame_rate). Prefer r_frame_rate for NTSC MP4."""
    path = str(video_path)
    payload = _ffprobe_json(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            path,
        ]
    )
    if not payload:
        return None, None
    streams = payload.get("streams") or []
    if not streams:
        return None, None
    st = streams[0]
    r_fps = _eval_fraction_fps(st.get("r_frame_rate"))
    a_fps = _eval_fraction_fps(st.get("avg_frame_rate"))
    if r_fps is not None and r_fps <= 0:
        r_fps = None
    if a_fps is not None and a_fps <= 0:
        a_fps = None
    return r_fps, a_fps


def probe_video_wh(video_path: str | Path) -> Tuple[Optional[int], Optional[int]]:
    path = str(video_path)
    payload = _ffprobe_json(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            path,
        ]
    )
    if not payload:
        return None, None
    streams = payload.get("streams") or []
    if not streams:
        return None, None
    st = streams[0]
    try:
        w = int(st.get("width") or 0)
        h = int(st.get("height") or 0)
    except (TypeError, ValueError):
        return None, None
    if w <= 0 or h <= 0:
        return None, None
    return w, h


def pick_analysis_fps_from_probes(
    opencv_fps: float = 0.0,
    r_fps: Optional[float] = None,
    avg_fps: Optional[float] = None,
) -> Optional[float]:
    """Prefer ffprobe r_frame_rate; avg is often ~24 on 29.97 MP4."""
    try:
        of = float(opencv_fps) if opencv_fps and float(opencv_fps) > 0 else None
    except (TypeError, ValueError):
        of = None
    rf = float(r_fps) if r_fps and float(r_fps) > 0 else None
    af = float(avg_fps) if avg_fps and float(avg_fps) > 0 else None
    if rf:
        if af and abs(rf - 29.0) < 0.02 and abs(af - 29.97002997002997) < 0.06:
            return af
        return rf
    if af and of and abs(af - of) >= 0.5 and _near_standard_cfr(of):
        return of
    if af:
        return af
    if of:
        return of
    return None


def probe_source_fps(video_path: str | Path, opencv_fps: float = 0.0) -> float:
    """Best-effort CFR for timeline + frame indices (matches Oxco compare.py)."""
    r_fps, avg_fps = probe_video_stream_fps_rates(video_path)
    picked = pick_analysis_fps_from_probes(opencv_fps, r_fps, avg_fps)
    if picked and picked > 0:
        return snap_standard_fps(float(picked))
    if opencv_fps and float(opencv_fps) > 0:
        return float(opencv_fps)
    return 30.0


def format_timeline_framerate_for_resolve(fps: float) -> str:
    """Strings for project/timeline SetSetting('timelineFrameRate', …)."""
    try:
        f = float(fps)
    except (TypeError, ValueError):
        return "30"
    if f <= 0:
        return "30"
    common = [
        (23.976023976023978, "23.976"),
        (24.0, "24"),
        (25.0, "25"),
        (29.97002997002997, "29.97"),
        (30.0, "30"),
        (48.0, "48"),
        (50.0, "50"),
        (59.94005994005994, "59.94"),
        (60.0, "60"),
    ]
    for val, label in common:
        if abs(f - val) < 0.04:
            return label
    if abs(f - round(f)) < 0.001:
        return str(int(round(f)))
    out = f"{f:.6f}".rstrip("0").rstrip(".")
    return out if out else "30"


def format_clip_fps_property_for_resolve(fps_float: float) -> str:
    try:
        f = float(fps_float)
    except (TypeError, ValueError):
        return "30.0"
    if f <= 0:
        return "30.0"
    if abs(f - round(f)) < 0.02:
        return f"{float(int(round(f))):.1f}"
    return format_timeline_framerate_for_resolve(f)


def _timeline_fps_string_candidates(timeline_rate: str, fps_float: float) -> List[str]:
    try:
        f = float(fps_float)
    except (TypeError, ValueError):
        f = 0.0
    out: List[str] = []
    seen: set[str] = set()

    def add(s: str) -> None:
        if s and s not in seen:
            seen.add(s)
            out.append(str(s))

    add(timeline_rate)
    add(format_timeline_framerate_for_resolve(f))
    if f > 0:
        raw = f"{f:.6f}".rstrip("0").rstrip(".")
        add(raw)
        if abs(f - round(f)) < 0.06:
            ri = int(round(f))
            add(str(ri))
            add(f"{ri}.0")
    return out


def _try_set_setting(target: Any, key: str, val: str) -> bool:
    try:
        ok = target.SetSetting(key, str(val))
    except Exception:
        return False
    return ok is not False


def set_project_master_before_import(
    project: Any,
    timeline_rate: str,
    analysis_fps: float,
    width: int,
    height: int,
) -> None:
    """Master FPS + resolution before ImportMedia (pool clips otherwise lock wrong FPS)."""
    candidates = _timeline_fps_string_candidates(timeline_rate, analysis_fps)
    for cand in candidates:
        if _try_set_setting(project, "timelineFrameRate", cand):
            break
    else:
        _try_set_setting(project, "timelineFrameRate", candidates[0])
    _try_set_setting(project, "timelineResolutionWidth", str(width))
    _try_set_setting(project, "timelineResolutionHeight", str(height))
    time.sleep(0.3)


def sync_active_timeline_fps(
    project: Any,
    timeline: Any,
    timeline_rate: str,
    analysis_fps: float,
    *,
    skip_if_close: bool = False,
) -> None:
    if skip_if_close:
        try:
            gt = timeline.GetSetting("timelineFrameRate")
            g = float(str(gt).replace(",", "."))
            if abs(g - float(analysis_fps)) < 0.06:
                return
        except Exception:
            pass
    candidates = _timeline_fps_string_candidates(timeline_rate, analysis_fps)
    for cand in candidates:
        if _try_set_setting(timeline, "timelineFrameRate", cand):
            break
    else:
        _try_set_setting(timeline, "timelineFrameRate", candidates[0])
    _ = project  # API symmetry with Oxco; project master already set


def override_clip_fps(clip: Any, analysis_fps: float, timeline_rate: str) -> None:
    """Resolve often tags clips as 23.976/24 while ffprobe reports 29.97."""
    clip_pv = format_clip_fps_property_for_resolve(analysis_fps)
    raw = f"{float(analysis_fps):.6f}".rstrip("0").rstrip(".")
    for val in (timeline_rate, clip_pv, raw, f"{float(analysis_fps):.3f}"):
        if not val:
            continue
        try:
            if clip.SetClipProperty("FPS", val) is not False:
                return
        except Exception:
            continue


def _normalise_fps(fps: Any) -> str:
    """Map a numeric FPS to Resolve's timelineFrameRate string."""
    try:
        fps_float = float(str(fps).strip().split()[0])
        if fps_float <= 0:
            raise ValueError
    except (TypeError, ValueError, IndexError):
        return "25"
    return format_timeline_framerate_for_resolve(fps_float)


def apply_project_timeline_settings(
    project: Any, fps: Any, resolution: Any
) -> Tuple[int, int, str]:
    """Force the project's timeline settings to match a source clip so any
    timeline created next inherits matching width/height/FPS.

    Must be called BEFORE ``media_pool.CreateEmptyTimeline(...)`` —
    ``CreateEmptyTimeline`` snapshots the project settings at creation
    time. Setting them afterwards has no effect on the active timeline.
    Also call :func:`cleanup_timelines` first if the project already
    has timelines at a different rate: Resolve refuses rate changes
    while timelines exist, and fails silently.

    Parses ``resolution`` as ``"1920x1080"`` / ``"1920 x 1080"`` (both
    are seen depending on codec). Falls back to 1920×1080 @ "25" on
    unparseable metadata (Rule 6).

    Returns
    -------
    (width, height, fps_str_actually_applied) — the FPS comes from a
    ``GetSetting`` read-back, so if Resolve silently rejected the
    change you see the stale value and can act on it.
    """
    # Parse resolution with whitespace tolerance.
    width, height = 1920, 1080
    try:
        w_str, h_str = str(resolution).lower().replace(" ", "").split("x", 1)
        width = int(w_str)
        height = int(h_str)
    except (ValueError, AttributeError):
        pass

    fps_str = _normalise_fps(fps)

    project.SetSetting("timelineResolutionWidth", str(width))
    project.SetSetting("timelineResolutionHeight", str(height))
    project.SetSetting("timelineFrameRate", fps_str)

    # Tiny settle — Resolve sometimes lags one poll behind when you
    # SetSetting + GetSetting + CreateEmptyTimeline back to back.
    time.sleep(0.3)

    try:
        applied = project.GetSetting("timelineFrameRate") or fps_str
        applied = str(applied).strip() or fps_str
    except Exception:
        applied = fps_str
    return width, height, applied


def list_render_presets(project: Any) -> List[str]:
    """Return every render-preset name available in the project (factory +
    user), de-duplicated and case-insensitively sorted.

    Useful for populating a render-preset dropdown in a UI. Returns an
    empty list on any API failure — callers should fall back to the
    default preset chain in :func:`render_with_preset`.
    """
    try:
        names = project.GetRenderPresetList() or []
    except Exception:
        return []

    seen: set = set()
    unique: List[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    unique.sort(key=str.lower)
    return unique


def render_with_preset(
    project: Any,
    *,
    output_dir: str,
    output_name: str,
    preset_name: Optional[str] = None,
    fallback_presets: Tuple[str, ...] = DEFAULT_RENDER_PRESETS,
    timeout_s: float = 3600.0,
    status_callback: Optional[Callable[[str], None]] = None,
    frame_rate: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    """Configure and execute a single render job, monitored with a timeout.

    Rule 7: purges the queue before queueing (``DeleteAllRenderJobs``).
    Rule 8: preset fallback chain + bounded polling loop.

    Tries ``preset_name`` first (if given); on failure walks
    ``fallback_presets`` in order until one loads. Raises if none of them
    do — surfaces the list of attempted names so the user can fix their
    preset selection.

    Parameters
    ----------
    output_dir:
        Directory for the rendered file. Converted to forward slashes
        automatically (Rule 3).
    output_name:
        Base filename (without extension) — the preset determines the
        extension via ``FormatStr``.
    preset_name:
        Preferred render preset. ``None`` means "skip straight to the
        fallback chain".
    fallback_presets:
        Presets to try if ``preset_name`` doesn't load. Defaults to
        ``("YouTube - 1080p", "H.264 Master")``.
    timeout_s:
        Hard stop for the render polling loop. Aborts the render on
        expiry so a frozen Resolve doesn't hang the caller forever.
    status_callback:
        Receives progress strings. Same shape as ``connect_resolve``'s.
    """
    def _log(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    project.DeleteAllRenderJobs()

    tried: List[str] = []
    loaded = False
    for candidate in (preset_name, *fallback_presets):
        if not candidate or candidate in tried:
            continue
        tried.append(candidate)
        if project.LoadRenderPreset(candidate):
            _log(f"Render preset loaded: {candidate}")
            loaded = True
            break
    if not loaded:
        raise ResolveError(
            "Could not load any render preset. Tried: "
            + ", ".join(tried)
            + ". Check the names in Resolve's Deliver page."
        )

    render_settings: dict[str, Any] = {
        "SelectAllFrames": True,
        "TargetDir": to_forward(output_dir),
        "CustomName": output_name,
    }
    if frame_rate is not None and float(frame_rate) > 0:
        render_settings["FrameRate"] = float(frame_rate)
    if width and height:
        render_settings["ResolutionWidth"] = int(width)
        render_settings["ResolutionHeight"] = int(height)
    project.SetRenderSettings(render_settings)

    job_id = project.AddRenderJob()
    if not job_id:
        raise ResolveError("Resolve refused to queue the render job.")
    project.StartRendering(job_id)

    started = time.time()
    while project.IsRenderingInProgress():
        if time.time() - started > timeout_s:
            project.StopRendering()
            raise ResolveError(
                f"Render exceeded the {timeout_s:.0f}s timeout and was aborted."
            )
        time.sleep(1.0)
    _log("Render finished.")


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Connect, print project info, and list render presets.

    Returns a shell-style exit code (0 on success, 1 on failure) so you
    can use it as a quick smoke test::

        python davinci_api.py && echo "all good"
    """
    with scripting_thread():
        try:
            resolve, project, media_pool, root_folder = connect_resolve(
                status_callback=lambda m: print(f"[status] {m}"),
                auto_launch=True,
            )
        except ResolveError as err:
            print(f"[error] {err}", file=sys.stderr)
            return 1

        clip_count = len(root_folder.GetClipList() or [])
        print(f"Connected — product: {resolve.GetProductName()}")
        print(f"           project: {project.GetName()}")
        print(f"     media-pool clips (root folder): {clip_count}")

        presets = list_render_presets(project)
        if presets:
            print(f"     render presets available ({len(presets)}):")
            for name in presets[:15]:
                print(f"       - {name}")
            if len(presets) > 15:
                print(f"       … and {len(presets) - 15} more")
        else:
            print("     render presets: (none reported by the API)")
        return 0


if __name__ == "__main__":
    raise SystemExit(_demo())

"""Deleting files: to the Windows Recycle Bin when they're on a Windows drive and we run
under WSL (removing them from Linux would be permanent), permanently otherwise."""

from __future__ import annotations

import os
import re
import subprocess
from functools import cache
from pathlib import Path

# One line out per file, as soon as it's done: "ok<TAB>path" or "error<TAB>path<TAB>why".
RECYCLE = r"""
[Console]::InputEncoding = [Text.Encoding]::UTF8
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName Microsoft.VisualBasic
foreach ($path in [Console]::In.ReadToEnd().Split("`n")) {
  $path = $path.Trim()
  if (-not $path) { continue }
  try {
    [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($path, 'OnlyErrorDialogs', 'SendToRecycleBin')
    [Console]::Out.WriteLine("ok`t$path")
  } catch {
    $why = if ($_.Exception.InnerException) { $_.Exception.InnerException.Message } else { $_.Exception.Message }
    [Console]::Out.WriteLine("error`t$path`t$why")
  }
  [Console]::Out.Flush()
}
"""


@cache
def _wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def windows_path(path: Path) -> str | None:
    """/mnt/d/Radno/x.NEF -> D:\\Radno\\x.NEF, for paths on a Windows drive under WSL."""
    m = re.fullmatch(r"/mnt/([a-zA-Z])(/.*)?", str(Path(path).absolute()))
    if not (m and _wsl()):
        return None
    return f"{m[1].upper()}:" + (m[2] or "/").replace("/", "\\")


def recycles(folder: Path) -> bool:
    """Whether remove() sends files from this folder to a Recycle Bin."""
    win = windows_path(folder)
    return win is not None and (Path(f"/mnt/{win[0].lower()}") / "$Recycle.Bin").is_dir()


def remove(paths: list[Path], done=lambda path, error: None) -> dict[Path, str]:
    """Delete the files, calling done(path, error or None) after each. Returns why each file
    that's still there couldn't be deleted."""
    errors = {}

    def finished(path: Path, error: str | None):
        if path.exists():
            errors[path] = error or "it's still there"
        done(path, errors.get(path))

    if paths and all(recycles(p.parent) for p in paths):
        by_windows = {windows_path(p): p for p in paths}
        proc = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", RECYCLE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        proc.stdin.write("\n".join(by_windows).encode())
        proc.stdin.close()
        for line in proc.stdout:
            status, _, rest = line.decode(errors="replace").lstrip("\ufeff").rstrip("\r\n").partition("\t")
            win, _, why = rest.partition("\t")
            if win in by_windows:
                finished(by_windows.pop(win), None if status == "ok" else why or "it couldn't be deleted")
        proc.wait()
        for path in by_windows.values():  # never reported back
            finished(path, "the Recycle Bin didn't take it")
    else:
        for p in paths:
            try:
                os.remove(p)
                finished(p, None)
            except OSError as e:
                finished(p, e.strerror or str(e))
    return errors

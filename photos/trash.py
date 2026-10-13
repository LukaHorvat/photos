"""Deleting files: to the Windows Recycle Bin when they're on a Windows drive and we run
under WSL (removing them from Linux would be permanent), permanently otherwise."""

from __future__ import annotations

import os
import re
import subprocess
from functools import cache
from pathlib import Path

RECYCLE = r"""
[Console]::InputEncoding = [Text.Encoding]::UTF8
Add-Type -AssemblyName Microsoft.VisualBasic
foreach ($path in [Console]::In.ReadToEnd().Split("`n")) {
  $path = $path.Trim()
  if ($path) {
    try { [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($path, 'OnlyErrorDialogs', 'SendToRecycleBin') }
    catch { Write-Output "$path`: $($_.Exception.Message)" }
  }
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


def remove(paths: list[Path]) -> list[str]:
    """Delete the files; returns what went wrong, one line per file that couldn't be."""
    if not paths:
        return []
    errors = []
    if all(recycles(p.parent) for p in paths):
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", RECYCLE],
            input="\n".join(windows_path(p) for p in paths).encode(),
            capture_output=True,
        )
        errors = [line for line in proc.stdout.decode(errors="replace").splitlines() if line.strip()]
    else:
        for p in paths:
            try:
                os.remove(p)
            except OSError as e:
                errors.append(f"{p}: {e}")
    still = [f"{p}: still there" for p in paths if p.exists() and not any(line.startswith(windows_path(p) or str(p)) for line in errors)]
    return errors + still

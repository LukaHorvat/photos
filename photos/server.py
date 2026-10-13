"""Local web server behind the culling and editing UI, which runs in the (Windows) browser.

State lives in <folder>/.photos.json:
    {"marks": {"DSC_0001.NEF": "keep" | "reject"},
     "edits": {"DSC_0001.NEF": {"compress": 40, "exposure": 0.3, "temp": 5200, "saturation": 10}}}
See edits.py for how edits apply to the following photos of a series.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import denoise, develop, edits
from .export import export_photo
from .raw import IMAGE_EXTS, RAW_EXTS, Photo, read_file

WEB = Path(__file__).parent / "web"
STATE_FILE = ".photos.json"
MARKS = {"keep", "reject"}
STATIC_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".vert": "text/plain", ".frag": "text/plain"}


def _natural_key(name: str):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", name)]


class Prefetcher:
    """Slow per-photo work (raw decoding), done in a pool around wherever the viewer is.

    The browser says which photos it's about to show; we work on the one on screen
    first, then the rest, and drop queued work it has moved away from. Results of the
    `keep` most recently requested keys stay cached.
    """

    def __init__(self, fn, workers: int, keep: int):
        self.fn = fn
        self.pool = ThreadPoolExecutor(workers)
        self.futures: OrderedDict[object, Future] = OrderedDict()
        self.keep = keep
        self.generation = 0
        self.lock = threading.Lock()

    def get(self, key) -> Future:
        with self.lock:
            f = self.futures.get(key)
            if f is None or f.cancelled():
                f = self.futures[key] = self.pool.submit(self.fn, key)
            self.futures.move_to_end(key)
            while len(self.futures) > self.keep:
                self.futures.popitem(last=False)[1].cancel()
            return f

    def want(self, keys: list):
        """keys[0] is on screen; the rest wait until it's done so they don't compete with it."""
        if not keys:
            return
        wanted = set(keys)
        with self.lock:
            self.generation += 1
            generation = self.generation
            for key, f in list(self.futures.items()):
                if key not in wanted and f.cancel():
                    del self.futures[key]

        def neighbours(_):
            if generation == self.generation:  # the viewer hasn't moved on since
                for key in keys[1:]:
                    self.get(key)

        self.get(keys[0]).add_done_callback(neighbours)

    def result(self, key):
        while True:
            try:
                return self.get(key).result()
            except CancelledError:  # dropped by want() while we waited; queue it again
                continue


class Library:
    def __init__(self, folder: Path):
        self.folder = folder
        by_stem: dict[str, Path] = {}
        for entry in os.scandir(folder):  # unlike Path.is_file(), no stat per file (slow on /mnt)
            p = Path(entry.path)
            if p.suffix.lower() not in IMAGE_EXTS or not entry.is_file():
                continue
            # RAW+JPEG pairs in the same folder show once, as the raw.
            prev = by_stem.get(p.stem)
            if prev is None or p.suffix.lower() in RAW_EXTS:
                by_stem[p.stem] = p
        paths = sorted(by_stem.values(), key=lambda p: _natural_key(p.name))
        self.photos = {p.name: Photo(p) for p in paths}
        self.names = list(self.photos)
        self.clips = Prefetcher(lambda name: self.photos[name].clipping_mask(), workers=6, keep=10**6)
        self.previews = Prefetcher(
            lambda key: develop.preview(self.photos[key[0]], read_file(self.photos[key[0]].path), key[1]), workers=4, keep=12
        )
        # The export's input at full size, by (name, noise reduction on), for the editor's final
        # views: ~200 MB and 1.5-3 s of work each, for the photo on screen and the next (which
        # waits for it). Two at a time, so a change to the photo on screen needn't wait for the next.
        self.fulls = Prefetcher(
            lambda key: develop.full(self.photos[key[0]], read_file(self.photos[key[0]].path), key[1]), workers=2, keep=4
        )
        self.state_path = folder / STATE_FILE
        self.state = self._load()
        self._save_lock = threading.Lock()
        self._groups: list[int] | None = None
        self._taken: list[str | None] = []
        self.export = Export(self)

    def _load(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text())
        except FileNotFoundError:
            state = {}
        state.setdefault("version", 1)
        state.setdefault("marks", {})
        state.setdefault("edits", {})
        return state

    def _save(self):
        tmp = self.state_path.with_name(STATE_FILE + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.state_path)

    def set_mark(self, name: str, mark: str | None):
        with self._save_lock:
            if mark:
                self.state["marks"][name] = mark
            else:
                self.state["marks"].pop(name, None)
            self._save()

    def groups(self) -> list[int]:
        """Series number of every photo; reads all headers the first time (~1 s)."""
        if self._groups is None:
            with ThreadPoolExecutor(8) as ex:
                metas = list(ex.map(lambda p: p.meta(), self.photos.values()))
            self._taken = [m.get("taken") for m in metas]
            self._groups = edits.group_ids(metas)
        return self._groups

    def set_edit(self, name: str, settings: dict) -> dict:
        with self._save_lock:
            self.state["edits"] = edits.apply(self.names, self.groups(), self.state["edits"], name, settings)
            self._save()
            return self.runs()

    def settings(self) -> dict[str, dict]:
        return dict(zip(self.names, edits.resolve(self.names, self.groups(), self.state["edits"])))

    def full_key(self, name: str, noise: str | None = None):
        """fulls key for a photo: its noise setting as the editor has it ("auto" or a number),
        or as saved."""
        if noise is None:
            setting = self.settings()[name]["noise"]
        else:
            setting = None if noise == "auto" else float(noise)
        return name, denoise.active(setting, self.photos[name].iso)

    def runs(self) -> dict:
        """What the editor shows: which photos share settings, and those settings."""
        ids, settings = edits.runs(self.names, self.groups(), self.state["edits"])
        return {"runs": ids, "runSettings": settings, "runEdited": [s != develop.DEFAULTS for s in settings]}

    def prewarm(self, workers: int = 8):
        """Parse headers and build thumbnails in the background so the grid fills instantly."""
        q: queue.Queue[Photo] = queue.Queue()
        for p in self.photos.values():
            q.put(p)

        def work():
            while True:
                try:
                    q.get_nowait().thumbnail()
                except queue.Empty:
                    return
                except Exception as e:  # a broken file shouldn't stop the rest
                    print(f"  thumbnail failed: {e}", file=sys.stderr)

        threads = [threading.Thread(target=work, daemon=True) for _ in range(workers)]
        for t in threads:
            t.start()
        threading.Thread(target=lambda: ([t.join() for t in threads], self.groups()), daemon=True).start()


class Export:
    """Background export of the kept photos to <folder>/export."""

    def __init__(self, lib: Library):
        self.lib = lib
        self.lock = threading.RLock()
        self.status = {"running": False, "total": 0, "done": 0, "failed": [], "outDir": str(lib.folder / "export")}

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.status, "failed": list(self.status["failed"])}

    def start(self, workers: int = 6) -> dict:
        with self.lock:
            if self.status["running"]:
                return self.snapshot()
            settings = self.lib.settings()
            kept = [n for n in self.lib.names if self.lib.state["marks"].get(n) == "keep"]
            self.status.update(running=True, total=len(kept), done=0, failed=[])
        out_dir = Path(self.status["outDir"])

        def one(name):
            try:
                export_photo(self.lib.photos[name], settings[name], out_dir)
            except Exception as e:
                print(f"  export {name} failed: {e!r}", file=sys.stderr)
                with self.lock:
                    self.status["failed"].append(name)
            with self.lock:
                self.status["done"] += 1

        def run():
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(one, kept))
            with self.lock:
                self.status["running"] = False

        threading.Thread(target=run, daemon=True).start()
        return self.snapshot()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    lib: Library
    # Part of every image URL, so the browser can cache freely within a run but
    # never shows another folder's DSC_0001.NEF from an earlier one.
    version = format(int(time.time()), "x")

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        if path == "/":
            return self._static("index.html")
        if path.startswith("/static/"):
            return self._static(path.removeprefix("/static/"))
        if path == "/api/state":
            return self._json(
                {
                    "folder": str(self.lib.folder),
                    "version": self.version,
                    "photos": self.lib.names,
                    "marks": self.lib.state["marks"],
                }
            )
        if path == "/api/edits":
            return self._json(
                {
                    "groups": self.lib.groups(),
                    "taken": self.lib._taken,
                    **self.lib.runs(),
                    "raw": [p.is_raw for p in self.lib.photos.values()],
                    "defaults": develop.DEFAULTS,
                    "limits": develop.LIMITS,
                    "autoNoiseIso": denoise.AUTO_ISO,
                    "tempStep": develop.TEMP_STEP,
                }
            )
        if path == "/api/export":
            return self._json(self.lib.export.snapshot())
        kind, _, name = path.lstrip("/").partition("/")
        photo = self.lib.photos.get(name)
        if photo is None:
            return self.send_error(HTTPStatus.NOT_FOUND)
        w = int(query.get("w", ["0"])[0])
        try:
            if kind == "img":
                return self._send(photo.jpeg(w), "image/jpeg", cache=True)
            if kind == "thumb":
                return self._send([photo.thumbnail()], "image/jpeg", cache=True)
            if kind == "meta":
                return self._json(photo.meta(), cache=True)
            if kind in ("clip", "develop", "full") and not photo.is_raw:
                return self.send_error(HTTPStatus.NOT_FOUND, "not a raw file")
            if kind == "clip":
                self.lib.clips.result(name)
                return self._send([photo.clipping(w)], "image/png", cache=True)
            if kind == "develop" and "final" in query:
                full = self.lib.fulls.get(self.lib.full_key(name, query.get("noise", [None])[0])).result()
                return self._send([develop.full_preview(full, preview_size(w))], "application/octet-stream")
            if kind == "develop":
                data = self.lib.previews.result((name, preview_size(w)))
                return self._send([data], "application/octet-stream")
            if kind == "full":
                i, j = (int(v) for v in query["t"][0].split(","))
                full = self.lib.fulls.get(self.lib.full_key(name, query.get("noise", [None])[0])).result()
                return self._send([full.tile(i, j)], "application/octet-stream")
        except CancelledError:  # the editor moved on while we waited (and won't read this)
            return self.send_error(HTTPStatus.CONFLICT, "dropped")
        except ConnectionError:
            raise
        except Exception as e:
            print(f"  {name}: {e!r}", file=sys.stderr)
            return self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        photos = self.lib.photos
        raws = [n for n in body.get("names", []) if n in photos and photos[n].is_raw]
        if self.path == "/api/clip-hint":
            self.lib.clips.want(raws)
            return self._json({"ok": True})
        if self.path == "/api/develop-hint":
            # names[0] is on screen, then the next, then others nearby; noise: their settings
            size = preview_size(int(body.get("w", 0)))
            self.lib.previews.want([(n, size) for n in raws])
            noise = [str(v) for v in body.get("noise", [])] + [None, None]
            self.lib.fulls.want([self.lib.full_key(n, v) for n, v in zip(raws[:2], noise)])
            return self._json({"ok": True})
        if self.path == "/api/edit":
            name = body.get("name")
            if name not in photos:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            return self._json(self.lib.set_edit(name, body.get("settings") or {}))
        if self.path == "/api/export":
            return self._json(self.lib.export.start())
        if self.path != "/api/mark":
            return self.send_error(HTTPStatus.NOT_FOUND)
        name, mark = body.get("name"), body.get("mark")
        if name not in photos or (mark is not None and mark not in MARKS):
            return self.send_error(HTTPStatus.BAD_REQUEST)
        self.lib.set_mark(name, mark)
        self._json({"ok": True})

    def _static(self, name: str):
        f = (WEB / name).resolve()
        if f.parent != WEB.resolve() or not f.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        self._send([f.read_bytes()], STATIC_TYPES.get(f.suffix, "application/octet-stream"))

    def _json(self, obj, cache=False):
        self._send([json.dumps(obj).encode()], "application/json", cache)

    def _send(self, chunks: list[bytes], ctype: str, cache=False):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(sum(len(c) for c in chunks)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        for c in chunks:
            self.wfile.write(c)


def preview_size(w: int) -> int:
    """Editor preview long side: what the browser asked for, in steps of 256, at most half-size raw."""
    return min(max(-(-w // 256) * 256, 1024), 3072)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # The browser cancels prefetches it no longer needs; that's not an error.
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def open_browser(url: str):
    try:
        if "microsoft" in Path("/proc/version").read_text().lower():
            subprocess.Popen(["explorer.exe", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    except OSError:
        pass
    import webbrowser

    webbrowser.open(url)


def serve(folder: Path, port: int, browser: bool, mode: str = "cull"):
    lib = Library(folder)
    if not lib.photos:
        sys.exit(f"No images found in {folder}")
    Handler.lib = lib
    server = Server(("127.0.0.1", port), Handler)
    url = f"http://localhost:{server.server_address[1]}/#{mode}"
    kept = sum(m == "keep" for m in lib.state["marks"].values())
    print(f"{len(lib.photos)} photos in {folder} ({kept} kept so far)")
    print(f"Open {url}  (Ctrl+C to stop)")
    lib.prewarm()
    if browser:
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        for p in (lib.clips, lib.previews, lib.fulls):
            p.pool.shutdown(wait=False, cancel_futures=True)

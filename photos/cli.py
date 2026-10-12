import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(prog="photos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help in [("cull", "pick which photos to keep"), ("edit", "develop the kept photos")]:
        p = sub.add_parser(name, help=help)
        p.add_argument("folder", type=Path)
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    p = sub.add_parser("export", help="develop the kept photos to <folder>/export as JPEG")
    p.add_argument("folder", type=Path)

    args = ap.parse_args()
    folder = args.folder.resolve()
    if args.cmd in ("cull", "edit"):
        from .server import serve

        serve(folder, args.port, not args.no_browser, mode=args.cmd)
    elif args.cmd == "export":
        from .server import Library

        lib = Library(folder)
        status = lib.export.start()
        print(f"Exporting {status['total']} photos to {status['outDir']}")
        while status["running"]:
            time.sleep(0.5)
            status = lib.export.snapshot()
            print(f"\r  {status['done']} / {status['total']}", end="", flush=True)
        print()
        if status["failed"]:
            sys.exit(f"Failed: {', '.join(status['failed'])}")

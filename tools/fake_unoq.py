"""Stand-in for the UNO Q: accepts AirPen frames on :7000/airpen and draws them in the terminal.

    .venv/bin/python tools/fake_unoq.py          # then run AirPen with --unoq localhost
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SHADES = " .:-=+*#%@"


class Handler(BaseHTTPRequestHandler):
    frames = 0

    def do_POST(self):
        if self.path != "/airpen":
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        frame = body.get("frame", [])
        ok = isinstance(frame, list) and len(frame) == 104
        if ok:
            Handler.frames += 1
            rows = ["".join(SHADES[min(9, v * 10 // 256)] * 2 for v in frame[r * 13:(r + 1) * 13]) for r in range(8)]
            sys.stdout.write("\x1b[H\x1b[2J" + f"fake UNO Q  frame {Handler.frames}  state {body.get('state')}\n"
                             + "+" + "-" * 26 + "+\n" + "\n".join(f"|{r}|" for r in rows) + "\n+" + "-" * 26 + "+\n")
            sys.stdout.flush()
        out = json.dumps({"ok": ok}).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7000
    print(f"fake UNO Q listening on :{port}")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()

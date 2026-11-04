"""Link to the Arduino UNO Q over Wi-Fi.

The UNO Q app (unoq_app/) runs App Lab's WebUI brick, which serves HTTP on port 7000. We POST
the latest LED frame plus a small status dict to /airpen. A background thread sends at a fixed
rate and always sends only the newest frame, so a slow network never stalls the vision loop.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np


class BoardLink:
    def __init__(self, host: str, rate_hz: float = 20.0, timeout: float = 0.4):
        if ":" not in host:
            host = f"{host}:7000"
        self.url = f"http://{host}/airpen"
        self.period = 1.0 / rate_hz
        self.timeout = timeout
        self.status = "connecting"
        self.rtt_ms = 0.0
        self._payload: bytes | None = None
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._running = True
        self._fails = 0
        self._thread = threading.Thread(target=self._loop, name="unoq", daemon=True)
        self._thread.start()

    def send(self, frame: np.ndarray, state: dict) -> None:
        body = json.dumps({"frame": frame.reshape(-1).tolist(), "state": state}).encode()
        with self._lock:
            self._payload = body
        self._new.set()

    def _loop(self) -> None:
        while self._running:
            if not self._new.wait(0.5):
                continue
            self._new.clear()
            with self._lock:
                body = self._payload
            start = time.perf_counter()
            try:
                req = urllib.request.Request(self.url, data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp.read()
                self.rtt_ms = (time.perf_counter() - start) * 1000
                self.status = "online"
                self._fails = 0
            except Exception:
                self._fails += 1
                self.status = "offline" if self._fails >= 3 else self.status
                time.sleep(min(2.0, 0.2 * self._fails))  # back off while the board is unreachable
            spare = self.period - (time.perf_counter() - start)
            if spare > 0:
                time.sleep(spare)

    def close(self) -> None:
        self._running = False
        self._new.set()
        self._thread.join(timeout=1.0)

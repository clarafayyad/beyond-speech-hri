import os
import threading
import time
import socket
from http.server import HTTPServer, SimpleHTTPRequestHandler
from sic_framework.devices import Pepper
from sic_framework.devices.common_pepper.pepper_tablet import (
    UrlMessage,
    ClearDisplayMessage,
)
from sic_framework.core import utils

DISPLAY_TIME = 5

class _ImageHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler for Pepper-safe static serving."""

    def log_error(self, format, *args):
        if "favicon.ico" in str(args) or "Broken pipe" in str(args):
            return
        super().log_error(format, *args)

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()


class PepperTabletDisplayService:
    """
    Service that:
    - starts a local HTTP server once
    - serves an idle screen by default
    - allows updating the displayed image
    - controls Pepper tablet reloads
    """

    def __init__(
        self,
        pepper: Pepper,
        port: int = 8000,
    ):
        self.pepper = pepper
        self.port = port

        self.work_dir = os.path.dirname(os.path.abspath(__file__))

        self.index_html = os.path.join(self.work_dir, "index.html")

        os.makedirs(self.work_dir, exist_ok=True)
        # os.chdir(self.work_dir)

        self._create_idle_screen()
        self._start_server()
        self.server_url = f"http://{utils.get_ip_adress()}:{self.port}"

        self._reload_tablet()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def show_idle(self):
        """Show idle screen on Pepper."""
        self._create_idle_screen()
        self._reload_tablet()

    def show_image(self, image_path: str):
        """
        Update the displayed image.

        image_path: path to an image file (any size/format)
        """
        image_path = os.path.join("images/", image_path)
        self._create_image_page(image_path)
        self._reload_tablet()

    def clear(self):
        """Clear Pepper tablet."""
        self.pepper.tablet.send_message(ClearDisplayMessage())

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _start_server(self):
        """Start HTTP server in background thread and block until ready."""

        server_ready = threading.Event()
        from functools import partial
        def _serve():
            handler = partial(
                _ImageHTTPRequestHandler,
                directory=self.work_dir
            )
            self._httpd = HTTPServer(("", self.port), handler)

            # 🔔 Socket is now bound and listening
            server_ready.set()

            self._httpd.serve_forever()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()

        # ⏳ Block until server thread signals readiness
        if not server_ready.wait(timeout=5.0):
            raise RuntimeError(
                f"Pepper tablet display server failed to start on port {self.port}"
            )

    def _reload_tablet(self):
        """Reload the tablet page."""
        self.pepper.tablet.send_message(UrlMessage(self.server_url))
        time.sleep(DISPLAY_TIME)

    def clear_display(self):
        self._create_idle_screen()
        self._reload_tablet()

    def _create_idle_screen(self):
        self._create_image_page("images/codenames.png")

    def _create_image_page(self, image_path: str):
        """Create HTML page showing current image."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            overflow: hidden;
        }}
        img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
    </style>
</head>
<body>
    <img src="{image_path}" alt="Displayed Image">
</body>
</html>
"""
        with open(self.index_html, "w") as f:
            f.write(html)


if __name__ == "__main__":
    pepper = Pepper(ip="10.0.0.168")
    display = PepperTabletDisplayService(pepper)
    display.show_idle()
    display.show_image("3.png")
    display.clear()
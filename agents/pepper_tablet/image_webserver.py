"""
Simple Image Web Server

This is a standalone web server that serves and displays an image file.

Usage:
    python image_webserver.py

Requirements:
    - demo_image.jpg (or png) in the same directory
    - You can replace the image file by changing the IMAGE_FILE variable.
"""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

from sic_framework.core import utils
from PIL import Image

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
HTTP_PORT = 8000
IMAGE_FILE = "demo_image.png"  # jpg / png / gif


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Handler
# ─────────────────────────────────────────────────────────────────────────────
class ImageHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Basic HTTP handler for serving images."""

    def log_error(self, format, *args):
        """Suppress favicon and broken pipe errors."""
        if "favicon.ico" in str(args) or "Broken pipe" in str(args):
            return
        super().log_error(format, *args)

    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def create_image_html(image_filename):
    """Create an HTML page that displays a centered image."""
    return f"""<!DOCTYPE html>
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
    <img src="{image_filename}" alt="Displayed Image">
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Start the image web server."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.exists(IMAGE_FILE):
        print(f"ERROR: Image file '{IMAGE_FILE}' not found!")
        return

    file_size_mb = os.path.getsize(IMAGE_FILE) / (1024 * 1024)
    local_ip = utils.get_ip_adress()

    # Create index.html
    with open("index.html", "w") as f:
        f.write(create_image_html(IMAGE_FILE))

    print("=" * 70)
    print("Simple Image Web Server")
    print("=" * 70)
    print(f"Image file: {IMAGE_FILE}")
    print(f"Image size: {file_size_mb:.2f} MB")
    print("")
    print(f"Server running on port {HTTP_PORT}")
    print("")
    print("Access the image:")
    print(f"  Local:   http://localhost:{HTTP_PORT}")
    print(f"  Network: http://{local_ip}:{HTTP_PORT}")
    print("")
    print("Direct image URL:")
    print(f"  http://{local_ip}:{HTTP_PORT}/{IMAGE_FILE}")
    print("=" * 70)

    server = HTTPServer(("", HTTP_PORT), ImageHTTPRequestHandler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


if __name__ == "__main__":
    main()

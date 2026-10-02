# Run guide

1. Install Python 3.10 or newer.
2. From this folder, run `python -m pip install -r requirements.txt`.
3. Run `python app.py`; stop with Ctrl+C.
4. Open `http://127.0.0.1:8765`.
5. For LAN access, use the laptop's private IP address and port 8765 on a trusted network.

Set `config/camera.yaml` device_id, width, and height. For LAN-only deployment set an appropriate network interface via `--host 192.168.x.x`; default `0.0.0.0` exposes the unauthenticated dashboard to reachable interfaces. Do not forward the port to the internet.

API: `/api/status`, `/api/tracks`, `/api/events`; live camera: `/video.mjpg`.

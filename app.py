"""Offline Vision-X local prototype. Run with: python app.py"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def config(name: str) -> dict:
    p = ROOT / "config" / name
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    return inter / max(1, aw*ah + bw*bh - inter)


class Tracker:
    """Greedy IoU tracker; IDs are session-local and may switch on crossings."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = int(config("tracker.yaml").get("max_lost_frames", 12))

    def update(self, detections):
        ids = list(self.tracks)
        pairs = sorted(((iou(self.tracks[k]["bbox"], d["bbox"]), k, j)
                        for k in ids for j, d in enumerate(detections)), reverse=True)
        used_t, used_d = set(), set()
        for score, k, j in pairs:
            if score < .15 or k in used_t or j in used_d: continue
            t, d = self.tracks[k], detections[j]
            old = t["bbox"]
            ox, oy = old[0]+old[2]/2, old[1]+old[3]/2
            nx, ny = d["bbox"][0]+d["bbox"][2]/2, d["bbox"][1]+d["bbox"][3]/2
            t.update(d); t["velocity_px_s"] = [round((nx-ox)*self.fps, 1), round((ny-oy)*self.fps, 1)]
            t["missed"] = 0; t["age"] += 1; used_t.add(k); used_d.add(j)
        for k in ids:
            if k not in used_t: self.tracks[k]["missed"] += 1
        for j, d in enumerate(detections):
            if j not in used_d:
                k = self.next_id; self.next_id += 1
                self.tracks[k] = {**d, "id": k, "age": 1, "missed": 0, "velocity_px_s": [0,0], "born": time.time()}
        self.tracks = {k:v for k,v in self.tracks.items() if v["missed"] <= self.max_lost}
        return [dict(v, state="LOST" if v["missed"] else ("NEW" if v["age"] == 1 else "TRACKING")) for v in self.tracks.values()]


class VisionApp:
    def __init__(self, device=0, mode="human"):
        self.device, self.mode = device, mode
        self.cap = None
        self.lock = threading.RLock()
        self.frame = None
        self.tracks = []
        self.events = deque(maxlen=200)
        self.industrial = None
        try:
            from industrial import IndustrialConfig, IndustrialMonitor
            self.industrial = IndustrialMonitor(IndustrialConfig.load(ROOT / "config" / "industrial.yaml"))
        except Exception as exc:
            self.events.append({"time": now(), "type": "industrial_config_error", "message": str(exc)})
        self.fps = 0.0; self.latency_ms = 0.0; self.processing_ms = 0.0
        self.camera_error = "OpenCV is not installed" if cv2 is None else "Camera not started"
        self.running = False
        self.tracker = Tracker(); self.tracker.fps = 15
        self.hog = None
        self.custom_detector = None
        self.detector_error = ""
        if cv2 is not None:
            try:
                self.hog = cv2.HOGDescriptor(); self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            except AttributeError as exc:
                self.detector_error = f"OpenCV build lacks the HOG people baseline: {exc}"
        det_cfg = config("detector.yaml")
        if det_cfg.get("backend") == "visionx_custom_grid":
            try:
                from visionx.detection import CustomGridDetector
                classes_path = ROOT / det_cfg.get("classes", "python/datasets/classes.txt")
                classes = [x.strip() for x in classes_path.read_text(encoding="utf-8").splitlines() if x.strip()]
                self.custom_detector = CustomGridDetector(str(ROOT / det_cfg.get("checkpoint", "models/detector.pt")), classes, float(det_cfg.get("confidence_threshold", .35)), float(det_cfg.get("nms_iou_threshold", .45)), str(det_cfg.get("device", "cpu")))
                self.hog = None
            except Exception as exc:
                self.detector_error = str(exc)
                self.hog = None
        self.homography = None
        self._load_homography()

    def _load_homography(self):
        if cv2 is None: return
        c = config("localization.yaml")
        pts = c.get("image_points", [])
        world = c.get("world_points_m", [])
        if len(pts) == len(world) and len(pts) >= 4:
            import numpy as np
            self.homography = cv2.getPerspectiveTransform(np.array(pts[:4], dtype="float32"), np.array(world[:4], dtype="float32"))

    def start(self):
        self.running = True
        if cv2 is None: return
        self.cap = cv2.VideoCapture(self.device)
        cam = config("camera.yaml")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam.get("width", 640)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam.get("height", 480)))
        if not self.cap.isOpened():
            self.camera_error = f"Could not open camera device {self.device}"; self.cap.release(); self.cap = None; return
        self.camera_error = ""
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = time.perf_counter()
        while self.running and self.cap:
            tick = time.perf_counter()
            ok, frame = self.cap.read()
            if not ok:
                self.camera_error = "Camera read failed"; time.sleep(.1); continue
            start = time.perf_counter()
            detections = []
            if self.custom_detector is not None:
                detections = [dict(class_id=d.class_id, class_name=d.class_name, confidence=round(float(d.confidence),3), bbox=[int(v) for v in d.bounding_box]) for d in self.custom_detector.detect(frame, time.time())]
            elif self.mode == "human" and self.hog is not None:
                rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
                detections = [dict(class_id=0, class_name="person", confidence=round(float(weights[i]), 3), bbox=list(map(int, r))) for i,r in enumerate(rects)]
            self.tracker.fps = 1 / max(.001, time.perf_counter()-prev); prev = time.perf_counter()
            tracks = self.tracker.update(detections)
            for t in tracks:
                x,y,w,h=t["bbox"]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(70,220,150),2)
                cv2.putText(frame,f"Person-{t['id']:03d} {t['confidence']:.2f}",(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(70,220,150),2)
            self.processing_ms=(time.perf_counter()-start)*1000
            self.fps=.9*self.fps+.1/(time.perf_counter()-tick) if self.fps else 1/(time.perf_counter()-tick)
            with self.lock:
                self.frame=frame; self.tracks=tracks; self.camera_error=""

    def track_payload(self):
        with self.lock:
            result=[]
            for t in self.tracks:
                x,y,w,h=t["bbox"]; px=x+w/2; py=y+h
                coord=self.localize(px,py)
                vx,vy=t["velocity_px_s"]
                result.append({"id":f"{self.mode.title()}-{t['id']:03d}","class":t["class_name"],"class_name":t["class_name"],"confidence":t["confidence"],"bbox":t["bbox"],"x_m":coord[0],"y_m":coord[1],"world_x":coord[0],"world_y":coord[1],"speed_px_s":round(math.hypot(vx,vy),1),"direction":self.direction(vx,vy),"state":t["state"],"age_frames":t["age"],"last_seen":now()})
            if self.mode == "industrial" and self.industrial is not None:
                raw = [{"id": t["id"], "class_name": t["class_name"], "x_m": self.localize(t["bbox"][0]+t["bbox"][2]/2, t["bbox"][1]+t["bbox"][3])[0], "y_m": self.localize(t["bbox"][0]+t["bbox"][2]/2, t["bbox"][1]+t["bbox"][3])[1], "state": t["state"]} for t in self.tracks]
                self.industrial.update(raw)
                self.events = deque(self.industrial.events(), maxlen=200)
            return result

    def localize(self,x,y):
        if cv2 is None or self.homography is None: return None,None
        import numpy as np
        p=cv2.perspectiveTransform(np.array([[[x,y]]],dtype="float32"),self.homography)[0][0]
        return round(float(p[0]),2),round(float(p[1]),2)

    @staticmethod
    def direction(vx,vy):
        if math.hypot(vx,vy)<8:return "STATIONARY"
        return ("S" if vy>0 else "N")+ ("E" if vx>0 else "W")

    def status(self):
        online=bool(self.cap and self.cap.isOpened() and not self.camera_error)
        det_name="custom trained detector" if self.custom_detector else ("OpenCV HOG people baseline" if self.hog else (self.detector_error or "Unavailable (install opencv-python)"))
        detector_active=self.custom_detector is not None or (self.hog is not None and self.mode=="human")
        return {"product":"OFFLINE VISION-X","mode":self.mode,"camera_online":online,"camera_status":"online" if online else "unavailable","camera_error":self.camera_error,"camera_message":self.camera_error or "Camera connected","detector":det_name if detector_active else ("No industrial detector configured" if self.mode=="industrial" else "Unavailable (install opencv-python)"),"model_status":"CUSTOM TRAINED MODEL" if self.custom_detector else ("BASELINE" if self.hog and self.mode=="human" else "NO DETECTOR"),"fps":round(self.fps,1),"processing_ms":round(self.processing_ms,1),"latency_ms":round(self.processing_ms,1),"active_tracks":len(self.tracks),"detection_count":len(self.tracks),"localization":"calibrated homography" if self.homography is not None else "not calibrated","localization_status":"Calibrated camera plane" if self.homography is not None else "Positions unavailable · camera calibration required","room_width_m":config("localization.yaml").get("room_width_m",10),"room_height_m":config("localization.yaml").get("room_height_m",8),"offline":True}


class Handler(BaseHTTPRequestHandler):
    app: VisionApp
    def log_message(self,*args): pass
    def send_json(self,obj):
        data=json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(data))); self.send_header("Access-Control-Allow-Origin","*"); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/status": return self.send_json(self.app.status())
        if path=="/api/tracks": return self.send_json(self.app.track_payload())
        if path=="/api/events": return self.send_json(list(self.app.events))
        if path=="/api/mode": return self.send_json({"mode":self.app.mode})
        if path=="/video.mjpg":
            if cv2 is None or self.app.cap is None or self.app.camera_error:
                self.send_error(503, "CAMERA UNAVAILABLE"); return
            self.send_response(200); self.send_header("Cache-Control","no-store"); self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame"); self.end_headers()
            while True:
                if self.app.camera_error or self.app.cap is None: break
                with self.app.lock: frame=self.app.frame.copy() if self.app.frame is not None else None
                if frame is None:
                    time.sleep(.25); continue
                ok,buf=cv2.imencode(".jpg",frame)
                if not ok: continue
                try: self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"); time.sleep(.05)
                except (BrokenPipeError,ConnectionResetError): break
            return
        if path.startswith("/dashboard/"):
            name=Path(path.removeprefix("/dashboard/")).name
            if name in {"app.js","styles.css"}:
                file=ROOT/"dashboard"/name
                if file.exists():
                    data=file.read_bytes(); self.send_response(200); self.send_header("Content-Type","text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
        if path in ("/",""):
            file=ROOT/"dashboard"/"index.html"
            if file.exists():
                data=file.read_bytes(); self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data); return
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/mode":
            self.send_error(404); return
        try:
            length=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(length) or b"{}")
            mode=body.get("mode")
            if mode not in ("human","industrial"):
                self.send_error(400,"mode must be human or industrial"); return
            self.app.mode=mode
            self.send_json({"mode":mode,"model_status":"BASELINE" if mode=="human" and self.app.hog else "NO DETECTOR"})
        except (ValueError,TypeError): self.send_error(400,"invalid JSON")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default=None); ap.add_argument("--port",type=int,default=8765); ap.add_argument("--camera",type=int,default=None); ap.add_argument("--mode",choices=["human","industrial"],default=None)
    args=ap.parse_args(); nc=config("network.yaml"); cc=config("camera.yaml")
    app=VisionApp(args.camera if args.camera is not None else int(cc.get("device_id",0)), args.mode or config("system.yaml").get("mode","human"))
    Handler.app=app; app.start()
    host=args.host or nc.get("host","0.0.0.0")
    server=ThreadingHTTPServer((host,args.port),Handler)
    print(f"Offline Vision-X: http://127.0.0.1:{args.port}  (LAN bind {host}; camera: {app.camera_error or 'online'})")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        app.running=False; server.server_close()
        if app.cap: app.cap.release()

if __name__=="__main__": main()

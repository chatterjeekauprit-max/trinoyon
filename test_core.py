import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from app import Tracker, iou, VisionApp
from industrial import IndustrialConfig, IndustrialMonitor


class CoreTests(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0,0,10,10],[5,0,10,10]), 1/3)

    def test_tracker_recovers_same_id(self):
        tracker=Tracker(); tracker.fps=10
        det=lambda x:{"bbox":[x,0,20,40],"class_id":0,"class_name":"person","confidence":.8}
        first=tracker.update([det(10)])[0]["id"]
        track=tracker.update([det(12)])[0]
        self.assertEqual(first, track["id"])
        self.assertEqual(track["state"], "TRACKING")

    def test_industrial_zone_event(self):
        cfg=IndustrialConfig.from_mapping({"zones":[{"name":"A","polygon":[[0,0],[2,0],[2,2],[0,2]]}], "rules_enabled":["zone_entry"]})
        engine=IndustrialMonitor(cfg)
        events=engine.update([{"id":"c1","class_name":"component","x_m":1,"y_m":1}])
        self.assertTrue(any(e.rule == "zone_entry" for e in events))

    def test_camera_absence_is_explicit(self):
        app=VisionApp()
        status=app.status()
        self.assertTrue(status["offline"])
        if not status["camera_online"]:
            self.assertTrue(status["camera_error"] or status["detector"] == "Unavailable (install opencv-python)")


if __name__ == "__main__": unittest.main()

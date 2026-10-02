"""Rule-based industrial event engine. Coordinates use the local map frame."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Tuple
import json

Point = Tuple[float, float]

@dataclass(frozen=True)
class Zone:
    name: str
    polygon: Tuple[Point, ...]
    def contains(self, point: Point) -> bool:
        x, y = point
        inside = False
        for i, (x1, y1) in enumerate(self.polygon):
            x2, y2 = self.polygon[(i + 1) % len(self.polygon)]
            cross = (x-x1)*(y2-y1) - (y-y1)*(x2-x1)
            if abs(cross) < 1e-9 and min(x1,x2)-1e-9 <= x <= max(x1,x2)+1e-9 and min(y1,y2)-1e-9 <= y <= max(y1,y2)+1e-9:
                return True
            if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
                inside = not inside
        return inside

@dataclass(frozen=True)
class IndustrialConfig:
    zones: Tuple[Zone, ...] = ()
    allowed_classes: Tuple[str, ...] = ()
    allowed_zones_by_class: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    expected_counts: Mapping[str, int] = field(default_factory=dict)
    missing_after_seconds: float = 3.0
    dwell_limit_seconds: Mapping[str, float] = field(default_factory=dict)
    max_events: int = 500
    rules_enabled: Tuple[str, ...] = ("zone_entry", "zone_exit", "excessive_dwell", "wrong_zone", "unexpected_object", "count_exceeded", "missing_object")
    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "IndustrialConfig":
        zones = tuple(Zone(str(z["name"]), tuple((float(p[0]),float(p[1])) for p in z["polygon"])) for z in raw.get("zones", []))
        dwell = {str(k):float(v) for k,v in raw.get("dwell_limit_seconds",{}).items()}
        if "dwell_seconds" in raw:
            dwell.update({z.name:float(raw["dwell_seconds"]) for z in zones if z.name not in dwell})
        enabled = tuple({"dwell":"excessive_dwell", "excess_count":"count_exceeded"}.get(str(r),str(r)) for r in raw.get("rules_enabled", cls.__dataclass_fields__["rules_enabled"].default))
        return cls(zones, tuple(raw.get("allowed_classes", [])), {str(k):tuple(v) for k,v in raw.get("allowed_zones_by_class",{}).items()}, {str(k):int(v) for k,v in raw.get("expected_counts",{}).items()}, float(raw.get("missing_after_seconds",3)), dwell, int(raw.get("max_events",500)), enabled)
    @classmethod
    def load(cls, path: str | Path) -> "IndustrialConfig":
        # JSON syntax is a YAML subset; this loader has no PyYAML dependency.
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8-sig")))

@dataclass
class IndustrialEvent:
    timestamp: str
    rule: str
    severity: str
    message: str
    track_id: Optional[str] = None
    class_name: Optional[str] = None
    zone: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    def as_dict(self) -> Dict[str, Any]:
        return {"timestamp":self.timestamp,"rule":self.rule,"severity":self.severity,"message":self.message,"track_id":self.track_id,"class_name":self.class_name,"zone":self.zone,"details":self.details}

class IndustrialMonitor:
    """Call update(tracks, timestamp_seconds) each frame; accepts dicts or objects.

    Tracks use id/track_id, class_name, world_x/world_y (or x/y), state.
    Rules are explicit heuristics; no trained anomaly/defect model is implied.
    """
    def __init__(self, config: IndustrialConfig):
        self.config=config
        self.history: Deque[IndustrialEvent]=deque(maxlen=max(1,config.max_events))
        self._track_zone: Dict[str,Optional[str]]={}
        self._zone_since: Dict[Tuple[str,str],float]={}
        self._active: set[Tuple[str,str]] = set()
        self._started: Optional[float]=None
    @staticmethod
    def _get(obj: Any, *names: str, default: Any=None) -> Any:
        for n in names:
            if isinstance(obj,Mapping) and n in obj: return obj[n]
            if hasattr(obj,n): return getattr(obj,n)
        return default
    def _emit(self, out: List[IndustrialEvent], now: float, rule: str, key: str, message: str, **kw: Any) -> None:
        k=(rule,key)
        if k in self._active: return
        self._active.add(k)
        if rule not in self.config.rules_enabled: return
        e=IndustrialEvent(datetime.fromtimestamp(now,timezone.utc).isoformat(),rule,kw.pop("severity","info"),message,**kw)
        out.append(e); self.history.append(e)
    def update(self, tracks: Iterable[Any], timestamp: Optional[float]=None) -> List[IndustrialEvent]:
        now=float(timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp())
        if self._started is None: self._started=now
        out: List[IndustrialEvent]=[]; counts: Dict[str,int]={}; conditions=set()
        for tr in tracks:
            tid=str(self._get(tr,"track_id","id",default=""))
            if not tid: continue
            if str(self._get(tr,"state",default="TRACKING")).upper() in ("LOST","REMOVED"): continue
            cls=str(self._get(tr,"class_name","class",default="unknown")); counts[cls]=counts.get(cls,0)+1
            x=self._get(tr,"world_x","x_m","x"); y=self._get(tr,"world_y","y_m","y")
            zone=None
            if x is not None and y is not None:
                zone=next((z.name for z in self.config.zones if z.contains((float(x),float(y)))),None)
            prior=self._track_zone.get(tid)
            if prior != zone:
                if prior: self._emit(out,now,"zone_exit",tid+":"+prior,f"{tid} exited {prior}",track_id=tid,class_name=cls,zone=prior)
                if zone: self._emit(out,now,"zone_entry",tid+":"+zone,f"{tid} entered {zone}",track_id=tid,class_name=cls,zone=zone)
                self._track_zone[tid]=zone
                for k in [k for k in self._zone_since if k[0]==tid]: self._zone_since.pop(k,None)
                if zone: self._zone_since[(tid,zone)]=now
            if zone:
                self._zone_since.setdefault((tid,zone),now)
                limit=self.config.dwell_limit_seconds.get(zone)
                if limit is not None and now-self._zone_since[(tid,zone)] >= limit:
                    k=("dwell",tid); conditions.add(k)
                    self._emit(out,now,"excessive_dwell",tid,f"{tid} exceeded dwell limit in {zone}",severity="warning",track_id=tid,class_name=cls,zone=zone,details={"dwell_seconds":now-self._zone_since[(tid,zone)],"limit_seconds":limit})
            allowed=self.config.allowed_zones_by_class.get(cls)
            if zone and allowed is not None and zone not in allowed:
                k=("wrong_zone",tid); conditions.add(k)
                self._emit(out,now,"wrong_zone",tid,f"{tid} ({cls}) observed in disallowed zone {zone}",severity="warning",track_id=tid,class_name=cls,zone=zone,details={"allowed_zones":list(allowed)})
            if self.config.allowed_classes and cls not in self.config.allowed_classes:
                k=("unexpected_object",tid); conditions.add(k)
                self._emit(out,now,"unexpected_object",tid,f"Unexpected class {cls} detected as {tid}",severity="warning",track_id=tid,class_name=cls,zone=zone)
        for cls,expected in self.config.expected_counts.items():
            actual=counts.get(cls,0)
            for rule,condition,msg in (("count_exceeded",actual>expected,f"{cls} count {actual} exceeds expected {expected}"),("missing_object",actual<expected and now-self._started>=self.config.missing_after_seconds,f"Expected {expected} {cls} object(s), observed {actual}")):
                k=(rule,cls)
                if condition:
                    conditions.add(k); self._emit(out,now,rule,cls,msg,severity="warning",class_name=cls,details={"actual":actual,"expected":expected})
        # Re-arm transition alerts after a condition clears.
        self._active={k for k in self._active if k[0] not in ("excessive_dwell","wrong_zone","unexpected_object","missing_object","count_exceeded") or k in conditions}
        return out
    def events(self, limit: Optional[int]=None) -> List[Dict[str,Any]]:
        items=list(self.history)
        if limit is not None: items=items[-max(0,int(limit)):]
        return [e.as_dict() for e in items]

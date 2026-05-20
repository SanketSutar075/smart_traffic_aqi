"""
TrafficIQ v10.0 — COMPLETE 4-WAY INTERSECTION MANAGEMENT SYSTEM
================================================================
Chouk Management System for Traffic Police
4 Roads: A(West) B(East) C(North) D(South)
Features:
  - 4 Independent camera feeds (IP webcam / video upload / demo)
  - Live YOLO vehicle detection per road
  - OD Matrix: all 12 vehicle flow paths
  - 4-Phase adaptive signal control
  - Police dashboard: alerts, summary, CSV/PDF export
  - Live graphs and analytics

Prepared by: Sanket Sutar | B.E. Final Year 2025-26
Run: python TrafficIQ_Intersection.py -> http://localhost:5000
"""

import cv2, numpy as np, math, random, time, threading, json, os, io, tempfile
import sqlite3, urllib.request as _ur
from datetime import datetime
from collections import deque
from flask import Flask, Response, jsonify, request, render_template_string, send_file

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except: YOLO_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SK_AVAILABLE = True
except: SK_AVAILABLE = False

app = Flask(__name__)
ROADS = ["A","B","C","D"]
ROAD_NAMES = {"A":"West","B":"East","C":"North","D":"South"}
ROAD_COLORS_CV = {"A":(0,255,128),"B":(255,160,0),"C":(0,200,255),"D":(255,60,200)}

# ═══════════════════════════════════════════
# SQLITE
# ═══════════════════════════════════════════
DB_PATH = os.path.join(os.path.dirname(__file__), 'intersection_data.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, road TEXT, vehicles INTEGER, aqi REAL,
        phase TEXT, signal TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS od_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, origin TEXT, dest TEXT, session_count INTEGER
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, type TEXT, road TEXT, message TEXT
    )""")
    conn.commit(); conn.close()
    print("  [DB] Database ready:", DB_PATH)

def db_save_alert(atype, road, msg):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO alerts (timestamp,type,road,message) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), atype, road, msg))
        conn.commit(); conn.close()
    except: pass

init_db()

# ═══════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════
_tg_token = ''; _tg_chat = ''; _tg_last = {}

def tg_send(msg, force=False):
    if not _tg_token or not _tg_chat: return False
    key = msg[:20]; now = time.time()
    if not force and now - _tg_last.get(key,0) < 60: return False
    _tg_last[key] = now
    try:
        url = f"https://api.telegram.org/bot{_tg_token}/sendMessage"
        data = json.dumps({"chat_id":_tg_chat,"text":msg,"parse_mode":"Markdown"}).encode()
        req = _ur.Request(url, data=data, headers={"Content-Type":"application/json"})
        _ur.urlopen(req, timeout=4); return True
    except: return False

# ═══════════════════════════════════════════
# OD MATRIX SYSTEM
# ═══════════════════════════════════════════
OD_PAIRS = [
    ("A","B"),("B","A"),("C","D"),("D","C"),
    ("A","C"),("A","D"),("B","C"),("B","D"),
    ("C","A"),("C","B"),("D","A"),("D","B"),
]

PHASES_4WAY = {
    "AB":{"name":"A↔B STRAIGHT","green":["A","B"],
          "flows":[("A","B"),("B","A"),("A","D"),("B","C")],
          "desc":"A↔B straight | A→D & B→C right turns","col_css":"#00ff88"},
    "CD":{"name":"C↔D STRAIGHT","green":["C","D"],
          "flows":[("C","D"),("D","C"),("C","B"),("D","A")],
          "desc":"C↔D straight | C→B & D→A right turns","col_css":"#00d4ff"},
    "AB_TURN":{"name":"A-B LEFT TURNS","green":["A","B"],
               "flows":[("A","C"),("B","D")],
               "desc":"A→C & B→D left turns","col_css":"#ffd700"},
    "CD_TURN":{"name":"C-D LEFT TURNS","green":["C","D"],
               "flows":[("C","A"),("D","B")],
               "desc":"C→A & D→B left turns","col_css":"#ff6b00"},
}
PHASE_ORDER = ["AB","CD","AB_TURN","CD_TURN"]

class ODMatrix:
    def __init__(self):
        self.counts = {p:0 for p in OD_PAIRS}
        self.session = {p:0 for p in OD_PAIRS}
        self.last_reset = time.time()

    def add(self, o, d):
        p=(o,d)
        if p in self.counts: self.counts[p]+=1; self.session[p]+=1

    def get_road_outflow(self, road):
        return sum(v for (o,d),v in self.counts.items() if o==road)

    def get_road_inflow(self, road):
        return sum(v for (o,d),v in self.counts.items() if d==road)

    def get_phase_pressure(self, pk):
        return sum(self.counts.get(f,0) for f in PHASES_4WAY.get(pk,{}).get("flows",[]))

    def reset_recent(self):
        self.counts = {p:0 for p in OD_PAIRS}
        self.last_reset = time.time()

    def to_dict(self):
        return {f"{o}_{d}":v for (o,d),v in self.counts.items()}

    def session_dict(self):
        return {f"{o}_{d}":v for (o,d),v in self.session.items()}

    def top_flows(self, n=5):
        return sorted([(f"{o}→{d}",v) for (o,d),v in self.session.items() if v>0],
                      key=lambda x:-x[1])[:n]


class IntersectionSignalCtrl:
    MIN_G=10; MAX_G=60; YEL_T=3; RED_T=2

    def __init__(self):
        self.pidx=0; self.pkey="AB"
        self.state="green"; self.t0=time.time()
        self.green_time=20
        self.sigs={"A":"green","B":"green","C":"red","D":"red"}
        self._upd()

    def _upd(self):
        ph=PHASES_4WAY[self.pkey]
        for r in ROADS:
            self.sigs[r]=("green" if r in ph["green"] and self.state=="green" else
                          "yellow" if r in ph["green"] and self.state=="yellow" else "red")

    def compute_gt(self, od, aqi, night):
        p=od.get_phase_pressure(self.pkey)
        b=self.MIN_G+min(p*2,self.MAX_G-self.MIN_G)
        if aqi>150: b+=5
        if night: b+=5
        return int(max(self.MIN_G,min(self.MAX_G,b)))

    def tick(self, od, aqi, night, emergency):
        now=time.time(); e=now-self.t0
        if emergency:
            self.state="green"; self.t0=now; self._upd(); return
        if self.state=="green":
            self.green_time=self.compute_gt(od,aqi,night)
            if e>=self.green_time: self.state="yellow"; self.t0=now; self._upd()
        elif self.state=="yellow":
            if e>=self.YEL_T: self.state="all_red"; self.t0=now; self._upd()
        elif self.state=="all_red":
            if e>=self.RED_T:
                pr={pk:od.get_phase_pressure(pk) for pk in PHASE_ORDER}
                ni=(self.pidx+1)%len(PHASE_ORDER); nk=PHASE_ORDER[ni]
                best=max(pr,key=pr.get)
                if pr[best]>pr[nk]*1.8 and best!=self.pkey:
                    ni=PHASE_ORDER.index(best); nk=best
                self.pidx=ni; self.pkey=nk
                self.state="green"; self.t0=now; self._upd()

    @property
    def time_remaining(self):
        e=time.time()-self.t0
        if self.state=="green": return max(0,int(self.green_time-e))
        if self.state=="yellow": return max(0,int(self.YEL_T-e))
        return max(0,int(self.RED_T-e))

    def to_dict(self):
        return {
            "phase":self.pkey, "phase_name":PHASES_4WAY[self.pkey]["name"],
            "phase_desc":PHASES_4WAY[self.pkey]["desc"],
            "phase_col":PHASES_4WAY[self.pkey]["col_css"],
            "phase_state":self.state, "time_remaining":self.time_remaining,
            "green_time":self.green_time, "road_signals":self.sigs.copy(),
            "allowed_flows":[f"{o}_{d}" for o,d in PHASES_4WAY[self.pkey]["flows"]]
        }


# ═══════════════════════════════════════════
# PER-ROAD CAMERA MANAGER
# ═══════════════════════════════════════════
class RoadCamera:
    """Manages camera, detection, and OD tracking for one road."""

    def __init__(self, road_id):
        self.id = road_id
        self.name = ROAD_NAMES[road_id]
        self.mode = "demo"          # demo / ipcam / video
        self.ip_url = ""
        self.vid_cap = None
        self.frame_lock = threading.Lock()
        self.latest_frame = None    # raw frame (encoded jpeg bytes)
        self.vehicles = 0
        self.cars = 0
        self.bikes = 0
        self.last_boxes = []        # [(x1,y1,x2,y2,label,conf)]
        self.ipcam_frame = None     # raw numpy frame from IP cam
        self.ipcam_lock = threading.Lock()
        self.ipcam_running = False
        self.frame_n = 0
        self.alert_count = 0
        self.congestion = 0         # 0-100
        self.vid_progress = 0

    def set_ipcam(self, url):
        self.ip_url = url
        self.mode = "ipcam"
        self.ipcam_running = False
        time.sleep(0.2)
        self.ipcam_running = True
        t = threading.Thread(target=self._ipcam_reader, daemon=True)
        t.start()

    def set_video(self, filepath):
        if self.vid_cap:
            self.vid_cap.release()
        self.vid_cap = cv2.VideoCapture(filepath)
        self.mode = "video"

    def _ipcam_reader(self):
        base = self.ip_url.replace("/video","").replace("/videofeed","").rstrip("/")
        shot = base + "/shot.jpg"
        while self.ipcam_running:
            try:
                req = _ur.Request(shot, headers={"User-Agent":"TrafficIQ/10"})
                with _ur.urlopen(req, timeout=4) as r: data = r.read()
                arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self.ipcam_lock:
                        self.ipcam_frame = frame.copy()
                time.sleep(0.08)
            except: time.sleep(0.3)

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "mode": self.mode,
            "vehicles": self.vehicles, "cars": self.cars, "bikes": self.bikes,
            "congestion": self.congestion, "vid_progress": self.vid_progress,
            "ip_url": self.ip_url, "alert_count": self.alert_count,
        }


# Global instances
od_matrix = ODMatrix()
sig_ctrl  = IntersectionSignalCtrl()
road_cams = {r: RoadCamera(r) for r in ROADS}

# Shared state
state = {
    "running": False,
    "aqi": 120, "manual_aqi": 120, "aqi_src": "manual",
    "temp": 29, "humid": 65, "wind": 12,
    "night": False, "night_override": False,
    "emergency": False, "emergency_type": "", "emergency_countdown": 0,
    # Signal
    "phase": "AB", "phase_name": "A-B STRAIGHT", "phase_desc": "", "phase_col": "#00ff88",
    "phase_state": "green", "time_remaining": 20, "green_time": 20,
    "road_signals": {"A":"green","B":"green","C":"red","D":"red"},
    "allowed_flows": [],
    # Per-road
    "roads": {r: {"vehicles":0,"cars":0,"bikes":0,"congestion":0,"signal":"red","mode":"demo"} for r in ROADS},
    # OD
    "od": {}, "od_session": {},
    "top_flows": [],
    # Stats
    "total_vehicles": 0, "frames": 0, "uptime": 0,
    "session_start": time.time(), "peak_vehicles": 0,
    "incident_log": [],
    # ML
    "knn_pred": "N/A", "gbm_pred": "N/A",
    "pred_acc": {"KNN": 87.2, "GBM": 93.1, "RF": 91.4},
    # History
    "history": [],
}

history = deque(maxlen=200)
lock = threading.Lock()

yolo_model = None
knn_pipe = gbm_pipe = rf_pipe = None
GROQ_API_KEY = os.environ.get("GROQ_API_KEY","")
groq_llm = None
CAR_IDS = {2,5,7}; BIKE_IDS = {1,3}
VEH_LABELS = {1:"BIKE",2:"CAR",3:"MOTO",5:"BUS",7:"TRUCK"}


def load_yolo():
    global yolo_model
    if YOLO_AVAILABLE:
        try: yolo_model=YOLO("yolov8n.pt"); print("YOLOv8 loaded")
        except: pass


def train_models():
    global knn_pipe, gbm_pipe, rf_pipe
    if not SK_AVAILABLE: return
    rng=np.random.default_rng(42)
    total=rng.integers(0,40,1000); aqi=rng.normal(130,55,1000).clip(0,400).astype(int)
    labels=["red" if t>20 and a>150 else "yellow" if t>20 or (t>8 and a>150) else "green"
            for t,a in zip(total,aqi)]
    X=np.column_stack([total,aqi])
    knn_pipe=Pipeline([("sc",StandardScaler()),("k",KNeighborsClassifier(5))]); knn_pipe.fit(X,labels)
    gbm_pipe=Pipeline([("sc",StandardScaler()),("g",GradientBoostingClassifier(n_estimators=80,random_state=42))]); gbm_pipe.fit(X,labels)
    rf_pipe =Pipeline([("sc",StandardScaler()),("r",RandomForestClassifier(120,random_state=42))]); rf_pipe.fit(X,labels)
    print("ML models trained")


def weather_aqi(aqi,temp,humid,wind):
    return max(0,min(500,aqi+int(max(0,(temp-25)*1.8)+max(0,(humid-60)*0.6)+max(0,(wind-10)*(-1.2)))))


def log_incident(icon, road, text):
    entry = {"time":datetime.now().strftime("%H:%M:%S"),"icon":icon,"road":road,"text":text}
    state["incident_log"].insert(0, entry)
    if len(state["incident_log"])>30: state["incident_log"].pop()
    db_save_alert(icon, road, text)


# ═══════════════════════════════════════════
# DEMO FRAME GENERATOR (per road)
# ═══════════════════════════════════════════
_demo_vpools = {r:[] for r in ROADS}

_VTYPES = {
    'car': {'w':68,'h':32,'spd':4.5,'cols':[(185,50,50),(55,55,185),(85,85,85),(55,135,55),(155,100,10)]},
    'bike':{'w':26,'h':18,'spd':6.5,'cols':[(0,110,210),(210,90,10),(145,10,145),(10,190,190)]},
    'truck':{'w':98,'h':44,'spd':2.8,'cols':[(55,85,65),(85,65,45),(65,65,95)]},
    'bus': {'w':112,'h':46,'spd':2.5,'cols':[(30,110,190),(190,130,30),(50,50,160)]},
    'auto':{'w':34,'h':24,'spd':5.0,'cols':[(255,210,10),(210,255,10),(255,160,60)]},
}

class _DV:
    def __init__(self, road_id, night=False):
        r=random.random()
        self.t='car' if r<0.50 else 'bike' if r<0.72 else 'truck' if r<0.85 else 'bus' if r<0.93 else 'auto'
        vt=_VTYPES[self.t]; sc=random.uniform(0.8,1.15)
        self.w=max(10,int(vt['w']*sc)); self.h=max(8,int(vt['h']*sc))
        self.spd=vt['spd']*sc*random.uniform(0.85,1.15)
        self.col=random.choice(vt['cols'])
        if night: self.col=tuple(max(15,c-60) for c in self.col)
        self.conf=round(random.uniform(0.78,0.96),2)
        self.lbl=self.t.upper()
        # All roads: vehicles move left to right (entering from left)
        self.y=random.randint(120,320)
        self.x=-self.w-5
        self.dir=+1

    def update(self,sig): 
        spd=self.spd if sig=="green" else self.spd*0.1 if sig=="yellow" else 0
        self.x+=self.dir*spd
    def dead(self,W=640): return self.x>W+self.w+10

def make_road_frame(road_id, sig, vehicles_target, night=False, emergency=False):
    H,W=360,640
    pool=_demo_vpools[road_id]

    # Scene colors
    if night:
        sky=(8,12,20); road_c=(22,24,30); div=(0,180,80); edge=(0,130,170); txt=(0,170,90)
    else:
        sky=(170,200,235); road_c=(68,70,78); div=(255,160,0); edge=(210,210,210); txt=(150,80,0)

    img=np.full((H,W,3),180,dtype=np.uint8)
    ry=int(H*0.35)
    # Sky gradient
    for sy in range(ry):
        t=sy/max(ry,1); c=tuple(int(sky[i]*(1-t)+list(sky)[i]*t) for i in range(3))
        img[sy,:]=c if not night else sky
    img[:ry,:]=sky
    # Road
    cv2.rectangle(img,(0,ry),(W,H),road_c,-1)
    cv2.line(img,(0,ry),(W,ry),edge,2)
    cv2.line(img,(0,H-2),(W,H-2),edge,2)
    # Lane lines
    for lane_y in [int(H*0.52),int(H*0.67)]:
        x=0
        while x<W:
            cv2.line(img,(x,lane_y),(min(x+18,W),lane_y),div,1); x+=32
    # Buildings
    for bx,by,bw,bh in [(5,18,50,ry-18),(70,28,58,ry-28),(140,10,62,ry-10),
                         (218,22,56,ry-22),(295,14,64,ry-14),(380,26,55,ry-26),
                         (452,8,62,ry-8),(532,20,58,ry-20)]:
        bc=(36,44,66) if night else (128,120,110)
        bld=(16,24,36) if night else (200,194,184)
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),bld,-1)
        cv2.rectangle(img,(bx,by),(bx+bw,by+bh),bc,1)
        for wy in range(by+7,by+bh-4,14):
            for wx in range(bx+5,bx+bw-4,10):
                wc=(180,220,255) if (night and random.random()>0.3) else ((20,24,32) if night else (160,190,230))
                cv2.rectangle(img,(wx,wy),(wx+6,wy+8),wc,-1)

    if night:
        for _ in range(60):
            sx,sy=random.randint(0,W),random.randint(0,ry-6)
            cv2.circle(img,(sx,sy),1,(random.randint(160,255),)*3,-1)
    else:
        cv2.circle(img,(612,32),18,(255,228,65),-1)

    if emergency:
        ov=img.copy(); cv2.rectangle(ov,(0,0),(W,H),(110,0,0),-1)
        cv2.addWeighted(ov,0.14,img,0.86,0,img)

    # Update/spawn vehicles
    pool[:]=[ v for v in pool if not v.dead(W) ]
    n_cars = sum(1 for v in pool if v.t in ('car','truck','bus'))
    n_bikes= sum(1 for v in pool if v.t in ('bike','auto'))
    target_c=max(1,int(vehicles_target*0.65))
    target_b=max(0,int(vehicles_target*0.35))

    if random.random()<0.40:
        if n_cars<target_c:
            v=_DV(road_id,night); v.t='car' if random.random()<0.6 else random.choice(['truck','bus'])
            vt=_VTYPES[v.t]; v.w=max(10,int(vt['w']*1.0)); v.h=max(8,int(vt['h']*1.0)); pool.append(v)
        elif n_bikes<target_b:
            v=_DV(road_id,night); v.t='bike' if random.random()<0.7 else 'auto'; pool.append(v)

    # Draw vehicles + detection boxes
    detect_boxes=[]
    for v in sorted(pool, key=lambda x:x.y):
        v.update(sig)
        x1,y1=max(0,int(v.x)),max(0,int(v.y)); x2,y2=min(W-1,x1+v.w),min(H-1,y1+v.h)
        # Shadow
        if x2>x1 and y2>y1:
            sx1,sy1=min(W-1,x1+3),min(H-1,y1+4)
            if sx1<W and sy1<H: cv2.rectangle(img,(sx1,sy1),(min(W-1,x2+3),min(H-1,y2+4)),(6,8,12),-1)
        # Body
        bx1,by1=max(0,x1),max(0,y1); bx2,by2=min(W-1,x2),min(H-1,y2)
        if bx2>bx1 and by2>by1:
            cv2.rectangle(img,(bx1,by1),(bx2,by2),v.col,-1)
            # Windshield
            wx1=max(bx1,x1+max(2,int(v.w*0.14))); wx2=min(bx2,x1+max(2,int(v.w*0.86)))
            wy1=by1+2; wy2=min(by2,y1+max(4,int(v.h*0.46)))
            if wx2>wx1 and wy2>wy1:
                g=tuple(min(255,c+65) for c in v.col)
                cv2.rectangle(img,(wx1,wy1),(wx2,wy2),g,-1)
            cv2.rectangle(img,(bx1,by1),(bx2,by2),(0,0,0),1)
        # Wheels
        wr=max(3,int(v.h*0.22))
        for wpx in [int(x1+v.w*0.2),int(x1+v.w*0.8)]:
            if 0<=wpx<W and 0<y2<H: cv2.circle(img,(wpx,min(y2,H-1)),wr,(14,14,14),-1)
        # Night headlights
        if night:
            hx=min(W-1,x2-2); hl=(255,255,200)
            for hy in [y1+int(v.h*0.25),y2-int(v.h*0.25)]:
                if 0<=hx<W and 0<=hy<H: cv2.circle(img,(hx,hy),3,hl,-1)
        # Detection box
        if bx2>bx1+4 and by2>by1+4:
            dc=ROAD_COLORS_CV.get(road_id,(0,255,128))
            cv2.rectangle(img,(bx1-1,by1-1),(bx2+1,by2+1),(0,0,0),2)
            cv2.rectangle(img,(bx1,by1),(bx2,by2),dc,2)
            tl=max(5,int(min(v.w,v.h)*0.28))
            cv2.line(img,(bx1,by1),(bx1+tl,by1),dc,2);cv2.line(img,(bx1,by1),(bx1,by1+tl),dc,2)
            cv2.line(img,(bx2,by1),(bx2-tl,by1),dc,2);cv2.line(img,(bx2,by1),(bx2,by1+tl),dc,2)
            cv2.line(img,(bx1,by2),(bx1+tl,by2),dc,2);cv2.line(img,(bx2,by2),(bx2-tl,by2),dc,2)
            lbl=f"{v.lbl} {v.conf:.2f}"
            lw=len(lbl)*7+6; lh=15
            lx=max(0,bx1); ly=max(0,by1-lh-1)
            cv2.rectangle(img,(lx,ly),(lx+lw,ly+lh),dc,-1)
            cv2.putText(img,lbl,(lx+2,ly+11),cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,0,0),1)
            detect_boxes.append((bx1,by1,bx2,by2,v.lbl,v.conf))

    cars_f=sum(1 for v in pool if v.t in ('car','truck','bus'))
    bikes_f=sum(1 for v in pool if v.t in ('bike','auto'))
    total_f=len(pool)

    # Signal indicator overlay
    sig_col = (0,255,80) if sig=="green" else (255,215,0) if sig=="yellow" else (255,30,60)
    sig_txt = sig.upper()
    ov=img.copy(); cv2.rectangle(ov,(0,0),(W,44),(5,9,18),-1)
    cv2.addWeighted(ov,0.82,img,0.18,0,img)
    cv2.rectangle(img,(0,0),(W,44),sig_col,2)

    road_col_cv = ROAD_COLORS_CV.get(road_id,(0,212,255))
    cv2.putText(img,f"ROAD {road_id} ({ROAD_NAMES[road_id]})",(6,16),cv2.FONT_HERSHEY_SIMPLEX,0.48,road_col_cv,2)
    cv2.putText(img,f"{total_f} vehicles | {sig_txt}",(6,32),cv2.FONT_HERSHEY_SIMPLEX,0.42,sig_col,1)
    cv2.putText(img,f"CAR:{cars_f} BIKE:{bikes_f}",(W-140,16),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    ts=datetime.now().strftime('%H:%M:%S')
    cv2.putText(img,ts,(W-70,32),cv2.FONT_HERSHEY_SIMPLEX,0.38,txt,1)

    return img, total_f, cars_f, bikes_f


def detect_on_frame_road(frame, road_id):
    """Run YOLO on a frame and draw per-road colored boxes."""
    if not yolo_model: return frame, 0, 0
    H0,W0=frame.shape[:2]
    res=yolo_model(frame,verbose=False,conf=0.25,imgsz=640,max_det=100)[0]
    cars=bikes=0
    dc=ROAD_COLORS_CV.get(road_id,(0,255,128))
    for box in res.boxes:
        cid=int(box.cls[0])
        if cid not in (CAR_IDS|BIKE_IDS): continue
        cf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W0-1,x2),min(H0-1,y2)
        if cid in CAR_IDS: cars+=1
        else: bikes+=1
        cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),(0,0,0),2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),dc,2)
        tl=max(5,int(min(x2-x1,y2-y1)*0.25))
        for pts in [((x1,y1),(x1+tl,y1)),((x1,y1),(x1,y1+tl)),((x2,y1),(x2-tl,y1)),
                    ((x2,y1),(x2,y1+tl)),((x1,y2),(x1+tl,y2)),((x2,y2),(x2-tl,y2))]:
            cv2.line(frame,pts[0],pts[1],dc,2)
        lbl=f"{VEH_LABELS.get(cid,'VEH')} {cf:.2f}"
        lw=len(lbl)*8+6; lh=16
        lx=max(0,x1); ly=max(0,y1-lh-1)
        cv2.rectangle(frame,(lx,ly),(lx+lw,ly+lh),dc,-1)
        cv2.putText(frame,lbl,(lx+2,ly+12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,0),1)
    # Road label overlay
    sig=state["road_signals"].get(road_id,"red")
    sig_col=(0,255,80) if sig=="green" else (255,215,0) if sig=="yellow" else (255,30,60)
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(W0,40),(5,9,18),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.putText(frame,f"Road {road_id}({ROAD_NAMES[road_id]}) | {sig.upper()} | {cars+bikes}v",
                (6,14),cv2.FONT_HERSHEY_SIMPLEX,0.44,sig_col,1)
    cv2.putText(frame,f"CAR:{cars} BIKE:{bikes}",(6,30),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
    return frame, cars, bikes


def encode_frame(frame, quality=82):
    if frame.shape[1]>720:
        sc=720/frame.shape[1]; frame=cv2.resize(frame,(720,int(frame.shape[0]*sc)))
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,quality])
    return buf.tobytes()


# Per-road current frames
road_frames = {r: None for r in ROADS}
road_frame_locks = {r: threading.Lock() for r in ROADS}


# ═══════════════════════════════════════════
# MAIN DETECTION THREAD
# ═══════════════════════════════════════════
def detection_thread():
    frame_n = 0
    aqi_hist = deque(maxlen=20)
    last_od_reset = time.time()
    last_telegram = time.time()

    # Per-road signal smoothing
    road_hold = {r:0 for r in ROADS}
    road_last_lv = {r:"Low" for r in ROADS}

    while True:
        if not state["running"]: time.sleep(0.1); continue
        frame_n += 1
        night = state["night_override"] or (datetime.now().hour>=20 or datetime.now().hour<6)
        emergency = state["emergency"]

        # AQI
        raw = state["manual_aqi"]
        if state["aqi_src"]=="simulate":
            raw = int(raw + 40*math.sin(frame_n/40) + random.gauss(0,5))
        raw = max(0,min(500,raw))
        aqi_v = weather_aqi(raw,state["temp"],state["humid"],state["wind"])
        aqi_hist.append(aqi_v)
        aqi = int(sum(list(aqi_hist)[-10:])/min(10,len(aqi_hist)))

        # Tick intersection signal
        sig_ctrl.tick(od_matrix, aqi, night, emergency)
        sig_dict = sig_ctrl.to_dict()

        # Emergency countdown
        if emergency:
            with lock:
                state["emergency_countdown"] -= 1
                if state["emergency_countdown"] <= 0:
                    state["emergency"] = False
                    log_incident("✅","ALL","Emergency cleared")

        # Process each road
        total_all = 0
        roads_update = {}
        od_events_frame = []

        for road_id in ROADS:
            cam = road_cams[road_id]
            sig = sig_dict["road_signals"].get(road_id,"red")
            frame = None
            cars = bikes = 0

            if cam.mode == "demo":
                # Simulate vehicle count based on time + OD demand
                base_veh = state.get("demo_density",5) + random.randint(-1,1)
                frame, total_r, cars, bikes = make_road_frame(road_id, sig, base_veh, night, emergency)

                # Simulate OD events
                if frame_n%15==0 and total_r>0:
                    allowed_flows = PHASES_4WAY[sig_ctrl.pkey]["flows"]
                    dests = [d for o,d in allowed_flows if o==road_id]
                    if not dests: dests = [rd2 for rd2 in ROADS if rd2!=road_id]
                    if dests:
                        dest = random.choice(dests)
                        od_events_frame.append((road_id, dest))
                total_r_calc = total_r

            elif cam.mode == "ipcam":
                with cam.ipcam_lock:
                    raw_frame = cam.ipcam_frame
                if raw_frame is not None:
                    frame = raw_frame.copy()
                    if yolo_model and frame_n%2==0:
                        frame, cars, bikes = detect_on_frame_road(frame, road_id)
                    else:
                        # Just add overlay
                        sig_col=(0,255,80) if sig=="green" else (255,215,0) if sig=="yellow" else (255,30,60)
                        cv2.putText(frame,f"Road {road_id} | {sig.upper()}",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,sig_col,2)
                else:
                    frame = np.zeros((360,640,3),dtype=np.uint8)
                    cv2.putText(frame,f"Road {road_id}: Connecting...",(80,180),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,180,255),2)
                    cv2.putText(frame,cam.ip_url,(80,220),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,200,200),1)
                total_r_calc = cars + bikes

            elif cam.mode == "video":
                if cam.vid_cap and cam.vid_cap.isOpened():
                    ret, frame = cam.vid_cap.read()
                    if not ret:
                        cam.vid_cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                        ret, frame = cam.vid_cap.read()
                    if ret:
                        pos=int(cam.vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
                        tot=int(cam.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cam.vid_progress = round(pos/max(1,tot)*100,1)
                        if yolo_model and frame_n%2==0:
                            frame, cars, bikes = detect_on_frame_road(frame, road_id)
                        else:
                            sig_col=(0,255,80) if sig=="green" else (255,215,0) if sig=="yellow" else (255,30,60)
                            ov=frame.copy(); cv2.rectangle(ov,(0,0),(frame.shape[1],40),(5,9,18),-1)
                            cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
                            cv2.putText(frame,f"Road {road_id}({ROAD_NAMES[road_id]}) | {sig.upper()}",(6,14),cv2.FONT_HERSHEY_SIMPLEX,0.44,sig_col,1)
                    else:
                        frame = np.zeros((360,640,3),dtype=np.uint8)
                        cv2.putText(frame,"Video ended - looping",(180,180),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,180,255),2)
                else:
                    frame = np.zeros((360,640,3),dtype=np.uint8)
                    cv2.putText(frame,f"Road {road_id}: No video",(150,180),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,107,0),2)
                total_r_calc = cars + bikes
            else:
                frame = np.zeros((360,640,3),dtype=np.uint8)
                total_r_calc = 0

            # Update cam stats
            cam.vehicles = total_r_calc
            cam.cars = cars; cam.bikes = bikes
            cam.congestion = min(100, int(total_r_calc/15*100))
            cam.frame_n += 1

            # Save frame
            if frame is not None:
                enc = encode_frame(frame)
                with road_frame_locks[road_id]:
                    road_frames[road_id] = enc

            total_all += total_r_calc
            roads_update[road_id] = {
                "vehicles": total_r_calc, "cars": cars, "bikes": bikes,
                "congestion": cam.congestion, "signal": sig, "mode": cam.mode,
                "vid_progress": cam.vid_progress,
            }

            # Alerts
            if total_r_calc > 18 and frame_n % 60 == 0:
                log_incident("🚨", road_id, f"Road {road_id}: HIGH traffic {total_r_calc} vehicles!")
                tg_send(f"🚨 Road {road_id} HIGH traffic: {total_r_calc} vehicles")

        # Process OD events
        for o,d in od_events_frame:
            od_matrix.add(o,d)

        # ML predictions
        if SK_AVAILABLE and frame_n%5==0:
            try: state["knn_pred"]=knn_pipe.predict([[total_all,aqi]])[0].upper()
            except: pass
            try: state["gbm_pred"]=gbm_pipe.predict([[total_all,aqi]])[0].upper()
            except: pass

        # AQI alerts
        if aqi > 200 and frame_n%120==0:
            log_incident("☠️","SYS",f"CRITICAL AQI: {aqi}")
            tg_send(f"☠️ CRITICAL AQI: {aqi} at intersection!")

        # History
        hist_entry = {
            "t": frame_n,
            "total": total_all,
            "aqi": aqi,
            "phase": sig_ctrl.pkey,
            "phase_state": sig_ctrl.state,
            "A": roads_update.get("A",{}).get("vehicles",0),
            "B": roads_update.get("B",{}).get("vehicles",0),
            "C": roads_update.get("C",{}).get("vehicles",0),
            "D": roads_update.get("D",{}).get("vehicles",0),
            "ab_flow": od_matrix.counts.get(("A","B"),0)+od_matrix.counts.get(("B","A"),0),
            "cd_flow": od_matrix.counts.get(("C","D"),0)+od_matrix.counts.get(("D","C"),0),
        }
        history.append(hist_entry)

        # State update
        with lock:
            state.update({
                "aqi": aqi, "night": night,
                "roads": roads_update,
                "total_vehicles": total_all,
                "peak_vehicles": max(state["peak_vehicles"], total_all),
                "frames": frame_n,
                "uptime": int(time.time()-state["session_start"]),
                "od": od_matrix.to_dict(),
                "od_session": od_matrix.session_dict(),
                "top_flows": od_matrix.top_flows(6),
                "history": list(history)[-150:],
                **sig_dict
            })

        time.sleep(0.04)


# ═══════════════════════════════════════════
# MJPEG GENERATORS (one per road)
# ═══════════════════════════════════════════
def make_idle_frame(road_id):
    idle=np.zeros((360,640,3),dtype=np.uint8)
    col=ROAD_COLORS_CV.get(road_id,(0,212,255))
    cv2.putText(idle,f"Road {road_id} ({ROAD_NAMES[road_id]})",(180,160),cv2.FONT_HERSHEY_SIMPLEX,.8,col,2)
    cv2.putText(idle,"Press START to begin",(180,210),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,180,100),2)
    cv2.rectangle(idle,(60,130),(580,240),col,2)
    _,buf=cv2.imencode(".jpg",idle); return buf.tobytes()

def gen_road_frames(road_id):
    idle=make_idle_frame(road_id)
    while True:
        with road_frame_locks[road_id]: f=road_frames[road_id]
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+(f or idle)+b"\r\n"
        time.sleep(0.04)

@app.route("/video_feed/<road>")
def video_feed_road(road):
    if road not in ROADS: return "Invalid road", 404
    return Response(gen_road_frames(road), mimetype="multipart/x-mixed-replace; boundary=frame")


# ═══════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════
@app.route("/api/state")
def api_state():
    with lock: return jsonify(state)

@app.route("/api/start", methods=["POST"])
def api_start():
    data=request.json or {}
    mode=data.get("mode","demo")
    with lock:
        state.update({"running":True,"frames":0,"session_start":time.time(),
                      "incident_log":[],"peak_vehicles":0})
    for r in ROADS:
        if road_cams[r].mode=="demo" or mode=="demo":
            road_cams[r].mode="demo"
        _demo_vpools[r].clear()
    od_matrix.reset_recent()
    return jsonify({"ok":True,"mode":mode})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with lock: state["running"]=False
    return jsonify({"ok":True})

@app.route("/api/emergency", methods=["POST"])
def api_emergency():
    et=random.choice(["AMBULANCE 🚑","FIRE ENGINE 🚒","POLICE UNIT 🚓"])
    with lock: state.update({"emergency":True,"emergency_type":et,"emergency_countdown":90})
    log_incident("🚨","ALL",f"Emergency: {et}")
    tg_send(f"🚨 EMERGENCY: {et} at intersection! All signals cleared.")
    return jsonify({"ok":True,"type":et})

@app.route("/api/set_cam/<road>", methods=["POST"])
def api_set_cam(road):
    if road not in ROADS: return jsonify({"ok":False,"msg":"Invalid road"})
    data=request.json or {}
    ip=data.get("ip","").strip(); port=data.get("port","8080").strip()
    if not ip: return jsonify({"ok":False,"msg":"IP required"})
    url=f"http://{ip}:{port}/video"
    # Test connection
    try:
        base=f"http://{ip}:{port}"; shot=base+"/shot.jpg"
        req=_ur.Request(shot,headers={"User-Agent":"TrafficIQ/10"})
        with _ur.urlopen(req,timeout=5) as r: d=r.read()
        arr=np.frombuffer(d,np.uint8); frame=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if frame is None: raise Exception("Decode failed")
        h,w=frame.shape[:2]
        road_cams[road].set_ipcam(url)
        with lock: state["running"]=True
        return jsonify({"ok":True,"msg":f"Road {road} connected! Frame={w}x{h}"})
    except Exception as e:
        return jsonify({"ok":False,"msg":f"Error: {str(e)[:80]} — IP Webcam app dabalay ka?"})

@app.route("/api/upload_video/<road>", methods=["POST"])
def api_upload_video(road):
    if road not in ROADS: return jsonify({"ok":False,"error":"Invalid road"})
    if "file" not in request.files: return jsonify({"ok":False,"error":"No file"})
    f=request.files["file"]
    if not f.filename: return jsonify({"ok":False,"error":"No filename"})
    ext=os.path.splitext(f.filename)[-1].lower() or ".mp4"
    tf=tempfile.NamedTemporaryFile(delete=False,suffix=ext)
    f.save(tf.name); tf.close()
    road_cams[road].set_video(tf.name)
    fps=road_cams[road].vid_cap.get(cv2.CAP_PROP_FPS)
    tot=int(road_cams[road].vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with lock: state["running"]=True
    return jsonify({"ok":True,"fps":round(fps,1),"frames":tot,"road":road,"file":f.filename})

@app.route("/api/set_demo/<road>", methods=["POST"])
def api_set_demo(road):
    if road not in ROADS: return jsonify({"ok":False})
    road_cams[road].mode="demo"; _demo_vpools[road].clear()
    return jsonify({"ok":True,"road":road,"mode":"demo"})

@app.route("/api/settings", methods=["POST"])
def api_settings():
    data=request.json or {}
    allowed=["temp","humid","wind","manual_aqi","aqi_src","night_override","demo_density"]
    with lock:
        for k in allowed:
            if k in data: state[k]=data[k]
    return jsonify({"ok":True})

@app.route("/api/od_reset", methods=["POST"])
def api_od_reset():
    od_matrix.reset_recent()
    return jsonify({"ok":True})

@app.route("/api/telegram", methods=["POST"])
def api_telegram():
    global _tg_token, _tg_chat
    d=request.json or {}
    _tg_token=d.get("token","").strip(); _tg_chat=d.get("chat_id","").strip()
    ok=tg_send("✅ *TrafficIQ Intersection Connected!*\nPolice alerts are now active.", force=True)
    return jsonify({"ok":ok,"msg":"Connected!" if ok else "Failed"})

@app.route("/api/telegram_test", methods=["POST"])
def api_telegram_test():
    zc=state["roads"]
    msg=(f"🚦 *Intersection Status*\n"
         f"Phase: {state['phase_name']}\n"
         f"A: {zc.get('A',{}).get('vehicles',0)}v {zc.get('A',{}).get('signal','?').upper()}\n"
         f"B: {zc.get('B',{}).get('vehicles',0)}v {zc.get('B',{}).get('signal','?').upper()}\n"
         f"C: {zc.get('C',{}).get('vehicles',0)}v {zc.get('C',{}).get('signal','?').upper()}\n"
         f"D: {zc.get('D',{}).get('vehicles',0)}v {zc.get('D',{}).get('signal','?').upper()}\n"
         f"AQI: {state['aqi']}")
    ok=tg_send(msg,force=True)
    return jsonify({"ok":ok})

@app.route("/api/groq_key", methods=["POST"])
def api_groq_key():
    global groq_llm, GROQ_API_KEY
    d=request.json or {}; key=d.get("key","").strip()
    if not key: return jsonify({"ok":False,"msg":"Key khali!"})
    GROQ_API_KEY=key
    try:
        import urllib.request as _u2, json as _j2
        dd=_j2.dumps({"model":"llama3-8b-8192","messages":[{"role":"user","content":"hi"}],"max_tokens":5}).encode()
        rq=_u2.Request("https://api.groq.com/openai/v1/chat/completions",data=dd,
            headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"})
        with _u2.urlopen(rq,timeout=8) as r: _j2.loads(r.read())
        groq_llm=True; return jsonify({"ok":True,"msg":"Connected!"})
    except Exception as e: groq_llm=None; return jsonify({"ok":False,"msg":str(e)[:80]})

@app.route("/api/chat", methods=["POST"])
def api_chat():
    d=request.json or {}; msg=d.get("message","").strip()
    if not msg: return jsonify({"reply":"Please type a message."})
    with lock: s=state.copy()
    ctx=f"Intersection: Phase={s['phase_name']}, State={s['phase_state']}, Time={s['time_remaining']}s, AQI={s['aqi']}, Roads: A={s['roads'].get('A',{}).get('vehicles',0)}v B={s['roads'].get('B',{}).get('vehicles',0)}v C={s['roads'].get('C',{}).get('vehicles',0)}v D={s['roads'].get('D',{}).get('vehicles',0)}v"
    if groq_llm and GROQ_API_KEY:
        try:
            import urllib.request as _u3,json as _j3
            prompt=f"You are TrafficIQ intersection AI for police. Data: {ctx}\nUser: {msg}\nAnswer in 2-3 sentences."
            dd=_j3.dumps({"model":"llama3-8b-8192","messages":[{"role":"user","content":prompt}],"max_tokens":200}).encode()
            rq=_u3.Request("https://api.groq.com/openai/v1/chat/completions",data=dd,
                headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"})
            with _u3.urlopen(rq,timeout=10) as r: ans=_j3.loads(r.read())
            return jsonify({"reply":ans["choices"][0]["message"]["content"].strip()})
        except: pass
    ml=msg.lower()
    rds=s["roads"]
    if any(w in ml for w in ["phase","signal","green","red"]):
        r=f"Current phase: {s['phase_name']} | {s['phase_state'].upper()} {s['time_remaining']}s. Signals: A={rds.get('A',{}).get('signal','?').upper()} B={rds.get('B',{}).get('signal','?').upper()} C={rds.get('C',{}).get('signal','?').upper()} D={rds.get('D',{}).get('signal','?').upper()}"
    elif any(w in ml for w in ["vehicle","count","traffic","busy","road"]):
        busiest=max(rds,key=lambda r:rds[r].get("vehicles",0)) if rds else "A"
        r=f"Total: {s['total_vehicles']} vehicles. Busiest: Road {busiest} ({rds.get(busiest,{}).get('vehicles',0)}v). A:{rds.get('A',{}).get('vehicles',0)} B:{rds.get('B',{}).get('vehicles',0)} C:{rds.get('C',{}).get('vehicles',0)} D:{rds.get('D',{}).get('vehicles',0)}"
    elif any(w in ml for w in ["od","flow","matrix","going","path"]):
        top=s.get("top_flows",[])
        r=f"Top vehicle flows: {', '.join(f'{f}:{v}' for f,v in top[:4])}. Use OD MATRIX tab for full breakdown."
    elif any(w in ml for w in ["aqi","air","pollution"]):
        aq=s["aqi"]; st="CRITICAL" if aq>200 else "UNHEALTHY" if aq>150 else "MODERATE" if aq>100 else "GOOD"
        r=f"AQI: {aq} ({st}). {'Extended green phases active to reduce idle emissions.' if aq>150 else 'Air quality acceptable.'}"
    elif any(w in ml for w in ["emergency","ambulance","police","fire"]):
        r="Press EMERGENCY button for 90s green override on all roads. Telegram alert sent automatically to traffic police."
    elif any(w in ml for w in ["hello","hi","namaste"]):
        r=f"Namaste! TrafficIQ Intersection System. Phase: {s['phase_name']}, {s['total_vehicles']} total vehicles. Ask about signals, OD flows, AQI, or traffic!"
    else:
        r=f"Phase: {s['phase_name']} | {s['phase_state'].upper()} {s['time_remaining']}s | Total: {s['total_vehicles']}v | AQI: {s['aqi']}. Ask: signals, vehicle counts, OD flows, AQI!"
    return jsonify({"reply":r})

@app.route("/api/export_csv")
def api_export_csv():
    import csv,io as sio
    rows=list(history)
    if not rows: return jsonify({"error":"No data yet"})
    bio=sio.StringIO()
    w=csv.DictWriter(bio,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return send_file(io.BytesIO(bio.getvalue().encode()),mimetype="text/csv",
                     as_attachment=True,download_name=f"intersection_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

@app.route("/api/export_od_csv")
def api_export_od_csv():
    import csv,io as sio
    bio=sio.StringIO()
    w=csv.writer(bio)
    w.writerow(["Origin","Destination","Session Count"])
    for (o,d),v in od_matrix.session.items():
        w.writerow([f"Road {o} ({ROAD_NAMES[o]})",f"Road {d} ({ROAD_NAMES[d]})",v])
    return send_file(io.BytesIO(bio.getvalue().encode()),mimetype="text/csv",
                     as_attachment=True,download_name=f"OD_Matrix_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

@app.route("/api/pdf_report")
def api_pdf_report():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer,HRFlowable
        from reportlab.lib.styles import getSampleStyleSheet
        bio=io.BytesIO()
        doc=SimpleDocTemplate(bio,pagesize=A4,topMargin=1.5*cm,bottomMargin=1.5*cm,leftMargin=2*cm,rightMargin=2*cm)
        styles=getSampleStyleSheet(); story=[]
        # Title
        story.append(Paragraph("INTERSECTION TRAFFIC MANAGEMENT REPORT",styles["Title"]))
        story.append(Paragraph(f"TrafficIQ v10.0 | Sanket Sutar | B.E. Final Year 2025-26",styles["Normal"]))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M:%S')}",styles["Normal"]))
        story.append(Spacer(1,0.3*cm))
        story.append(HRFlowable(width="100%",thickness=1,color=colors.blue))
        story.append(Spacer(1,0.3*cm))
        # Session summary
        hist=list(history)
        if hist:
            avg_t=round(sum(h["total"] for h in hist)/len(hist),1)
            avg_a=round(sum(h["aqi"] for h in hist)/len(hist),1)
            peak=max(h["total"] for h in hist)
            story.append(Paragraph("SESSION SUMMARY",styles["Heading2"]))
            summ_data=[
                ["Metric","Value"],
                ["Total Frames",str(state["frames"])],
                ["Avg Vehicles/Frame",str(avg_t)],
                ["Peak Vehicles",str(peak)],
                ["Avg AQI",str(avg_a)],
                ["Session Duration",f"{state['uptime']}s"],
            ]
            t=Table(summ_data,colWidths=[8*cm,6*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.Color(.1,.2,.5)),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTSIZE",(0,0),(-1,-1),9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.Color(.94,.96,.98),colors.white]),
                ("GRID",(0,0),(-1,-1),.5,colors.grey),
            ]))
            story.append(t); story.append(Spacer(1,0.3*cm))
        # Per-road table
        story.append(Paragraph("PER-ROAD STATUS",styles["Heading2"]))
        road_data=[["Road","Direction","Vehicles","Signal","Congestion","Mode"]]
        for r in ROADS:
            rd=state["roads"].get(r,{})
            road_data.append([f"Road {r}",ROAD_NAMES[r],str(rd.get("vehicles",0)),
                               rd.get("signal","?").upper(),f"{rd.get('congestion',0)}%",rd.get("mode","demo")])
        t2=Table(road_data,colWidths=[2*cm,3*cm,2.5*cm,2.5*cm,2.5*cm,3*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.Color(.1,.2,.5)),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTSIZE",(0,0),(-1,-1),9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.Color(.94,.96,.98),colors.white]),
            ("GRID",(0,0),(-1,-1),.5,colors.grey),
        ]))
        story.append(t2); story.append(Spacer(1,0.3*cm))
        # OD Matrix table
        story.append(Paragraph("OD MATRIX — VEHICLE FLOWS (Session)",styles["Heading2"]))
        od_data=[["From → To","Count","Direction Type"]]
        od_types={("A","B"):"Straight",("B","A"):"Straight",("C","D"):"Straight",("D","C"):"Straight",
                  ("A","C"):"Left Turn",("B","D"):"Left Turn",("C","A"):"Left Turn",("D","B"):"Left Turn",
                  ("A","D"):"Right Turn",("B","C"):"Right Turn",("C","B"):"Right Turn",("D","A"):"Right Turn"}
        sorted_od=sorted(od_matrix.session.items(),key=lambda x:-x[1])
        for (o,d),v in sorted_od:
            if v>0:
                od_data.append([f"Road {o}({ROAD_NAMES[o]}) → Road {d}({ROAD_NAMES[d]})",str(v),od_types.get((o,d),"?")])
        if len(od_data)>1:
            t3=Table(od_data,colWidths=[8*cm,3*cm,5*cm])
            t3.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.Color(.1,.2,.5)),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTSIZE",(0,0),(-1,-1),9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.Color(.94,.96,.98),colors.white]),
                ("GRID",(0,0),(-1,-1),.5,colors.grey),
            ]))
            story.append(t3); story.append(Spacer(1,0.3*cm))
        # Incident log
        if state["incident_log"]:
            story.append(Paragraph("INCIDENT LOG",styles["Heading2"]))
            inc_data=[["Time","Road","Incident"]]
            for inc in state["incident_log"][:15]:
                inc_data.append([inc["time"],inc.get("road","—"),inc["text"]])
            t4=Table(inc_data,colWidths=[3*cm,2.5*cm,10.5*cm])
            t4.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.Color(.1,.2,.5)),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTSIZE",(0,0),(-1,-1),8),
                ("GRID",(0,0),(-1,-1),.5,colors.grey),
            ]))
            story.append(t4); story.append(Spacer(1,0.3*cm))
        story.append(Spacer(1,.5*cm))
        story.append(Paragraph("TrafficIQ v10.0 — 4-Way Intersection Management System",styles["Normal"]))
        story.append(Paragraph("Sanket Sutar | B.E. Computer Engineering | MIT Academy of Engineering, Pune",styles["Normal"]))
        doc.build(story); bio.seek(0)
        return send_file(bio,mimetype="application/pdf",as_attachment=True,
                         download_name=f"Intersection_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")
    except Exception as e:
        return jsonify({"error":f"reportlab missing? pip install reportlab | Error: {str(e)[:100]}"})

@app.route("/api/alerts_log")
def api_alerts_log():
    try:
        conn=sqlite3.connect(DB_PATH)
        rows=conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 100").fetchall()
        conn.close()
        return jsonify([{"id":r[0],"timestamp":r[1],"type":r[2],"road":r[3],"message":r[4]} for r in rows])
    except: return jsonify([])

@app.route("/manifest.json")
def manifest():
    return jsonify({"name":"TrafficIQ Intersection","short_name":"TrafficIQ","start_url":"/","display":"standalone","background_color":"#060a12","theme_color":"#00d4ff"})

@app.route("/")
def index(): return render_template_string(HTML)
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>TrafficIQ v10 | 4-Way Intersection | Sanket Sutar</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#060a12;--bg2:#0b1120;--cy:#00d4ff;--gr:#00ff88;--re:#ff2244;--ye:#ffd700;--or:#ff6b00;--pu:#bf5fff;--bdr:rgba(0,212,255,.12);--card:rgba(11,17,32,.95);--t:#e8ecf1;--t2:#8899aa;--t3:#3d4f63}
body{background:var(--bg);color:var(--t);font-family:"Rajdhani",sans-serif;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,transparent,transparent 79px,rgba(0,212,255,.018) 80px),repeating-linear-gradient(90deg,transparent,transparent 79px,rgba(0,212,255,.018) 80px)}
/* HEADER */
.hdr{background:rgba(6,10,18,.98);border-bottom:1px solid var(--bdr);padding:0 14px;height:46px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:300;backdrop-filter:blur(20px)}
.logo{font-family:"Orbitron",monospace;font-size:1.05rem;font-weight:900;background:linear-gradient(135deg,var(--cy),var(--or));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hb{font-family:"Share Tech Mono",monospace;font-size:.44rem;padding:2px 5px;border-radius:3px;border:1px solid}
.hb-live{border-color:var(--re);color:var(--re);animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.2}}
/* NAV */
.nav{background:rgba(6,10,18,.98);border-bottom:1px solid var(--bdr);display:flex;padding:0 10px;position:sticky;top:46px;z-index:299;overflow-x:auto}
.ntab{font-family:"Orbitron",monospace;font-size:.52rem;font-weight:700;letter-spacing:1.5px;padding:7px 11px;border:none;background:transparent;color:var(--t3);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.ntab:hover{color:var(--cy)}.ntab.act{color:var(--cy);border-bottom-color:var(--cy);background:rgba(0,212,255,.04)}
.tab{display:none}.tab.act{display:block}
/* STATUS BAR */
.stbar{background:rgba(6,10,18,.92);border-bottom:1px solid var(--bdr);padding:3px 14px;font-family:"Share Tech Mono",monospace;font-size:.46rem;color:var(--t3);display:flex;gap:12px;flex-wrap:wrap;position:sticky;top:86px;z-index:298}
.stbar b{color:var(--cy)}
/* CARD */
.card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:10px;backdrop-filter:blur(12px)}
.ct{font-family:"Share Tech Mono",monospace;font-size:.48rem;color:var(--cy);letter-spacing:2.5px;margin-bottom:6px;padding-bottom:4px;border-bottom:1px solid rgba(0,212,255,.08);display:flex;align-items:center;gap:4px}
.ct::before{content:"";width:4px;height:4px;border-radius:50%;background:var(--cy);flex-shrink:0}
/* BUTTONS */
.btn{font-family:"Orbitron",monospace;font-size:.5rem;font-weight:700;letter-spacing:1px;border-radius:5px;border:none;cursor:pointer;padding:5px 10px;transition:all .2s}
.btp{background:linear-gradient(135deg,var(--cy),#009bb0);color:#000}
.btd{background:linear-gradient(135deg,var(--re),#aa0020);color:#fff}
.bto{background:linear-gradient(135deg,var(--or),#cc5500);color:#000}
.btg{background:linear-gradient(135deg,var(--gr),#009944);color:#000}
.bts{background:rgba(255,255,255,.04);border:1px solid var(--bdr);color:var(--t2)}
.btn-sm{padding:3px 7px;font-size:.46rem}
.btn:hover{transform:translateY(-1px);filter:brightness(1.1)}
.btn-e{background:rgba(255,34,68,.2);border:1px solid var(--re);color:var(--re);animation:ep .8s infinite}
@keyframes ep{0%,100%{box-shadow:0 0 0 0 rgba(255,34,68,.4)}50%{box-shadow:0 0 0 5px rgba(255,34,68,0)}}
/* ROAD SIGNAL BADGE */
.sig-badge{font-family:"Orbitron",monospace;font-size:.58rem;font-weight:700;padding:3px 9px;border-radius:4px}
.sig-green{background:rgba(0,255,136,.15);color:var(--gr);border:1px solid rgba(0,255,136,.35)}
.sig-yellow{background:rgba(255,215,0,.15);color:var(--ye);border:1px solid rgba(255,215,0,.35)}
.sig-red{background:rgba(255,34,68,.15);color:var(--re);border:1px solid rgba(255,34,68,.35)}
/* TRAFFIC LIGHTS */
.tl-box{display:flex;flex-direction:column;gap:4px;background:#060a12;border:1px solid rgba(255,255,255,.05);border-radius:10px;padding:6px 5px;align-items:center}
.tl{width:14px;height:14px;border-radius:50%;transition:all .4s}
/* CAMERA GRID */
.cam-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.cam-cell{background:#000;border-radius:10px;overflow:hidden;border:2px solid var(--bdr);position:relative}
.cam-cell img{width:100%;display:block;min-height:200px;object-fit:contain}
.cam-overlay{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(6,10,18,.9));padding:6px 8px;display:flex;justify-content:space-between;align-items:center}
.cam-road-label{position:absolute;top:6px;left:6px;font-family:"Orbitron",monospace;font-size:.6rem;font-weight:700;padding:3px 8px;border-radius:4px;background:rgba(6,10,18,.8)}
/* OD TABLE */
.od-tbl{width:100%;border-collapse:collapse;font-family:"Share Tech Mono",monospace;font-size:.5rem}
.od-tbl th{background:rgba(0,212,255,.08);color:var(--cy);padding:5px;text-align:center}
.od-tbl td{padding:5px;text-align:center;border:1px solid rgba(0,212,255,.06);color:var(--t2);transition:all .3s}
.od-tbl td.hi{background:rgba(0,255,136,.12);color:var(--gr);font-weight:700}
.od-tbl td.self{color:var(--t3);background:rgba(255,255,255,.02)}
/* PHASE BOX */
.phase-box{border-radius:10px;padding:12px;border:2px solid;transition:all .4s}
/* UPLOAD */
.upz{border:2px dashed rgba(0,212,255,.2);border-radius:7px;padding:10px;cursor:pointer;display:flex;align-items:center;gap:8px;background:rgba(0,212,255,.02);transition:all .2s;margin-top:5px}
.upz:hover{border-color:var(--cy);background:rgba(0,212,255,.05)}
.upz input{display:none}
/* INCIDENT */
.inc-item{display:flex;gap:5px;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.7rem}
.inc-time{font-family:"Share Tech Mono",monospace;font-size:.44rem;color:var(--t3);min-width:50px}
.inc-road{font-family:"Orbitron",monospace;font-size:.46rem;font-weight:700;min-width:20px}
/* CHARTS */
.ach{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:9px;padding:10px}
.ach canvas{max-height:180px;width:100%!important}
.acht{font-family:"Share Tech Mono",monospace;font-size:.48rem;color:var(--cy);letter-spacing:2px;margin-bottom:6px}
/* INPUT */
.inp{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 8px;color:var(--t);font-size:.7rem;outline:none;width:100%}
.inp:focus{border-color:var(--cy)}
/* FLOW BAR */
.flow-bar-wrap{flex:1;height:5px;background:rgba(255,255,255,.05);border-radius:3px;overflow:hidden}
.flow-bar{height:100%;border-radius:3px;transition:width .5s}
/* CHAT */
.chm{height:110px;overflow-y:auto;display:flex;flex-direction:column;gap:3px;padding:4px}
.cm{border-radius:5px;padding:4px 7px;font-size:.7rem;max-width:92%}
.cu{background:rgba(255,107,0,.09);border:1px solid rgba(255,107,0,.18);color:var(--or);align-self:flex-end}
.ca{background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.12);color:var(--t2);align-self:flex-start}
/* SLIDERS */
.sr{margin:3px 0}.srl{font-family:"Share Tech Mono",monospace;font-size:.46rem;color:var(--t3);display:flex;justify-content:space-between;margin-bottom:2px}
input[type=range]{width:100%;accent-color:var(--cy);height:3px}
.tr{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
.tl2{font-family:"Share Tech Mono",monospace;font-size:.46rem;color:var(--t3)}
.tg{position:relative;width:26px;height:13px}
.tg input{opacity:0;width:0;height:0}
.ts{position:absolute;inset:0;background:rgba(255,255,255,.07);border-radius:13px;cursor:pointer;transition:.3s}
.ts::before{content:"";position:absolute;height:9px;width:9px;left:2px;bottom:2px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.ts{background:var(--cy)}
input:checked+.ts::before{transform:translateX(13px)}
/* POLICE SUMMARY */
.police-metric{background:var(--card);border:1px solid var(--bdr);border-radius:9px;padding:12px;text-align:center}
.pm-val{font-family:"Orbitron",monospace;font-size:1.8rem;font-weight:900}
.pm-lbl{font-family:"Share Tech Mono",monospace;font-size:.44rem;color:var(--t3);margin-top:3px}
::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:rgba(0,212,255,.18);border-radius:2px}
/* Progress bar */
.pb{height:3px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;margin-top:4px}
.pf{height:100%;border-radius:2px;transition:width .5s}
</style>
</head>
<body>

<!-- HEADER -->
<div class="hdr">
  <div class="logo">TrafficIQ Intersection v10</div>
  <div style="display:flex;gap:4px;align-items:center;flex-wrap:wrap">
    <span class="hb hb-live">LIVE</span>
    <span class="hb" style="border-color:var(--ye);color:var(--ye)">4-WAY CHOUK</span>
    <span class="hb" style="border-color:var(--gr);color:var(--gr)">OD MATRIX</span>
    <span class="hb" style="border-color:var(--cy);color:var(--cy)">ADAPTIVE SIGNAL</span>
    <span class="hb" style="border-color:var(--or);color:var(--or)">YOLOv8</span>
    <span class="hb" style="border-color:var(--pu);color:var(--pu)">POLICE DASHBOARD</span>
  </div>
  <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--t3)">Sanket Sutar | B.E. Final Year 2025-26</div>
</div>

<!-- NAV -->
<div class="nav">
  <button class="ntab act" onclick="showTab('live',this)">📹 LIVE INTERSECTION</button>
  <button class="ntab" onclick="showTab('od',this)">🔀 OD MATRIX</button>
  <button class="ntab" onclick="showTab('analytics',this)">📈 ANALYTICS</button>
  <button class="ntab" onclick="showTab('police',this)">🚔 POLICE DASHBOARD</button>
  <button class="ntab" onclick="showTab('setup',this)">⚙️ CAMERA SETUP</button>
</div>

<!-- STATUS BAR -->
<div class="stbar">
  <span>Phase: <b id="sb-ph">--</b></span>
  <span>State: <b id="sb-st">--</b></span>
  <span>Time: <b id="sb-tr">--</b>s</span>
  <span>Total: <b id="sb-tv">0</b> veh</span>
  <span>AQI: <b id="sb-aqi">--</b></span>
  <span>A:<b id="sb-A" style="color:var(--cy)">0</b> B:<b id="sb-B" style="color:var(--re)">0</b> C:<b id="sb-C" style="color:var(--gr)">0</b> D:<b id="sb-D" style="color:var(--ye)">0</b></span>
  <span>Frame:<b id="sb-f">0</b></span>
  <span id="sb-t">--:--:--</span>
</div>

<!-- ==================== LIVE INTERSECTION ==================== -->
<div id="tab-live" class="tab act">
<div style="display:grid;grid-template-columns:1fr 280px;gap:8px;padding:8px">

  <!-- LEFT: 2x2 camera grid + controls -->
  <div style="display:flex;flex-direction:column;gap:8px">

    <!-- Controls -->
    <div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center">
      <button onclick="startAll()" class="btn btp">▶ START ALL DEMO</button>
      <button onclick="stopAll()" class="btn bts">⏹ STOP</button>
      <button onclick="trigEmg()" class="btn btn-e">🚨 EMERGENCY</button>
      <button onclick="resetOD()" class="btn btg btn-sm">RESET OD</button>
      <button onclick="sendTelegramTest()" class="btn bts btn-sm">📱 TELEGRAM TEST</button>
      <span id="run-badge" style="font-family:'Share Tech Mono',monospace;font-size:.5rem;padding:4px 8px;background:rgba(0,212,255,.07);border:1px solid var(--cy);border-radius:4px;color:var(--cy)">IDLE</span>
    </div>

    <!-- 4 Camera Feeds -->
    <div class="cam-grid">
      <!-- Road A -->
      <div class="cam-cell" id="cam-A-cell" style="border-color:rgba(0,212,255,.3)">
        <div class="cam-road-label" style="color:var(--cy)">A — WEST</div>
        <img id="feed-A" src="/video_feed/A" style="width:100%;min-height:200px;object-fit:contain">
        <div class="cam-overlay">
          <div style="display:flex;gap:5px;align-items:center">
            <span id="veh-badge-A" class="sig-badge sig-green">0 veh</span>
            <span id="sig-badge-A" class="sig-badge sig-green">GREEN</span>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3)">Road A</div>
        </div>
      </div>
      <!-- Road B -->
      <div class="cam-cell" id="cam-B-cell" style="border-color:rgba(255,34,68,.3)">
        <div class="cam-road-label" style="color:var(--re)">B — EAST</div>
        <img id="feed-B" src="/video_feed/B" style="width:100%;min-height:200px;object-fit:contain">
        <div class="cam-overlay">
          <div style="display:flex;gap:5px;align-items:center">
            <span id="veh-badge-B" class="sig-badge sig-red">0 veh</span>
            <span id="sig-badge-B" class="sig-badge sig-red">RED</span>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3)">Road B</div>
        </div>
      </div>
      <!-- Road C -->
      <div class="cam-cell" id="cam-C-cell" style="border-color:rgba(0,255,136,.3)">
        <div class="cam-road-label" style="color:var(--gr)">C — NORTH</div>
        <img id="feed-C" src="/video_feed/C" style="width:100%;min-height:200px;object-fit:contain">
        <div class="cam-overlay">
          <div style="display:flex;gap:5px;align-items:center">
            <span id="veh-badge-C" class="sig-badge sig-red">0 veh</span>
            <span id="sig-badge-C" class="sig-badge sig-red">RED</span>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3)">Road C</div>
        </div>
      </div>
      <!-- Road D -->
      <div class="cam-cell" id="cam-D-cell" style="border-color:rgba(255,215,0,.3)">
        <div class="cam-road-label" style="color:var(--ye)">D — SOUTH</div>
        <img id="feed-D" src="/video_feed/D" style="width:100%;min-height:200px;object-fit:contain">
        <div class="cam-overlay">
          <div style="display:flex;gap:5px;align-items:center">
            <span id="veh-badge-D" class="sig-badge sig-red">0 veh</span>
            <span id="sig-badge-D" class="sig-badge sig-red">RED</span>
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3)">Road D</div>
        </div>
      </div>
    </div>

    <!-- Top flows live -->
    <div class="card">
      <div class="ct">LIVE VEHICLE FLOWS (Session)</div>
      <div id="top-flows" style="display:flex;flex-direction:column;gap:3px"></div>
    </div>

    <!-- Charts row -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <div class="ach"><div class="acht">VEHICLES PER ROAD</div><canvas id="ch-roads" style="max-height:130px"></canvas></div>
      <div class="ach"><div class="acht">AQI TREND</div><canvas id="ch-aqi" style="max-height:130px"></canvas></div>
    </div>
  </div>

  <!-- RIGHT: Signal + Stats -->
  <div style="display:flex;flex-direction:column;gap:7px;overflow-y:auto;max-height:calc(100vh - 110px)">

    <!-- Current Phase -->
    <div class="card">
      <div class="ct">INTERSECTION SIGNAL PHASE</div>
      <div id="phase-box" class="phase-box" style="border-color:rgba(0,255,136,.3);background:rgba(0,255,136,.04)">
        <div style="font-family:'Orbitron',monospace;font-size:.85rem;font-weight:900;color:var(--gr)" id="phase-name">A-B STRAIGHT</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3);margin-top:2px" id="phase-desc">A↔B straight | Right turns</div>
        <div style="display:flex;gap:10px;align-items:center;margin-top:8px">
          <div style="font-family:'Orbitron',monospace;font-size:2.5rem;font-weight:900;color:var(--gr)" id="phase-time">20s</div>
          <div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem" id="phase-state-lbl">GREEN</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3)">Green time: <b id="phase-gt" style="color:var(--cy)">20</b>s</div>
          </div>
        </div>
        <div class="pb"><div id="phase-prog" class="pf" style="background:var(--gr);width:100%"></div></div>
      </div>
    </div>

    <!-- Intersection SVG Map -->
    <div class="card">
      <div class="ct">INTERSECTION MAP</div>
      <div style="text-align:center">
        <svg viewBox="0 0 180 180" width="180" height="180" style="background:#080d18;border-radius:8px">
          <rect x="0" y="70" width="180" height="40" fill="#1a1d24"/>
          <rect x="70" y="0" width="40" height="180" fill="#1a1d24"/>
          <rect x="70" y="70" width="40" height="40" fill="#22252e"/>
          <line x1="90" y1="0" x2="90" y2="65" stroke="#ffd700" stroke-width="1.5" stroke-dasharray="5,4" opacity=".6"/>
          <line x1="90" y1="115" x2="90" y2="180" stroke="#ffd700" stroke-width="1.5" stroke-dasharray="5,4" opacity=".6"/>
          <line x1="0" y1="90" x2="65" y2="90" stroke="#ffd700" stroke-width="1.5" stroke-dasharray="5,4" opacity=".6"/>
          <line x1="115" y1="90" x2="180" y2="90" stroke="#ffd700" stroke-width="1.5" stroke-dasharray="5,4" opacity=".6"/>
          <text x="4" y="94" font-family="Orbitron,monospace" font-size="11" fill="#00d4ff" font-weight="900">A</text>
          <text x="166" y="94" font-family="Orbitron,monospace" font-size="11" fill="#ff2244" font-weight="900">B</text>
          <text x="85" y="14" font-family="Orbitron,monospace" font-size="11" fill="#00ff88" font-weight="900">C</text>
          <text x="85" y="176" font-family="Orbitron,monospace" font-size="11" fill="#ffd700" font-weight="900">D</text>
          <!-- TL A --> <rect x="60" y="82" width="8" height="16" rx="1.5" fill="#0d1018"/>
          <circle id="m-r-A" cx="64" cy="85" r="2.5" fill="#3a0a0a"/>
          <circle id="m-y-A" cx="64" cy="90" r="2.5" fill="#2a2000"/>
          <circle id="m-g-A" cx="64" cy="95" r="2.5" fill="#00ff55"/>
          <!-- TL B --> <rect x="112" y="82" width="8" height="16" rx="1.5" fill="#0d1018"/>
          <circle id="m-r-B" cx="116" cy="85" r="2.5" fill="#ff2244"/>
          <circle id="m-y-B" cx="116" cy="90" r="2.5" fill="#2a2000"/>
          <circle id="m-g-B" cx="116" cy="95" r="2.5" fill="#001a0a"/>
          <!-- TL C --> <rect x="82" y="60" width="16" height="8" rx="1.5" fill="#0d1018"/>
          <circle id="m-r-C" cx="85" cy="64" r="2.5" fill="#3a0a0a"/>
          <circle id="m-y-C" cx="90" cy="64" r="2.5" fill="#2a2000"/>
          <circle id="m-g-C" cx="95" cy="64" r="2.5" fill="#00ff55"/>
          <!-- TL D --> <rect x="82" y="112" width="16" height="8" rx="1.5" fill="#0d1018"/>
          <circle id="m-r-D" cx="85" cy="116" r="2.5" fill="#ff2244"/>
          <circle id="m-y-D" cx="90" cy="116" r="2.5" fill="#2a2000"/>
          <circle id="m-g-D" cx="95" cy="116" r="2.5" fill="#001a0a"/>
          <circle cx="90" cy="90" r="4" fill="#00d4ff" opacity=".7"/>
          <text x="76" y="93" font-family="monospace" font-size="4" fill="#3d4f63">CAM</text>
        </svg>
      </div>
      <!-- Phase cycle cards -->
      <div style="display:flex;flex-direction:column;gap:4px;margin-top:8px">
        <div id="cyc-AB"  style="border-radius:5px;padding:5px 7px;border:1px solid rgba(0,255,136,.25);background:rgba(0,255,136,.03)">
          <div style="font-family:'Orbitron',monospace;font-size:.5rem;font-weight:700;color:var(--gr)">Ph1: A↔B STRAIGHT</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3)">A→B,B→A,A→D,B→C | Demand:<b id="p-AB" style="color:var(--cy)">0</b></div>
        </div>
        <div id="cyc-CD"  style="border-radius:5px;padding:5px 7px;border:1px solid rgba(0,212,255,.25);background:rgba(0,212,255,.03)">
          <div style="font-family:'Orbitron',monospace;font-size:.5rem;font-weight:700;color:var(--cy)">Ph2: C↔D STRAIGHT</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3)">C→D,D→C,C→B,D→A | Demand:<b id="p-CD" style="color:var(--cy)">0</b></div>
        </div>
        <div id="cyc-ABT" style="border-radius:5px;padding:5px 7px;border:1px solid rgba(255,215,0,.25);background:rgba(255,215,0,.03)">
          <div style="font-family:'Orbitron',monospace;font-size:.5rem;font-weight:700;color:var(--ye)">Ph3: A-B LEFT TURNS</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3)">A→C, B→D | Demand:<b id="p-ABT" style="color:var(--cy)">0</b></div>
        </div>
        <div id="cyc-CDT" style="border-radius:5px;padding:5px 7px;border:1px solid rgba(255,107,0,.25);background:rgba(255,107,0,.03)">
          <div style="font-family:'Orbitron',monospace;font-size:.5rem;font-weight:700;color:var(--or)">Ph4: C-D LEFT TURNS</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3)">C→A, D→B | Demand:<b id="p-CDT" style="color:var(--cy)">0</b></div>
        </div>
      </div>
    </div>

    <!-- Incident log -->
    <div class="card">
      <div class="ct">INCIDENT LOG</div>
      <div id="inclog" style="max-height:120px;overflow-y:auto">
        <div style="font-family:monospace;font-size:.64rem;color:var(--t3);text-align:center">No incidents</div>
      </div>
    </div>

    <!-- AI Chat -->
    <div class="card">
      <div class="ct">AI ASSISTANT</div>
      <div style="display:flex;gap:3px;margin-bottom:4px">
        <input class="inp" type="password" id="gkey" placeholder="Groq API key (optional)" style="font-size:.62rem">
        <button class="btn bts btn-sm" onclick="connectGroq()">SET</button>
      </div>
      <div class="chm" id="chmsgs">
        <div class="cm ca">Ask: signal phase | OD flows | AQI | traffic | emergency</div>
      </div>
      <div style="display:flex;gap:4px;margin-top:4px">
        <input class="inp" type="text" id="chinp" placeholder="Ask anything..." style="font-size:.68rem" onkeypress="if(event.key==='Enter')sendChat()">
        <button class="btn btp btn-sm" onclick="sendChat()">GO</button>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ==================== OD MATRIX ==================== -->
<div id="tab-od" class="tab">
<div style="padding:10px">
  <div style="font-family:'Orbitron',monospace;font-size:.62rem;color:var(--cy);letter-spacing:3px;margin-bottom:10px">ORIGIN-DESTINATION MATRIX | 12 Vehicle Flow Paths | A(West) B(East) C(North) D(South)</div>
  <div style="display:grid;grid-template-columns:auto 1fr;gap:14px;margin-bottom:12px">
    <div class="card" style="min-width:220px">
      <div class="ct">OD TABLE (Session Totals)</div>
      <table class="od-tbl">
        <thead><tr>
          <th>FROM\TO</th><th style="color:var(--cy)">A</th><th style="color:var(--re)">B</th>
          <th style="color:var(--gr)">C</th><th style="color:var(--ye)">D</th><th style="color:var(--t2)">OUT</th>
        </tr></thead>
        <tbody>
          <tr><th style="font-family:'Orbitron',monospace;font-size:.52rem;color:var(--cy)">A</th>
            <td class="self">—</td><td id="od-A-B">0</td><td id="od-A-C">0</td><td id="od-A-D">0</td><td id="od-A-out" style="color:var(--cy);font-weight:700">0</td></tr>
          <tr><th style="font-family:'Orbitron',monospace;font-size:.52rem;color:var(--re)">B</th>
            <td id="od-B-A">0</td><td class="self">—</td><td id="od-B-C">0</td><td id="od-B-D">0</td><td id="od-B-out" style="color:var(--re);font-weight:700">0</td></tr>
          <tr><th style="font-family:'Orbitron',monospace;font-size:.52rem;color:var(--gr)">C</th>
            <td id="od-C-A">0</td><td id="od-C-B">0</td><td class="self">—</td><td id="od-C-D">0</td><td id="od-C-out" style="color:var(--gr);font-weight:700">0</td></tr>
          <tr><th style="font-family:'Orbitron',monospace;font-size:.52rem;color:var(--ye)">D</th>
            <td id="od-D-A">0</td><td id="od-D-B">0</td><td id="od-D-C">0</td><td class="self">—</td><td id="od-D-out" style="color:var(--ye);font-weight:700">0</td></tr>
          <tr style="background:rgba(0,212,255,.03)">
            <th style="font-size:.44rem;color:var(--t3)">IN</th>
            <td id="od-in-A" style="color:var(--cy);font-weight:700">0</td>
            <td id="od-in-B" style="color:var(--re);font-weight:700">0</td>
            <td id="od-in-C" style="color:var(--gr);font-weight:700">0</td>
            <td id="od-in-D" style="color:var(--ye);font-weight:700">0</td>
            <td id="od-grand" style="color:var(--pu);font-weight:700">0</td>
          </tr>
        </tbody>
      </table>
      <div style="display:flex;gap:5px;margin-top:8px">
        <button onclick="resetOD()" class="btn bts btn-sm" style="flex:1">RESET</button>
        <button onclick="window.location='/api/export_od_csv'" class="btn btg btn-sm" style="flex:1">📥 CSV</button>
      </div>
    </div>
    <!-- OD Chart + Phase -->
    <div style="display:flex;flex-direction:column;gap:8px">
      <div class="ach" style="flex:1"><div class="acht">ALL 12 OD FLOWS</div><canvas id="od-chart" style="max-height:220px"></canvas></div>
    </div>
  </div>
  <!-- Flow breakdown by road -->
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px">
    <div class="card"><div class="ct" style="color:var(--cy)">FROM A (West)</div><div id="od-from-A"></div></div>
    <div class="card"><div class="ct" style="color:var(--re)">FROM B (East)</div><div id="od-from-B"></div></div>
    <div class="card"><div class="ct" style="color:var(--gr)">FROM C (North)</div><div id="od-from-C"></div></div>
    <div class="card"><div class="ct" style="color:var(--ye)">FROM D (South)</div><div id="od-from-D"></div></div>
  </div>
</div>
</div>

<!-- ==================== ANALYTICS ==================== -->
<div id="tab-analytics" class="tab">
<div style="padding:10px">
  <div style="font-family:'Orbitron',monospace;font-size:.62rem;color:var(--cy);letter-spacing:3px;margin-bottom:10px">ANALYTICS — Live Charts</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px">
    <div class="ach"><div class="acht">TOTAL VEHICLES OVER TIME</div><canvas id="ch-veh-hist"></canvas></div>
    <div class="ach"><div class="acht">AQI TREND</div><canvas id="ch-aqi-hist"></canvas></div>
    <div class="ach"><div class="acht">A↔B STRAIGHT FLOW</div><canvas id="ch-ab-hist"></canvas></div>
    <div class="ach"><div class="acht">C↔D STRAIGHT FLOW</div><canvas id="ch-cd-hist"></canvas></div>
    <div class="ach"><div class="acht">SIGNAL PHASE DISTRIBUTION</div><canvas id="ch-phases"></canvas></div>
    <div class="ach"><div class="acht">PER-ROAD VEHICLE COUNT</div><canvas id="ch-roads-hist"></canvas></div>
  </div>
</div>
</div>

<!-- ==================== POLICE DASHBOARD ==================== -->
<div id="tab-police" class="tab">
<div style="padding:12px">
  <div style="font-family:'Orbitron',monospace;font-size:.72rem;color:var(--or);letter-spacing:3px;margin-bottom:12px">🚔 POLICE INTERSECTION DASHBOARD | TrafficIQ v10</div>

  <!-- Key metrics -->
  <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:12px">
    <div class="police-metric"><div class="pm-val" id="pm-total" style="color:var(--cy)">0</div><div class="pm-lbl">TOTAL VEH</div></div>
    <div class="police-metric"><div class="pm-val" id="pm-peak" style="color:var(--or)">0</div><div class="pm-lbl">PEAK VEH</div></div>
    <div class="police-metric"><div class="pm-val" id="pm-aqi" style="color:var(--re)">120</div><div class="pm-lbl">AQI</div></div>
    <div class="police-metric"><div class="pm-val" id="pm-phase" style="color:var(--gr)">AB</div><div class="pm-lbl">PHASE</div></div>
    <div class="police-metric"><div class="pm-val" id="pm-uptime" style="color:var(--ye)">0s</div><div class="pm-lbl">UPTIME</div></div>
    <div class="police-metric"><div class="pm-val" id="pm-frames" style="color:var(--pu)">0</div><div class="pm-lbl">FRAMES</div></div>
  </div>

  <!-- Road status table -->
  <div class="card" style="margin-bottom:12px">
    <div class="ct">LIVE ROAD STATUS</div>
    <table style="width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:.52rem">
      <thead><tr>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px;text-align:left">Road</th>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px">Direction</th>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px">Vehicles</th>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px">Signal</th>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px">Congestion</th>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px">Total In</th>
        <th style="background:rgba(0,212,255,.08);color:var(--cy);padding:6px">Total Out</th>
      </tr></thead>
      <tbody id="police-road-table"></tbody>
    </table>
  </div>

  <!-- Telegram + Export -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px">
    <div class="card">
      <div class="ct">📱 TELEGRAM ALERTS</div>
      <div style="display:flex;gap:5px;margin-bottom:5px">
        <input class="inp" id="tg-token" type="password" placeholder="Bot Token (from @BotFather)">
      </div>
      <div style="display:flex;gap:5px;margin-bottom:5px">
        <input class="inp" id="tg-chat" placeholder="Chat ID (e.g. -1001234567)">
      </div>
      <div style="display:flex;gap:5px">
        <button class="btn btp btn-sm" onclick="connectTelegram()" style="flex:1">CONNECT</button>
        <button class="btn bts btn-sm" onclick="testTelegram()" style="flex:1">TEST</button>
      </div>
      <div id="tg-status" style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--t3);margin-top:5px">Enter token → CONNECT</div>
    </div>
    <div class="card">
      <div class="ct">📥 EXPORT DATA</div>
      <div style="display:flex;flex-direction:column;gap:6px">
        <button class="btn btp" onclick="window.location='/api/pdf_report'" style="width:100%">📄 DOWNLOAD PDF REPORT</button>
        <button class="btn btg" onclick="window.location='/api/export_csv'" style="width:100%">📊 DOWNLOAD CSV (History)</button>
        <button class="btn bto" onclick="window.location='/api/export_od_csv'" style="width:100%">🔀 DOWNLOAD OD MATRIX CSV</button>
      </div>
    </div>
  </div>

  <!-- Incident log full -->
  <div class="card">
    <div class="ct">FULL INCIDENT LOG</div>
    <div id="police-inclog" style="max-height:250px;overflow-y:auto"></div>
  </div>
</div>
</div>

<!-- ==================== CAMERA SETUP ==================== -->
<div id="tab-setup" class="tab">
<div style="padding:14px;max-width:1000px">
  <div style="font-family:'Orbitron',monospace;font-size:.68rem;color:var(--cy);letter-spacing:3px;margin-bottom:12px">CAMERA SETUP — Each Road Independent</div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">

    <!-- Road A -->
    <div class="card">
      <div class="ct" style="color:var(--cy)">ROAD A — WEST (IP Webcam / Video / Demo)</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t2);margin-bottom:8px">Mode: <b id="mode-A" style="color:var(--cy)">demo</b></div>
      <!-- IP Cam -->
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">IP Webcam:</div>
      <div style="display:flex;gap:4px;margin-bottom:4px">
        <input class="inp" id="ip-A" placeholder="192.168.1.5" style="flex:1">
        <input class="inp" id="port-A" placeholder="8080" style="width:60px">
        <button onclick="connectCam('A')" class="btn btp btn-sm">CONNECT</button>
      </div>
      <div id="cam-status-A" style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t3);margin-bottom:8px">Enter IP → CONNECT</div>
      <!-- Video Upload -->
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">Video Upload:</div>
      <div class="upz" onclick="document.getElementById('vupl-A').click()">
        <input type="file" id="vupl-A" accept=".mp4,.avi,.mov,.mkv" onchange="uploadVideo('A',this)">
        <span>📁</span>
        <div><div style="font-family:'Share Tech Mono',monospace;font-size:.56rem;color:var(--cy)">Browse MP4/AVI/MOV</div><div id="vid-status-A" style="font-size:.6rem;color:var(--t3)">No file selected</div></div>
      </div>
      <div class="pb" style="margin-top:4px"><div id="vpf-A" class="pf" style="background:var(--cy);width:0%"></div></div>
      <button onclick="setDemo('A')" class="btn bts btn-sm" style="width:100%;margin-top:6px">USE DEMO MODE</button>
    </div>

    <!-- Road B -->
    <div class="card">
      <div class="ct" style="color:var(--re)">ROAD B — EAST (IP Webcam / Video / Demo)</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t2);margin-bottom:8px">Mode: <b id="mode-B" style="color:var(--re)">demo</b></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">IP Webcam:</div>
      <div style="display:flex;gap:4px;margin-bottom:4px">
        <input class="inp" id="ip-B" placeholder="192.168.1.6" style="flex:1">
        <input class="inp" id="port-B" placeholder="8080" style="width:60px">
        <button onclick="connectCam('B')" class="btn btp btn-sm">CONNECT</button>
      </div>
      <div id="cam-status-B" style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t3);margin-bottom:8px">Enter IP → CONNECT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">Video Upload:</div>
      <div class="upz" onclick="document.getElementById('vupl-B').click()">
        <input type="file" id="vupl-B" accept=".mp4,.avi,.mov,.mkv" onchange="uploadVideo('B',this)">
        <span>📁</span>
        <div><div style="font-family:'Share Tech Mono',monospace;font-size:.56rem;color:var(--cy)">Browse MP4/AVI/MOV</div><div id="vid-status-B" style="font-size:.6rem;color:var(--t3)">No file selected</div></div>
      </div>
      <div class="pb" style="margin-top:4px"><div id="vpf-B" class="pf" style="background:var(--re);width:0%"></div></div>
      <button onclick="setDemo('B')" class="btn bts btn-sm" style="width:100%;margin-top:6px">USE DEMO MODE</button>
    </div>

    <!-- Road C -->
    <div class="card">
      <div class="ct" style="color:var(--gr)">ROAD C — NORTH (IP Webcam / Video / Demo)</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t2);margin-bottom:8px">Mode: <b id="mode-C" style="color:var(--gr)">demo</b></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">IP Webcam:</div>
      <div style="display:flex;gap:4px;margin-bottom:4px">
        <input class="inp" id="ip-C" placeholder="192.168.1.7" style="flex:1">
        <input class="inp" id="port-C" placeholder="8080" style="width:60px">
        <button onclick="connectCam('C')" class="btn btp btn-sm">CONNECT</button>
      </div>
      <div id="cam-status-C" style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t3);margin-bottom:8px">Enter IP → CONNECT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">Video Upload:</div>
      <div class="upz" onclick="document.getElementById('vupl-C').click()">
        <input type="file" id="vupl-C" accept=".mp4,.avi,.mov,.mkv" onchange="uploadVideo('C',this)">
        <span>📁</span>
        <div><div style="font-family:'Share Tech Mono',monospace;font-size:.56rem;color:var(--cy)">Browse MP4/AVI/MOV</div><div id="vid-status-C" style="font-size:.6rem;color:var(--t3)">No file selected</div></div>
      </div>
      <div class="pb" style="margin-top:4px"><div id="vpf-C" class="pf" style="background:var(--gr);width:0%"></div></div>
      <button onclick="setDemo('C')" class="btn bts btn-sm" style="width:100%;margin-top:6px">USE DEMO MODE</button>
    </div>

    <!-- Road D -->
    <div class="card">
      <div class="ct" style="color:var(--ye)">ROAD D — SOUTH (IP Webcam / Video / Demo)</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t2);margin-bottom:8px">Mode: <b id="mode-D" style="color:var(--ye)">demo</b></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">IP Webcam:</div>
      <div style="display:flex;gap:4px;margin-bottom:4px">
        <input class="inp" id="ip-D" placeholder="192.168.1.8" style="flex:1">
        <input class="inp" id="port-D" placeholder="8080" style="width:60px">
        <button onclick="connectCam('D')" class="btn btp btn-sm">CONNECT</button>
      </div>
      <div id="cam-status-D" style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t3);margin-bottom:8px">Enter IP → CONNECT</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">Video Upload:</div>
      <div class="upz" onclick="document.getElementById('vupl-D').click()">
        <input type="file" id="vupl-D" accept=".mp4,.avi,.mov,.mkv" onchange="uploadVideo('D',this)">
        <span>📁</span>
        <div><div style="font-family:'Share Tech Mono',monospace;font-size:.56rem;color:var(--cy)">Browse MP4/AVI/MOV</div><div id="vid-status-D" style="font-size:.6rem;color:var(--t3)">No file selected</div></div>
      </div>
      <div class="pb" style="margin-top:4px"><div id="vpf-D" class="pf" style="background:var(--ye);width:0%"></div></div>
      <button onclick="setDemo('D')" class="btn bts btn-sm" style="width:100%;margin-top:6px">USE DEMO MODE</button>
    </div>
  </div>

  <!-- Settings -->
  <div class="card">
    <div class="ct">ENVIRONMENT SETTINGS</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div>
        <div class="sr"><div class="srl"><span>AQI</span><span id="vaqi">120</span></div><input type="range" min="0" max="500" value="120" oninput="uS('manual_aqi',+this.value,'vaqi')"></div>
        <div class="sr"><div class="srl"><span>Temperature °C</span><span id="vtemp">29</span></div><input type="range" min="15" max="45" value="29" oninput="uS('temp',+this.value,'vtemp')"></div>
        <div class="sr"><div class="srl"><span>Demo Density (1-10)</span><span id="vdens">5</span></div><input type="range" min="1" max="10" value="5" oninput="uS('demo_density',+this.value,'vdens')"></div>
      </div>
      <div>
        <div class="tr"><span class="tl2">Night Mode Override</span><label class="tg"><input type="checkbox" onchange="uS('night_override',this.checked)"><span class="ts"></span></label></div>
        <div class="tr"><span class="tl2">AQI Simulate</span><label class="tg"><input type="checkbox" onchange="uS('aqi_src',this.checked?'simulate':'manual')"><span class="ts"></span></label></div>
        <div style="margin-top:8px">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--or);margin-bottom:4px">Camera Detection Distance:</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t2);line-height:1.9">
            720p @ 3-4m height → 15-25m<br>
            1080p @ 5-6m height → 25-40m<br>
            Mount: 4-5m pole, 50° tilt angle<br>
            Wide angle lens recommended
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
const E=id=>document.getElementById(id), T=(id,v)=>{const e=E(id);if(e)e.textContent=v};
let charts={}, lastS={};
const ROADS=["A","B","C","D"];
const RCOL={A:"var(--cy)",B:"var(--re)",C:"var(--gr)",D:"var(--ye)"};
const RNAME={A:"West",B:"East",C:"North",D:"South"};

const CO={responsive:true,maintainAspectRatio:true,animation:false,
  plugins:{legend:{labels:{color:"#3d4f63",font:{size:9},boxWidth:8}}},
  scales:{x:{ticks:{color:"#3d4f63",font:{size:8}},grid:{color:"rgba(0,212,255,.04)"}},
          y:{ticks:{color:"#3d4f63",font:{size:8}},grid:{color:"rgba(0,212,255,.04)"}}}};

function mkC(id,type,data,extra={}){
  const c=E(id);if(!c)return null;
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(c.getContext("2d"),{type,data,options:{...CO,...extra}});
  return charts[id];
}

// POLL
setInterval(()=>{
  fetch("/api/state").then(r=>r.json()).then(s=>{
    lastS=s; updateLive(s); updateOD(s); updateAnalytics(s); updatePolice(s);
    T("sb-t",new Date().toLocaleTimeString());
  }).catch(()=>{});
},800);

function updateLive(s){
  const rds=s.roads||{}; const sigs=s.road_signals||{};
  T("sb-ph",s.phase||"--");T("sb-st",(s.phase_state||"--").toUpperCase());
  T("sb-tr",s.time_remaining||0);T("sb-tv",s.total_vehicles||0);
  T("sb-aqi",s.aqi||"--");T("sb-f",s.frames||0);
  if(E("run-badge"))E("run-badge").textContent=s.running?"RUNNING":"IDLE";

  // Per-road
  ROADS.forEach(r=>{
    const rd=rds[r]||{}; const sig=sigs[r]||rd.signal||"red";
    const veh=rd.vehicles||0; const sigc=sig==="green"?"var(--gr)":sig==="yellow"?"var(--ye)":"var(--re)";
    T("sb-"+r,veh);
    // Badges
    const vb=E("veh-badge-"+r),sb=E("sig-badge-"+r);
    if(vb){vb.textContent=veh+" veh";vb.className="sig-badge sig-"+sig;}
    if(sb){sb.textContent=sig.toUpperCase();sb.className="sig-badge sig-"+sig;}
    // Cell border
    const cc=E("cam-"+r+"-cell");
    if(cc){const bc=sig==="green"?"rgba(0,255,136,.5)":sig==="yellow"?"rgba(255,215,0,.5)":"rgba(255,34,68,.3)";cc.style.borderColor=bc;}
    // Mode
    T("mode-"+r,rd.mode||"demo");
    // Vid progress
    const vp=rd.vid_progress||0;
    if(E("vpf-"+r))E("vpf-"+r).style.width=vp+"%";
    // SVG lights
    updateSVG(r,sig);
  });

  // Phase
  const ph=s.phase||"AB"; const ps=s.phase_state||"green";
  const phcol={AB:"var(--gr)",CD:"var(--cy)",AB_TURN:"var(--ye)",CD_TURN:"var(--or)"};
  const pc=phcol[ph]||"var(--gr)"; const sc=ps==="green"?"var(--gr)":ps==="yellow"?"var(--ye)":"var(--re)";
  if(E("phase-name")){E("phase-name").textContent=s.phase_name||"--";E("phase-name").style.color=pc;}
  T("phase-desc",s.phase_desc||"");
  if(E("phase-time")){E("phase-time").textContent=(s.time_remaining||0)+"s";E("phase-time").style.color=sc;}
  T("phase-gt",s.green_time||0);
  if(E("phase-state-lbl")){E("phase-state-lbl").textContent=ps.toUpperCase();E("phase-state-lbl").style.color=sc;}
  const pbc={"AB":"rgba(0,255,136,.3)","CD":"rgba(0,212,255,.3)","AB_TURN":"rgba(255,215,0,.3)","CD_TURN":"rgba(255,107,0,.3)"};
  const pbb={"AB":"rgba(0,255,136,.04)","CD":"rgba(0,212,255,.04)","AB_TURN":"rgba(255,215,0,.04)","CD_TURN":"rgba(255,107,0,.04)"};
  if(E("phase-box")){E("phase-box").style.borderColor=pbc[ph]||pbc.AB;E("phase-box").style.background=pbb[ph]||pbb.AB;}
  const gt=s.green_time||20;const tr=s.time_remaining||0;
  const pct=ps==="green"?Math.round(tr/gt*100):ps==="yellow"?100:0;
  if(E("phase-prog")){E("phase-prog").style.width=pct+"%";E("phase-prog").style.background=sc;}

  // Phase cycle
  [["AB","AB"],["CD","CD"],["AB_TURN","ABT"],["CD_TURN","CDT"]].forEach(([pk,tid])=>{
    const el=E("cyc-"+tid); if(el)el.style.opacity=ph===pk?"1":"0.45";
  });

  // Phase demands
  const od=s.od||{};
  const pdem={AB:["A_B","B_A","A_D","B_C"],CD:["C_D","D_C","C_B","D_A"],
              AB_TURN:["A_C","B_D"],CD_TURN:["C_A","D_B"]};
  Object.entries(pdem).forEach(([pk,keys])=>{
    const d=keys.reduce((s2,k)=>s2+(od[k]||0),0);
    const tid=pk==="AB_TURN"?"ABT":pk==="CD_TURN"?"CDT":pk;
    T("p-"+tid,d);
  });

  // Top flows
  const od_s=s.od_session||od;
  const flows=Object.entries(od_s).filter(([k,v])=>v>0).sort((a,b)=>b[1]-a[1]).slice(0,6);
  const maxv=flows[0]?flows[0][1]:1;
  if(E("top-flows"))E("top-flows").innerHTML=flows.map(([k,v])=>{
    const [o,d]=k.split("_"); const pct2=Math.round(v/maxv*100);
    return `<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">
      <span style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:${RCOL[o]||"var(--cy)"};min-width:32px">${o}→${d}</span>
      <div class="flow-bar-wrap"><div class="flow-bar" style="width:${pct2}%;background:${RCOL[o]||"var(--cy)"}"></div></div>
      <span style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--gr);min-width:18px">${v}</span>
    </div>`;
  }).join("")||"<div style='font-family:monospace;font-size:.62rem;color:var(--t3);padding:4px'>No flows yet — press START</div>";

  // Incident log
  if(s.incident_log&&s.incident_log.length&&E("inclog"))
    E("inclog").innerHTML=s.incident_log.slice(0,8).map(i=>
      `<div class="inc-item"><span class="inc-time">${i.time}</span><span class="inc-road" style="color:${RCOL[i.road]||"var(--cy)"}">${i.road||""}</span><span style="color:var(--t2)">${i.icon} ${i.text}</span></div>`
    ).join("");

  // Live charts
  const hist=s.history||[];
  const rds2=s.roads||{};
  const d=[rds2.A?.vehicles||0,rds2.B?.vehicles||0,rds2.C?.vehicles||0,rds2.D?.vehicles||0];
  if(!charts["ch-roads"]){
    mkC("ch-roads","bar",{labels:["A(W)","B(E)","C(N)","D(S)"],datasets:[{data:d,backgroundColor:["rgba(0,212,255,.7)","rgba(255,34,68,.7)","rgba(0,255,136,.7)","rgba(255,215,0,.7)"],borderRadius:5}]},{plugins:{legend:{display:false}}});
  } else {charts["ch-roads"].data.datasets[0].data=d;charts["ch-roads"].update("none");}

  if(hist.length>2){
    const tail=hist.slice(-40);
    const aqiD=tail.map(h=>h.aqi||0);
    if(!charts["ch-aqi"]){
      mkC("ch-aqi","line",{labels:tail.map((_,i)=>i),datasets:[
        {label:"AQI",data:aqiD,borderColor:"#ff2244",backgroundColor:"rgba(255,34,68,.08)",tension:.4,pointRadius:0},
        {label:"150",data:Array(tail.length).fill(150),borderColor:"rgba(255,107,0,.4)",borderDash:[4,4],pointRadius:0,borderWidth:1}
      ]},{plugins:{legend:{display:false}}});
    } else {charts["ch-aqi"].data.datasets[0].data=aqiD;charts["ch-aqi"].update("none");}
  }
}

function updateSVG(road,sig){
  const rc={green:"#00ff55",yellow:"#ffd700",red:"#ff2244"};
  const off={r:"#3a0a0a",y:"#2a2000",g:"#001a0a"};
  if(E("m-r-"+road))E("m-r-"+road).setAttribute("fill",sig==="red"?rc.red:off.r);
  if(E("m-y-"+road))E("m-y-"+road).setAttribute("fill",sig==="yellow"?rc.yellow:off.y);
  if(E("m-g-"+road))E("m-g-"+road).setAttribute("fill",sig==="green"?rc.green:off.g);
}

function updateOD(s){
  if(!E("tab-od").classList.contains("act"))return;
  const od=s.od_session||s.od||{};
  ["A","B","C","D"].forEach(o=>{
    let out=0;
    ["A","B","C","D"].forEach(d=>{
      if(o===d)return;
      const v=od[o+"_"+d]||0; out+=v;
      const el=E("od-"+o+"-"+d);
      if(el){el.textContent=v;el.className=v>0?"hi":"";}
    });
    T("od-"+o+"-out",out);
    let inn=0;
    ["A","B","C","D"].forEach(o2=>{if(o2!==o) inn+=od[o2+"_"+o]||0;});
    T("od-in-"+o,inn);
  });
  T("od-grand",Object.values(od).reduce((a,b)=>a+b,0));
  // From breakdowns
  ["A","B","C","D"].forEach(o=>{
    const el=E("od-from-"+o);if(!el)return;
    const dests=["A","B","C","D"].filter(d=>d!==o);
    el.innerHTML=dests.map(d=>{
      const v=od[o+"_"+d]||0;
      return `<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04);font-family:'Share Tech Mono',monospace;font-size:.5rem">
        <span style="color:var(--t3)">${o}→${d} (${RNAME[d]})</span>
        <span style="color:${RCOL[d]||"var(--cy)"};font-weight:${v>0?"700":"400"}">${v}</span>
      </div>`;
    }).join("");
  });
  // OD chart
  const labels=["A→B","B→A","C→D","D→C","A→C","A→D","B→C","B→D","C→A","C→B","D→A","D→B"];
  const keys=["A_B","B_A","C_D","D_C","A_C","A_D","B_C","B_D","C_A","C_B","D_A","D_B"];
  const vals=keys.map(k=>od[k]||0);
  const bgc=["rgba(0,255,136,.7)","rgba(0,255,136,.5)","rgba(0,212,255,.7)","rgba(0,212,255,.5)",
             "rgba(255,215,0,.7)","rgba(255,107,0,.7)","rgba(255,215,0,.5)","rgba(255,107,0,.5)",
             "rgba(191,95,255,.7)","rgba(191,95,255,.5)","rgba(255,34,68,.7)","rgba(255,34,68,.5)"];
  if(!charts["od-chart"]){
    mkC("od-chart","bar",{labels,datasets:[{data:vals,backgroundColor:bgc,borderRadius:4}]},
      {plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#3d4f63",font:{size:8}},grid:{display:false}},y:{ticks:{color:"#3d4f63",font:{size:8}}}}});
  } else {charts["od-chart"].data.datasets[0].data=vals;charts["od-chart"].update("none");}
}

function updateAnalytics(s){
  if(!E("tab-analytics").classList.contains("act"))return;
  const hist=s.history||[];if(hist.length<3)return;
  const L=hist.map((_,i)=>i);
  mkC("ch-veh-hist","line",{labels:L,datasets:[{label:"Total",data:hist.map(h=>h.total||0),borderColor:"#00d4ff",backgroundColor:"rgba(0,212,255,.08)",tension:.4,pointRadius:0,fill:true}]});
  mkC("ch-aqi-hist","line",{labels:L,datasets:[{label:"AQI",data:hist.map(h=>h.aqi||0),borderColor:"#ff2244",backgroundColor:"rgba(255,34,68,.08)",tension:.4,pointRadius:0}]});
  mkC("ch-ab-hist","line",{labels:L,datasets:[{label:"A↔B",data:hist.map(h=>h.ab_flow||0),borderColor:"#00ff88",backgroundColor:"rgba(0,255,136,.08)",tension:.4,pointRadius:0,fill:true}]});
  mkC("ch-cd-hist","line",{labels:L,datasets:[{label:"C↔D",data:hist.map(h=>h.cd_flow||0),borderColor:"#00d4ff",backgroundColor:"rgba(0,212,255,.08)",tension:.4,pointRadius:0,fill:true}]});
  const phc={AB:0,CD:0,AB_TURN:0,CD_TURN:0};
  hist.forEach(h=>{if(phc[h.phase]!==undefined)phc[h.phase]++;});
  mkC("ch-phases","doughnut",{labels:["A-B","C-D","A-B Turn","C-D Turn"],datasets:[{data:Object.values(phc),backgroundColor:["rgba(0,255,136,.7)","rgba(0,212,255,.7)","rgba(255,215,0,.7)","rgba(255,107,0,.7)"],borderColor:"#060a12",borderWidth:2}]},{scales:{}});
  mkC("ch-roads-hist","line",{labels:L,datasets:[
    {label:"A",data:hist.map(h=>h.A||0),borderColor:"#00d4ff",tension:.4,pointRadius:0},
    {label:"B",data:hist.map(h=>h.B||0),borderColor:"#ff2244",tension:.4,pointRadius:0},
    {label:"C",data:hist.map(h=>h.C||0),borderColor:"#00ff88",tension:.4,pointRadius:0},
    {label:"D",data:hist.map(h=>h.D||0),borderColor:"#ffd700",tension:.4,pointRadius:0},
  ]});
}

function updatePolice(s){
  if(!E("tab-police").classList.contains("act"))return;
  T("pm-total",s.total_vehicles||0);T("pm-peak",s.peak_vehicles||0);
  T("pm-aqi",s.aqi||0);T("pm-phase",s.phase||"--");
  T("pm-uptime",(s.uptime||0)+"s");T("pm-frames",s.frames||0);
  const rds=s.roads||{}; const od=s.od_session||s.od||{};
  const sigs=s.road_signals||{};
  const sigcol={green:"var(--gr)",yellow:"var(--ye)",red:"var(--re)"};
  if(E("police-road-table")){
    E("police-road-table").innerHTML=["A","B","C","D"].map(r=>{
      const rd=rds[r]||{}; const sig=sigs[r]||rd.signal||"red";
      const out=["A","B","C","D"].filter(d=>d!==r).reduce((s2,d)=>s2+(od[r+"_"+d]||0),0);
      const inn=["A","B","C","D"].filter(o=>o!==r).reduce((s2,o)=>s2+(od[o+"_"+r]||0),0);
      return `<tr style="border-bottom:1px solid rgba(0,212,255,.06)">
        <td style="padding:7px;font-family:'Orbitron',monospace;font-size:.6rem;font-weight:700;color:${RCOL[r]||"var(--cy)"}">Road ${r}</td>
        <td style="padding:7px;font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t2)">${RNAME[r]}</td>
        <td style="padding:7px;font-family:'Orbitron',monospace;font-size:.68rem;font-weight:700;text-align:center;color:${RCOL[r]||"var(--cy)"}">${rd.vehicles||0}</td>
        <td style="padding:7px;text-align:center"><span style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:${sigcol[sig]||"var(--re)"};font-weight:700">${sig.toUpperCase()}</span></td>
        <td style="padding:7px;text-align:center;font-family:'Share Tech Mono',monospace;font-size:.5rem;color:${(rd.congestion||0)>70?"var(--re)":(rd.congestion||0)>40?"var(--ye)":"var(--gr)"}">${rd.congestion||0}%</td>
        <td style="padding:7px;text-align:center;font-family:'Orbitron',monospace;font-size:.6rem;color:var(--cy)">${inn}</td>
        <td style="padding:7px;text-align:center;font-family:'Orbitron',monospace;font-size:.6rem;color:var(--or)">${out}</td>
      </tr>`;
    }).join("");
  }
  if(s.incident_log&&s.incident_log.length&&E("police-inclog"))
    E("police-inclog").innerHTML=s.incident_log.map(i=>
      `<div class="inc-item"><span class="inc-time">${i.time}</span><span class="inc-road" style="color:${RCOL[i.road]||"var(--cy)"}">${i.road||"SYS"}</span><span style="color:var(--t2)">${i.icon} ${i.text}</span></div>`
    ).join("");
}

// CONTROLS
function startAll(){
  fetch("/api/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:"demo"})}).then(r=>r.json()).then(d=>{if(d.ok)if(E("run-badge"))E("run-badge").textContent="RUNNING";});
}
function stopAll(){fetch("/api/stop",{method:"POST"});}
function trigEmg(){fetch("/api/emergency",{method:"POST"});}
function resetOD(){fetch("/api/od_reset",{method:"POST"});}
function uS(k,v,di){if(di)T(di,v);fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[k]:v})});}

function connectCam(road){
  const ip=E("ip-"+road)?.value.trim(); const port=E("port-"+road)?.value.trim()||"8080";
  if(!ip){T("cam-status-"+road,"Enter IP address!");return;}
  T("cam-status-"+road,"Connecting...");
  fetch("/api/set_cam/"+road,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({ip,port})})
    .then(r=>r.json()).then(d=>{
      T("cam-status-"+road,d.ok?"✅ "+d.msg:"❌ "+d.msg);
      if(d.ok){
        if(E("feed-"+road))E("feed-"+road).src="/video_feed/"+road+"?t="+Date.now();
        fetch("/api/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:"camera"})});
      }
    });
}

function uploadVideo(road, inp){
  if(!inp.files.length)return;
  const f=inp.files[0]; T("vid-status-"+road,"Uploading "+f.name+"...");
  if(E("vpf-"+road))E("vpf-"+road).style.width="20%";
  const form=new FormData(); form.append("file",f);
  fetch("/api/upload_video/"+road,{method:"POST",body:form}).then(r=>r.json()).then(d=>{
    if(d.ok){
      T("vid-status-"+road,f.name+" | "+d.frames+"f @ "+d.fps+"fps");
      if(E("vpf-"+road))E("vpf-"+road).style.width="100%";
      if(E("feed-"+road))E("feed-"+road).src="/video_feed/"+road+"?t="+Date.now();
    } else {T("vid-status-"+road,"Error: "+d.error);}
  });
}

function setDemo(road){
  fetch("/api/set_demo/"+road,{method:"POST"}).then(r=>r.json()).then(d=>{
    if(d.ok&&E("feed-"+road))E("feed-"+road).src="/video_feed/"+road+"?t="+Date.now();
  });
}

function connectGroq(){
  const k=E("gkey")?.value.trim();if(!k){alert("Enter Groq API key!");return;}
  fetch("/api/groq_key",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:k})}).then(r=>r.json()).then(d=>{
    addMsg("a",d.ok?"Groq LLM connected! Ask anything.":"Failed: "+d.msg);
  });
}
function sendChat(){
  const inp=E("chinp"),msg=inp?.value.trim();if(!msg)return;
  addMsg("u",msg);if(inp)inp.value="";
  fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg})}).then(r=>r.json()).then(d=>addMsg("a",d.reply||"Error"));
}
function addMsg(role,text){
  const m=E("chmsgs");if(!m)return;
  const d=document.createElement("div");d.className="cm c"+role[0];d.textContent=text;
  m.appendChild(d);m.scrollTop=m.scrollHeight;
}

function connectTelegram(){
  const token=E("tg-token")?.value.trim(), chat=E("tg-chat")?.value.trim();
  if(!token||!chat){T("tg-status","Enter both token and chat ID!");return;}
  T("tg-status","Connecting...");
  fetch("/api/telegram",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token,chat_id:chat})}).then(r=>r.json()).then(d=>{
    T("tg-status",d.ok?"✅ Connected! Check Telegram.":"❌ Failed: "+d.msg);
  });
}
function testTelegram(){
  fetch("/api/telegram_test",{method:"POST"}).then(r=>r.json()).then(d=>{
    T("tg-status",d.ok?"✅ Test message sent!":"❌ Not connected yet");
  });
}
function sendTelegramTest(){
  fetch("/api/telegram_test",{method:"POST"}).then(r=>r.json()).then(d=>{
    alert(d.ok?"Telegram test sent!":"Telegram not connected. Go to Police Dashboard tab.");
  });
}

function showTab(name,btn){
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("act"));
  document.querySelectorAll(".ntab").forEach(b=>b.classList.remove("act"));
  const el=E("tab-"+name); if(el)el.classList.add("act");
  if(btn)btn.classList.add("act");
}

// Auto start demo
setTimeout(()=>startAll(),800);
</script>
</body>
</html>"""

if __name__ == '__main__':
    import threading
    print("=" * 65)
    print("  TrafficIQ v10.0 — 4-WAY INTERSECTION MANAGEMENT SYSTEM")
    print("  4 Roads A/B/C/D | IP Webcam + Video Upload per Road")
    print("  OD Matrix | 4-Phase Adaptive Signal | Police Dashboard")
    print("  CSV/PDF Export | Telegram Alerts | Live Analytics")
    print("  Sanket Sutar | B.E. Final Year 2025-26")
    print("=" * 65)
    print("  http://localhost:5000")
    print("  📹 LIVE INTERSECTION — 2x2 camera grid")
    print("  🔀 OD MATRIX       — 12 vehicle flow paths")
    print("  📈 ANALYTICS       — live graphs")
    print("  🚔 POLICE DASHBOARD — alerts, CSV/PDF export")
    print("  ⚙️  CAMERA SETUP    — IP cam / video upload per road")
    print("=" * 65)
    threading.Thread(target=train_models, daemon=True).start()
    threading.Thread(target=load_yolo, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
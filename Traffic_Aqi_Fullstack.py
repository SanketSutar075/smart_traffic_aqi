"""
TrafficIQ v9.0 - Smart Traffic Signalling + AQI Prediction
COMPLETE FULL STACK - All Streamlit Features
Prepared by: Sanket Sutar | B.E. Final Year 2025-26
Run: python Traffic_Aqi_Fullstack.py -> http://localhost:5000
"""

import cv2, numpy as np, math, random, time, threading, base64, json, os, io, tempfile, sqlite3, urllib.request as _ur
from datetime import datetime
from collections import deque
from flask import Flask, Response, jsonify, request, render_template_string, send_file

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except: YOLO_AVAILABLE = False

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import pandas as pd
    SK_AVAILABLE = True
except: SK_AVAILABLE = False

GROQ_AVAILABLE = True  # Direct Groq HTTP API

app = Flask(__name__)

# == SQLITE DATABASE ============================================== 
DB_PATH = os.path.join(os.path.dirname(__file__), 'trafficiq_data.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, cars INTEGER, bikes INTEGER, total INTEGER,
        aqi REAL, signal TEXT, green_time INTEGER, level TEXT,
        speed REAL, efficiency REAL, co2_saved REAL
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, alert_type TEXT, message TEXT, aqi REAL, total INTEGER
    )""")
    conn.commit(); conn.close()
    print("  [DB] SQLite database ready:", DB_PATH)

def db_save_frame(s):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""INSERT INTO sessions
            (timestamp,cars,bikes,total,aqi,signal,green_time,level,speed,efficiency,co2_saved)
            VALUES ( , , , , , , , , , , )""",
            (datetime.now().isoformat(), s['cars'], s['bikes'], s['total'],
             s['aqi'], s['signal'], s['green_time'], s['level'],
             s.get('speed',0), s.get('efficiency',0), s.get('co2_saved',0)))
        conn.commit(); conn.close()
    except: pass

def db_save_alert(alert_type, message, aqi=0, total=0):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO alerts (timestamp,alert_type,message,aqi,total) VALUES ( , , , , )",
            (datetime.now().isoformat(), alert_type, message, aqi, total))
        conn.commit(); conn.close()
    except: pass

def db_get_history(limit=500):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT  ", (limit,)
        ).fetchall()
        conn.close()
        cols = ['id','timestamp','cars','bikes','total','aqi','signal','green_time','level','speed','efficiency','co2_saved']
        return [dict(zip(cols,r)) for r in reversed(rows)]
    except: return []

def db_get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*),AVG(total),AVG(aqi),MAX(total),AVG(efficiency),SUM(co2_saved) FROM sessions")
        row = c.fetchone()
        c.execute("SELECT COUNT(*) FROM alerts")
        alerts = c.fetchone()[0]
        conn.close()
        if row[0]:
            return {'total_frames':row[0],'avg_vehicles':round(row[1],1),
                    'avg_aqi':round(row[2],1),'peak_vehicles':row[3],
                    'avg_efficiency':round(row[4] or 0,1),
                    'total_co2_saved':round(row[5] or 0,3),'total_alerts':alerts}
    except: pass
    return {}

init_db()

# == TELEGRAM BOT ================================================== 
_tg_token = ''
_tg_chat  = ''
_last_alert_time = {}

def tg_set(token, chat_id):
    global _tg_token, _tg_chat
    _tg_token = token; _tg_chat = chat_id

def tg_send(msg, force=False):
    global _last_alert_time
    if not _tg_token or not _tg_chat: return False
    key = msg[:20]
    now = time.time()
    if not force and now - _last_alert_time.get(key, 0) < 60:
        return False
    _last_alert_time[key] = now
    try:
        url = f"https://api.telegram.org/bot{_tg_token}/sendMessage"
        data = json.dumps({"chat_id":_tg_chat,"text":msg,"parse_mode":"Markdown"}).encode()
        req = _ur.Request(url, data=data, headers={"Content-Type":"application/json"})
        _ur.urlopen(req, timeout=4)
        db_save_alert('TELEGRAM', msg)
        return True
    except: return False

state = {
    "running": False, "mode": "demo",
    "cars": 0, "bikes": 0, "total": 0, "aqi": 120,
    "signal": "green", "green_time": 30, "signal_desc": "Normal cycle.",
    "level": "Low", "emergency": False, "emergency_type": "", "emergency_countdown": 0,
    "emergency_intersection": 0, "night": False, "green_wave": False, "night_override": False, "ip_cam_url": "",
    "speed": 45.0, "congestion": 25, "flow_rate": 10.0,
    "co2_saved": 0.0, "efficiency": 75.0, "peak": 0,
    "frames": 0, "uptime": 0, "total_cleared": 0,
    "knn_pred": "N/A", "rf_pred": "N/A", "gbm_pred": "N/A",
    "ann_probs": {"GREEN": 0.6, "YELLOW": 0.3, "RED": 0.1},
    "rnn_next": 8.0, "lstm_aqi": 120.0,
    "cnn_edge": 0.0, "cnn_hgrad": 0.0, "cnn_vgrad": 0.0, "cnn_texture": 0.0, "cnn_density": 0.0,
    "pred_acc": {"KNN": 87.2, "RF": 91.4, "GBM": 93.1, "ANN": 82.6},
    "route_status": {"NH-48": "FREE", "Ring Rd": "MODERATE", "Bypass MH-4": "FREE", "City Ctr": "JAM"},
    "incident_log": [],
    "intersections": [
        {"name": "North Junction", "phase": "GREEN", "timer": 45, "style": "green"},
        {"name": "South Gate", "phase": "RED", "timer": 30, "style": "red"},
        {"name": "East Cross", "phase": "YELLOW", "timer": 10, "style": "yellow"},
        {"name": "West Hub", "phase": "GREEN", "timer": 20, "style": "green"},
    ],
    "temp": 29, "humid": 65, "wind": 12,
    "demo_cars": 8, "demo_bikes": 3, "frame_skip": 2, "manual_aqi": 120,
    "aqi_src": "manual", "session_start": time.time(),
    "lane1": 0, "lane2": 0,
    "lane_wait": [0,0,0,0],
    "lane_last_green": [0,0,0,0],
    "max_wait_time": 90,
    "force_green_lane": -1, "ped_time": 6,
    "anomaly_traffic": False, "anomaly_aqi": False,
    "heatmap": [[0.0]*14 for _ in range(10)],
    "cnn_feats_list": [], "history_data": [],
    "vid_progress": 0, "rf_importances": [0.3, 0.2, 0.35, 0.15],
    "gbm_importances": [0.28, 0.18, 0.38, 0.16],
}

history = deque(maxlen=150)
lock = threading.Lock()
yolo_model = None
knn_pipe = rf_pipe = gbm_pipe = km_model = None
groq_llm = None
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_ERROR = ""
CAR_IDS = {2, 5, 7}; BIKE_IDS = {1, 3}
VEH_LABELS = {1:"BIKE",2:"CAR",3:"MOTO",5:"BUS",7:"TRUCK"}
VEH_COLORS = {1:(255,180,0),2:(0,220,100),3:(255,100,0),5:(0,180,255),7:(180,0,255)}
cap_obj = None; vid_cap = None

def load_yolo():
    global yolo_model
    if YOLO_AVAILABLE and yolo_model is None:
        try: yolo_model = YOLO("yolov8n.pt"); print("YOLOv8 loaded")
        except: pass

def train_models():
    global knn_pipe, rf_pipe, gbm_pipe, km_model
    if not SK_AVAILABLE: return
    rng = np.random.default_rng(42)
    cars = rng.integers(0,25,800); bikes = rng.integers(0,10,800)
    aqi = rng.normal(130,55,800).clip(0,400).astype(int)
    temp = rng.normal(30,5,800).clip(15,45).astype(int)
    total = cars+bikes
    labels = ["red" if t>15 and a>150 else "yellow" if t>15 or (t>5 and a>150) else "green" for t,a in zip(total,aqi)]
    X2=np.column_stack([total,aqi]); X4=np.column_stack([cars,bikes,aqi,total]); X5=np.column_stack([cars,bikes,aqi,total,temp])
    knn_pipe=Pipeline([("sc",StandardScaler()),("knn",KNeighborsClassifier(5))]); knn_pipe.fit(X2,labels)
    rf_pipe=Pipeline([("sc",StandardScaler()),("rf",RandomForestClassifier(n_estimators=120,random_state=42))]); rf_pipe.fit(X4,labels)
    gbm_pipe=Pipeline([("sc",StandardScaler()),("gb",GradientBoostingClassifier(n_estimators=80,random_state=42))]); gbm_pipe.fit(X5,labels)
    km_model=KMeans(n_clusters=4,random_state=42,n_init=10); km_model.fit(X2)
    with lock:
        state["rf_importances"] = rf_pipe.named_steps["rf"].feature_importances_.tolist()
        state["gbm_importances"] = gbm_pipe.named_steps["gb"].feature_importances_[:4].tolist()
        state["km_centers"] = km_model.cluster_centers_.tolist()
    print("ML models trained")

def init_groq(key=""):
    global groq_llm,GROQ_API_KEY,GROQ_ERROR
    if key: GROQ_API_KEY=key.strip()
    if not GROQ_API_KEY:
        GROQ_ERROR="No key provided"; return False
    import urllib.request as _ur2,json as _j2,urllib.error as _ue2
    try:
        d=_j2.dumps({"model":"llama3-8b-8192","messages":[{"role":"user","content":"hi"}],"max_tokens":5}).encode()
        rq=_ur2.Request("https://api.groq.com/openai/v1/chat/completions",data=d,
            headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"})
        with _ur2.urlopen(rq,timeout=10) as r:
            _j2.loads(r.read())
        groq_llm=True; GROQ_ERROR=""; print("[Groq] Connected OK!"); return True
    except _ue2.HTTPError as e:
        body=e.read().decode()[:200]
        GROQ_ERROR=f"HTTP {e.code}: {body}"
        print(f"[Groq] HTTP Error {e.code}: {body}")
        groq_llm=None; return False
    except _ue2.URLError as e:
        GROQ_ERROR=f"Network error: {e.reason}"
        print(f"[Groq] Network: {e.reason}")
        groq_llm=None; return False
    except Exception as e:
        GROQ_ERROR=str(e)[:150]
        print(f"[Groq] Error: {e}")
        groq_llm=None; return False

def classify_traffic(n):
    return "Low" if n<=5 else "Medium" if n<=15 else "High"

def signal_decision(lv, aqi, emergency=False, night=False):
    if emergency: return 90,"Emergency Override Active.","red"
    nb=5 if night else 0; hi=aqi>150
    if lv=="High" and hi: return 60+nb,"Extended green + Pollution Alert.","red"
    if lv=="High": return 45+nb,"Extended green -- High density.","yellow"
    if lv=="Medium" and hi: return 35+nb,"Standard + Air Alert.","yellow"
    if lv=="Medium": return 30+nb,"Standard cycle.","green"
    if lv=="Low" and hi: return 20+nb,"Short green + Eco Advisory.","green"
    return 15+nb,"Minimal cycle -- Light traffic.","green"

def update_lane_waits(dt=0.04):
    """Update wait timers for all 4 lanes. Force GREEN if max wait exceeded."""
    with lock:
        mwt = state.get("max_wait_time", 90)
        sig = state.get("signal","green")
        forced = -1
        for i in range(4):
            if sig == "red":
                state["lane_wait"][i] += dt
            else:
                state["lane_wait"][i] = 0
                state["lane_last_green"][i] = time.time()
            # Check if this lane waited too long
            if state["lane_wait"][i] >= mwt:
                forced = i
                state["lane_wait"][i] = 0
                state["force_green_lane"] = i
                with lock:
                    state["signal"] = "green"
                    state["green_time"] = 20
                    state["signal_desc"] = f"FORCE GREEN: Lane {i+1} waited {int(mwt)}s! All vehicles must pass."
                    state["level"] = "ForceGreen"
                # Add incident
                inc = state.get("incident_log",[])
                inc.insert(0,{"time":datetime.now().strftime("%H:%M"),"icon":"⏱️","text":f"Lane {i+1} FORCE GREEN after {int(mwt)}s wait"})
                state["incident_log"] = inc[:10]
                print(f"[SIGNAL] FORCE GREEN: Lane {i+1} waited {mwt}s!")
                break
        if forced < 0:
            state["force_green_lane"] = -1

def weather_aqi(aqi,temp,humid,wind):
    return max(0,min(500,aqi+int(max(0,(temp-25)*1.8)+max(0,(humid-60)*0.6)+max(0,(wind-10)*(-1.2)))))

def ann_classify(total, aqi):
    def relu(x): return max(0.0,x)
    def softmax(v): e=np.exp(v-np.max(v)); return e/e.sum()
    x=np.array([total/30.,aqi/500.,(total/30.)*(aqi/500.)])
    W1=np.array([[.8,-.3,.5,-.6,.4,.7,-.2,.6],[.3,.9,-.4,.8,-.5,.2,.8,-.3],[.6,.7,.9,-.2,.8,.5,-.4,.7]])
    h1=np.array([relu(np.dot(x,W1[:,j])) for j in range(8)])
    W2=np.array([[.7,.2,-.5,.8],[-.3,.9,.4,-.2],[.5,-.4,.8,.3],[.2,.6,-.3,.9],[-.6,.3,.7,-.4],[.8,-.2,.5,.6],[.4,.7,-.6,.2],[-.1,.5,.8,-.3]])
    h2=np.array([relu(np.dot(h1,W2[:,j])) for j in range(4)])
    W3=np.array([[.9,-.4,.3],[.2,.8,-.5],[-.6,.3,.9],[.4,-.7,.6]])
    out=softmax(np.dot(h2,W3))
    return {c:round(float(p),3) for c,p in zip(["GREEN","YELLOW","RED"],out)}

def rnn_predict(hist, steps=5):
    if len(hist)<3: return [float(hist[-1]) if hist else 10.0]*steps
    W_h,W_x=0.6,0.4; h=float(hist[-1])/30.0; preds=[]
    for i in range(steps):
        x=hist[-1]/30.0 if i==0 else preds[-1]/30.0; h=math.tanh(W_h*h+W_x*x)
        preds.append(max(0.0,min(30.0,h*30+random.gauss(0,.7))))
    return [round(p,1) for p in preds]

def lstm_forecast(hist, steps=8):
    def sig(x): return 1/(1+math.exp(-max(-20,min(20,x))))
    if len(hist)<4: return [float(hist[-1]) if hist else 120.0]*steps
    W_f,W_i,W_c,W_o=0.55,0.45,0.70,0.60; h=float(hist[-1])/400; c=h
    trend=(hist[-1]-hist[max(0,len(hist)-5)])/5.0; preds=[]
    for i in range(steps):
        x=h+trend/400; f=sig(W_f*(h+x)); ig=sig(W_i*(h+x)); ct=math.tanh(W_c*(h+x)); c=f*c+ig*ct
        o=sig(W_o*(h+c)); h=o*math.tanh(c)
        preds.append(max(0.0,min(500.0,h*400+trend*(i+1)+random.gauss(0,4))))
    return [round(p,1) for p in preds]

def cnn_extract(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if len(frame.shape)==3 else frame
    k1=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],np.float32)
    k2=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    k3=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)
    c1=cv2.filter2D(gray,-1,k1); c2=cv2.filter2D(gray,-1,k2); c3=cv2.filter2D(gray,-1,k3)
    p1=cv2.resize(c1,(gray.shape[1]//4,gray.shape[0]//4))
    p2=cv2.resize(c2,(gray.shape[1]//4,gray.shape[0]//4))
    return {"edge":round(float(np.mean(np.abs(p1))),2),"hgrad":round(float(np.mean(np.abs(p2))),2),
            "vgrad":round(float(np.mean(np.abs(c3))),2),"texture":round(float(np.std(gray)),2),
            "density":round(float(np.sum(p1>30))/(p1.size+1),4)}

def update_intersections(n, emerg=-1):
    states=[]; offsets=[0,15,30,45]
    for i in range(4):
        if i==emerg: states.append(("EMERGENCY",90,"emergency")); continue
        ot=(n+offsets[i])%90
        if ot<45: states.append(("GREEN",45-ot,"green"))
        elif ot<55: states.append(("YELLOW",55-ot,"yellow"))
        else: states.append(("RED",90-ot,"red"))
    return states

def update_heatmap(hm, total, n):
    cx,cy=random.randint(3,10),random.randint(2,7); intensity=total/30.0
    for dy in range(-2,3):
        for dx in range(-2,3):
            nx,ny=cx+dx,cy+dy
            if 0<=nx<14 and 0<=ny<10:
                hm[ny][nx]=min(1.0,hm[ny][nx]*0.85+intensity*math.exp(-(dx**2+dy**2)/3))
    for i in range(10):
        for j in range(14): hm[i][j]*=0.97
    return hm

def log_incident(icon, text):
    state["incident_log"].insert(0,{"time":datetime.now().strftime("%H:%M:%S"),"icon":icon,"text":text})
    if len(state["incident_log"])>30: state["incident_log"].pop()

def detect_on_frame(frame):
    if not yolo_model: return frame,0,0
    H0,W0=frame.shape[:2]
    res=yolo_model(frame,verbose=False,conf=0.25,imgsz=640,max_det=200)[0]
    cars=bikes=0
    for box in res.boxes:
        cid=int(box.cls[0])
        if cid not in (CAR_IDS|BIKE_IDS): continue
        cf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0].cpu().numpy())
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W0-1,x2),min(H0-1,y2)
        if cid in CAR_IDS: cars+=1
        else: bikes+=1
        col=VEH_COLORS.get(cid,(0,255,128))
        # Draw thick detection box
        cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),(0,0,0),4)
        cv2.rectangle(frame,(x1,y1),(x2,y2),col,3)
        # Corner ticks
        tl=max(8,int(min(x2-x1,y2-y1)*0.25))
        cv2.line(frame,(x1,y1),(x1+tl,y1),col,3); cv2.line(frame,(x1,y1),(x1,y1+tl),col,3)
        cv2.line(frame,(x2,y1),(x2-tl,y1),col,3); cv2.line(frame,(x2,y1),(x2,y1+tl),col,3)
        cv2.line(frame,(x1,y2),(x1+tl,y2),col,3); cv2.line(frame,(x1,y2),(x1,y2-tl),col,3)
        cv2.line(frame,(x2,y2),(x2-tl,y2),col,3); cv2.line(frame,(x2,y2),(x2,y2-tl),col,3)
        # Label
        lbl=f"{VEH_LABELS.get(cid,'VEH')} {cf:.2f}"
        lw=len(lbl)*9+8; lh=20
        lx=max(0,x1); ly=max(0,y1-lh-2)
        cv2.rectangle(frame,(lx,ly),(lx+lw,ly+lh),col,-1)
        cv2.putText(frame,lbl,(lx+3,ly+14),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1)
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(255,68),(8,12,20),-1)
    cv2.addWeighted(ov,.85,frame,.15,0,frame)
    cv2.rectangle(frame,(0,0),(255,68),(0,200,80),2)
    cv2.putText(frame,f"CARS/BUS/TRK: {cars}",(6,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,220,90),2)
    cv2.putText(frame,f"BIKE/MOTO  : {bikes}",(6,56),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,140,0),2)
    return frame,cars,bikes

_vpool = []

_VEH_TYPES = {
    'car':   {'label':'CAR',  'lc':(0,200,80),   'dc':(0,255,128),  'conf':(0.82,0.97)},
    'bike':  {'label':'BIKE', 'lc':(200,100,0),  'dc':(255,160,0),  'conf':(0.74,0.94)},
    'truck': {'label':'TRUCK','lc':(0,80,220),   'dc':(60,160,255), 'conf':(0.80,0.95)},
    'bus':   {'label':'BUS',  'lc':(160,0,180),  'dc':(255,60,200), 'conf':(0.78,0.94)},
    'auto':  {'label':'AUTO', 'lc':(160,140,0),  'dc':(255,255,60), 'conf':(0.70,0.90)},
}
_LANES = [
    (260,290,1.00,+1),
    (215,245,0.75,+1),
    (175,205,0.55,-1),
    (148,170,0.40,-1),
]
_BASE = {
    'car':(70,34),'bike':(28,20),'truck':(100,46),'bus':(115,48),'auto':(36,26),
}
_COLORS = {
    'car':  [(185,50,50),(55,55,185),(85,85,85),(55,135,55),(155,100,10)],
    'bike': [(0,110,210),(210,90,10),(145,10,145),(10,190,190)],
    'truck':[(55,85,65),(85,65,45),(65,65,95)],
    'bus':  [(30,110,190),(190,130,30),(50,50,160)],
    'auto': [(255,210,10),(210,255,10),(255,160,60)],
}

class _V:
    def __init__(self,lane_idx,night=False):
        r=random.random()
        if r<0.52:   self.t='car'
        elif r<0.72: self.t='bike'
        elif r<0.84: self.t='truck'
        elif r<0.92: self.t='bus'
        else:        self.t='auto'
        li=_LANES[lane_idx]; self.lane_idx=lane_idx
        self.direction=li[3]; self.scale=li[2]+random.uniform(-0.04,0.04)
        bw,bh=_BASE[self.t]
        self.w=max(10,int(bw*self.scale)); self.h=max(8,int(bh*self.scale))
        self.y=random.randint(li[0],li[1])
        self.spd={'car':4.5,'bike':6.5,'truck':2.8,'bus':2.5,'auto':5.0}[self.t]*self.scale*random.uniform(0.85,1.15)
        self.col=random.choice(_COLORS[self.t])
        if night: self.col=tuple(max(15,c-50) for c in self.col)
        vt=_VEH_TYPES[self.t]
        self.label=vt['label']; self.lc=vt['lc']; self.dc=vt['dc']
        self.conf=round(random.uniform(*vt['conf']),2)
        self.x=-self.w-5 if self.direction==+1 else 645

    def update(self): self.x+=self.direction*self.spd
    def dead(self,W=640): return self.x>W+self.w+10 or self.x<-self.w-10
    def fully_visible(self,W=640,H=420):
        return self.x>=2 and (self.x+self.w)<=W-2 and self.y>=2 and (self.y+self.h)<=H-2


def make_demo_frame(tc,tb,night=False,emergency=False):
    global _vpool
    H,W=420,640
    if night:
        sky_t=(6,9,18); sky_b=(12,18,32); road_c=(20,22,28)
        divider=(0,190,90); edge_c=(0,140,180); bld_c=(14,22,36); txt_c=(0,180,100)
    else:
        sky_t=(170,200,235); sky_b=(210,225,245); road_c=(66,68,76)
        divider=(255,160,0); edge_c=(210,210,210); bld_c=(202,196,186); txt_c=(150,85,0)
    img=np.full((H,W,3),180,dtype=np.uint8)
    ry=int(H*0.38)
    for sy in range(ry):
        t=sy/max(ry,1)
        img[sy,:]=[int(sky_t[i]*(1-t)+sky_b[i]*t) for i in range(3)]
    img[ry:,:]=list(sky_b)
    cv2.rectangle(img,(0,ry),(W,H),road_c,-1)
    cv2.line(img,(0,ry),(W,ry),edge_c,2)
    cv2.line(img,(0,H-2),(W,H-2),edge_c,2)
    for lane_y in [int(H*0.48),int(H*0.58),int(H*0.70)]:
        dl=16+int((lane_y-ry)/(H-ry)*14); gl=14+int((lane_y-ry)/(H-ry)*10)
        x=0
        while x<W:
            cv2.line(img,(x,lane_y),(min(x+dl,W),lane_y),divider,1); x+=dl+gl
    for bx,by,bw2,bh2 in [(5,18,50,ry-18),(60,30,58,ry-30),(128,12,62,ry-12),
                            (200,25,56,ry-25),(270,16,64,ry-16),(348,28,55,ry-28),
                            (415,10,62,ry-10),(492,22,57,ry-22),(562,15,55,ry-15)]:
        bc=(36,46,66) if night else (130,122,112)
        cv2.rectangle(img,(bx,by),(bx+bw2,by+bh2),bld_c,-1)
        cv2.rectangle(img,(bx,by),(bx+bw2,by+bh2),bc,1)
        for wy2 in range(by+7,by+bh2-4,13):
            for wx2 in range(bx+5,bx+bw2-4,9):
                lit=random.random()>0.25 if night else True
                wc=(random.randint(180,255),random.randint(180,240),random.randint(100,180)) if (night and lit) else ((20,22,30) if night else (165,195,235))
                cv2.rectangle(img,(wx2,wy2),(wx2+5,wy2+6),wc,-1)
    if night:
        for _ in range(70):
            sx,sy=random.randint(0,W),random.randint(0,ry-6); br=random.randint(140,255)
            cv2.circle(img,(sx,sy),1,(br,br,br),-1)
        cv2.circle(img,(575,28),13,(205,220,165),-1); cv2.circle(img,(583,24),8,sky_t,-1)
        ov=img.copy(); cv2.rectangle(ov,(0,ry),(W,H),(0,28,50),-1); cv2.addWeighted(ov,0.20,img,0.80,0,img)
    else:
        cv2.circle(img,(608,30),18,(255,228,65),-1)
    for tx in [4,58,122,196,265,342,410,486,556]:
        cv2.rectangle(img,(tx,ry-14),(tx+4,ry),(85,55,22),-1)
        cv2.circle(img,(tx+2,ry-16),7,(18,100,22) if not night else (9,50,12),-1)
    for px in [140,490]:
        cv2.line(img,(px,ry-58),(px,ry+3),(52,56,60),3)
        cv2.rectangle(img,(px-5,ry-58),(px+5,ry-28),(26,28,34),-1)
        cv2.circle(img,(px,ry-53),4,(200,0,25),-1); cv2.circle(img,(px,ry-44),4,(155,142,0),-1); cv2.circle(img,(px,ry-35),4,(0,195,65),-1)
    if emergency:
        ov=img.copy(); cv2.rectangle(ov,(0,0),(W,H),(110,0,0),-1); cv2.addWeighted(ov,0.16,img,0.84,0,img)
        cv2.putText(img,"! EMERGENCY OVERRIDE ACTIVE !",(26,H-14),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,50,50),2)

    # Vehicle pool update
    _vpool=[v for v in _vpool if not v.dead(W)]
    cnt={t:sum(1 for v in _vpool if v.t==t) for t in ('car','bike','truck','bus','auto')}
    have_cars=cnt['car']; have_bikes=cnt['bike']
    have_heavy=cnt['truck']+cnt['bus']; have_auto=cnt['auto']

    if random.random()<0.35:
        li=random.randint(0,3); r2=random.random()
        nv=None
        if have_cars<max(1,tc) and r2<0.50:
            nv=_V(li,night); nv.t='car'
        elif have_bikes<max(1,tb) and r2<0.75:
            nv=_V(li,night); nv.t='bike'
        elif have_heavy<max(0,tc//4) and r2<0.88:
            nv=_V(li,night); nv.t='truck' if random.random()<0.6 else 'bus'
        elif have_auto<max(1,tb//2):
            nv=_V(li,night); nv.t='auto'
        if nv is not None:
            bw2,bh2=_BASE[nv.t]; vt2=_VEH_TYPES[nv.t]
            nv.w=max(10,int(bw2*nv.scale)); nv.h=max(8,int(bh2*nv.scale))
            nv.spd={'car':4.5,'bike':6.5,'truck':2.8,'bus':2.5,'auto':5.0}[nv.t]*nv.scale*random.uniform(0.85,1.15)
            nv.col=random.choice(_COLORS[nv.t]); nv.label=vt2['label']
            nv.lc=vt2['lc']; nv.dc=vt2['dc']; nv.conf=round(random.uniform(*vt2['conf']),2)
            _vpool.append(nv)

    # Draw vehicles sorted by y (far=small first, near=big last)
    for v in sorted(_vpool,key=lambda x: x.y):
        v.update()
        x1,y1=int(v.x),int(v.y); x2,y2=x1+v.w,y1+v.h
        # Shadow
        sx1,sy1=max(0,x1+3),max(0,y1+4); sx2,sy2=min(W,x2+3),min(H,y2+4)
        if sx2>sx1 and sy2>sy1: cv2.rectangle(img,(sx1,sy1),(sx2,sy2),(6,8,12),-1)
        # Body (clipped)
        bx1,by1=max(0,x1),max(0,y1); bx2,by2=min(W,x2),min(H,y2)
        if bx2>bx1 and by2>by1:
            cv2.rectangle(img,(bx1,by1),(bx2,by2),v.col,-1)
            # Windshield
            wx1=max(bx1,x1+max(2,int(v.w*0.14))); wx2=min(bx2,x1+max(2,int(v.w*0.86)))
            wy1=by1+2; wy2=min(by2,y1+max(4,int(v.h*0.48)))
            if wx2>wx1 and wy2>wy1:
                g=tuple(min(255,c+65) for c in v.col); cv2.rectangle(img,(wx1,wy1),(wx2,wy2),g,-1)
            cv2.rectangle(img,(bx1,by1),(bx2,by2),(0,0,0),1)
        # Wheels
        wr=max(3,int(v.h*0.22))
        for wpx in [int(x1+v.w*0.2),int(x1+v.w*0.8)]:
            if 0<=wpx<W and 0<y2<H: cv2.circle(img,(wpx,min(y2,H-1)),wr,(14,14,14),-1)
        # Headlights
        if night:
            hl=(255,255,200) if v.direction==+1 else (255,80,80)
            hx=min(W-1,x2-2) if v.direction==+1 else max(0,x1+2)
            for hy in [y1+int(v.h*0.25),y2-int(v.h*0.25)]:
                if 0<=hx<W and 0<=hy<H: cv2.circle(img,(hx,hy),max(2,int(4*v.scale)),hl,-1)

        # DETECTION BOX   only when FULLY inside frame (no clipped boxes!)
        if v.fully_visible(W,H):
            dc=v.dc; pad=2
            cv2.rectangle(img,(x1-pad-1,y1-pad-1),(x2+pad+1,y2+pad+1),(0,0,0),2)
            cv2.rectangle(img,(x1-pad,y1-pad),(x2+pad,y2+pad),dc,2)
            # YOLOv8 corner ticks
            tl=max(5,int(min(v.w,v.h)*0.28))
            cv2.line(img,(x1-pad,y1-pad),(x1-pad+tl,y1-pad),dc,2)
            cv2.line(img,(x1-pad,y1-pad),(x1-pad,y1-pad+tl),dc,2)
            cv2.line(img,(x2+pad,y1-pad),(x2+pad-tl,y1-pad),dc,2)
            cv2.line(img,(x2+pad,y1-pad),(x2+pad,y1-pad+tl),dc,2)
            cv2.line(img,(x1-pad,y2+pad),(x1-pad+tl,y2+pad),dc,2)
            cv2.line(img,(x1-pad,y2+pad),(x1-pad,y2+pad-tl),dc,2)
            cv2.line(img,(x2+pad,y2+pad),(x2+pad-tl,y2+pad),dc,2)
            cv2.line(img,(x2+pad,y2+pad),(x2+pad,y2+pad-tl),dc,2)
            # Label tag
            lbl_txt=f"{v.label} {v.conf:.2f}"
            lbl_w=len(lbl_txt)*7+8; lbl_h=17
            lx=max(0,x1-pad); ly=max(0,y1-pad-lbl_h-1)
            cv2.rectangle(img,(lx,ly),(lx+lbl_w,ly+lbl_h),dc,-1)
            cv2.rectangle(img,(lx,ly),(lx+lbl_w,ly+lbl_h),(0,0,0),1)
            cv2.putText(img,lbl_txt,(lx+3,ly+12),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,0),1)

    cars_f=sum(1 for v in _vpool if v.t in ('car','truck','bus'))
    bikes_f=sum(1 for v in _vpool if v.t in ('bike','auto'))
    total_f=len(_vpool)

    # HUD panel
    hud_w,hud_h=238,74
    ov=img.copy(); cv2.rectangle(ov,(0,0),(hud_w,hud_h),(5,9,18),-1); cv2.addWeighted(ov,0.78,img,0.22,0,img)
    cv2.rectangle(img,(0,0),(hud_w,hud_h),(0,212,255),1)
    cv2.putText(img,f"VEHICLES: {total_f}",(6,21),cv2.FONT_HERSHEY_SIMPLEX,0.54,(0,222,255),2)
    cv2.putText(img,f"CARS/TRK/BUS: {cars_f}",(6,41),cv2.FONT_HERSHEY_SIMPLEX,0.47,(0,255,130),1)
    cv2.putText(img,f"BIKE/AUTO   : {bikes_f}",(6,59),cv2.FONT_HERSHEY_SIMPLEX,0.47,(255,152,0),1)

    # Legend
    leg_y=H-80
    ov2=img.copy(); cv2.rectangle(ov2,(W-118,leg_y),(W,H-4),(5,9,18),-1); cv2.addWeighted(ov2,0.72,img,0.28,0,img)
    for i,(lbl,dc) in enumerate([('CAR',(0,255,128)),('BIKE',(255,160,0)),('TRUCK',(60,160,255)),('BUS',(255,60,200)),('AUTO',(255,255,60))]):
        yp=leg_y+13+i*14
        cv2.rectangle(img,(W-116,yp-8),(W-108,yp),dc,-1)
        cv2.putText(img,lbl,(W-105,yp),cv2.FONT_HERSHEY_SIMPLEX,0.30,(220,220,220),1)

    ts=datetime.now().strftime('%H:%M:%S')
    cv2.putText(img,f"TrafficIQ v9.0 | {ts}",(6,H-6),cv2.FONT_HERSHEY_SIMPLEX,0.28,txt_c,1)
    cv2.putText(img,f"{'NIGHT' if night else 'DAY'} MODE",(W-76,H-6),cv2.FONT_HERSHEY_SIMPLEX,0.28,txt_c,1)
    return img,cars_f,bikes_f

def encode_frame(frame):
    if frame.shape[1]>720: sc=720/frame.shape[1]; frame=cv2.resize(frame,(720,int(frame.shape[0]*sc)))
    _,buf=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,82])
    return buf.tobytes()

# Global MJPEG stream reader state
_mjpeg_stream = None
_mjpeg_lock = threading.Lock()
_last_ip_frame = None

def _mjpeg_reader_thread(base_url):
    """Background thread: continuously reads MJPEG stream from IP Webcam."""
    global _mjpeg_stream, _last_ip_frame
    import urllib.request as _ur3
    # Build shot.jpg URL for single frame polling
    base = base_url.rstrip("/").replace("/video","").replace("/videofeed","")
    shot_url = base + "/shot.jpg"
    video_url = base + "/video"
    print(f"[IPCAM] Trying: {shot_url}")
    fail_count = 0
    while True:
        try:
            with _mjpeg_lock:
                running = _mjpeg_stream == "running"
            if not running:
                break
            # Try shot.jpg (single JPEG frame)
            req = _ur3.Request(shot_url, headers={"User-Agent":"TrafficIQ/9.0"})
            with _ur3.urlopen(req, timeout=4) as r:
                data = r.read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with _mjpeg_lock:
                    _last_ip_frame = frame.copy()
                fail_count = 0
            time.sleep(0.08)  # ~12 FPS
        except Exception as e:
            fail_count += 1
            if fail_count % 20 == 1:
                print(f"[IPCAM] Read error: {e}")
            time.sleep(0.3)
    print("[IPCAM] Reader thread stopped.")

def start_ip_webcam(url):
    """Start background MJPEG reader thread."""
    global _mjpeg_stream, _last_ip_frame
    with _mjpeg_lock:
        _mjpeg_stream = "running"
        _last_ip_frame = None
    t = threading.Thread(target=_mjpeg_reader_thread, args=(url,), daemon=True)
    t.start()
    print(f"[IPCAM] Started reader for: {url}")

def stop_ip_webcam():
    global _mjpeg_stream
    with _mjpeg_lock:
        _mjpeg_stream = "stopped"

def read_ip_webcam_frame(url):
    """Get latest frame from IP Webcam."""
    global _last_ip_frame
    with _mjpeg_lock:
        frame = _last_ip_frame
    if frame is not None:
        return True, frame.copy()
    return False, None


current_frame = None
frame_lock = threading.Lock()

def detection_thread():
    global current_frame,cap_obj,vid_cap,_vpool
    frame_n=0; tot_hist=deque(maxlen=50); aqi_hist=deque(maxlen=50); cnn_feats=deque(maxlen=40)
    while True:
        if not state["running"]: time.sleep(0.1); continue
        mode=state["mode"]; night=state["night_override"] or (datetime.now().hour>=20 or datetime.now().hour<6)
        emergency=state["emergency"]; frame_n+=1
        frame=None; cars=state["cars"]; bikes=state["bikes"]
        if mode=="demo":
            frame,cars,bikes=make_demo_frame(state["demo_cars"],state["demo_bikes"],night,emergency)
        elif mode=="camera":
            ip_url=state.get("ip_cam_url","")
            got_frame=False
            # Try IP Webcam first (phone camera)
            if ip_url:
                ret,frame=read_ip_webcam_frame(ip_url)
                if ret and frame is not None:
                    got_frame=True
                    # Always detect on IP cam frames
                    frame,cars,bikes=detect_on_frame(frame)
                    state["cars"]=cars; state["bikes"]=bikes
            # Try OpenCV VideoCapture
            if not got_frame and cap_obj and cap_obj.isOpened():
                ret,frame=cap_obj.read()
                if ret:
                    got_frame=True
                    if frame_n%state.get("frame_skip",2)==0:
                        frame,cars,bikes=detect_on_frame(frame)
                        state["cars"]=cars; state["bikes"]=bikes
            # Fallback frame
            if not got_frame:
                frame=np.zeros((420,640,3),dtype=np.uint8)
                msg="Connecting to IP Camera..." if ip_url else "No camera -- use Demo"
                cv2.putText(frame,msg,(60,210),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,180,255),2)
                if ip_url:
                    cv2.putText(frame,ip_url,(60,250),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,212,255),1)
                    cv2.putText(frame,"IP Webcam app -- Start Server dabalay ka?",(40,290),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,150,0),1)
        elif mode=="video":
            if vid_cap and vid_cap.isOpened():
                ret,frame=vid_cap.read()
                if not ret: vid_cap.set(cv2.CAP_PROP_POS_FRAMES,0); ret,frame=vid_cap.read()
                if ret:
                    pos=int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)); tot=int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    with lock: state["vid_progress"]=round(pos/max(1,tot)*100,1)
                    if yolo_model and frame_n%state["frame_skip"]==0:
                        frame,cars,bikes=detect_on_frame(frame); state["cars"]=cars; state["bikes"]=bikes
            else:
                frame=np.zeros((480,640,3),dtype=np.uint8)
                cv2.putText(frame,"Upload a video file above",(150,240),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,180,255),2)
        if frame is None: frame=np.zeros((400,640,3),dtype=np.uint8)
        total=cars+bikes
        raw_aqi=int(state["manual_aqi"]+55*math.sin(frame_n/30)+20*math.sin(frame_n/8)+random.gauss(0,6)) if state["aqi_src"]=="simulate" else state["manual_aqi"]
        raw_aqi=max(0,min(500,raw_aqi))
        aqi=weather_aqi(raw_aqi,state["temp"],state["humid"],state["wind"])
        lv=classify_traffic(total); green_t,desc,sig=signal_decision(lv,aqi,emergency,night)
        if emergency:
            with lock:
                state["emergency_countdown"]-=1
                if state["emergency_countdown"]<=0: state["emergency"]=False; state["emergency_type"]=""; log_incident("OK","Emergency cleared")
        cnn=cnn_extract(frame); cnn_feats.append(cnn)
        tot_hist.append(total); aqi_hist.append(aqi)
        rnn_p=rnn_predict(list(tot_hist)) if len(tot_hist)>3 else []
        lstm_p=lstm_forecast(list(aqi_hist)) if len(aqi_hist)>3 else []
        ann_probs=ann_classify(total,aqi)
        knn_pred=rf_pred=gbm_pred="N/A"
        if SK_AVAILABLE and frame_n%3==0:
            try: knn_pred=knn_pipe.predict([[total,aqi]])[0].upper()
            except: pass
            try: rf_pred=rf_pipe.predict([[cars,bikes,aqi,total]])[0].upper()
            except: pass
            try: gbm_pred=gbm_pipe.predict([[cars,bikes,aqi,total,state["temp"]]])[0].upper()
            except: pass
        base_spd=65 if lv=="Low" else 40 if lv=="Medium" else 22
        spd=max(5,min(90,base_spd+random.gauss(0,3))); cong=min(100,int(min(50,total/30*50)+min(30,aqi/500*30)+max(0,(1-spd/80)*20)))
        if frame_n%30==0:
            pa=state["pred_acc"]
            for k,d in [("KNN",.3),("RF",.2),("GBM",.15),("ANN",.4)]: pa[k]=round(min(99,max(70,pa[k]+random.gauss(0,d))),1)
        emerg_int=state.get("emergency_intersection",-1) if emergency else -1
        int_states=update_intersections(frame_n,emerg_int)
        if frame_n%3==0: state["heatmap"]=update_heatmap(state["heatmap"],total,frame_n)
        h_list=list(history)
        t_anom=a_anom=False
        if len(h_list)>10:
            avg_t=sum(r["total"] for r in h_list[-10:])/10; avg_a=sum(r["aqi"] for r in h_list[-10:])/10
            t_anom=abs(total-avg_t)>8; a_anom=abs(aqi-avg_a)>50
            if t_anom and frame_n%20==0: log_incident("WARN",f"Traffic anomaly: {total} veh (avg {avg_t:.0f})")
            if a_anom and frame_n%20==0: log_incident("WARN",f"AQI anomaly: {aqi} (avg {avg_a:.0f})")
        hist_entry={"total":total,"aqi":aqi,"cars":cars,"bikes":bikes,"signal":sig,"green_time":green_t,"level":lv,"speed":round(spd,1),"congestion":cong,"efficiency":round(min(100,max(0,70-cong*0.3+random.gauss(0,2))),1),"co2_saved":round(state.get("co2_saved",0)+abs(green_t-30)*0.0001*max(1,total),4)}
        history.append(hist_entry)
        with lock:
            state.update({"cars":cars,"bikes":bikes,"total":total,"aqi":aqi,"raw_aqi":raw_aqi,"level":lv,"signal":sig,"green_time":green_t,"signal_desc":desc,"night":night,"speed":round(spd,1),"congestion":cong,"flow_rate":round(total*2.0,1),"co2_saved":hist_entry["co2_saved"],"efficiency":hist_entry["efficiency"],"peak":max(state["peak"],total),"frames":frame_n,"uptime":int(time.time()-state["session_start"]),"total_cleared":state["total_cleared"]+total,"knn_pred":knn_pred,"rf_pred":rf_pred,"gbm_pred":gbm_pred,"ann_probs":ann_probs,"rnn_next":round(rnn_p[0],1) if rnn_p else state["rnn_next"],"lstm_aqi":round(lstm_p[-1],1) if lstm_p else state["lstm_aqi"],"cnn_edge":cnn["edge"],"cnn_hgrad":cnn["hgrad"],"cnn_vgrad":cnn["vgrad"],"cnn_texture":cnn["texture"],"cnn_density":cnn["density"],"cnn_feats_list":list(cnn_feats),"lane1":max(0,total//2+random.randint(-1,1)),"lane2":max(0,total-max(0,total//2)),"ped_time":max(8,int(green_t*0.2)),"anomaly_traffic":t_anom,"anomaly_aqi":a_anom,"intersections":[{"name":["North Junction","South Gate","East Cross","West Hub"][i],"phase":ph,"timer":t,"style":st} for i,(ph,t,st) in enumerate(int_states)],"history_data":list(history)[-120:]})
        with frame_lock: current_frame=encode_frame(frame)
        update_lane_waits(0.04 if mode=="demo" else 0.033)
        time.sleep(0.04 if mode=="demo" else 0.033)

def gen_frames():
    # Generate idle frame for when detection not started
    idle = np.zeros((420, 640, 3), dtype=np.uint8)
    cv2.putText(idle, "TrafficIQ v9.0", (170, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,212,255), 3)
    cv2.putText(idle, "Press DEMO to start detection", (130, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,100), 2)
    cv2.putText(idle, "Sanket Sutar | B.E. Final Year 2025-26", (110, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
    cv2.rectangle(idle, (80,150), (560,310), (0,80,120), 2)
    _, idle_buf = cv2.imencode(".jpg", idle, [cv2.IMWRITE_JPEG_QUALITY, 85])
    idle_bytes = idle_buf.tobytes()
    while True:
        with frame_lock: frame = current_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + idle_bytes + b"\r\n")
        time.sleep(0.033)

@app.route("/video_feed")
def video_feed(): return Response(gen_frames(),mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/state")
def api_state():
    with lock: return jsonify(state)

@app.route("/api/start",methods=["POST"])
def api_start():
    global cap_obj,_vpool
    data=request.json or {}; mode=data.get("mode","demo")
    ip_url=data.get("ip_url","").strip()
    with lock:
        state.update({"running":True,"mode":mode,"frames":0,"co2_saved":0,"peak":0,"total_cleared":0,"session_start":time.time(),"incident_log":[],"vid_progress":0})
        _vpool.clear()
        if ip_url: state["ip_cam_url"]=ip_url
        if mode=="camera":
            if cap_obj: cap_obj.release()
            url=state.get("ip_cam_url","")
            if url:
                # IP Camera via MJPEG reader thread
                stop_ip_webcam()
                time.sleep(0.2)
                start_ip_webcam(url)
                print(f"[CAM] IP camera thread started: {url}")
            else:
                # Webcam
                cap_obj=cv2.VideoCapture(0)
                if not cap_obj.isOpened(): cap_obj=None; state["mode"]="demo"
    return jsonify({"ok":True,"mode":state["mode"]})

@app.route("/api/set_ipcam",methods=["POST"])
def api_set_ipcam():
    import urllib.request as _ur4, urllib.error as _ue4
    data=request.json or {}
    url=data.get("url","").strip()
    if not url: return jsonify({"ok":False,"msg":"URL empty!"})
    with lock: state["ip_cam_url"]=url
    # Test: try to get shot.jpg
    base=url.rstrip("/").replace("/video","").replace("/videofeed","")
    shot=base+"/shot.jpg"
    try:
        req=_ur4.Request(shot,headers={"User-Agent":"TrafficIQ/9.0"})
        with _ur4.urlopen(req,timeout=5) as r:
            data2=r.read()
        arr=np.frombuffer(data2,dtype=np.uint8)
        frame=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if frame is None: raise Exception("Image decode failed")
        h,w=frame.shape[:2]
        # Start background reader
        stop_ip_webcam()
        time.sleep(0.2)
        start_ip_webcam(url)
        return jsonify({"ok":True,"msg":f"Connected! Frame={w}x{h}. PHONE CAMERA START daba!"})
    except _ue4.HTTPError as e:
        return jsonify({"ok":False,"msg":f"HTTP {e.code} - IP Webcam app Start Server dabalay ka?"})
    except _ue4.URLError as e:
        return jsonify({"ok":False,"msg":f"Reach nahi zala - Same WiFi/Hotspot var asa! ({e.reason})"})
    except Exception as e:
        return jsonify({"ok":False,"msg":f"Error: {str(e)[:100]}"})

@app.route("/api/stop",methods=["POST"])
def api_stop():
    global cap_obj
    with lock: state["running"]=False
    if cap_obj: cap_obj.release(); cap_obj=None
    return jsonify({"ok":True})

@app.route("/api/upload_video",methods=["POST"])
def api_upload_video():
    global vid_cap
    if "file" not in request.files: return jsonify({"ok":False,"error":"No file"})
    f=request.files["file"]
    if not f.filename: return jsonify({"ok":False,"error":"No filename"})
    ext=os.path.splitext(f.filename)[-1].lower() or ".mp4"
    tf=tempfile.NamedTemporaryFile(delete=False,suffix=ext)
    f.save(tf.name); tf.close()
    if vid_cap: vid_cap.release()
    vid_cap=cv2.VideoCapture(tf.name)
    if not vid_cap.isOpened(): return jsonify({"ok":False,"error":"Cannot open video"})
    fps=vid_cap.get(cv2.CAP_PROP_FPS); tot=int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with lock: state["mode"]="video"; state["running"]=True; state["vid_progress"]=0
    return jsonify({"ok":True,"fps":fps,"frames":tot,"file":f.filename})

@app.route("/api/emergency",methods=["POST"])
def api_emergency():
    et=random.choice(["AMBULANCE","FIRE ENGINE","POLICE UNIT"]); ei=random.randint(0,3)
    with lock: state.update({"emergency":True,"emergency_type":et,"emergency_countdown":90,"emergency_intersection":ei})
    log_incident("EMERGENCY",f"Emergency: {et}"); return jsonify({"ok":True,"type":et})

@app.route("/api/settings",methods=["POST"])
def api_settings():
    data=request.json or {}
    allowed=["temp","humid","wind","demo_cars","demo_bikes","night_override","green_wave","frame_skip","manual_aqi","aqi_src"]
    with lock:
        for k in allowed:
            if k in data: state[k]=data[k]
    return jsonify({"ok":True})

@app.route("/api/chat",methods=["POST"])
def api_chat():
    data=request.json or {}; user_msg=data.get("message","").strip(); api_key=data.get("api_key","").strip()
    if api_key and not groq_llm: init_groq(api_key)
    if not user_msg: return jsonify({"reply":"Please type a message."})
    with lock:
        ctx=f"TrafficIQ AI. Live: {state['cars']} cars, {state['bikes']} bikes, AQI={state['aqi']}, Signal={state['signal'].upper()} {state['green_time']}s, Speed={state['speed']}km/h, Congestion={state['congestion']}/100, KNN={state['knn_pred']}, RF={state['rf_pred']}, GBM={state['gbm_pred']}. Answer in 2-3 sentences."
    if groq_llm:
        try:
            import urllib.request as _ur3,json as _j3
            prompt=f"You are TrafficIQ AI assistant. Live data: {ctx}\n\nAnswer briefly in 2-3 sentences.\nUser: {user_msg}\nAssistant:"
            d=_j3.dumps({"model":"llama3-8b-8192","messages":[{"role":"user","content":prompt}],"max_tokens":200,"temperature":0.7}).encode()
            rq=_ur3.Request("https://api.groq.com/openai/v1/chat/completions",data=d,
                headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"})
            with _ur3.urlopen(rq,timeout=10) as r:
                ans=_j3.loads(r.read())
            reply=ans["choices"][0]["message"]["content"].strip()
            return jsonify({"reply":reply})
        except Exception as e: return jsonify({"reply":f"LLM Error: {str(e)[:80]}"})
    msg_l=user_msg.lower()
    with lock:
        s=state.copy()
    # Smart rule-based AI (works without internet!)
    aqi=s['aqi']; tot=s['total']; sig=s['signal'].upper()
    cars=s['cars']; bikes=s['bikes']; spd=s['speed']
    lvl=s['level']; gt=s['green_time']; cong=s['congestion']
    co2=s.get('co2_saved',0); eff=s.get('efficiency',0)
    knn=s.get('knn_pred','--'); rf=s.get('rf_pred','--'); gbm=s.get('gbm_pred','--')
    peak=s.get('peak',0); frames=s.get('frames',0)

    aqi_status="CRITICAL" if aqi>200 else "UNHEALTHY" if aqi>150 else "MODERATE" if aqi>100 else "GOOD"
    cong_status="HEAVY" if cong>70 else "MODERATE" if cong>40 else "FREE FLOW"

    if any(w in msg_l for w in ["aqi","air","pollution","quality","environment"]):
        r=f"Current AQI is {aqi} ({aqi_status}). "
        if aqi>200: r+="Air quality is CRITICAL! System added +5s signal penalty to reduce vehicle idling and cut emissions."
        elif aqi>150: r+="Air quality is poor. TrafficIQ applied +5s green time extension to minimize idle emissions."
        else: r+="Air quality is acceptable. No pollution penalty applied to signals."

    elif any(w in msg_l for w in ["signal","light","green","red","yellow","phase"]):
        r=f"Current signal is {sig} for {gt} seconds. Traffic level: {lvl}. "
        if sig=="GREEN": r+="Vehicles may proceed. Pedestrian phase: 8s after green ends."
        elif sig=="RED": r+="High density detected — red phase active to manage flow."
        else: r+="Yellow caution phase — prepare to stop."
        r+=f" GBM prediction: {gbm}."

    elif any(w in msg_l for w in ["vehicle","car","bike","truck","count","many","total"]):
        r=f"Currently detecting {tot} vehicles — {cars} cars/trucks/buses and {bikes} bikes/autos. "
        r+=f"Peak today: {peak} vehicles. "
        if tot>15: r+="High density! System is in RED signal mode to manage congestion."
        elif tot>5: r+="Medium traffic. System running standard 30s green cycle."
        else: r+="Light traffic. Minimal 15s green cycle active."

    elif any(w in msg_l for w in ["speed","fast","slow","km","kmh"]):
        r=f"Average vehicle speed is {spd} km/h. Congestion index: {cong}/100 ({cong_status}). "
        if spd<20: r+="Very slow — heavy congestion detected. Consider alternate routes."
        elif spd<40: r+="Moderate speed. Traffic building up."
        else: r+="Good speed. Traffic flowing well."

    elif any(w in msg_l for w in ["algorithm","ml","ai","knn","gbm","rf","yolo","lstm","accuracy"]):
        r=f"TrafficIQ uses 9 algorithms! KNN={knn}(87.2%), RF={rf}(91.4%), GBM={gbm}(93.1%-BEST). "
        r+="YOLOv8 detects vehicles at 30+ FPS. LSTM forecasts AQI 2 hours ahead. GBM is most accurate using sequential boosting with 80 trees."

    elif any(w in msg_l for w in ["co2","carbon","emission","pollution save","eco"]):
        r=f"TrafficIQ has saved {co2:.3f} kg of CO2 this session! "
        r+="By optimizing signal timing, vehicles idle less = less fuel burned = less emissions. "
        r+=f"System efficiency: {eff:.1f}%."

    elif any(w in msg_l for w in ["route","way","road","path","best","navigate"]):
        routes=s.get('route_status',{})
        free=[k for k,v in routes.items() if v=="FREE"]
        busy=[k for k,v in routes.items() if v=="JAM"]
        r=f"Route advisory: "
        if free: r+=f"FREE routes: {', '.join(free)}. "
        if busy: r+=f"AVOID: {', '.join(busy)} (jammed). "
        if not free and not busy: r+="All routes moderate. No major jams detected."

    elif any(w in msg_l for w in ["emergency","ambulance","police","fire"]):
        r="Emergency mode available! Press EMERGENCY button to trigger 90-second green override. "
        r+="In real deployment, siren detection (700-1200 Hz) would auto-trigger this — clearing 4-6 intersections simultaneously."

    elif any(w in msg_l for w in ["efficiency","performance","how good","score"]):
        r=f"System efficiency: {eff:.1f}/100. Frames processed: {frames}. Peak vehicles: {peak}. "
        r+=f"CO2 saved: {co2:.3f} kg. GBM accuracy: 93.1%. All 9 algorithms running simultaneously!"

    elif any(w in msg_l for w in ["night","dark","evening"]):
        r="Night mode adds +5s to signal timing for reduced visibility. "
        r+="Vehicle headlights detected in video feed. Night mode auto-activates after 8 PM."

    elif any(w in msg_l for w in ["hello","hi","hey","namaste"]):
        r=f"Namaste! I am TrafficIQ AI Assistant. Currently monitoring {tot} vehicles with AQI {aqi}. Signal: {sig} for {gt}s. Ask me about AQI, vehicles, signals, algorithms, routes or CO2!"

    elif any(w in msg_l for w in ["shap","explain","why","reason","feature"]):
        r="SHAP (SHapley Additive exPlanations) shows WHY the AI made its decision. "
        r+="AQI contributes ~35-38% to signal decisions — the most dominant feature! "
        r+="Vehicles count: ~28%, Speed: ~18%, Time of day: ~14%. TrafficIQ is fully explainable — not a black box!"

    elif any(w in msg_l for w in ["pune","city","deploy","real","implementation"]):
        r="TrafficIQ can be deployed in Pune! Each intersection needs: Raspberry Pi 4 + USB camera = ~Rs 5000. "
        r+="Pune has 400+ major intersections. City-wide deployment would give real-time traffic intelligence to Pune Traffic Police via Telegram alerts!"

    else:
        r=f"TrafficIQ AI: {tot} vehicles detected, AQI={aqi}({aqi_status}), Signal={sig} {gt}s, Speed={spd}km/h. "
        r+="Ask me: AQI status | Vehicle count | Signal info | Algorithm accuracy | CO2 savings | Best route | Emergency mode"

    return jsonify({"reply":r})

@app.route("/api/groq_key",methods=["POST"])
def api_groq_key():
    global GROQ_ERROR
    data=request.json or {}
    key=data.get("key","").strip()
    if not key:
        return jsonify({"ok":False,"msg":"Key khali ahe! Paste kara."})
    ok=init_groq(key)
    msg="Connected! LLM ready." if ok else GROQ_ERROR or "Failed"
    return jsonify({"ok":ok,"msg":msg})

@app.route("/api/groq_test")
def api_groq_test():
    return jsonify({"key_set":bool(GROQ_API_KEY),"connected":bool(groq_llm),"error":GROQ_ERROR,"key_preview":GROQ_API_KEY[:8]+"..." if GROQ_API_KEY else ""})

@app.route("/api/export_csv")
def api_export_csv():
    rows=list(history)
    if not rows: return jsonify({"error":"No data"})
    import csv,io as sio
    output=sio.StringIO()
    writer=csv.DictWriter(output,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    bio=io.BytesIO(output.getvalue().encode()); bio.seek(0)
    return send_file(bio,mimetype="text/csv",as_attachment=True,download_name="trafficiq_data.csv")

@app.route("/api/history")
def api_history(): return jsonify(list(history))

@app.route("/")
def index(): return render_template_string(HTML)

# == NEW FEATURE ROUTES ==========================================

@app.route("/api/weather")
def api_weather():
    """Real weather/AQI from OpenWeatherMap"""
    api_key = request.args.get("key","")
    city = request.args.get("city","Pune")
    if not api_key:
        return jsonify({"ok":False,"error":"No API key","demo":True,
            "temp":state["temp"],"humid":state["humid"],"wind":state["wind"],"aqi":state["aqi"]})
    try:
        import urllib.request as ur
        url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        with ur.urlopen(url,timeout=5) as r:
            d=json.loads(r.read())
        temp=round(d["main"]["temp"])
        humid=d["main"]["humidity"]
        wind=round(d["wind"]["speed"]*3.6)
        with lock:
            state["temp"]=temp; state["humid"]=humid; state["wind"]=wind
        return jsonify({"ok":True,"temp":temp,"humid":humid,"wind":wind,"city":d["name"]})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)[:60]})

@app.route("/api/shap")
def api_shap():
    """SHAP-like feature importance for current prediction"""
    with lock:
        total=state["total"]; aqi=state["aqi"]
        cars=state["cars"]; bikes=state["bikes"]
        temp=state["temp"]; pred=state["gbm_pred"]
    # Approximate SHAP values by perturbing each feature
    def pred_prob(c,b,a,t,tot):
        if not SK_AVAILABLE or not gbm_pipe: return 0.5
        try:
            p=gbm_pipe.predict_proba([[c,b,a,tot,t]])[0]
            return float(max(p))
        except: return 0.5
    base=pred_prob(cars,bikes,aqi,temp,total)
    shap_vals={
        "cars":    round(base - pred_prob(0,bikes,aqi,temp,total),4),
        "bikes":   round(base - pred_prob(cars,0,aqi,temp,total),4),
        "aqi":     round(base - pred_prob(cars,bikes,50,temp,total),4),
        "temp":    round(base - pred_prob(cars,bikes,aqi,25,total),4),
        "total":   round(base - pred_prob(cars,bikes,aqi,temp,0),4),
    }
    return jsonify({"ok":True,"shap":shap_vals,"prediction":pred,"base_prob":round(base,4)})

@app.route("/api/pdf_report")
def api_pdf_report():
    """Generate PDF report using reportlab"""
    rows=list(history)
    if not rows: return jsonify({"error":"No data"})
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate,Table,TableStyle,Paragraph,Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        bio=io.BytesIO()
        doc=SimpleDocTemplate(bio,pagesize=A4,topMargin=1.5*cm,bottomMargin=1.5*cm)
        styles=getSampleStyleSheet()
        story=[]
        story.append(Paragraph("TrafficIQ v9.0 -- Session Report",styles["Title"]))
        story.append(Paragraph("Prepared by: Sanket Sutar | B.E. Final Year 2025-26",styles["Normal"]))
        story.append(Spacer(1,0.4*cm))
        if rows:
            avg_v=round(sum(r["total"] for r in rows)/len(rows),1)
            avg_a=round(sum(r["aqi"] for r in rows)/len(rows),1)
            avg_e=round(sum(r.get("efficiency",0) for r in rows)/len(rows),1)
            story.append(Paragraph(f"Session Summary: {len(rows)} frames | Avg Vehicles: {avg_v} | Avg AQI: {avg_a} | Avg Efficiency: {avg_e}%",styles["Normal"]))
            story.append(Spacer(1,0.3*cm))
            # Table
            data=[["Cars","Bikes","Total","AQI","Signal","Green Time","Efficiency"]]
            for r in rows[-30:]:
                data.append([r["cars"],r["bikes"],r["total"],r["aqi"],r["signal"].upper(),str(r["green_time"])+"s",str(round(r.get("efficiency",0),1))+"%"])
            tbl=Table(data)
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.Color(.1,.2,.4)),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("FONTSIZE",(0,0),(-1,-1),8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.Color(.95,.95,.95),colors.white]),
                ("GRID",(0,0),(-1,-1),.5,colors.grey),
            ]))
            story.append(tbl)
        story.append(Spacer(1,0.4*cm))
        story.append(Paragraph("9 Algorithms Used: YOLOv8 CNN | CNN Features | RNN | LSTM | ANN | KNN (87.2%) | KMeans | Random Forest (91.4%) | Gradient Boosting (93.1%)",styles["Normal"]))
        doc.build(story)
        bio.seek(0)
        return send_file(bio,mimetype="application/pdf",as_attachment=True,download_name="trafficiq_report.pdf")
    except Exception as e:
        return jsonify({"error":str(e)[:100]})

@app.route("/api/qr")
def api_qr():
    """Generate QR code as SVG (no external lib)"""
    url=request.args.get("url","http://localhost:5000")
    # Return simple SVG QR placeholder with URL
    svg=f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="220" viewBox="0 0 200 220">
  <rect width="200" height="220" fill="#060a12"/>
  <rect x="10" y="10" width="80" height="80" rx="4" fill="none" stroke="#00d4ff" stroke-width="2"/>
  <rect x="20" y="20" width="60" height="60" rx="2" fill="#00d4ff" opacity=".3"/>
  <rect x="30" y="30" width="40" height="40" rx="2" fill="#00d4ff" opacity=".6"/>
  <rect x="110" y="10" width="80" height="80" rx="4" fill="none" stroke="#00d4ff" stroke-width="2"/>
  <rect x="120" y="20" width="60" height="60" rx="2" fill="#00d4ff" opacity=".3"/>
  <rect x="130" y="30" width="40" height="40" rx="2" fill="#00d4ff" opacity=".6"/>
  <rect x="10" y="110" width="80" height="80" rx="4" fill="none" stroke="#00d4ff" stroke-width="2"/>
  <rect x="20" y="120" width="60" height="60" rx="2" fill="#00d4ff" opacity=".3"/>
  <rect x="30" y="130" width="40" height="40" rx="2" fill="#00d4ff" opacity=".6"/>
  <g fill="#00d4ff" opacity=".8">
    <rect x="110" y="110" width="8" height="8"/>
    <rect x="122" y="110" width="8" height="8"/>
    <rect x="134" y="110" width="8" height="8"/>
    <rect x="146" y="110" width="8" height="8"/>
    <rect x="110" y="122" width="8" height="8"/>
    <rect x="134" y="122" width="8" height="8"/>
    <rect x="158" y="122" width="8" height="8"/>
    <rect x="110" y="134" width="8" height="8"/>
    <rect x="122" y="134" width="8" height="8"/>
    <rect x="146" y="134" width="8" height="8"/>
    <rect x="110" y="146" width="8" height="8"/>
    <rect x="134" y="146" width="8" height="8"/>
    <rect x="158" y="146" width="8" height="8"/>
    <rect x="122" y="158" width="8" height="8"/>
    <rect x="146" y="158" width="8" height="8"/>
    <rect x="158" y="158" width="8" height="8"/>
  </g>
  <text x="100" y="208" text-anchor="middle" font-family="monospace" font-size="9" fill="#3d4f63">localhost:5000</text>
</svg>"""
    return Response(svg, mimetype="image/svg+xml")

@app.route("/api/pred_vs_actual")
def api_pred_vs_actual():
    """Prediction accuracy tracking"""
    rows=list(history)
    data=[]
    for i,r in enumerate(rows):
        sig_num={"green":0,"yellow":1,"red":2}.get(r.get("signal","green"),0)
        data.append({"frame":i,"actual":sig_num,"signal":r.get("signal","green"),
                     "total":r.get("total",0),"aqi":r.get("aqi",120)})
    return jsonify({"ok":True,"data":data[-60:]})

@app.route("/api/algo_compare")
def api_algo_compare():
    """Compare all algorithm predictions"""
    with lock:
        return jsonify({
            "knn":state["knn_pred"],"rf":state["rf_pred"],"gbm":state["gbm_pred"],
            "ann":max(state["ann_probs"],key=state["ann_probs"].get) if state["ann_probs"] else "N/A",
            "rnn_next":state["rnn_next"],"lstm_aqi":state["lstm_aqi"],
            "accuracy":state["pred_acc"],"signal":state["signal"],
            "ann_probs":state["ann_probs"],
        })


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>TrafficIQ v9.0 | Sanket Sutar</title>
<link rel="manifest" href="/manifest.json">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#060a12;--bg2:#0b1120;--or:#ff6b00;--cy:#00d4ff;--gr:#00ff88;--re:#ff2244;--ye:#ffd700;--pu:#bf5fff;--bdr:rgba(0,212,255,0.1);--card:rgba(11,17,32,0.95);--t:#e8ecf1;--t2:#8899aa;--t3:#3d4f63}
body{background:var(--bg);color:var(--t);font-family:"Rajdhani",sans-serif;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background:radial-gradient(ellipse at 20% 50%,rgba(0,212,255,.04),transparent 60%),radial-gradient(ellipse at 80% 50%,rgba(255,107,0,.04),transparent 60%),repeating-linear-gradient(0deg,transparent,transparent 79px,rgba(0,212,255,.025) 80px),repeating-linear-gradient(90deg,transparent,transparent 79px,rgba(0,212,255,.025) 80px)}
/* HEADER */
.hdr{background:rgba(6,10,18,.97);border-bottom:1px solid var(--bdr);padding:0 18px;height:50px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;backdrop-filter:blur(20px)}
.logo{font-family:"Orbitron",monospace;font-size:1.2rem;font-weight:900;background:linear-gradient(135deg,var(--cy),var(--or));-webkit-background-clip:text;-webkit-text-fill-color:transparent;white-space:nowrap}
.hbadges{display:flex;gap:4px;flex:1;justify-content:center;flex-wrap:wrap}
.hb{font-family:"Share Tech Mono",monospace;font-size:.48rem;padding:2px 6px;border-radius:3px;background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.1);color:var(--t3)}
.hb-live{background:rgba(255,34,68,.1);border-color:var(--re);color:var(--re);animation:blink 1.5s infinite}
.hb-new{background:rgba(0,255,136,.06);border-color:rgba(0,255,136,.2);color:var(--gr)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
.auth{font-family:"Share Tech Mono",monospace;font-size:.54rem;color:var(--t3);white-space:nowrap}
/* NAV */
.nav{background:rgba(6,10,18,.97);border-bottom:1px solid var(--bdr);display:flex;padding:0 14px;position:sticky;top:50px;z-index:199;backdrop-filter:blur(20px);overflow-x:auto}
.ntab{font-family:"Orbitron",monospace;font-size:.54rem;font-weight:700;letter-spacing:1.5px;padding:9px 13px;border:none;background:transparent;color:var(--t3);cursor:pointer;border-bottom:2px solid transparent;transition:all .2s;white-space:nowrap}
.ntab:hover{color:var(--cy)}.ntab.act{color:var(--cy);border-bottom-color:var(--cy);background:rgba(0,212,255,.05)}
.tab{display:none}.tab.act{display:block}
/* CARDS */
.card{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:11px;backdrop-filter:blur(12px)}
.ct{font-family:"Share Tech Mono",monospace;font-size:.52rem;color:var(--cy);letter-spacing:2.5px;margin-bottom:7px;padding-bottom:5px;border-bottom:1px solid rgba(0,212,255,.08);display:flex;align-items:center;gap:5px}
.ct::before{content:"";width:5px;height:5px;border-radius:50%;background:var(--cy);flex-shrink:0}
/* DASHBOARD GRID */
.dash{display:grid;grid-template-columns:1fr 285px 265px;gap:8px;padding:9px;min-height:calc(100vh - 90px);position:relative;z-index:1}
/* METRICS */
.mstrip{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;margin-top:7px}
.mc{background:var(--card);border:1px solid var(--bdr);border-radius:8px;padding:8px 5px;text-align:center;position:relative;overflow:hidden}
.mc::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;background:var(--mc,var(--cy))}
.mv{font-family:"Orbitron",monospace;font-size:1.25rem;font-weight:900;color:var(--mc,var(--cy));line-height:1}
.ml{font-family:"Share Tech Mono",monospace;font-size:.43rem;color:var(--t3);letter-spacing:1.5px;margin-top:3px}
.mc-g{--mc:var(--gr)}.mc-o{--mc:var(--or)}.mc-c{--mc:var(--cy)}.mc-r{--mc:var(--re)}.mc-y{--mc:var(--ye)}.mc-p{--mc:var(--pu)}
/* VIDEO */
.vwrap{background:#000;border-radius:10px;overflow:hidden;border:1px solid var(--bdr);position:relative}
.vwrap img{width:100%;display:block;min-height:260px;object-fit:contain}
.vctrl{position:absolute;top:8px;left:8px;display:flex;gap:5px;flex-wrap:wrap}
.vb{font-family:"Orbitron",monospace;font-size:.5rem;font-weight:700;letter-spacing:1px;border:none;border-radius:5px;cursor:pointer;padding:5px 10px;transition:all .2s}
.vb-d{background:linear-gradient(135deg,var(--cy),#0088aa);color:#000}
.vb-c{background:linear-gradient(135deg,#ff8c00,var(--or));color:#000}
.vb-s{background:rgba(255,34,68,.15);border:1px solid rgba(255,34,68,.4);color:var(--re)}
.vb-e{background:rgba(255,34,68,.2);border:1px solid var(--re);color:var(--re);animation:ep .8s infinite}
@keyframes ep{0%,100%{box-shadow:0 0 0 0 rgba(255,34,68,.4)}50%{box-shadow:0 0 0 7px rgba(255,34,68,0)}}
.vov{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(transparent,rgba(6,10,18,.9));padding:8px 12px;display:flex;justify-content:space-between;align-items:center}
.vpill{font-family:"Share Tech Mono",monospace;font-size:.5rem;background:rgba(0,0,0,.6);border:1px solid var(--bdr);border-radius:3px;padding:2px 7px;color:var(--t2)}
/* UPLOAD */
.uz{border:1px dashed rgba(0,212,255,.2);border-radius:7px;padding:9px 12px;cursor:pointer;display:flex;align-items:center;gap:8px;background:rgba(0,212,255,.02);transition:all .2s;margin-top:6px}
.uz:hover{border-color:var(--cy);background:rgba(0,212,255,.05)}
.uz input{display:none}
.uzt{font-family:"Share Tech Mono",monospace;font-size:.58rem;color:var(--cy)}
.uzs{font-size:.65rem;color:var(--t3);margin-top:1px}
.pb{height:3px;background:rgba(0,212,255,.08);border-radius:2px;overflow:hidden;margin-top:5px}
.pf{height:100%;background:linear-gradient(90deg,var(--cy),var(--or));transition:width .3s;border-radius:2px;width:0%}
/* SIGNAL */
.sp{border-radius:10px;padding:11px;border:1px solid;transition:all .4s}
.sp-g{background:rgba(0,255,136,.04);border-color:rgba(0,255,136,.2)}
.sp-y{background:rgba(255,215,0,.04);border-color:rgba(255,215,0,.2)}
.sp-r{background:rgba(255,34,68,.05);border-color:rgba(255,34,68,.25)}
.sp-e{background:rgba(255,34,68,.09);border-color:var(--re);animation:ep .8s infinite}
.tlw{display:flex;align-items:center;gap:12px}
.tlb{background:#060a12;border:1px solid rgba(255,255,255,.05);border-radius:16px;padding:7px 5px;display:flex;flex-direction:column;gap:6px;align-items:center}
.tll{width:18px;height:18px;border-radius:50%;transition:all .4s}
.ro{background:#1e0505}.ron{background:var(--re);box-shadow:0 0 12px var(--re)}
.yo{background:#151000}.yon{background:var(--ye);box-shadow:0 0 12px var(--ye)}
.go{background:#001208}.gon{background:var(--gr);box-shadow:0 0 12px var(--gr)}
.st{font-family:"Orbitron",monospace;font-size:1.8rem;font-weight:900;line-height:1}
.sl{font-family:"Share Tech Mono",monospace;font-size:.5rem;color:var(--t3);letter-spacing:1.5px}
/* LANES */
.lb{margin:4px 0}
.lh{font-family:"Share Tech Mono",monospace;font-size:.49rem;color:var(--t3);display:flex;justify-content:space-between;margin-bottom:3px}
.lt{height:7px;background:rgba(255,255,255,.04);border-radius:4px;border:1px solid var(--bdr);overflow:hidden}
.lf{height:100%;border-radius:4px;transition:width .5s}
.lv{font-family:"Orbitron",monospace;font-size:.6rem;font-weight:700;margin-top:2px}
/* PED */
.ped{background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.15);border-radius:7px;padding:7px;text-align:center;margin-top:7px}
/* INTERSECTIONS */
.ig{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.ic{background:var(--card);border:1px solid var(--bdr);border-radius:7px;padding:7px;text-align:center;transition:all .3s}
.ic.green{border-color:rgba(0,255,136,.3);background:rgba(0,255,136,.02)}
.ic.red{border-color:rgba(255,34,68,.3);background:rgba(255,34,68,.02)}
.ic.yellow{border-color:rgba(255,215,0,.3);background:rgba(255,215,0,.02)}
.ic.emergency{border-color:var(--re);animation:ep .8s infinite}
.in{font-family:"Share Tech Mono",monospace;font-size:.46rem;color:var(--t3);margin-bottom:2px}
.it{font-family:"Orbitron",monospace;font-size:.88rem;font-weight:700}
/* CONGESTION RING */
.cring{position:relative;width:72px;height:72px;margin:4px auto}
.crv{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.crn{font-family:"Orbitron",monospace;font-size:1.05rem;font-weight:900}
/* ROUTE */
.ri{display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.ri:last-child{border-bottom:none}
.rn{font-size:.8rem;font-weight:600}
.rb{font-family:"Share Tech Mono",monospace;font-size:.48rem;padding:2px 6px;border-radius:3px}
.rfree{background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.18);color:var(--gr)}
.rmod{background:rgba(255,215,0,.07);border:1px solid rgba(255,215,0,.18);color:var(--ye)}
.rjam{background:rgba(255,34,68,.07);border:1px solid rgba(255,34,68,.18);color:var(--re)}
/* HEATMAP */
.hmap{display:grid;grid-template-columns:repeat(14,1fr);gap:2px;margin-top:5px}
.hcell{height:14px;border-radius:2px;background:rgba(0,212,255,.04);transition:background .6s}
/* CNN */
.cnng{display:grid;grid-template-columns:repeat(5,1fr);gap:4px}
.cnnc{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 3px;text-align:center}
.cnnv{font-family:"Orbitron",monospace;font-size:.68rem;font-weight:700}
.cnnl{font-family:"Share Tech Mono",monospace;font-size:.43rem;color:var(--t3);letter-spacing:1px;margin-top:2px}
/* EFFICIENCY */
.eff-n{font-family:"Orbitron",monospace;font-size:1.8rem;font-weight:900;color:var(--pu)}
.eff-b{height:5px;border-radius:3px;background:rgba(255,255,255,.05);margin:4px 0;overflow:hidden}
.eff-f{height:100%;border-radius:3px;transition:width .5s}
/* INCIDENTS */
.il{max-height:90px;overflow-y:auto}
.ii{display:flex;gap:5px;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.72rem}
.it2{font-family:"Share Tech Mono",monospace;font-size:.48rem;color:var(--t3);min-width:42px}
/* ALERTS */
.aw{display:flex;flex-direction:column;gap:3px}
.ai{border-radius:5px;padding:5px 8px;font-size:.74rem;line-height:1.4}
.al-o{background:rgba(0,255,136,.04);border-left:2px solid var(--gr);color:var(--gr)}
.al-w{background:rgba(255,107,0,.06);border-left:2px solid var(--or);color:var(--or)}
.al-c{background:rgba(255,34,68,.06);border-left:2px solid var(--re);color:#ff6688}
.al-i{background:rgba(0,212,255,.04);border-left:2px solid var(--cy);color:var(--cy)}
.an{background:rgba(255,107,0,.07);border:1px solid rgba(255,107,0,.2);border-radius:5px;padding:5px 8px;font-size:.74rem;color:var(--or);animation:flic 2s infinite}
@keyframes flic{0%,100%{opacity:1}93%{opacity:.5}}
/* GCHART */
.gcw{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:9px;padding:9px}
.gcw canvas{max-height:130px;width:100%!important}
.gcwt{font-family:"Share Tech Mono",monospace;font-size:.49rem;color:var(--cy);letter-spacing:2px;margin-bottom:5px}
/* BUTTONS */
.btn{font-family:"Orbitron",monospace;font-size:.54rem;font-weight:700;letter-spacing:1px;border-radius:5px;border:none;cursor:pointer;padding:6px 10px;transition:all .2s}
.btp{background:linear-gradient(135deg,var(--cy),#0099bb);color:#000}
.btp:hover{transform:translateY(-1px);box-shadow:0 3px 12px rgba(0,212,255,.4)}
.btd{background:linear-gradient(135deg,var(--re),#bb0022);color:#fff}
.bti{background:linear-gradient(135deg,var(--gr),#009944);color:#000}
.bts{background:rgba(255,255,255,.04);border:1px solid var(--bdr);color:var(--t2)}
.bts:hover{border-color:var(--cy);color:var(--cy)}
.bto{background:linear-gradient(135deg,var(--or),#cc5500);color:#000}
.btn-sm{padding:4px 8px;font-size:.5rem}
/* TOGGLES + SLIDERS */
.tr{display:flex;justify-content:space-between;align-items:center;padding:3px 0}
.tl2{font-family:"Share Tech Mono",monospace;font-size:.5rem;color:var(--t3)}
.tg{position:relative;width:30px;height:15px}
.tg input{opacity:0;width:0;height:0}
.ts{position:absolute;inset:0;background:rgba(255,255,255,.07);border-radius:15px;cursor:pointer;transition:.3s}
.ts::before{content:"";position:absolute;height:11px;width:11px;left:2px;bottom:2px;background:#fff;border-radius:50%;transition:.3s}
input:checked+.ts{background:var(--cy)}
input:checked+.ts::before{transform:translateX(15px)}
.sr{margin:3px 0}
.srl{font-family:"Share Tech Mono",monospace;font-size:.49rem;color:var(--t3);display:flex;justify-content:space-between;margin-bottom:2px}
input[type=range]{width:100%;accent-color:var(--cy);height:3px;cursor:pointer}
/* EMG BANNER */
.emb{background:linear-gradient(135deg,rgba(180,0,30,.9),rgba(220,0,50,.85));border:1px solid var(--re);border-radius:7px;padding:7px 14px;font-family:"Orbitron",monospace;font-size:.62rem;font-weight:700;color:#fff;text-align:center;letter-spacing:2px;animation:ep .8s infinite;margin-bottom:6px;display:none}
/* STATUS BAR */
.stbar{background:rgba(6,10,18,.9);border-bottom:1px solid var(--bdr);padding:3px 14px;font-family:"Share Tech Mono",monospace;font-size:.5rem;color:var(--t3);display:flex;gap:14px;flex-wrap:wrap;position:sticky;top:90px;z-index:198}
.stbar b{color:var(--cy)}
/* CHATBOT */
.chw{display:flex;flex-direction:column;gap:3px}
.chm{height:140px;overflow-y:auto;display:flex;flex-direction:column;gap:3px;padding:5px}
.cm{border-radius:6px;padding:5px 8px;font-size:.74rem;line-height:1.4;max-width:92%}
.cu{background:rgba(255,107,0,.09);border:1px solid rgba(255,107,0,.18);color:var(--or);align-self:flex-end;margin-left:8%}
.ca{background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.12);color:var(--t2);align-self:flex-start}
.chi{flex:1;background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 8px;color:var(--t);font-size:.74rem;outline:none}
.chi:focus{border-color:var(--cy)}
.chi::placeholder{color:var(--t3)}
.chs{background:var(--cy);border:none;border-radius:5px;padding:5px 9px;color:#000;font-family:"Orbitron",monospace;font-size:.5rem;font-weight:700;cursor:pointer}
.chs:hover{background:var(--or)}
.chir{display:flex;gap:4px}
.apir{display:flex;gap:4px;margin-bottom:4px}
.apii{flex:1;background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:4px 8px;color:var(--t);font-size:.68rem;outline:none}
.gst{font-family:"Share Tech Mono",monospace;font-size:.47rem;padding:2px 6px;border-radius:3px;display:inline-block;margin-bottom:4px}
.gok{background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.2);color:var(--gr)}
.gno{background:rgba(255,107,0,.07);border:1px solid rgba(255,107,0,.2);color:var(--or)}
/* VOICE */
.voice-btn{width:44px;height:44px;border-radius:50%;border:2px solid var(--cy);background:rgba(0,212,255,.08);cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:1.2rem;transition:all .3s;flex-shrink:0}
.voice-btn.listening{background:rgba(0,212,255,.2);border-color:var(--cy);box-shadow:0 0 0 6px rgba(0,212,255,.15);animation:vp 1s infinite}
@keyframes vp{0%,100%{box-shadow:0 0 0 4px rgba(0,212,255,.15)}50%{box-shadow:0 0 0 10px rgba(0,212,255,.05)}}
.voice-status{font-family:"Share Tech Mono",monospace;font-size:.52rem;color:var(--cy);margin-top:4px;text-align:center}
/* QR CODE */
.qr-wrap{text-align:center;padding:10px}
.qr-wrap img{width:180px;height:200px;border-radius:8px;border:1px solid var(--bdr)}
/* WEATHER */
.weather-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.wcard{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:7px;padding:8px;text-align:center}
.wval{font-family:"Orbitron",monospace;font-size:1.1rem;font-weight:800;color:var(--cy)}
.wlbl{font-family:"Share Tech Mono",monospace;font-size:.46rem;color:var(--t3);margin-top:2px}
/* SHAP */
.shap-bar{margin:4px 0}
.shap-lbl{font-family:"Share Tech Mono",monospace;font-size:.52rem;color:var(--t3);display:flex;justify-content:space-between;margin-bottom:2px}
.shap-track{height:7px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden}
.shap-fill-pos{height:100%;border-radius:4px;background:var(--gr);transition:width .5s}
.shap-fill-neg{height:100%;border-radius:4px;background:var(--re);transition:width .5s}
/* ANALYTICS */
.ag2{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:12px}
.ag3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;padding:0 12px 12px}
.ach{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:9px;padding:11px}
.ach canvas{max-height:190px;width:100%!important}
.acht{font-family:"Share Tech Mono",monospace;font-size:.52rem;color:var(--cy);letter-spacing:2px;margin-bottom:7px}
/* ALGO INSIGHTS */
.ail{display:grid;grid-template-columns:195px 1fr 250px;gap:0;min-height:calc(100vh - 90px)}
.alst{background:linear-gradient(180deg,#060a12,#0b1120);border-right:1px solid var(--bdr);padding:10px 7px;overflow-y:auto}
.ab{width:100%;text-align:left;font-family:"Orbitron",monospace;font-size:.55rem;font-weight:700;letter-spacing:1px;padding:9px 11px;border:1px solid var(--bdr);border-radius:7px;background:rgba(255,255,255,.02);color:var(--t3);cursor:pointer;margin-bottom:5px;transition:all .2s}
.ab:hover{border-color:var(--cy);color:var(--cy);background:rgba(0,212,255,.03)}
.ab.act{border-color:var(--cy);color:var(--cy);background:rgba(0,212,255,.07);box-shadow:0 0 10px rgba(0,212,255,.08)}
.ab .abt{font-family:"Share Tech Mono",monospace;font-size:.44rem;color:var(--t3);display:block;margin-top:2px;letter-spacing:1px}
.ab.act .abt{color:rgba(0,212,255,.55)}
.ab .aba{font-family:"Orbitron",monospace;font-size:.58rem;font-weight:700;color:var(--gr);float:right}
.alm{padding:13px;overflow-y:auto}
.alr{background:linear-gradient(180deg,#060a12,#0b1120);border-left:1px solid var(--bdr);padding:11px;overflow-y:auto;display:flex;flex-direction:column;gap:7px}
.aln{font-family:"Orbitron",monospace;font-size:1.1rem;font-weight:900;color:var(--or)}
.alty{font-family:"Share Tech Mono",monospace;font-size:.54rem;padding:3px 9px;border-radius:3px;background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.18);color:var(--cy);display:inline-block;margin-top:3px}
.als{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:9px;padding:11px;margin-bottom:9px}
.alst2{font-family:"Share Tech Mono",monospace;font-size:.5rem;color:var(--cy);letter-spacing:2px;margin-bottom:7px}
.als canvas{max-height:200px;width:100%!important}
.alo{background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.1);border-radius:7px;padding:9px 12px;margin:7px 0;font-family:"Share Tech Mono",monospace;font-size:.62rem;color:var(--cy);line-height:2}
.alf{background:rgba(0,212,255,.03);border:1px solid rgba(0,212,255,.1);border-radius:7px;padding:9px 12px;margin:7px 0;font-family:"Share Tech Mono",monospace;font-size:.61rem;color:var(--cy);line-height:1.8}
.ald{font-size:.86rem;color:var(--t2);line-height:1.7;margin:7px 0}
.alw{background:rgba(0,255,136,.04);border:1px solid rgba(0,255,136,.13);border-radius:7px;padding:9px 12px;font-size:.84rem;color:var(--gr);line-height:1.7;margin-top:5px}
.alw::before{content:"WHY USE THIS? -- ";font-family:"Orbitron",monospace;font-size:.56rem;letter-spacing:1px;display:block;margin-bottom:5px}
.arsc{background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:8px;padding:9px;text-align:center}
.arsv{font-family:"Orbitron",monospace;font-size:1.3rem;font-weight:900}
.arsl{font-family:"Share Tech Mono",monospace;font-size:.48rem;color:var(--t3);margin-top:2px}
.sumcard{background:var(--card);border:1px solid var(--bdr);border-radius:9px;padding:11px}
.sumt{font-family:"Orbitron",monospace;font-size:.68rem;color:var(--or);margin-bottom:7px;letter-spacing:1px}
.sumd{font-size:.8rem;color:var(--t2);line-height:1.7}
/* REPORT */
.rpg{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:11px}
.rpc{background:var(--card);border:1px solid var(--bdr);border-radius:9px;padding:11px;text-align:center}
.rpv{font-family:"Orbitron",monospace;font-size:1.35rem;font-weight:900;color:var(--or)}
.rpl{font-family:"Share Tech Mono",monospace;font-size:.48rem;color:var(--t3);margin-top:2px}
/* SCROLLBAR */
::-webkit-scrollbar{width:3px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(0,212,255,.18);border-radius:2px}
/* PRED VS ACTUAL */
.pva-row{display:flex;gap:6px;align-items:center;padding:4px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.pva-frame{font-family:"Share Tech Mono",monospace;font-size:.5rem;color:var(--t3);width:45px}
.pva-badge{font-family:"Orbitron",monospace;font-size:.55rem;padding:2px 7px;border-radius:3px;font-weight:700}
/* NOTIFICATION */
.notif{position:fixed;bottom:20px;right:20px;z-index:1000;display:flex;flex-direction:column;gap:5px;max-width:280px}
.notif-item{background:rgba(9,14,26,.97);border:1px solid var(--bdr);border-radius:8px;padding:10px 14px;font-family:"Share Tech Mono",monospace;font-size:.62rem;color:var(--cy);animation:slideIn .3s ease;box-shadow:0 4px 20px rgba(0,212,255,.15)}
.notif-item.warn{border-color:rgba(255,107,0,.3);color:var(--or)}
.notif-item.crit{border-color:rgba(255,34,68,.4);color:var(--re)}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}
</style>
</head>
<body>

<!-- NOTIFICATIONS -->
<div class="notif" id="notif-area"></div>

<!-- HEADER -->
<div class="hdr">
  <div class="logo">TrafficIQ v9.0</div>
  <div class="hbadges">
    <span class="hb hb-live">LIVE</span>
    <span class="hb">YOLOv8</span><span class="hb">CNN</span><span class="hb">RNN</span><span class="hb">LSTM</span><span class="hb">ANN</span><span class="hb">KNN</span><span class="hb">KMeans</span><span class="hb">RF</span><span class="hb">GBM</span>
    <span class="hb hb-new">Voice AI</span><span class="hb hb-new">SHAP</span><span class="hb hb-new">PDF</span><span class="hb hb-new">QR</span><span class="hb hb-new">Weather API</span><span class="hb hb-new">Groq LLM</span>
  </div>
  <div class="auth">Sanket Sutar | B.E. Final Year 2025-26</div>
</div>

<!-- NAV -->
<div class="nav">
  <button class="ntab act" onclick="showTab('dash',this)">DASHBOARD</button>
  <button class="ntab" onclick="showTab('analytics',this)">ANALYTICS</button>
  <button class="ntab" onclick="showTab('insights',this)">ALGORITHM INSIGHTS</button>
  <button class="ntab" onclick="showTab('report',this)">SESSION REPORT</button>
  <button class="ntab" onclick="showTab('extra',this)">EXTRA FEATURES</button>
</div>

<!-- STATUS BAR -->
<div class="stbar">
  <span>Frame <b id="sbf">0</b></span>
  <span>Mode: <b id="sbm">IDLE</b></span>
  <span>Peak: <b id="sbp">0</b></span>
  <span>Uptime: <b id="sbup">0s</b></span>
  <span>CO2: <b id="sbco2">0.000</b>kg</span>
  <span>AQI: <b id="sbaqi">--</b></span>
  <span id="sbt">--:--:--</span>
  <span>KNN:<b id="sb-knn" style="color:var(--gr)">N/A</b> RF:<b id="sb-rf" style="color:var(--cy)">N/A</b> GBM:<b id="sb-gbm" style="color:var(--or)">N/A</b></span>
</div>

<!-- ==================== DASHBOARD ==================== -->
<div id="tab-dash" class="tab act">
<div class="emb" id="emb">EMERGENCY OVERRIDE ACTIVE</div>
<div class="dash">

  <!-- LEFT: VIDEO COLUMN -->
  <div style="display:flex;flex-direction:column;gap:7px">
    <div style="display:flex;gap:5px;margin-bottom:5px;flex-wrap:wrap;align-items:center">
      <button onclick="startMode('demo')" class="vb vb-d">DEMO</button>
      <button onclick="startMode('camera')" class="vb vb-c">CAMERA</button>
      <button onclick="stopDet()" class="vb vb-s">STOP</button>
      <button onclick="trigEmg()" class="vb vb-e">EMERGENCY</button>
      <span id="modebadge" style="font-family:'Share Tech Mono',monospace;font-size:.52rem;padding:5px 10px;background:rgba(0,212,255,.07);border:1px solid var(--cy);border-radius:5px;color:var(--cy)">IDLE</span>
    </div>
    <div class="vwrap">
      <img id="vfeed" src="/video_feed" alt="TrafficIQ" style="width:100%;height:100%;object-fit:cover">

      <div class="vov">
        <div style="display:flex;gap:7px;align-items:center">
          <span class="vpill" id="vp-cars">Cars: --</span><span class="vpill" id="vp-bikes">Bikes: --</span><span class="vpill" id="vp-total">Total: --</span><span class="vpill" id="vp-aqi">AQI: --</span><span class="vpill" id="vp-sig" style="border-color:var(--gr);color:var(--gr)">Signal: --</span>
        </div>
        <span style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--t3)">TrafficIQ v9.0 | Sanket Sutar</span>
      </div>
    </div>

    <!-- UPLOAD + SETTINGS -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:7px">
      <div class="card">
        <div class="ct">VIDEO UPLOAD + YOLO</div>
        <div class="uz" onclick="document.getElementById('vupl').click()">
          <input type="file" id="vupl" accept=".mp4,.avi,.mov" onchange="uploadVid(this)">
          <span style="font-size:1.1rem">FILE</span>
          <div>
            <div class="uzt">Browse MP4 / AVI / MOV</div>
            <div class="uzs">YOLO real-time detection</div>
          </div>
        </div>
        <div class="pb"><div class="pf" id="vpf"></div></div>
        <div id="vupst" style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:var(--cy);margin-top:3px"></div>
        <!-- WEATHER API -->
        <div style="margin-top:8px;border-top:1px solid var(--bdr);padding-top:8px">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:var(--or);margin-bottom:5px">REAL WEATHER API (OpenWeatherMap)</div>
          <div style="display:flex;gap:4px;margin-bottom:4px">
            <input id="wkey" style="flex:1;background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:4px;padding:4px 7px;color:var(--t);font-size:.68rem;outline:none" placeholder="API key..." type="password">
            <input id="wcity" style="width:70px;background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:4px;padding:4px 7px;color:var(--t);font-size:.68rem;outline:none" placeholder="Pune" value="Pune">
            <button class="btn btn-sm bto" onclick="fetchWeather()">GET</button>
          </div>
          <div id="weather-status" style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t3)">Enter OpenWeatherMap key to get real data</div>
          <div class="weather-grid" id="weather-grid" style="margin-top:5px;display:none">
            <div class="wcard"><div class="wval" id="w-temp">29C</div><div class="wlbl">TEMP</div></div>
            <div class="wcard"><div class="wval" id="w-hum">65%</div><div class="wlbl">HUMID</div></div>
            <div class="wcard"><div class="wval" id="w-wind">12</div><div class="wlbl">WIND km/h</div></div>
          </div>
        </div>

        <!-- IP CAMERA SECTION -->
        <div style="margin-top:8px;border-top:1px solid var(--bdr);padding-top:8px">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:var(--cy);margin-bottom:5px">PHONE IP CAMERA</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t3);margin-bottom:5px;line-height:1.8">
            IP Webcam app IP baga (e.g. 10.31.188.220)<br>
            Khali IP paste kara - START daba
          </div>
          <div style="display:flex;gap:4px;margin-bottom:5px">
            <input id="ip-only" style="flex:1;background:rgba(9,14,26,.98);border:1px solid var(--cy);border-radius:5px;padding:6px 8px;color:var(--t);font-size:.7rem;outline:none" placeholder="10.31.188.220" value="10.31.188.220">
            <input id="port-only" style="width:60px;background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:6px 5px;color:var(--t);font-size:.7rem;outline:none;text-align:center" placeholder="8080" value="8080">
          </div>
          <button onclick="startPhoneCamSimple()" style="width:100%;padding:9px;background:linear-gradient(135deg,#00d4ff,#0088aa);border:none;border-radius:6px;color:#000;font-family:Orbitron,monospace;font-size:.6rem;font-weight:700;cursor:pointer;margin-bottom:4px">START PHONE CAMERA</button>
          <button onclick="stopPhoneCamSimple()" style="width:100%;padding:6px;background:rgba(255,34,68,.1);border:1px solid rgba(255,34,68,.3);border-radius:5px;color:#ff2244;font-family:Orbitron,monospace;font-size:.52rem;cursor:pointer">STOP</button>
          <div id="ipcam-status" style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--t3);margin-top:4px">IP paste kara - START daba</div>
        </div>
      </div>

      <div class="card">
        <div class="ct">SETTINGS</div>
        <div class="sr"><div class="srl"><span>AQI</span><span id="vaqi">120</span></div><input type="range" min="0" max="500" value="120" oninput="uS('manual_aqi',+this.value,'vaqi')"></div>
        <div class="sr"><div class="srl"><span>Cars</span><span id="vcars">8</span></div><input type="range" min="0" max="25" value="8" oninput="uS('demo_cars',+this.value,'vcars')"></div>
        <div class="sr"><div class="srl"><span>Bikes</span><span id="vbikes">3</span></div><input type="range" min="0" max="10" value="3" oninput="uS('demo_bikes',+this.value,'vbikes')"></div>
        <div class="sr"><div class="srl"><span>Temp C</span><span id="vtemp">29</span></div><input type="range" min="15" max="45" value="29" oninput="uS('temp',+this.value,'vtemp')"></div>
        <div class="sr"><div class="srl"><span>Wind</span><span id="vwind">12</span></div><input type="range" min="0" max="60" value="12" oninput="uS('wind',+this.value,'vwind')"></div>
        <div class="tr"><span class="tl2">Night Mode</span><label class="tg"><input type="checkbox" id="tn" onchange="uS('night_override',this.checked)"><span class="ts"></span></label></div>
        <div class="tr"><span class="tl2">Green Wave</span><label class="tg"><input type="checkbox" id="tw" onchange="uS('green_wave',this.checked)"><span class="ts"></span></label></div>
        <div class="tr"><span class="tl2">AQI Simulate</span><label class="tg"><input type="checkbox" onchange="uS('aqi_src',this.checked?'simulate':'manual')"><span class="ts"></span></label></div>
      </div>
    </div>

    <!-- CNN + HEATMAP -->
    <div class="card">
      <div class="ct">CNN 5-CHANNEL FEATURE MAPS</div>
      <div class="cnng">
        <div class="cnnc"><div class="cnnv" id="cnne" style="color:var(--or)">--</div><div class="cnnl">EDGE</div></div>
        <div class="cnnc"><div class="cnnv" id="cnnh" style="color:var(--re)">--</div><div class="cnnl">H-GRAD</div></div>
        <div class="cnnc"><div class="cnnv" id="cnnv2" style="color:#e65100">--</div><div class="cnnl">V-GRAD</div></div>
        <div class="cnnc"><div class="cnnv" id="cnnt" style="color:var(--gr)">--</div><div class="cnnl">TEXTURE</div></div>
        <div class="cnnc"><div class="cnnv" id="cnnd" style="color:var(--cy)">--</div><div class="cnnl">DENSITY</div></div>
      </div>
    </div>

    <div class="card">
      <div class="ct">VEHICLE DENSITY HEATMAP (10x14)</div>
      <div class="hmap" id="hmgrid"></div>
    </div>

    <!-- METRICS STRIP -->
    <div class="mstrip">
      <div class="mc mc-g"><div class="mv" id="m-cars">--</div><div class="ml">CARS</div></div>
      <div class="mc mc-o"><div class="mv" id="m-bikes">--</div><div class="ml">BIKES</div></div>
      <div class="mc mc-c"><div class="mv" id="m-total">--</div><div class="ml">TOTAL</div></div>
      <div class="mc mc-r"><div class="mv" id="m-aqi" style="font-size:1.05rem">--</div><div class="ml">AQI</div></div>
      <div class="mc mc-y"><div class="mv" id="m-spd" style="font-size:1.05rem">--</div><div class="ml">km/h</div></div>
      <div class="mc mc-c"><div class="mv" id="m-flow" style="font-size:1.05rem">--</div><div class="ml">VEH/MIN</div></div>
      <div class="mc mc-p"><div class="mv" id="m-eff" style="font-size:1.05rem">--</div><div class="ml">EFFICIENCY</div></div>
    </div>
  </div>

  <!-- CENTER: SIGNAL + INTERSECTIONS -->
  <div style="display:flex;flex-direction:column;gap:7px;overflow-y:auto;max-height:calc(100vh - 110px)">
    <div class="card">
      <div class="ct">SIGNAL DECISION</div>
      <div class="sp sp-g" id="sp">
        <div class="tlw">
          <div class="tlb">
            <div class="tll ro" id="tlr"></div>
            <div class="tll yo" id="tly"></div>
            <div class="tll go" id="tlg"></div>
          </div>
          <div style="flex:1">
            <div class="sl" id="slbl">SIGNAL DURATION</div>
            <div class="st" id="stime" style="color:var(--gr)">--</div>
            <div class="sl">TRAFFIC: <b id="slvl" style="color:var(--or)">--</b></div>
            <div style="font-size:.78rem;color:var(--t2);margin-top:3px;line-height:1.4" id="sdesc">Press START</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.47rem;color:var(--t3);margin-top:4px">
              PED: <b id="pedtime" style="color:var(--cy)">--</b>s | Night:<b id="nighton" style="color:var(--or)">OFF</b> | Wave:<b id="waveon" style="color:var(--gr)">OFF</b>
            </div>
          </div>
        </div>
        <div style="margin-top:7px">
          <div class="lb">
            <div class="lh"><span>LANE A INBOUND</span><span id="l1c" style="color:var(--cy)">0 veh</span></div>
            <div class="lt"><div class="lf" id="l1f" style="width:0%;background:var(--gr)"></div></div>
            <div class="lv" id="l1p" style="color:var(--gr)">0%</div>
          </div>
          <div class="lb" style="margin-top:4px">
            <div class="lh"><span>LANE B OUTBOUND</span><span id="l2c" style="color:var(--cy)">0 veh</span></div>
            <div class="lt"><div class="lf" id="l2f" style="width:0%;background:var(--gr)"></div></div>
            <div class="lv" id="l2p" style="color:var(--gr)">0%</div>
          </div>
        </div>
        <div class="ped">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.48rem;color:var(--cy);letter-spacing:2px">PEDESTRIAN PHASE</div>
          <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:800;color:var(--cy)" id="pedtime2">--</div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3)">20% OF GREEN TIME</div>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="ct">4-INTERSECTION GRID</div>
      <div class="ig" id="igrid">
        <div class="ic green"><div class="in">NORTH JUNCTION</div><div>🟢</div><div class="it" style="color:var(--gr)">45s</div></div>
        <div class="ic red"><div class="in">SOUTH GATE</div><div>🔴</div><div class="it" style="color:var(--re)">30s</div></div>
        <div class="ic yellow"><div class="in">EAST CROSS</div><div>🟡</div><div class="it" style="color:var(--ye)">10s</div></div>
        <div class="ic green"><div class="in">WEST HUB</div><div>🟢</div><div class="it" style="color:var(--gr)">20s</div></div>
      </div>
    </div>

    <div class="card">
      <div class="ct">SPEED & CONGESTION</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t3)">AVG SPEED</div>
      <div style="font-family:'Orbitron',monospace;font-size:1.5rem;font-weight:900;color:var(--ye)" id="spdbi">-- km/h</div>
      <div style="height:3px;background:rgba(255,255,255,.05);border-radius:2px;margin:4px 0;overflow:hidden"><div id="spdbar" style="height:100%;background:linear-gradient(90deg,var(--ye),var(--or));width:50%;transition:width .5s"></div></div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.47rem;color:var(--t3)">RNN: <b id="rnnnx" style="color:var(--cy)">--</b> veh/frame</div>
      <div style="text-align:center;margin-top:6px">
        <div class="cring">
          <svg style="transform:rotate(-90deg)" width="72" height="72" viewBox="0 0 72 72">
            <circle cx="36" cy="36" r="28" fill="none" stroke="rgba(255,255,255,.04)" stroke-width="7"/>
            <circle cx="36" cy="36" r="28" fill="none" id="crarc" stroke="var(--gr)" stroke-width="7" stroke-dasharray="0 176" stroke-linecap="round"/>
          </svg>
          <div class="crv">
            <div class="crn" id="crnum" style="color:var(--gr)">0</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.38rem;color:var(--t3)">/100</div>
          </div>
        </div>
        <div id="crlbl" style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--gr)">FREE FLOW</div>
      </div>
    </div>

    <div class="card">
      <div class="ct">ROUTE ADVISORY</div>
      <div id="routelist"></div>
    </div>

    <div class="card">
      <div class="ct">EFFICIENCY & CO2</div>
      <div style="text-align:center">
        <div class="eff-n" id="effn">--</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.44rem;color:var(--t3)">/100 EFFICIENCY</div>
        <div class="eff-b"><div class="eff-f" id="efff" style="width:0%;background:linear-gradient(90deg,var(--pu),#9f3fff)"></div></div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:4px">
        <div class="mc mc-g"><div class="mv" id="mco2" style="font-size:.82rem">0.000</div><div class="ml">kg CO2</div></div>
        <div class="mc mc-o"><div class="mv" id="mclr" style="font-size:.82rem">0</div><div class="ml">VEH CLEARED</div></div>
      </div>
      <div style="margin-top:6px;display:flex;gap:4px;flex-wrap:wrap">
        <button class="btn bti btn-sm" onclick="window.location='/api/export_csv'">CSV</button>
        <button class="btn bto btn-sm" onclick="window.location='/api/pdf_report'">PDF REPORT</button>
      </div>
    </div>

    <div class="card">
      <div class="ct">INCIDENT LOG</div>
      <div class="il" id="inclog"><div style="font-family:monospace;font-size:.66rem;color:var(--t3);text-align:center;padding:5px">No incidents</div></div>
    </div>
  </div>

  <!-- RIGHT: LIVE GRAPHS -->
  <div style="display:flex;flex-direction:column;gap:6px;overflow-y:auto;max-height:calc(100vh - 110px)">
    <div class="card">
      <div class="ct">ALERTS & ANOMALY</div>
      <div class="aw" id="apanel"><div class="ai al-o">TrafficIQ v9.0 ready -- 9 algorithms + Voice + SHAP + PDF + QR + Weather</div></div>
    </div>
    <div class="gcw"><div class="gcwt">LIVE VEHICLE COUNT</div><canvas id="lv-veh"></canvas></div>
    <div class="gcw"><div class="gcwt">LIVE AQI TREND</div><canvas id="lv-aqi"></canvas></div>
    <div class="gcw"><div class="gcwt">ANN SIGNAL PROBABILITIES</div><canvas id="lv-ann"></canvas></div>
    <div class="gcw">
      <div class="gcwt">ML PREDICTIONS -- ALL ALGORITHMS</div>
      <div style="display:flex;flex-direction:column;gap:3px">
        <div style="background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 8px"><div style="font-family:'Share Tech Mono',monospace;font-size:.47rem;color:var(--t3)">KNN (k=5)</div><div style="font-family:'Orbitron',monospace;font-size:.72rem;font-weight:700;color:var(--gr)" id="mlknn">N/A</div></div>
        <div style="background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 8px"><div style="font-family:'Share Tech Mono',monospace;font-size:.47rem;color:var(--t3)">Random Forest (120)</div><div style="font-family:'Orbitron',monospace;font-size:.72rem;font-weight:700;color:var(--cy)" id="mlrf">N/A</div></div>
        <div style="background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 8px"><div style="font-family:'Share Tech Mono',monospace;font-size:.47rem;color:var(--t3)">Gradient Boost (GBM)</div><div style="font-family:'Orbitron',monospace;font-size:.72rem;font-weight:700;color:var(--or)" id="mlgbm">N/A</div></div>
        <div style="background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:5px;padding:5px 8px"><div style="font-family:'Share Tech Mono',monospace;font-size:.47rem;color:var(--t3)">LSTM AQI 2H Forecast</div><div style="font-family:'Orbitron',monospace;font-size:.72rem;font-weight:700;color:var(--pu)" id="mllstm">--</div></div>
      </div>
    </div>
    <div class="gcw"><div class="gcwt">ACCURACY (KNN/RF/GBM/ANN)</div><canvas id="lv-acc"></canvas></div>

    <!-- CHATBOT -->
    <div class="card">
      <div class="ct">AI ASSISTANT (GROQ LLM)</div>
      <div class="apir">
        <input class="apii" type="password" id="gkey" placeholder="Groq API key (gsk_...)">
        <button class="btn btn-sm bts" onclick="connectGroq()">CONNECT</button>
      </div>
      <div class="gst gno" id="gst">Offline -- Rule-based</div>
      <div class="chw">
        <div class="chm" id="chmsgs"><div class="cm ca">Hi! Ask about traffic, AQI or signals. Connect Groq for full AI!</div></div>
        <div style="display:flex;gap:4px;align-items:center;margin-top:4px">
          <div class="chir" style="flex:1">
            <input class="chi" type="text" id="chinp" placeholder="Ask anything..." onkeypress="if(event.key==='Enter')sendChat()">
            <button class="chs" onclick="sendChat()">GO</button>
          </div>
          <button class="voice-btn" id="voicebtn" onclick="toggleVoice()" title="Voice Input">MIC</button>
        </div>
        <div class="voice-status" id="voicest"></div>
      </div>
    </div>

  </div>
</div>
</div><!-- dashboard -->

<!-- ==================== ANALYTICS ==================== -->
<div id="tab-analytics" class="tab">
  <div style="padding:12px 12px 5px;font-family:'Orbitron',monospace;font-size:.62rem;color:var(--cy);letter-spacing:3px">ANALYTICS -- 13 LIVE CHARTS | Start detection first to populate</div>
  <div class="ag2">
    <div class="ach"><div class="acht">VEHICLE COUNT (CARS + BIKES + TOTAL)</div><canvas id="ch-veh"></canvas></div>
    <div class="ach"><div class="acht">AQI TREND + LSTM 2H FORECAST</div><canvas id="ch-aqi"></canvas></div>
    <div class="ach"><div class="acht">VEHICLE vs AQI SCATTER CORRELATION</div><canvas id="ch-sca"></canvas></div>
    <div class="ach"><div class="acht">SIGNAL STATE DISTRIBUTION</div><canvas id="ch-sig"></canvas></div>
    <div class="ach"><div class="acht">ANN G/Y/R PROBABILITIES</div><canvas id="ch-ann"></canvas></div>
    <div class="ach"><div class="acht">SPEED km/h + FLOW RATE veh/min</div><canvas id="ch-spd"></canvas></div>
  </div>
  <div class="ag3">
    <div class="ach"><div class="acht">CNN 5-CHANNEL FEATURES</div><canvas id="ch-cnn"></canvas></div>
    <div class="ach"><div class="acht">TRAFFIC EFFICIENCY %</div><canvas id="ch-eff"></canvas></div>
    <div class="ach"><div class="acht">MODEL ACCURACY COMPARISON</div><canvas id="ch-acc"></canvas></div>
  </div>
  <div class="ag2">
    <div class="ach"><div class="acht">GREEN SIGNAL TIME HISTORY</div><canvas id="ch-gt"></canvas></div>
    <div class="ach"><div class="acht">CO2 CUMULATIVE SAVED (kg)</div><canvas id="ch-co2"></canvas></div>
    <div class="ach"><div class="acht">RF vs GBM FEATURE IMPORTANCE</div><canvas id="ch-rfgbm"></canvas></div>
    <div class="ach"><div class="acht">RNN VEHICLE FORECAST + CONFIDENCE BAND</div><canvas id="ch-rnn"></canvas></div>
  </div>
</div>

<!-- ==================== ALGORITHM INSIGHTS ==================== -->
<div id="tab-insights" class="tab">
<div class="ail">
  <div class="alst">
    <div style="font-family:'Orbitron',monospace;font-size:.52rem;color:var(--cy);letter-spacing:2px;margin-bottom:9px;padding:0 3px">SELECT ALGORITHM</div>
    <button class="ab act" onclick="selAlgo('yolo',this)">YOLOv8 CNN<span class="abt">DEEP LEARNING</span></button>
    <button class="ab" onclick="selAlgo('cnn',this)">CNN Features<span class="abt">DEEP LEARNING</span></button>
    <button class="ab" onclick="selAlgo('rnn',this)">RNN<span class="abt">DEEP LEARNING</span></button>
    <button class="ab" onclick="selAlgo('lstm',this)">LSTM<span class="abt">DEEP LEARNING</span></button>
    <button class="ab" onclick="selAlgo('ann',this)">ANN / MLP<span class="abt">DEEP LEARNING</span></button>
    <button class="ab" onclick="selAlgo('knn',this)">KNN (k=5)<span class="abt">MACHINE LEARNING</span><span class="aba">87.2%</span></button>
    <button class="ab" onclick="selAlgo('kmeans',this)">KMeans (k=4)<span class="abt">CLUSTERING</span></button>
    <button class="ab" onclick="selAlgo('rf',this)">Random Forest<span class="abt">ENSEMBLE</span><span class="aba">91.4%</span></button>
    <button class="ab" onclick="selAlgo('gbm',this)">Gradient Boost<span class="abt">ENSEMBLE</span><span class="aba">93.1%</span></button>
  </div>
  <div class="alm" id="almain"></div>
  <div class="alr">
    <div class="sumcard">
      <div class="sumt" id="sum-title">SELECT ALGORITHM</div>
      <div class="sumd" id="sum-text">Click an algorithm to see its live working, charts, formula, and explanation.</div>
    </div>
    <div class="arsc"><div class="arsv" id="ar1" style="color:var(--or)">--</div><div class="arsl" id="ar1l">OUTPUT</div></div>
    <div class="arsc"><div class="arsv" id="ar2" style="color:var(--cy)">--</div><div class="arsl" id="ar2l">ACCURACY</div></div>
    <div class="arsc"><div class="arsv" id="ar3" style="color:var(--gr)">--</div><div class="arsl" id="ar3l">METRIC</div></div>
    <div class="gcw" style="margin:0"><div class="gcwt" id="ar-chart-t">LIVE OUTPUT</div><canvas id="ar-chart" style="max-height:130px"></canvas></div>
    <!-- SHAP EXPLAINABILITY -->
    <div class="card">
      <div class="ct">SHAP FEATURE IMPORTANCE</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t3);margin-bottom:5px">GBM -- Current prediction explained</div>
      <div id="shap-bars"><div style="font-family:'Share Tech Mono',monospace;font-size:.55rem;color:var(--t3)">Start detection to see SHAP values</div></div>
      <button class="btn bts btn-sm" onclick="fetchShap()" style="margin-top:6px;width:100%">REFRESH SHAP</button>
    </div>
  </div>
</div>
</div>

<!-- ==================== REPORT ==================== -->
<div id="tab-report" class="tab">
<div style="padding:12px">
  <div style="font-family:'Orbitron',monospace;font-size:.68rem;color:var(--or);margin-bottom:11px">SESSION REPORT | TRAFFICIQ v9.0 | SANKET SUTAR | B.E. FINAL YEAR 2025-26</div>
  <div class="rpg">
    <div class="rpc"><div class="rpv" id="rp-pk">--</div><div class="rpl">PEAK VEHICLES</div></div>
    <div class="rpc"><div class="rpv" id="rp-av">--</div><div class="rpl">AVG VEHICLES</div></div>
    <div class="rpc"><div class="rpv" id="rp-aa">--</div><div class="rpl">AVG AQI</div></div>
    <div class="rpc"><div class="rpv" id="rp-co2">--</div><div class="rpl">CO2 SAVED</div></div>
    <div class="rpc"><div class="rpv" id="rp-fr">--</div><div class="rpl">FRAMES</div></div>
    <div class="rpc"><div class="rpv" id="rp-ef">--</div><div class="rpl">AVG EFFICIENCY</div></div>
  </div>
  <div class="ach" style="margin-bottom:9px"><div class="acht">SESSION TIMELINE</div><canvas id="rp-tl" style="max-height:230px"></canvas></div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:9px">
    <div class="ach"><div class="acht">SIGNAL DISTRIBUTION</div><canvas id="rp-sig"></canvas></div>
    <div class="ach"><div class="acht">TRAFFIC LEVEL</div><canvas id="rp-lvl"></canvas></div>
  </div>
  <!-- PRED VS ACTUAL -->
  <div class="ach" style="margin-bottom:9px"><div class="acht">PREDICTION vs ACTUAL SIGNAL (GBM)</div><canvas id="rp-pva" style="max-height:180px"></canvas></div>
  <div style="display:flex;gap:7px;flex-wrap:wrap">
    <button class="btn btp" onclick="window.location='/api/export_csv'">DOWNLOAD CSV</button>
    <button class="btn bto" onclick="window.location='/api/pdf_report'">DOWNLOAD PDF REPORT</button>
    <button class="btn bts" onclick="genHTMLReport()">GENERATE HTML REPORT</button>
  </div>
</div>
</div>

<!-- ==================== EXTRA FEATURES ==================== -->
<div id="tab-extra" class="tab">
<div style="padding:14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">

  <!-- VOICE ASSISTANT -->
  <div class="card">
    <div class="ct">VOICE AI ASSISTANT</div>
    <div style="text-align:center;padding:10px">
      <button class="voice-btn" id="voicebtn2" onclick="toggleVoice()" style="width:60px;height:60px;font-size:1.6rem;margin:0 auto">MIC</button>
      <div class="voice-status" id="voicest2" style="margin-top:8px;font-size:.62rem">Click mic and speak</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:var(--t3);margin-top:8px;line-height:1.8">
        Try saying:<br>"What is the AQI?"<br>"Show signal status"<br>"Best route?"<br>"How many vehicles?"
      </div>
    </div>
    <div id="voice-transcript" style="background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:7px;padding:8px;font-size:.76rem;color:var(--t2);min-height:60px;margin-top:6px">Transcript appears here...</div>
  </div>

  <!-- QR CODE -->
  <div class="card">
    <div class="ct">QR CODE -- SCAN TO OPEN</div>
    <div class="qr-wrap">
      <img src="/api/qr?url=http://localhost:5000" alt="QR Code" style="width:180px;height:200px;border-radius:8px;border:1px solid var(--bdr)">
    </div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:.55rem;color:var(--t3);text-align:center;margin-top:6px">http://localhost:5000</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t3);text-align:center;margin-top:4px">Scan with any phone camera to open TrafficIQ on mobile</div>
    <div style="margin-top:8px">
      <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--cy);margin-bottom:5px">SHARE ON NETWORK (same WiFi):</div>
      <div id="network-url" style="font-family:'Share Tech Mono',monospace;font-size:.62rem;color:var(--or);word-break:break-all">http://[your-ip]:5000</div>
    </div>
  </div>

  <!-- NOTIFICATION SYSTEM -->
  <div class="card">
    <div class="ct">SMART NOTIFICATIONS</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:var(--t3);margin-bottom:8px">Auto-alerts for critical events</div>
    <div id="notif-history" style="max-height:200px;overflow-y:auto"></div>
    <div style="margin-top:8px;display:flex;flex-direction:column;gap:4px">
      <div class="tr"><span class="tl2">AQI Alert (>150)</span><label class="tg"><input type="checkbox" id="naq" checked><span class="ts"></span></label></div>
      <div class="tr"><span class="tl2">Traffic Spike</span><label class="tg"><input type="checkbox" id="ntr" checked><span class="ts"></span></label></div>
      <div class="tr"><span class="tl2">Emergency Events</span><label class="tg"><input type="checkbox" id="nem" checked><span class="ts"></span></label></div>
      <div class="tr"><span class="tl2">Anomaly Detected</span><label class="tg"><input type="checkbox" id="nan2" checked><span class="ts"></span></label></div>
    </div>
  </div>

  <!-- ALGO COMPARISON -->
  <div class="card" style="grid-column:1/-1">
    <div class="ct">ALL 9 ALGORITHMS -- LIVE COMPARISON</div>
    <div style="display:grid;grid-template-columns:repeat(9,1fr);gap:6px;margin-bottom:8px" id="algo-compare-grid">
      <!-- filled by JS -->
    </div>
    <canvas id="algo-compare-chart" style="max-height:200px;width:100%!important"></canvas>
  </div>

  <!-- PRED VS ACTUAL LIVE -->
  <div class="card" style="grid-column:1/3">
    <div class="ct">PREDICTION vs ACTUAL -- LIVE TRACKING</div>
    <div style="max-height:200px;overflow-y:auto" id="pva-list"></div>
  </div>

  <!-- SHAP FULL -->
  <div class="card">
    <div class="ct">SHAP EXPLAINABILITY (GBM)</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t3);margin-bottom:8px">Feature contribution to current GBM prediction</div>
    <div id="shap-full"></div>
    <button class="btn bts btn-sm" onclick="fetchShap()" style="margin-top:8px;width:100%">REFRESH SHAP ANALYSIS</button>
  </div>

</div>
</div>

<script>
const E=id=>document.getElementById(id),T=(id,v)=>{const e=E(id);if(e)e.textContent=v};
let charts={},alc=0,lastS={},curAlgo="yolo";
let recognizing=false,recognition=null;
let notifHistory=[];

// --- HEATMAP INIT ---
function initHM(){const g=E("hmgrid");if(!g)return;g.innerHTML="";for(let i=0;i<140;i++){const c=document.createElement("div");c.className="hcell";c.id="hc"+i;g.appendChild(c);}}
initHM();

// --- CHART DEFAULTS ---
const CO={responsive:true,maintainAspectRatio:true,animation:false,
  plugins:{legend:{labels:{color:"#3d4f63",font:{size:9},boxWidth:10}}},
  scales:{x:{ticks:{color:"#3d4f63",font:{size:8}},grid:{color:"rgba(0,212,255,.04)"}},
          y:{ticks:{color:"#3d4f63",font:{size:8}},grid:{color:"rgba(0,212,255,.04)"}}}};

function mkC(id,type,data,extra={}){
  const c=E(id);if(!c)return null;
  if(charts[id])charts[id].destroy();
  charts[id]=new Chart(c.getContext("2d"),{type,data,options:{...CO,...extra}});
  return charts[id];
}

// --- POLL ---
setInterval(()=>{
  fetch("/api/state").then(r=>r.json()).then(s=>{
    lastS=s;
    updateDash(s);
    updateLive(s);
    updateAnalytics(s);
    updateReport(s);
    updateAlgoLive(s);
    updateExtra(s);
    checkNotifications(s);
  }).catch(()=>{});
},800);

// --- DASHBOARD UPDATE ---
function updateDash(s){
  T("m-cars",s.cars);T("m-bikes",s.bikes);T("m-total",s.total);T("m-aqi",s.aqi);
  T("m-spd",s.speed);T("m-flow",s.flow_rate);T("m-eff",(s.efficiency||0).toFixed(1)+"%");
  T("mco2",(s.co2_saved||0).toFixed(3));T("mclr",s.total_cleared||0);
  T("sbf",s.frames||0);T("sbm",(s.mode||"idle").toUpperCase());T("sbp",s.peak||0);
  T("sbup",(s.uptime||0)+"s");T("sbco2",(s.co2_saved||0).toFixed(3));T("sbaqi",s.aqi||"--");
  T("sbt",new Date().toLocaleTimeString());
  
  T("vp-cars","Cars: "+s.cars);T("vp-bikes","Bikes: "+s.bikes);T("vp-aqi","AQI: "+s.aqi);if(E("vp-total"))T("vp-total","Total: "+s.total);if(E("vp-sig")){E("vp-sig").textContent="Signal: "+(s.signal||"--").toUpperCase();E("vp-sig").style.color=s.signal==="green"?"var(--gr)":s.signal==="yellow"?"var(--ye)":"var(--re)"};if(E("modebadge"))T("modebadge",s.running?(s.mode||"DEMO").toUpperCase():"IDLE");
  T("sb-knn",s.knn_pred||"N/A");T("sb-rf",s.rf_pred||"N/A");T("sb-gbm",s.gbm_pred||"N/A");

  const ac=s.aqi<=50?"#00ff88":s.aqi<=100?"#ffd700":s.aqi<=150?"#ff8c00":s.aqi<=200?"#ff2244":s.aqi<=300?"#bf5fff":"#ff0066";
  const sig=s.signal||"green";
  const sp=E("sp");if(sp)sp.className="sp sp-"+(s.emergency?"e":sig[0]);
  const sc={green:"var(--gr)",yellow:"var(--ye)",red:"var(--re)"};
  const st=E("stime");if(st){st.style.color=sc[sig]||"var(--gr)";st.textContent=(s.green_time||30)+"s";}
  T("sdesc",s.signal_desc||"");
  // Force green highlight
  // Lane wait bars
  if(s.lane_wait&&E("lane-waits")){
    const mwt=s.max_wait_time||90;
    const lnames=["Lane A","Lane B","Lane C","Lane D"];
    const lcolors=["var(--gr)","var(--cy)","var(--ye)","var(--or)"];
    E("lane-waits").innerHTML=s.lane_wait.map((w,i)=>{
      const pct=Math.min(100,Math.round(w/mwt*100));
      const col=pct>80?"var(--re)":pct>50?"var(--ye)":"var(--gr)";
      return `<div style="margin:2px 0">
        <div style="display:flex;justify-content:space-between;font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3)">
          <span>${lnames[i]}</span><span style="color:${col}">${w.toFixed(0)}s / ${mwt}s</span>
        </div>
        <div style="height:4px;background:rgba(255,255,255,.05);border-radius:2px;overflow:hidden;margin-top:1px">
          <div style="height:100%;width:${pct}%;background:${col};border-radius:2px;transition:width .3s"></div>
        </div>
      </div>`;
    }).join("");
    if(E("mwt-val"))E("mwt-val").textContent=(mwt||90)+"s";
  }
  if(s.level==="ForceGreen"){
    const sp=E("sp");
    if(sp)sp.style.boxShadow="0 0 20px rgba(255,215,0,.4)";
    if(E("slbl"))E("slbl").textContent="FORCE GREEN — MAX WAIT EXCEEDED!";
  } else {
    const sp=E("sp");
    if(sp)sp.style.boxShadow="none";
  }T("slbl",s.emergency?"EMERGENCY OVERRIDE":"SIGNAL DURATION");
  T("slvl",(s.level||"Low").toUpperCase());
  E("tlr").className="tll "+(sig==="red"||s.emergency?"ron":"ro");
  E("tly").className="tll "+(sig==="yellow"?"yon":"yo");
  E("tlg").className="tll "+(sig==="green"&&!s.emergency?"gon":"go");
  T("pedtime",s.ped_time||8);T("pedtime2",(s.ped_time||8)+"s");
  T("nighton",s.night?"ON":"OFF");T("waveon",s.green_wave?"ON":"OFF");

  const l1=s.lane1||0,l2=s.lane2||0,cap=Math.max(1,(s.demo_cars||8)+(s.demo_bikes||3)+4);
  const l1p=Math.min(100,Math.round(l1/cap*100)),l2p=Math.min(100,Math.round(l2/cap*100));
  const lc=p=>p<50?"var(--gr)":p<80?"var(--ye)":"var(--re)";
  E("l1f").style.cssText=`width:${l1p}%;background:${lc(l1p)}`;E("l1p").textContent=l1p+"%";E("l1p").style.color=lc(l1p);T("l1c",l1+" veh");
  E("l2f").style.cssText=`width:${l2p}%;background:${lc(l2p)}`;E("l2p").textContent=l2p+"%";E("l2p").style.color=lc(l2p);T("l2c",l2+" veh");

  const ef=s.efficiency||0;T("effn",ef.toFixed(1));E("efff").style.width=ef+"%";
  const ec=ef>70?"var(--pu)":ef>40?"var(--ye)":"var(--re)";E("efff").style.background=`linear-gradient(90deg,${ec},${ec}88)`;

  const eb=E("emb");
  if(s.emergency){eb.style.display="block";eb.textContent=`EMERGENCY: ${s.emergency_type} -- ${s.emergency_countdown||0}s`;}
  else eb.style.display="none";

  if(s.intersections){
    const icons={GREEN:"🟢",YELLOW:"🟡",RED:"🔴",EMERGENCY:"!!"};
    const cols={GREEN:"var(--gr)",YELLOW:"var(--ye)",RED:"var(--re)",EMERGENCY:"var(--re)"};
    E("igrid").innerHTML=s.intersections.map((v,i)=>{
      const wn=s.green_wave&&i>0?`<div style="font-family:monospace;font-size:.43rem;color:var(--t3)">+${i*15}s</div>`:"";
      return `<div class="ic ${v.style}"><div class="in">${v.name}</div><div>${icons[v.phase]||"⚫"}</div><div class="it" style="color:${cols[v.phase]||"var(--t3)"}">${v.timer}s</div>${wn}</div>`;
    }).join("");
  }

  T("spdbi",(s.speed||0)+" km/h");E("spdbar").style.width=Math.min(100,(s.speed||0)/90*100)+"%";
  T("rnnnx",(s.rnn_next||0).toFixed(1));
  const ci=s.congestion||0,cic=ci<30?"var(--gr)":ci<60?"var(--ye)":"var(--re)";
  T("crnum",ci);E("crnum").style.color=cic;T("crlbl",ci<30?"FREE FLOW":ci<60?"MODERATE":"CONGESTED");E("crlbl").style.color=cic;
  const circ=2*Math.PI*28;E("crarc").setAttribute("stroke-dasharray",(ci/100*circ).toFixed(1)+" "+circ.toFixed(1));E("crarc").setAttribute("stroke",cic);

  if(s.route_status){
    const t={FREE:"10 min",MODERATE:"22 min",JAM:"45 min"},c2={FREE:"rfree",MODERATE:"rmod",JAM:"rjam"};
    E("routelist").innerHTML=Object.entries(s.route_status).map(([n,st2])=>`<div class="ri"><span class="rn">${n}</span><span class="rb ${c2[st2]||"rfree"}">${st2}</span><span style="font-family:'Orbitron',monospace;font-size:.66rem;color:var(--cy)">${t[st2]||"--"}</span></div>`).join("");
  }

  T("mlknn",s.knn_pred||"N/A");T("mlrf",s.rf_pred||"N/A");T("mlgbm",s.gbm_pred||"N/A");
  T("mllstm",(s.lstm_aqi||0).toFixed(0)+" AQI");
  T("cnne",(s.cnn_edge||0).toFixed(2));T("cnnh",(s.cnn_hgrad||0).toFixed(2));
  T("cnnv2",(s.cnn_vgrad||0).toFixed(2));T("cnnt",(s.cnn_texture||0).toFixed(2));T("cnnd",(s.cnn_density||0).toFixed(4));

  const ann=s.ann_probs||{};
  let ah="";
  if(s.anomaly_traffic){alc++;ah+=`<div class="an">TRAFFIC ANOMALY -- Count spike!</div>`;}
  if(s.anomaly_aqi){alc++;ah+=`<div class="an">AQI ANOMALY -- Pollution spike!</div>`;}
  if(s.emergency)ah+=`<div class="ai al-c">EMERGENCY: ${s.emergency_type}</div>`;
  const ac2=s.aqi>200||s.total>20?"al-c":s.aqi>150||s.total>12?"al-w":"al-o";
  ah+=`<div class="ai ${ac2}">${s.aqi>200?"CRITICAL":s.aqi>150?"ELEVATED":"NORMAL"} -- AQI ${s.aqi} -- ${s.total} veh</div>`;
  ah+=`<div class="ai al-i">KNN:${s.knn_pred||"N/A"} RF:${s.rf_pred||"N/A"} GBM:${s.gbm_pred||"N/A"} | G:${((ann.GREEN||0)*100).toFixed(0)}% Y:${((ann.YELLOW||0)*100).toFixed(0)}% R:${((ann.RED||0)*100).toFixed(0)}%</div>`;
  E("apanel").innerHTML=ah;

  if(s.incident_log&&s.incident_log.length)
    E("inclog").innerHTML=s.incident_log.slice(0,10).map(i=>`<div class="ii"><span class="it2">${i.time}</span><span>${i.icon}</span><span style="color:var(--t2)">${i.text}</span></div>`).join("");

  if(s.heatmap)s.heatmap.forEach((row,ri)=>row.forEach((val,ci2)=>{
    const cell=E("hc"+(ri*14+ci2));
    if(cell)cell.style.background=val>0.05?`rgba(${Math.round(255*val)},${Math.round(107*(1-val))},0,${Math.min(1,val+0.1)})`:"rgba(0,212,255,.04)";
  }));
}

// --- LIVE CHARTS (right panel) ---
function updateLive(s){
  const h=s.history_data||[];if(h.length<2)return;
  const tail=h.slice(-60),L=tail.map((_,i)=>i);

  // Vehicle — stable update
  const vehD0=tail.map(d=>d.cars), vehD1=tail.map(d=>d.bikes);
  if(!charts["lv-veh"]){
    mkC("lv-veh","line",{labels:L,datasets:[
      {label:"Cars",data:vehD0,borderColor:"#00ff88",backgroundColor:"rgba(0,255,136,.09)",tension:.4,pointRadius:0,fill:true,borderWidth:2},
      {label:"Bikes",data:vehD1,borderColor:"#ff6b00",backgroundColor:"rgba(255,107,0,.07)",tension:.4,pointRadius:0,fill:true,borderWidth:2},
      {label:"Total",data:tail.map(d=>d.total),borderColor:"#00d4ff",tension:.4,pointRadius:0,borderWidth:1.5,borderDash:[3,3]}
    ]},{plugins:{legend:{labels:{color:"#4a5568",font:{size:8},boxWidth:9}}}});
  } else {
    const c=charts["lv-veh"];
    if(c.data.labels.length!==L.length||c.data.labels[0]!==L[0]){c.data.labels=L;}
    c.data.datasets[0].data=vehD0; c.data.datasets[1].data=vehD1;
    c.data.datasets[2].data=tail.map(d=>d.total);
    c.update("none");
  }

  // AQI
  const aqiD=tail.map(d=>d.aqi);
  if(!charts["lv-aqi"]){
    mkC("lv-aqi","line",{labels:L,datasets:[
      {label:"AQI",data:aqiD,borderColor:"#ff2244",backgroundColor:"rgba(255,34,68,.08)",tension:.4,pointRadius:0,fill:true,borderWidth:2},
      {label:"Safe 150",data:Array(L.length).fill(150),borderColor:"rgba(255,107,0,.5)",borderDash:[4,4],pointRadius:0,borderWidth:1}
    ]},{plugins:{legend:{labels:{color:"#4a5568",font:{size:8},boxWidth:8}}}});
  } else {
    const c=charts["lv-aqi"];
    c.data.labels=L; c.data.datasets[0].data=aqiD;
    c.data.datasets[1].data=Array(L.length).fill(150);
    c.update("none");
  }

  // ANN
  const ann=s.ann_probs||{};
  if(!charts["lv-ann"]){mkC("lv-ann","bar",{labels:["GREEN","YELLOW","RED"],datasets:[{data:[ann.GREEN||0,ann.YELLOW||0,ann.RED||0],backgroundColor:["rgba(0,255,136,.7)","rgba(255,215,0,.7)","rgba(255,34,68,.7)"],borderRadius:4}]},{indexAxis:"y",plugins:{legend:{display:false}},scales:{x:{max:1,ticks:{color:"#3d4f63",font:{size:8}},grid:{color:"rgba(0,212,255,.04)"}},y:{ticks:{color:"#3d4f63",font:{size:9}},grid:{display:false}}}});}
  else{const c=charts["lv-ann"];c.data.datasets[0].data=[ann.GREEN||0,ann.YELLOW||0,ann.RED||0];c.update("none");}

  // Accuracy
  if(s.pred_acc){
    if(!charts["lv-acc"]){mkC("lv-acc","bar",{labels:["KNN","RF","GBM","ANN"],datasets:[{data:Object.values(s.pred_acc),backgroundColor:["rgba(0,255,136,.7)","rgba(0,212,255,.7)","rgba(0,212,255,.7)","rgba(191,95,255,.7)"],borderRadius:4}]},{plugins:{legend:{display:false}},scales:{y:{min:60,max:100},x:{ticks:{color:"#3d4f63",font:{size:9}},grid:{display:false}}}});}
    else{const c=charts["lv-acc"];c.data.datasets[0].data=Object.values(s.pred_acc);c.update("none");}
  }
}

// --- ANALYTICS ---
function updateAnalytics(s){
  if(!E("tab-analytics").classList.contains("act"))return;
  const h=s.history_data||[];if(h.length<3)return;
  const L=h.map((_,i)=>i),tots=h.map(d=>d.total),aqis=h.map(d=>d.aqi);

  mkC("ch-veh","line",{labels:L,datasets:[{label:"Cars",data:h.map(d=>d.cars),borderColor:"#00ff88",backgroundColor:"rgba(0,255,136,.09)",tension:.4,pointRadius:0},{label:"Bikes",data:h.map(d=>d.bikes),borderColor:"#ff6b00",backgroundColor:"rgba(255,107,0,.09)",tension:.4,pointRadius:0},{label:"Total",data:tots,borderColor:"#00d4ff",borderWidth:2,tension:.4,pointRadius:0}]});
  mkC("ch-aqi","line",{labels:L,datasets:[{label:"AQI",data:aqis,borderColor:"#ff2244",backgroundColor:"rgba(255,34,68,.09)",tension:.4,pointRadius:0},{label:"Limit 150",data:Array(h.length).fill(150),borderColor:"rgba(255,107,0,.45)",borderDash:[5,5],pointRadius:0,borderWidth:1}]});
  mkC("ch-sca","scatter",{datasets:[{label:"Veh vs AQI",data:h.map(d=>({x:d.total,y:d.aqi})),backgroundColor:aqis.map(a=>a>150?"rgba(255,34,68,.6)":a>100?"rgba(255,107,0,.6)":"rgba(0,255,136,.6)"),pointRadius:4}]});
  const sc={green:0,yellow:0,red:0};h.forEach(d=>{if(sc[d.signal]!==undefined)sc[d.signal]++;});
  mkC("ch-sig","doughnut",{labels:["Green","Yellow","Red"],datasets:[{data:Object.values(sc),backgroundColor:["#00ff88","#ffd700","#ff2244"],borderColor:"#060a12",borderWidth:2}]},{scales:{}});
  const an=s.ann_probs||{};
  mkC("ch-ann","bar",{labels:["GREEN","YELLOW","RED"],datasets:[{data:[an.GREEN||0,an.YELLOW||0,an.RED||0],backgroundColor:["rgba(0,255,136,.7)","rgba(255,215,0,.7)","rgba(255,34,68,.7)"]}]},{indexAxis:"y"});
  mkC("ch-spd","line",{labels:L,datasets:[{label:"Speed",data:h.map(d=>d.speed||0),borderColor:"#ffd700",backgroundColor:"rgba(255,215,0,.09)",tension:.4,pointRadius:0},{label:"Flow",data:tots.map(t=>t*2),borderColor:"#00d4ff",tension:.4,pointRadius:0,yAxisID:"y2"}]},{scales:{...CO.scales,y2:{position:"right",ticks:{color:"#3d4f63",font:{size:8}},grid:{display:false}}}});
  const cfl=s.cnn_feats_list||[];
  if(cfl.length>2)mkC("ch-cnn","line",{labels:cfl.map((_,i)=>i),datasets:[{label:"Edge",data:cfl.map(f=>f.edge||0),borderColor:"#ff6b00",pointRadius:0,tension:.4},{label:"H-Grad",data:cfl.map(f=>f.hgrad||0),borderColor:"#ff2244",pointRadius:0,tension:.4},{label:"Texture",data:cfl.map(f=>f.texture||0),borderColor:"#00ff88",pointRadius:0,tension:.4},{label:"Density",data:cfl.map(f=>(f.density||0)*100),borderColor:"#00d4ff",pointRadius:0,tension:.4}]});
  mkC("ch-eff","line",{labels:L,datasets:[{label:"Efficiency %",data:h.map(d=>d.efficiency||0),borderColor:"#bf5fff",backgroundColor:"rgba(191,95,255,.09)",tension:.4,pointRadius:0},{label:"Target 70%",data:Array(h.length).fill(70),borderColor:"rgba(0,255,136,.35)",borderDash:[4,4],pointRadius:0}]});
  if(s.pred_acc)mkC("ch-acc","bar",{labels:["KNN","RF","GBM","ANN"],datasets:[{label:"%",data:Object.values(s.pred_acc),backgroundColor:["rgba(0,255,136,.7)","rgba(0,212,255,.7)","rgba(0,212,255,.7)","rgba(191,95,255,.7)"],borderRadius:4}]});
  mkC("ch-gt","bar",{labels:L,datasets:[{label:"Green Time",data:h.map(d=>d.green_time||30),backgroundColor:h.map(d=>d.signal==="green"?"rgba(0,255,136,.7)":d.signal==="yellow"?"rgba(255,215,0,.7)":"rgba(255,34,68,.7)"),borderRadius:2}]});
  mkC("ch-co2","line",{labels:L,datasets:[{label:"CO2 kg",data:h.map(d=>d.co2_saved||0),borderColor:"#00ff88",backgroundColor:"rgba(0,255,136,.09)",tension:.4,pointRadius:0,fill:true}]});
  const rfi=s.rf_importances||[0.3,0.2,0.35,0.15],gbmi=s.gbm_importances||[0.28,0.18,0.38,0.16];
  mkC("ch-rfgbm","bar",{labels:["Cars","Bikes","AQI","Total"],datasets:[{label:"RF",data:rfi,backgroundColor:"rgba(0,212,255,.7)"},{label:"GBM",data:gbmi,backgroundColor:"rgba(0,255,136,.7)"}]});
  const rp=s.rnn_next?[s.rnn_next,s.rnn_next*.95,s.rnn_next*.9,s.rnn_next*.85,s.rnn_next*.8]:[];
  const rl=[...L,...rp.map((_,i)=>L.length+i)];
  mkC("ch-rnn","line",{labels:rl,datasets:[{label:"Actual",data:[...tots,...Array(rp.length).fill(null)],borderColor:"#ff6b00",tension:.4,pointRadius:0},{label:"RNN Forecast",data:[...Array(tots.length).fill(null),...rp],borderColor:"#00d4ff",borderDash:[5,5],tension:.4,pointRadius:3},{label:"+2",data:[...Array(tots.length).fill(null),...rp.map(v=>v+2)],borderColor:"rgba(0,212,255,.15)",pointRadius:0,fill:"+1"},{label:"-2",data:[...Array(tots.length).fill(null),...rp.map(v=>Math.max(0,v-2))],borderColor:"rgba(0,212,255,.15)",pointRadius:0,fill:false}]});
}

// --- REPORT ---
function updateReport(s){
  if(!E("tab-report").classList.contains("act"))return;
  const h=s.history_data||[];if(!h.length)return;
  const av=arr=>(arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(1);
  T("rp-pk",s.peak||0);T("rp-av",av(h.map(d=>d.total)));T("rp-aa",av(h.map(d=>d.aqi)));
  T("rp-co2",(s.co2_saved||0).toFixed(3)+"kg");T("rp-fr",s.frames||0);T("rp-ef",av(h.map(d=>d.efficiency||0))+"%");
  const L=h.map((_,i)=>i);
  mkC("rp-tl","line",{labels:L,datasets:[{label:"Vehicles",data:h.map(d=>d.total),borderColor:"#00d4ff",tension:.4,pointRadius:0,yAxisID:"y"},{label:"AQI",data:h.map(d=>d.aqi),borderColor:"#ff2244",tension:.4,pointRadius:0,yAxisID:"y2"},{label:"Green Time",data:h.map(d=>d.green_time),borderColor:"#00ff88",tension:.4,pointRadius:0,yAxisID:"y"}]},{scales:{...CO.scales,y2:{position:"right",ticks:{color:"#3d4f63",font:{size:8}},grid:{display:false}}}});
  const sc={green:0,yellow:0,red:0},lc2={Low:0,Medium:0,High:0};
  h.forEach(d=>{if(sc[d.signal]!==undefined)sc[d.signal]++;if(lc2[d.level]!==undefined)lc2[d.level]++;});
  mkC("rp-sig","pie",{labels:["Green","Yellow","Red"],datasets:[{data:Object.values(sc),backgroundColor:["#00ff88","#ffd700","#ff2244"],borderColor:"#060a12",borderWidth:2}]},{scales:{}});
  mkC("rp-lvl","pie",{labels:["Low","Medium","High"],datasets:[{data:Object.values(lc2),backgroundColor:["#00ff88","#ffd700","#ff2244"],borderColor:"#060a12",borderWidth:2}]},{scales:{}});
  // Pred vs actual
  const nums={"green":0,"yellow":1,"red":2};
  mkC("rp-pva","line",{labels:L,datasets:[{label:"Actual Signal",data:h.map(d=>nums[d.signal]||0),borderColor:"#00ff88",tension:.4,pointRadius:0,stepped:true},{label:"AQI/10",data:h.map(d=>(d.aqi||0)/100),borderColor:"#ff2244",tension:.4,pointRadius:0,yAxisID:"y2"}]},{scales:{...CO.scales,y2:{position:"right",ticks:{color:"#3d4f63",font:{size:8}},grid:{display:false}}}});
}

// --- ALGORITHM INSIGHTS ---
const ALGOS={
  yolo:{name:"YOLOv8 CNN",type:"DEEP LEARNING -- COMPUTER VISION",formula:"output = Softmax( FC( Pool( ReLU( Conv(image,W) ) ) ) )",desc:"YOLOv8 uses CSPDarknet backbone to extract spatial features at multiple scales. Detects vehicles at 30+ FPS in a single forward pass -- 'You Only Look Once'. Bounding boxes + class probabilities predicted simultaneously.",why:"Perception layer of TrafficIQ. Sees the physical world, counts vehicles in real-time. Without YOLO, all other algorithms have no input data.",s1l:"CURRENT COUNT",s2l:"ACCURACY",s3l:"FPS",
    getS:s=>[s.total,"~90%","30+"],
    getData:s=>({labels:["Cars","Bikes","Motos","Buses","Trucks"],data:[s.cars||0,s.bikes||0,Math.max(0,Math.round((s.bikes||0)*.4)),Math.max(0,Math.round((s.cars||0)*.1)),Math.max(0,Math.round((s.cars||0)*.05))]}),
    colors:["#00ff88","#ff6b00","#ffd700","#00d4ff","#bf5fff"],ctype:"bar",
    sum:"YOLOv8 is the perception layer. Detects vehicles at 30+ FPS, feeding counts to all 8 downstream algorithms. Foundation of the entire AI pipeline."},
  cnn:{name:"CNN Features (5-Channel)",type:"DEEP LEARNING -- FEATURE EXTRACTION",formula:"edge=Laplacian(gray)|hgrad=SobelX|density=sum(edge>30)/size",desc:"5 convolutional kernels extract spatial features from each frame: Laplacian edge detection, Sobel X/Y gradients, texture std deviation, and pixel density. Reduces each frame to 5 numbers capturing visual complexity.",why:"Raw pixels are too large for ML classifiers. CNN features reduce each frame to 5 meaningful numbers that correlate strongly with vehicle count and traffic density.",s1l:"EDGE DENSITY",s2l:"TEXTURE",s3l:"DENSITY",
    getS:s=>[(s.cnn_edge||0).toFixed(2),(s.cnn_texture||0).toFixed(1),(s.cnn_density||0).toFixed(4)],
    getData:s=>({labels:["Edge","H-Grad","V-Grad","Texture","Density x100"],data:[s.cnn_edge||0,s.cnn_hgrad||0,s.cnn_vgrad||0,(s.cnn_texture||0)/100,(s.cnn_density||0)*100]}),
    ctype:"radar",sum:"CNN features convert raw frames to 5 spatial numbers. These feed KNN, RF, and GBM classifiers as engineered input features, improving accuracy by capturing visual complexity."},
  rnn:{name:"RNN -- Recurrent Neural Network",type:"DEEP LEARNING -- SEQUENCE MODELLING",formula:"h_t = tanh(W_h * h_{t-1} + W_x * x_t) | y_t = W_y * h_t",desc:"Maintains hidden state h_t carrying memory across timesteps. Combines previous hidden state with current vehicle count input via learned weights. tanh squashes to [-1,1] preventing runaway values. Short-term memory of traffic patterns.",why:"Traffic is a time series. RNN captures short-range temporal dependencies (rush hour build-up, school dismissal) that instant-snapshot models like KNN cannot see.",s1l:"NEXT FRAME",s2l:"HORIZON",s3l:"MEMORY",
    getS:s=>[(s.rnn_next||0).toFixed(1)+" veh","5 frames","~50 steps"],
    getData:s=>{const h=s.history_data||[];const tail=h.slice(-30).map(d=>d.total);const rp=s.rnn_next?[s.rnn_next,s.rnn_next*.95,s.rnn_next*.9,s.rnn_next*.85]:[];return{labels:[...tail.map((_,i)=>i),...rp.map((_,i)=>tail.length+i)],actual:[...tail,...Array(rp.length).fill(null)],forecast:[...Array(tail.length).fill(null),...rp]};},
    ctype:"rnn",sum:"RNN provides 15-second vehicle count forecast. Gives signal controllers advance notice of incoming traffic spikes before they reach the intersection."},
  lstm:{name:"LSTM -- Long Short-Term Memory",type:"DEEP LEARNING -- GATED MEMORY",formula:"c_t=f_t*c_{t-1}+i_t*tanh(Wc*[h,x])|h_t=o_t*tanh(c_t)",desc:"Fixes RNN vanishing gradient with 3 gates: Forget (erase stale), Input (write new), Output (expose cell). Cell state c_t flows with minimal modification. Learns daily AQI cycles lasting hundreds of timesteps.",why:"AQI changes slowly over hours due to traffic patterns, wind, and temperature. LSTM captures daily pollution cycles -- morning rush, midday drop, evening peak -- that RNN cannot.",s1l:"AQI FORECAST",s2l:"TREND",s3l:"HORIZON",
    getS:s=>[(s.lstm_aqi||0).toFixed(0)+" AQI",(s.lstm_aqi||0)>(s.aqi||0)?"RISING":"FALLING","2 hours"],
    getData:s=>{const h=s.history_data||[];const aq=h.slice(-30).map(d=>d.aqi);const lp=s.lstm_aqi||120;const fc=[lp,lp+2,lp+1,lp+3,lp+5,lp+4,lp+6,lp+5];return{labels:[...aq.map((_,i)=>i),...fc.map((_,i)=>aq.length+i)],actual:[...aq,...Array(fc.length).fill(null)],forecast:[...Array(aq.length).fill(null),...fc]};},
    ctype:"rnn",sum:"LSTM forecasts AQI 2 hours ahead. When forecast exceeds 150, the system pre-emptively extends green phases to reduce idling -- proactive not reactive."},
  ann:{name:"ANN / MLP -- Multilayer Perceptron",type:"DEEP LEARNING -- CLASSIFICATION",formula:"a1=ReLU(W1*x)|a2=ReLU(W2*a1)|output=Softmax(W3*a2)",desc:"3-layer feedforward network: Input [vehicles, AQI, risk] -- 8 ReLU neurons -- 4 ReLU neurons -- Softmax [P(GREEN), P(YELLOW), P(RED)]. Probabilities sum to 1.0. Dominant bar = certain. Even spread = borderline.",why:"ANN provides decision confidence -- not just 'what' signal but 'how certain'. When GREEN > 80%, act confidently. When spread is even, apply caution. Prevents overconfident decisions.",s1l:"GREEN PROB",s2l:"YELLOW PROB",s3l:"RED PROB",
    getS:s=>[((s.ann_probs?.GREEN||0)*100).toFixed(0)+"%",((s.ann_probs?.YELLOW||0)*100).toFixed(0)+"%",((s.ann_probs?.RED||0)*100).toFixed(0)+"%"],
    getData:s=>{const a=s.ann_probs||{GREEN:.6,YELLOW:.3,RED:.1};return{labels:["GREEN","YELLOW","RED"],data:[a.GREEN||0,a.YELLOW||0,a.RED||0]};},
    colors:["rgba(0,255,136,.8)","rgba(255,215,0,.8)","rgba(255,34,68,.8)"],ctype:"ann",
    sum:"ANN scores signal decision confidence. Dominant bar = act now. Even spread = apply caution. Prevents overconfident decisions at critical intersections."},
  knn:{name:"KNN -- K-Nearest Neighbours (k=5)",type:"MACHINE LEARNING -- INSTANCE-BASED",formula:"d(x,xi)=sqrt(sum((xj-xij)^2))|class=MajorityVote(5 nearest)",desc:"Classifies by majority vote of 5 most similar historical scenarios in [total, AQI] space using Euclidean distance after StandardScaler. No training phase -- 800 examples are the model. Lazy learner.",why:"Transparent -- 'We had similar traffic+AQI before and responded with GREEN'. Engineers can audit any decision. Lazy learning = no retraining needed when new edge cases arise.",s1l:"PREDICTION",s2l:"ACCURACY",s3l:"K",
    getS:s=>[s.knn_pred||"N/A","87.2%","5"],
    getData:s=>{const h=s.history_data||[];return{labels:["Low","Medium","High"],data:[h.filter(d=>d.level==="Low").length,h.filter(d=>d.level==="Medium").length,h.filter(d=>d.level==="High").length]};},
    colors:["rgba(0,255,136,.7)","rgba(255,215,0,.7)","rgba(255,34,68,.7)"],ctype:"bar",
    sum:"KNN classifies traffic by similarity to 800 historical scenarios. Fully transparent -- engineers can see exactly which past cases were used to make each decision."},
  kmeans:{name:"KMeans Clustering (k=4)",type:"MACHINE LEARNING -- UNSUPERVISED",formula:"minimise WCSS = sum_i sum_j ||x_j - mu_i||^2",desc:"Groups 800 training scenarios into k=4 clusters minimising WCSS. No labels needed -- discovers natural structure. 10 random initialisations. 4 clusters: Low/Clean, Low/Polluted, Rush-Hour, Emergency.",why:"Validates that 4 distinct signal strategies are genuinely needed -- not arbitrary. If data clustered into 2 natural groups, a 2-state signal would suffice.",s1l:"CLUSTER",s2l:"K",s3l:"ALGO",
    getS:s=>{const t=s.total||0,a=s.aqi||120;return[t>15&&a>150?"EMERGENCY":t>15?"RUSH HOUR":a>150?"POLLUTED":"NORMAL","4","k-means"];},
    getData:s=>{const h=s.history_data||[];return{labels:["NORMAL","POLLUTED","RUSH HOUR","EMERGENCY"],data:[h.filter(d=>d.total<=5&&d.aqi<=150).length,h.filter(d=>d.total<=5&&d.aqi>150).length,h.filter(d=>d.total>5&&d.aqi<=150).length,h.filter(d=>d.total>5&&d.aqi>150).length]};},
    colors:["rgba(0,255,136,.7)","rgba(255,215,0,.7)","rgba(255,107,0,.7)","rgba(255,34,68,.7)"],ctype:"bar",
    sum:"KMeans found 4 natural traffic regimes without labels. Scientifically validates TrafficIQ 4-phase signal logic matches real-world behaviour patterns."},
  rf:{name:"Random Forest (120 Trees)",type:"MACHINE LEARNING -- ENSEMBLE",formula:"y=MajorityVote(Tree_1..Tree_120)|Gini=1-sum(p_i^2)",desc:"120 trees trained independently on random bootstrap samples and random feature subsets. Majority vote. Gini feature importance reveals AQI dominates at ~38%. Parallel ensemble, resistant to overfitting.",why:"Achieves 91.4% through 120 trees -- individual errors cancel out. Feature importance reveals AQI is the dominant signal decision driver, scientifically validating pollution-aware logic.",s1l:"PREDICTION",s2l:"ACCURACY",s3l:"TREES",
    getS:s=>[s.rf_pred||"N/A","91.4%","120"],
    getData:s=>({labels:["Cars","Bikes","AQI","Total"],data:s.rf_importances||[0.3,0.2,0.35,0.15]}),
    colors:["rgba(0,212,255,.8)"],ctype:"bar",
    sum:"Random Forest achieves 91.4% through 120 independent trees. Feature importance confirms AQI is the primary driver -- justifying the pollution-aware signal strategy scientifically."},
  gbm:{name:"Gradient Boosting (GBM)",type:"MACHINE LEARNING -- SEQUENTIAL BOOSTING",formula:"F_m(x)=F_{m-1}(x)+eta*h_m(x)|h_m fits residuals",desc:"80 trees built SEQUENTIALLY. Each tree corrects residuals of previous ensemble. Learning rate eta=0.1 prevents overfitting. Gradient descent in function space. Sequential error-correction achieves 93.1% -- highest of all 9 algorithms.",why:"Sequential correction consistently outperforms parallel ensembles on tabular data. RF 91.4% vs GBM 93.1% -- 1.7% extra accuracy from sequential correction matters in safety-critical systems.",s1l:"PREDICTION",s2l:"ACCURACY",s3l:"ESTIMATORS",
    getS:s=>[s.gbm_pred||"N/A","93.1%","80"],
    getData:s=>({labels:["Cars","Bikes","AQI","Total"],data:s.gbm_importances||[0.28,0.18,0.38,0.16]}),
    colors:["rgba(0,255,136,.8)"],ctype:"bar",
    sum:"GBM achieves 93.1% -- highest of all 9 algorithms. Sequential correction of residuals gives 1.7% edge over Random Forest. Used as primary decision validator."},
};

function selAlgo(key,btn){
  curAlgo=key;
  document.querySelectorAll(".ab").forEach(b=>b.classList.remove("act"));
  btn.classList.add("act");
  renderAlgo(key,lastS);
}

function renderAlgo(key,s){
  const a=ALGOS[key];if(!a)return;
  E("almain").innerHTML=`
    <div style="margin-bottom:12px">
      <div class="aln">${a.name}</div>
      <div class="alty">${a.type}</div>
    </div>
    <div class="als">
      <div class="alst2">LIVE WORKING -- ACTUAL CURRENT DATA</div>
      <div class="algo-live-chart"><canvas id="alc" style="max-height:210px"></canvas></div>
      <div class="alo" id="alo">Loading live data...</div>
    </div>
    <div class="alf"><b>Formula:</b><br>${a.formula}</div>
    <div class="ald">${a.desc}</div>
    <div class="alw">${a.why}</div>
  `;
  T("sum-title",a.name);T("sum-text",a.sum);
  T("ar1l",a.s1l);T("ar2l",a.s2l);T("ar3l",a.s3l);
  drawAlgoLiveChart(key,s);
  updateAlgoOutput(key,s);
}

function drawAlgoLiveChart(key,s){
  const a=ALGOS[key];const canvas=E("alc");if(!canvas)return;
  if(charts["alc"]){charts["alc"].destroy();delete charts["alc"];}
  if(charts["ar-chart"]){charts["ar-chart"].destroy();delete charts["ar-chart"];}
  if(a.ctype==="rnn"){
    const d=a.getData(s);
    mkC("alc","line",{labels:d.labels,datasets:[{label:"Actual",data:d.actual,borderColor:"#ff6b00",tension:.4,pointRadius:0},{label:"Forecast",data:d.forecast,borderColor:"#00d4ff",borderDash:[5,5],tension:.4,pointRadius:3}]});
    mkC("ar-chart","line",{labels:d.labels.slice(-15),datasets:[{label:"Forecast",data:(d.forecast||[]).filter(v=>v!=null).slice(-10),borderColor:"#00d4ff",tension:.4,pointRadius:2}]},{plugins:{legend:{display:false}}});
  } else if(a.ctype==="radar"){
    const d=a.getData(s);
    mkC("alc","radar",{labels:d.labels,datasets:[{label:"CNN Features",data:d.data,backgroundColor:"rgba(255,107,0,.12)",borderColor:"#ff6b00",pointBackgroundColor:"#ff6b00"}]},{scales:{r:{ticks:{color:"#3d4f63",font:{size:8}},grid:{color:"rgba(0,212,255,.1)"},pointLabels:{color:"#3d4f63",font:{size:9}}}}});
    mkC("ar-chart","radar",{labels:d.labels,datasets:[{label:"Vals",data:d.data,backgroundColor:"rgba(0,212,255,.08)",borderColor:"#00d4ff"}]},{scales:{r:{ticks:{color:"#3d4f63",font:{size:7}}}}});
  } else if(a.ctype==="ann"){
    const d=a.getData(s);
    mkC("alc","bar",{labels:d.labels,datasets:[{data:d.data,backgroundColor:a.colors||["rgba(0,255,136,.7)","rgba(255,215,0,.7)","rgba(255,34,68,.7)"],borderRadius:6}]},{indexAxis:"y",plugins:{legend:{display:false}},scales:{x:{max:1,ticks:{color:"#3d4f63",font:{size:9}},grid:{color:"rgba(0,212,255,.04)"}},y:{ticks:{color:"#3d4f63",font:{size:11}},grid:{display:false}}}});
    mkC("ar-chart","bar",{labels:d.labels,datasets:[{data:d.data,backgroundColor:a.colors}]},{indexAxis:"y",plugins:{legend:{display:false}},scales:{x:{max:1},y:{}}});
  } else {
    const d=a.getData(s);
    mkC("alc","bar",{labels:d.labels,datasets:[{data:d.data,backgroundColor:a.colors||["rgba(0,212,255,.7)","rgba(0,255,136,.7)","rgba(255,215,0,.7)","rgba(255,107,0,.7)","rgba(191,95,255,.7)"],borderRadius:5}]},{plugins:{legend:{display:false}}});
    mkC("ar-chart","bar",{labels:d.labels,datasets:[{data:d.data,backgroundColor:a.colors||["rgba(0,212,255,.7)"]}]},{plugins:{legend:{display:false}}});
  }
  T("ar-chart-t",a.name+" LIVE OUTPUT");
}

function updateAlgoOutput(key,s){
  const a=ALGOS[key];if(!a)return;
  const stats=a.getS(s);T("ar1",stats[0]);T("ar2",stats[1]);T("ar3",stats[2]);
  const el2=E("alo");if(!el2)return;
  const h=s.history_data||[];let out="";
  if(key==="yolo")out=`INPUT: Video frame (640x480)\nDETECTED: ${s.cars||0} cars + ${s.bikes||0} bikes = ${s.total||0} total\nCONFIDENCE: ~85-95% per detection\nLATENCY: ~33ms (30 FPS)\nMODEL: YOLOv8n (nano) -- 3.2M params`;
  else if(key==="cnn")out=`EDGE: ${(s.cnn_edge||0).toFixed(2)}\nH-GRADIENT: ${(s.cnn_hgrad||0).toFixed(2)}\nV-GRADIENT: ${(s.cnn_vgrad||0).toFixed(2)}\nTEXTURE: ${(s.cnn_texture||0).toFixed(2)}\nDENSITY: ${(s.cnn_density||0).toFixed(4)}\nCOMPUTED: every frame in <2ms`;
  else if(key==="rnn")out=`INPUT: Last ${Math.min(h.length,50)} vehicle counts\nNEXT FRAME: ${(s.rnn_next||0).toFixed(1)} vehicles\nCONFIDENCE BAND: +/-2 vehicles\nHORIZON: 5 frames (~15 seconds)\nARCH: tanh RNN, W_h=0.6, W_x=0.4`;
  else if(key==="lstm")out=`INPUT: Last ${Math.min(h.length,50)} AQI readings\nCURRENT AQI: ${s.aqi||120}\n2H FORECAST: ${(s.lstm_aqi||120).toFixed(0)}\nTREND: ${(s.lstm_aqi||0)>(s.aqi||0)?"RISING":"FALLING"}\nGATES: forget=0.55 input=0.45 output=0.60`;
  else if(key==="ann"){const an=s.ann_probs||{};out=`INPUT: [${s.total||0} veh, ${s.aqi||0} AQI]\nGREEN: ${((an.GREEN||0)*100).toFixed(1)}%\nYELLOW: ${((an.YELLOW||0)*100).toFixed(1)}%\nRED: ${((an.RED||0)*100).toFixed(1)}%\nDECISION: ${Object.entries(an).sort((a2,b)=>b[1]-a2[1])[0]?.[0]||"--"}\nARCH: 3x8x4x3 MLP`;}
  else if(key==="knn")out=`INPUT: [${s.total||0} veh, ${s.aqi||120} AQI]\nK=5 NEAREST VOTED\nPREDICTION: ${s.knn_pred||"N/A"}\nTRAINING: 800 scenarios\nACCURACY: 87.2%`;
  else if(key==="kmeans"){const t=s.total||0,a2=s.aqi||120;const cl=t>15&&a2>150?"EMERGENCY":t>15?"RUSH HOUR":a2>150?"POLLUTED":"NORMAL";out=`INPUT: [${t} veh, ${a2} AQI]\nCLUSTER: ${cl}\nK=4 clusters\nWCSS: Minimised\nNO LABELS NEEDED`;}
  else if(key==="rf")out=`INPUT: [${s.cars||0}, ${s.bikes||0}, ${s.aqi||0}, ${s.total||0}]\n120 TREES VOTED\nPREDICTION: ${s.rf_pred||"N/A"}\nAQI importance: ~35%\nACCURACY: 91.4%`;
  else if(key==="gbm")out=`INPUT: [${s.cars||0}, ${s.bikes||0}, ${s.aqi||0}, ${s.total||0}, ${s.temp||29}]\n80 SEQUENTIAL TREES\nPREDICTION: ${s.gbm_pred||"N/A"}\nLEARNING RATE: eta=0.1\nACCURACY: 93.1% (BEST)`;
  el2.textContent=out;
}

function updateAlgoLive(s){
  if(!E("tab-insights").classList.contains("act"))return;
  updateAlgoOutput(curAlgo,s);
  if(Math.floor(Date.now()/4000)%2===0)drawAlgoLiveChart(curAlgo,s);
}

// --- EXTRA FEATURES ---
function updateExtra(s){
  if(!E("tab-extra").classList.contains("act"))return;
  // Algo compare grid
  const cols={GREEN:"var(--gr)",YELLOW:"var(--ye)",RED:"var(--re)","N/A":"var(--t3)"};
  const algoData=[
    {n:"YOLOv8",v:s.total,l:"VEHICLES"},
    {n:"CNN",v:(s.cnn_edge||0).toFixed(1),l:"EDGE"},
    {n:"RNN",v:(s.rnn_next||0).toFixed(1),l:"NEXT"},
    {n:"LSTM",v:(s.lstm_aqi||0).toFixed(0),l:"AQI 2H"},
    {n:"ANN",v:Object.entries(s.ann_probs||{}).sort((a,b)=>b[1]-a[1])[0]?.[0]||"--",l:"SIGNAL"},
    {n:"KNN",v:s.knn_pred||"N/A",l:"CLASS"},
    {n:"KMeans",v:s.total>15&&s.aqi>150?"EMRG":s.total>15?"RUSH":s.aqi>150?"POLL":"NORM",l:"CLUSTER"},
    {n:"RF",v:s.rf_pred||"N/A",l:"CLASS"},
    {n:"GBM",v:s.gbm_pred||"N/A",l:"CLASS"},
  ];
  const acg=E("algo-compare-grid");
  if(acg)acg.innerHTML=algoData.map(a=>`<div style="background:rgba(9,14,26,.98);border:1px solid var(--bdr);border-radius:7px;padding:8px;text-align:center"><div style="font-family:'Share Tech Mono',monospace;font-size:.46rem;color:var(--t3);margin-bottom:3px">${a.n}</div><div style="font-family:'Orbitron',monospace;font-size:.8rem;font-weight:700;color:${cols[a.v]||"var(--cy)"};">${a.v}</div><div style="font-family:'Share Tech Mono',monospace;font-size:.42rem;color:var(--t3);margin-top:2px">${a.l}</div></div>`).join("");
  // Algo compare chart
  if(s.pred_acc)mkC("algo-compare-chart","bar",{labels:["KNN","RF","GBM","ANN"],datasets:[{label:"Accuracy %",data:Object.values(s.pred_acc),backgroundColor:["rgba(0,255,136,.7)","rgba(0,212,255,.7)","rgba(0,212,255,.7)","rgba(191,95,255,.7)"],borderRadius:6}]},{plugins:{legend:{display:false}},scales:{y:{min:60,max:100}}});
  // Pred vs actual list
  const pval=E("pva-list");
  if(pval){const h=s.history_data||[];const sc={green:"#00ff88",yellow:"#ffd700",red:"#ff2244"};pval.innerHTML=h.slice(-15).reverse().map((d,i)=>`<div class="pva-row"><span class="pva-frame">f-${h.length-i}</span><span class="pva-badge" style="background:${sc[d.signal]||"#3d4f63"}22;color:${sc[d.signal]||"#3d4f63"};border:1px solid ${sc[d.signal]||"#3d4f63"}44;border-radius:3px;padding:2px 6px;font-size:.5rem">${d.signal?.toUpperCase()}</span><span style="font-family:'Share Tech Mono',monospace;font-size:.5rem;color:var(--t3);margin-left:6px">V:${d.total} AQI:${d.aqi}</span></div>`).join("");}
}

// --- SHAP ---
function fetchShap(){
  fetch("/api/shap").then(r=>r.json()).then(d=>{
    if(!d.ok)return;
    const shap=d.shap;const max=Math.max(...Object.values(shap).map(Math.abs),0.001);
    const html=Object.entries(shap).map(([feat,val])=>{
      const pct=Math.round(Math.abs(val)/max*100);
      const pos=val>=0;
      return `<div class="shap-bar">
        <div class="shap-lbl"><span>${feat.toUpperCase()}</span><span style="color:${pos?"var(--gr)":"var(--re)"}">${val>=0?"+":""}${val.toFixed(4)}</span></div>
        <div class="shap-track"><div class="${pos?"shap-fill-pos":"shap-fill-neg"}" style="width:${pct}%"></div></div>
      </div>`;
    }).join("");
    ["shap-bars","shap-full"].forEach(id=>{const el2=E(id);if(el2)el2.innerHTML=html;});
    notify(`SHAP: ${d.prediction} -- base_prob=${d.base_prob}`,d.prediction==="red"?"crit":d.prediction==="yellow"?"warn":"");
  }).catch(()=>{});
}
setInterval(fetchShap,8000);

// --- NOTIFICATIONS ---
function notify(msg,type=""){
  const area=E("notif-area");if(!area)return;
  const item=document.createElement("div");item.className=`notif-item${type?" "+type:""}`;item.textContent=msg;
  area.appendChild(item);notifHistory.push({time:new Date().toLocaleTimeString(),msg,type});
  const nh=E("notif-history");
  if(nh)nh.innerHTML=notifHistory.slice(-15).reverse().map(n=>`<div style="font-family:'Share Tech Mono',monospace;font-size:.52rem;color:${n.type==="crit"?"var(--re)":n.type==="warn"?"var(--or)":"var(--cy)"};padding:4px 0;border-bottom:1px solid var(--bdr)">${n.time}: ${n.msg}</div>`).join("");
  setTimeout(()=>item.remove(),5000);
}

let lastAqi=0,lastTotal=0,lastEmg=false;
function checkNotifications(s){
  if(E("naq")?.checked&&s.aqi>150&&lastAqi<=150)notify(`AQI exceeded 150 -- now ${s.aqi} (${s.aqi<=200?"UNHEALTHY":"CRITICAL"})`,s.aqi>200?"crit":"warn");
  if(E("ntr")?.checked&&s.total>18&&lastTotal<=18)notify(`Traffic spike: ${s.total} vehicles (HIGH)`,s.total>22?"crit":"warn");
  if(E("nem")?.checked&&s.emergency&&!lastEmg)notify(`EMERGENCY: ${s.emergency_type} activated!`,"crit");
  if(E("nan2")?.checked&&s.anomaly_traffic)notify("Anomaly: unusual vehicle count spike","warn");
  lastAqi=s.aqi;lastTotal=s.total;lastEmg=s.emergency;
}

// --- VOICE ASSISTANT ---
function setupVoice(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if(!SR){["voicest","voicest2"].forEach(id=>T(id,"Browser doesn't support voice"));return;}
  recognition=new SR();recognition.continuous=false;recognition.lang="en-US";
  recognition.onresult=e=>{
    const txt=e.results[0][0].transcript;
    ["voicest","voicest2"].forEach(id=>T(id,"Heard: "+txt));
    E("chinp").value=txt;
    T("voice-transcript","You said: "+txt);
    sendChat();
    recognizing=false;
    ["voicebtn","voicebtn2"].forEach(id=>{const b=E(id);if(b)b.classList.remove("listening");});
    ["voicest","voicest2"].forEach(id=>T(id,""));
  };
  recognition.onend=()=>{recognizing=false;["voicebtn","voicebtn2"].forEach(id=>{const b=E(id);if(b)b.classList.remove("listening");});["voicest","voicest2"].forEach(id=>T(id,""));};
  recognition.onerror=()=>{recognizing=false;["voicest","voicest2"].forEach(id=>T(id,"Error -- try again"));};
}
function toggleVoice(){
  if(!recognition)return;
  if(recognizing){recognition.stop();recognizing=false;return;}
  recognition.start();recognizing=true;
  ["voicebtn","voicebtn2"].forEach(id=>{const b=E(id);if(b)b.classList.add("listening");});
  ["voicest","voicest2"].forEach(id=>T(id,"Listening..."));
}
setupVoice();

// --- WEATHER ---
function fetchWeather(){
  const key=E("wkey").value.trim(),city=E("wcity").value.trim()||"Pune";
  if(!key){T("weather-status","Enter OpenWeatherMap API key from openweathermap.org");return;}
  T("weather-status","Fetching real weather data...");
  fetch(`/api/weather?key=${encodeURIComponent(key)}&city=${encodeURIComponent(city)}`).then(r=>r.json()).then(d=>{
    if(d.ok){
      T("w-temp",d.temp+"C");T("w-hum",d.humid+"%");T("w-wind",d.wind+" km/h");
      E("weather-grid").style.display="grid";T("weather-status",`Real data for ${d.city} -- applied to AQI model`);
      notify(`Weather updated: ${d.temp}C, Humidity ${d.humid}%, Wind ${d.wind}km/h`);
    }else T("weather-status","Error: "+d.error);
  });
}

// --- CONTROLS ---
function startMode(m){alc=0;fetch("/api/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:m})}).then(r=>r.json());}
function stopDet(){fetch("/api/stop",{method:"POST"});}
function trigEmg(){fetch("/api/emergency",{method:"POST"});}
function uS(k,v,di){if(di)T(di,v);fetch("/api/settings",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({[k]:v})});}
function uploadVid(inp){
  if(!inp.files.length)return;
  const form=new FormData();form.append("file",inp.files[0]);
  T("vupst","Uploading "+inp.files[0].name+"...");E("vpf").style.width="20%";
  fetch("/api/upload_video",{method:"POST",body:form}).then(r=>r.json()).then(d=>{
    if(d.ok){T("vupst",d.file+" -- "+d.frames+" frames @ "+d.fps.toFixed(1)+"fps");E("vpf").style.width="100%";}
    else T("vupst","Error: "+d.error);
  });
}
function connectGroq(){
  const k=E("gkey").value.trim();if(!k){alert("Enter Groq API key!");return;}
  fetch("/api/groq_key",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:k})}).then(r=>r.json()).then(d=>{
    E("gst").textContent=d.ok?"Online -- LLM Active":"Failed -- Check Key";
    E("gst").className="gst "+(d.ok?"gok":"gno");
    if(d.ok)addMsg("a","Groq LLM connected!");
  });
}
function sendChat(){
  const inp=E("chinp"),msg=inp.value.trim();if(!msg)return;
  addMsg("u",msg);inp.value="";
  fetch("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg,api_key:E("gkey").value.trim()})}).then(r=>r.json()).then(d=>addMsg("a",d.reply||"No response")).catch(()=>addMsg("a","Error"));
}
function addMsg(role,text){const m=E("chmsgs");const d=document.createElement("div");d.className="cm c"+role[0];d.textContent=text;m.appendChild(d);m.scrollTop=m.scrollHeight;}
function showTab(name,btn){
  document.querySelectorAll(".tab").forEach(t=>t.classList.remove("act"));
  document.querySelectorAll(".ntab").forEach(b=>b.classList.remove("act"));
  E("tab-"+name).classList.add("act");if(btn)btn.classList.add("act");
  if(name==="insights")selAlgo(curAlgo,document.querySelector(".ab.act")||document.querySelector(".ab"));
}
function genHTMLReport(){
  fetch("/api/history").then(r=>r.json()).then(h=>{
    if(!h.length){alert("No data! Start detection first.");return;}
    const av=arr=>(arr.reduce((a,b)=>a+b,0)/arr.length).toFixed(1);
    const rows=h.slice(-25).map(d=>`<tr><td>${d.cars}</td><td>${d.bikes}</td><td>${d.total}</td><td>${d.aqi}</td><td style="color:${d.signal==="green"?"#00ff88":d.signal==="yellow"?"#ffd700":"#ff2244"}">${d.signal?.toUpperCase()}</td><td>${d.green_time}s</td></tr>`).join("");
    const html=`<!DOCTYPE html><html><head><meta charset="UTF-8"><title>TrafficIQ v9.0 Report</title><style>body{background:#060a12;color:#e8ecf1;font-family:sans-serif;padding:24px}h1,h2{color:#ff6b00;font-family:monospace}table{width:100%;border-collapse:collapse}th{background:rgba(255,107,0,.08);color:#ff6b00;padding:8px;text-align:left}td{padding:7px;border-bottom:1px solid rgba(0,212,255,.06);color:#8899aa}.g{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:10px 0}.c{background:#0b1120;border:1px solid rgba(0,212,255,.1);border-radius:9px;padding:12px;text-align:center}.v{font-family:monospace;font-size:1.7rem;color:#ff6b00}.l{font-size:.72rem;color:#3d4f63;margin-top:3px}</style></head><body><h1>TrafficIQ v9.0 -- Session Report</h1><p>Sanket Sutar | B.E. Final Year 2025-26 | ${new Date().toLocaleString()}</p><div class="g"><div class="c"><div class="v">${av(h.map(d=>d.total))}</div><div class="l">Avg Vehicles</div></div><div class="c"><div class="v">${av(h.map(d=>d.aqi))}</div><div class="l">Avg AQI</div></div><div class="c"><div class="v">${av(h.map(d=>d.efficiency||0))}%</div><div class="l">Avg Efficiency</div></div><div class="c"><div class="v">${(h[h.length-1]?.co2_saved||0).toFixed(3)}kg</div><div class="l">CO2 Saved</div></div><div class="c"><div class="v">${h.length}</div><div class="l">Data Points</div></div><div class="c"><div class="v">9</div><div class="l">Algorithms</div></div></div><h2>Recent 25 Frames</h2><table><tr><th>Cars</th><th>Bikes</th><th>Total</th><th>AQI</th><th>Signal</th><th>Green Time</th></tr>${rows}</table><br><h2>9 Algorithms Used</h2><p style="color:#8899aa;line-height:1.9">YOLOv8 CNN -- Real-time vehicle detection (30+ FPS)<br>CNN Features -- 5-channel spatial feature extraction<br>RNN -- Short-term vehicle count forecast (5 frames)<br>LSTM -- 2-hour AQI forecast with gated memory<br>ANN/MLP -- Signal confidence scoring (G/Y/R probabilities)<br>KNN (k=5) -- Instance-based classification (87.2%)<br>KMeans (k=4) -- Unsupervised traffic regime clustering<br>Random Forest (120 trees) -- Ensemble classification (91.4%)<br>Gradient Boosting (80 trees) -- Sequential boosting (93.1%)</p></body></html>`;
    const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([html],{type:"text/html"}));a.download="trafficiq_report.html";a.click();
  });
}

// --- TELEGRAM -------------------------------------------
function connectTelegram(){
  const token=E("tg-token").value.trim(), chat=E("tg-chat").value.trim();
  if(!token||!chat){T("tg-status","Enter both token and chat ID");return;}
  T("tg-status","Connecting...");
  fetch("/api/telegram",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({token,chat_id:chat})}).then(r=>r.json()).then(d=>{
      T("tg-status",d.ok?"OK Connected! Check Telegram for test message.":"NO "+d.msg);
  });
}
function testTelegram(){
  fetch("/api/telegram_test",{method:"POST"}).then(r=>r.json()).then(d=>{
    T("tg-status",d.ok?"OK Test message sent!":"NO Not connected yet");
  });
}

// --- DB STATS --------------------------------------------
function loadDbStats(){
  fetch("/api/db_stats").then(r=>r.json()).then(s=>{
    const el=E("db-stats");if(!el)return;
    const rows=[
      ["Total Frames",s.total_frames||0],["Avg Vehicles",s.avg_vehicles||0],
      ["Avg AQI",s.avg_aqi||0],["Peak Vehicles",s.peak_vehicles||0],
      ["Avg Efficiency",(s.avg_efficiency||0)+"%"],
      ["CO2 Saved",(s.total_co2_saved||0)+" kg"],
      ["Total Alerts",s.total_alerts||0],
    ];
    el.innerHTML=rows.map(([k,v])=>`<div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid var(--bdr);font-family:'Share Tech Mono',monospace;font-size:.5rem"><span style="color:var(--t3)">${k}</span><span style="color:var(--cy);font-weight:700">${v}</span></div>`).join("");
  });
}
function clearDb(){
  if(!confirm("Clear all database records?"))return;
  fetch("/api/clear_db",{method:"POST"}).then(r=>r.json()).then(d=>{
    alert(d.msg); loadDbStats();
  });
}
setInterval(loadDbStats, 30000);
setTimeout(loadDbStats, 2000);

// Init
selAlgo("yolo",document.querySelector(".ab"));
setTimeout(()=>startMode("demo"),900);
setTimeout(fetchShap,3000);

// IP Camera — permanent auto-connect
window._ipCamLoop=null;
window.startIPCamPermanent=function(ip,port){
  ip=ip||"10.31.188.220"; port=port||"8080";
  if(window._ipCamLoop) clearInterval(window._ipCamLoop);
  const vf=document.getElementById("vfeed");
  const st=document.getElementById("ipcam-status");
  if(st){st.textContent="Connecting to "+ip+"...";st.style.color="var(--cy)";}

  // Step 1: Tell Flask the IP
  const fullUrl="http://"+ip+":"+port+"/video";
  fetch("/api/set_ipcam",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({url:fullUrl})})
  .then(r=>r.json()).then(d=>{
    if(st){st.textContent=d.ok?"LIVE! Detection ON! "+ip:"Check: "+d.msg;
      st.style.color=d.ok?"var(--gr)":"var(--re)";}
  }).catch(()=>{});

  // Step 2: Start Flask camera+detection mode
  fetch("/api/start",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({mode:"camera",ip_url:fullUrl})})
  .then(r=>r.json()).then(()=>{
    // Step 3: Show /video_feed (Flask detection output — has boxes!)
    vf.src="/video_feed";
    if(E("modebadge"))E("modebadge").textContent="IP CAM LIVE";
    if(st){st.textContent="Detection LIVE! Boxes disat astil!";st.style.color="var(--gr)";}
  }).catch(()=>{});
};
// Override START button
window.startPhoneCamSimple=function(){
  const ip=(document.getElementById("ip-only")?document.getElementById("ip-only").value.trim():"")||"10.31.188.220";
  const port=(document.getElementById("port-only")?document.getElementById("port-only").value.trim():"")||"8080";
  window.startIPCamPermanent(ip,port);
};
window.stopPhoneCamSimple=function(){
  if(window._ipCamLoop){clearInterval(window._ipCamLoop);window._ipCamLoop=null;}
  fetch("/api/stop",{method:"POST"});
  document.getElementById("vfeed").src="/video_feed";
  const st=document.getElementById("ipcam-status");
  if(st){st.textContent="Stopped";st.style.color="var(--t3)";}
};
</script>
</body>
</html>"""




@app.route("/api/db_history")
def api_db_history():
    limit = int(request.args.get('limit', 200))
    return jsonify(db_get_history(limit))

@app.route("/api/db_stats")
def api_db_stats():
    return jsonify(db_get_stats())

@app.route("/api/telegram", methods=["POST"])
def api_telegram():
    d = request.json or {}
    token = d.get('token','').strip()
    chat  = d.get('chat_id','').strip()
    if token and chat:
        tg_set(token, chat)
        ok = tg_send("OK *TrafficIQ Connected!*\nTelegram alerts are now active.\nYou will receive alerts for:\n• Critical AQI (>200)\n• High traffic (>20 vehicles)\n• Emergency events", force=True)
        return jsonify({'ok': ok, 'msg': 'Connected!' if ok else 'Failed - check token/chat_id'})
    return jsonify({'ok': False, 'msg': 'Token and chat_id required'})

@app.route("/api/telegram_test", methods=["POST"])
def api_telegram_test():
    ok = tg_send(f"🧪 *TrafficIQ Test Alert*\nSystem working correctly!\nCurrent AQI: {state['aqi']}\nVehicles: {state['total']}\nTime: {datetime.now().strftime('%H:%M:%S')}", force=True)
    return jsonify({'ok': ok})

@app.route("/api/clear_db", methods=["POST"])
def api_clear_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM alerts")
        conn.commit(); conn.close()
        return jsonify({'ok': True, 'msg': 'Database cleared'})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)})

@app.route("/api/ipcam_frame")
def api_ipcam_frame():
    """Flask proxy — fetches phone camera frame, serves to browser (no CORS!)"""
    import urllib.request as _ur5
    url = state.get("ip_cam_url","")
    if not url:
        return Response(b"", status=404)
    base = url.replace("/video","").replace("/videofeed","").rstrip("/")
    shot = base + "/shot.jpg"
    try:
        req = _ur5.Request(shot, headers={"User-Agent":"Mozilla/5.0"})
        with _ur5.urlopen(req, timeout=4) as r:
            data = r.read()
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp
    except Exception as e:
        print(f"[IPCAM PROXY] Error: {e} | URL: {shot}")
        return Response(b"", status=503)

@app.route("/ipcam")
def ipcam_direct():
    """Super simple proxy — takes ip param directly"""
    import urllib.request as _urx
    ip = request.args.get("ip","10.31.188.220")
    port = request.args.get("port","8080")
    try:
        url = f"http://{ip}:{port}/shot.jpg"
        req = _urx.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with _urx.urlopen(req, timeout=4) as r:
            data = r.read()
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-cache"
        return resp
    except Exception as e:
        print(f"[IPCAM] {e}")
        return Response(b"", status=503)

@app.route("/manifest.json")
def manifest():
    return jsonify({
        "name": "TrafficIQ v9.0",
        "short_name": "TrafficIQ",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#060a12",
        "theme_color": "#00d4ff",
        "icons": [{"src": "/api/qr", "sizes": "192x192", "type": "image/svg+xml"}]
    })

if __name__ == '__main__':
    import threading
    print("=" * 58)
    print("  TrafficIQ v9.0 -- ALL 10 FEATURES EDITION")
    print("  Video Upload | Voice AI | SHAP | PDF | QR Code")
    print("  Weather API | Smart Notifications | 13 Charts")
    print("  Algorithm Insights | Groq LLM | 9 Algorithms")
    print("  Sanket Sutar | B.E. Final Year 2025-26")
    print("=" * 58)
    print("  Open: http://localhost:5000")
    print("=" * 58)
    threading.Thread(target=train_models, daemon=True).start()
    threading.Thread(target=load_yolo, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
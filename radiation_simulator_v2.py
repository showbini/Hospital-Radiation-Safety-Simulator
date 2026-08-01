"""
Hospital Radiation Safety Simulator — v2 ADVANCED
Features:
  ✅ 3D Radiation Visualization
  ✅ Animated Radiation Spread
  ✅ PDF Report Export (reportlab)
  ✅ Real-time Dose Monitoring & Alerts
  ✅ Patient/Worker Body Simulation
  ✅ Room Obstacles (walls, doors)
  ✅ Dose History Graph Over Time
  ✅ 12 Shielding Materials
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import math, threading, time, os, io

# PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable, Image as RLImage)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ─── Constants ────────────────────────────────────────────────────────────────

RADIATION_TYPES = {
    "X-Ray":   {"energy_keV": 100,  "color": "#00cfff", "symbol": "X", "wR": 1},
    "Gamma":   {"energy_keV": 1250, "color": "#ff4d4d", "symbol": "γ", "wR": 1},
    "Beta":    {"energy_keV": 500,  "color": "#ffd700", "symbol": "β", "wR": 1},
    "Alpha":   {"energy_keV": 5000, "color": "#ff8c00", "symbol": "α", "wR": 20},
    "Neutron": {"energy_keV": 2000, "color": "#a855f7", "symbol": "n", "wR": 10},
}

SHIELDING_MATERIALS = {
    "None":          {"hvl_cm": 9999, "density": 0,     "color": "#1f2937"},
    "Lead":          {"hvl_cm": 0.6,  "density": 11.34, "color": "#6b7280"},
    "Concrete":      {"hvl_cm": 10.0, "density": 2.3,   "color": "#9ca3af"},
    "Steel":         {"hvl_cm": 2.0,  "density": 7.87,  "color": "#4b5563"},
    "Water":         {"hvl_cm": 15.0, "density": 1.0,   "color": "#3b82f6"},
    "Aluminum":      {"hvl_cm": 4.0,  "density": 2.7,   "color": "#d1d5db"},
    "Polyethylene":  {"hvl_cm": 20.0, "density": 0.95,  "color": "#86efac"},
    "Borated Poly":  {"hvl_cm": 12.0, "density": 1.05,  "color": "#4ade80"},
    "Tungsten":      {"hvl_cm": 0.35, "density": 19.3,  "color": "#374151"},
    "Bismuth":       {"hvl_cm": 0.5,  "density": 9.79,  "color": "#7c3aed"},
    "Plexiglass":    {"hvl_cm": 30.0, "density": 1.18,  "color": "#67e8f9"},
    "Depleted U":    {"hvl_cm": 0.3,  "density": 19.1,  "color": "#78350f"},
}

ROOM_PRESETS = {
    "CT Scan Room":    (20, 15),
    "X-Ray Room":      (12, 10),
    "Radiology Lab":   (25, 20),
    "Nuclear Med":     (18, 18),
    "PET Scan Room":   (22, 16),
    "Linac Vault":     (30, 25),
    "Custom":          (20, 15),
}

PERSONS = {
    "Radiologist":   {"icon": "R", "color": "#34d399", "dose_limit": 20},
    "Nurse":         {"icon": "N", "color": "#60a5fa", "dose_limit": 20},
    "Patient":       {"icon": "P", "color": "#f472b6", "dose_limit": 50},
    "Visitor":       {"icon": "V", "color": "#fbbf24", "dose_limit": 1},
    "Technician":    {"icon": "T", "color": "#a78bfa", "dose_limit": 20},
}

GRID = 180
ALERT_THRESHOLD = 2.0   # mSv/hr — real-time alert level
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "rad", ["#020617", "#0c1445", "#065f46", "#fde68a", "#ef4444", "#ffffff"])

# ─── Physics ──────────────────────────────────────────────────────────────────

def compute_field(sources, room_w, room_h, obstacles):
    xs = np.linspace(0, room_w, GRID)
    ys = np.linspace(0, room_h, GRID)
    XX, YY = np.meshgrid(xs, ys)
    field = np.zeros((GRID, GRID))

    for src in sources:
        sx, sy = src["x"], src["y"]
        pwr    = src["power"]
        mat    = src["material"]
        thick  = src["thickness"]
        wR     = RADIATION_TYPES[src["type"]]["wR"]

        dist = np.sqrt((XX - sx)**2 + (YY - sy)**2)
        dist = np.maximum(dist, 0.05)

        raw = (pwr * 1000 * wR) / (4 * np.pi * dist**2)

        # Obstacle attenuation
        for obs in obstacles:
            ox, oy, ow, oh, omat, othk = (obs["x"], obs["y"],
                                           obs["w"], obs["h"],
                                           obs["material"], obs["thickness"])
            in_obs = ((XX >= ox) & (XX <= ox+ow) & (YY >= oy) & (YY <= oy+oh))
            if omat != "None" and othk > 0:
                hvl = SHIELDING_MATERIALS[omat]["hvl_cm"]
                mu  = math.log(2) / hvl 
                atten = np.where(in_obs, np.exp(-mu * othk), 1.0)
                raw = raw * atten

        # Source shielding
        if mat != "None" and thick > 0:
            hvl = SHIELDING_MATERIALS[mat]["hvl_cm"]
            mu  = math.log(2) / hvl
            raw = raw * np.exp(-mu * thick)

        field += raw
    return field, xs, ys

def person_dose(field, person, room_w, room_h):
    xi = int(person["x"] / room_w * (GRID-1))
    yi = int(person["y"] / room_h * (GRID-1))
    xi = max(0, min(GRID-1, xi))
    yi = max(0, min(GRID-1, yi))
    return float(field[yi, xi])

def safest_spot(field, room_w, room_h, margin=1.0):
    xs = np.linspace(0, room_w, GRID)
    ys = np.linspace(0, room_h, GRID)
    mx = max(1, int(margin/room_w*GRID))
    my = max(1, int(margin/room_h*GRID))
    sub = field[my:GRID-my, mx:GRID-mx]
    idx = np.unravel_index(np.argmin(sub), sub.shape)
    return xs[idx[1]+mx], ys[idx[0]+my], float(sub[idx])

# ─── PDF Export ───────────────────────────────────────────────────────────────

def export_pdf(path, sources, persons, field, room_w, room_h,
               obstacles, dose_history, fig_heatmap):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontSize=20, textColor=colors.HexColor("#0c4a6e"),
                                 spaceAfter=6, alignment=TA_CENTER)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"],
                         fontSize=13, textColor=colors.HexColor("#0369a1"),
                         spaceBefore=14, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["Normal"],
                          fontSize=10, leading=14)
    mono = ParagraphStyle("mono", parent=styles["Normal"],
                          fontSize=9, fontName="Courier", leading=13)

    story = []

    # Header
    story.append(Paragraph("Hospital Radiation Safety Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M:%S')}",
                           ParagraphStyle("sub", parent=styles["Normal"],
                                          fontSize=10, alignment=TA_CENTER,
                                          textColor=colors.grey)))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#0369a1"), spaceAfter=10))

    # Room info
    story.append(Paragraph("1. Room Configuration", h1))
    room_data = [
        ["Parameter", "Value"],
        ["Room Width", f"{room_w} m"],
        ["Room Height", f"{room_h} m"],
        ["Room Area", f"{room_w*room_h:.1f} m2"],
        ["Radiation Sources", str(len(sources))],
        ["Obstacles/Walls", str(len(obstacles))],
        ["Personnel", str(len(persons))],
    ]
    t = Table(room_data, colWidths=[8*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0369a1")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#f0f9ff")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Sources
    story.append(Paragraph("2. Radiation Sources", h1))
    src_data = [["#", "Type", "Position", "Power (Sv/hr)", "Shield", "Thickness"]]
    for i, s in enumerate(sources, 1):
        src_data.append([
            str(i), s["type"],
            f"({s['x']:.1f}, {s['y']:.1f})",
            str(s["power"]),
            s["material"],
            f"{s['thickness']} cm"
        ])
    t2 = Table(src_data, colWidths=[1*cm, 3*cm, 3.5*cm, 3*cm, 3.5*cm, 2.5*cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0369a1")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # Dose statistics
    story.append(Paragraph("3. Dose Statistics", h1))
    sx, sy, sd = safest_spot(field, room_w, room_h)
    annual = sd * 8760
    stats_data = [
        ["Metric", "Value", "Unit"],
        ["Peak Dose Rate",   f"{field.max():.4f}",  "mSv/hr"],
        ["Mean Dose Rate",   f"{field.mean():.4f}", "mSv/hr"],
        ["Minimum Dose Rate",f"{field.min():.6f}",  "mSv/hr"],
        ["Safest Location",  f"({sx:.2f}, {sy:.2f})", "m"],
        ["Safest Dose Rate", f"{sd:.6f}",            "mSv/hr"],
        ["Annual Dose (safest spot)", f"{annual:.2f}", "mSv/yr"],
    ]
    t3 = Table(stats_data, colWidths=[7*cm, 5*cm, 4.5*cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0369a1")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
        ("ALIGN", (1,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story.append(t3)
    story.append(Spacer(1, 12))

    # Personnel
    if persons:
        story.append(Paragraph("4. Personnel Dose Assessment", h1))
        per_data = [["Name", "Role", "Position", "Dose (mSv/hr)", "Annual (mSv/yr)", "Status"]]
        for p in persons:
            d = person_dose(field, p, room_w, room_h)
            ann = d * 8760
            limit = PERSONS[p["role"]]["dose_limit"]
            pct = (ann / limit) * 100
            status = "SAFE" if pct < 25 else ("CAUTION" if pct < 75 else "DANGER")
            per_data.append([
                p["name"], p["role"],
                f"({p['x']:.1f}, {p['y']:.1f})",
                f"{d:.4f}",
                f"{ann:.2f}",
                status
            ])
        t4 = Table(per_data, colWidths=[3*cm, 3*cm, 3*cm, 3*cm, 3*cm, 2.5*cm])
        t4.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0369a1")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1),
             [colors.HexColor("#f0f9ff"), colors.white]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(t4)
        story.append(Spacer(1, 12))

    # ICRP limits
    story.append(Paragraph("5. ICRP Dose Limits Reference", h1))
    limits = [
        ["Category", "Annual Limit", "Notes"],
        ["Radiation Workers", "20 mSv/yr", "Averaged over 5 years"],
        ["Radiation Workers (single yr)", "50 mSv/yr", "Maximum in any single year"],
        ["General Public", "1 mSv/yr", "Excluding natural background"],
        ["Pregnant Workers", "1 mSv", "For declared pregnancy period"],
        ["Emergency Responders", "500 mSv", "Life-saving actions only"],
    ]
    t5 = Table(limits, colWidths=[6*cm, 5*cm, 5.5*cm])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0369a1")),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.HexColor("#f0f9ff"), colors.white]),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(t5)
    story.append(Spacer(1, 12))

    # Verdict
    pct_occ = (annual / 20) * 100
    if pct_occ < 10:
        verdict = "SAFE — Dose levels are well within all regulatory limits."
        vcolor  = colors.HexColor("#065f46")
    elif pct_occ < 75:
        verdict = "CAUTION — Monitor exposure time carefully. Consider additional shielding."
        vcolor  = colors.HexColor("#92400e")
    else:
        verdict = "DANGER — Dose levels exceed safe thresholds. Immediate shielding required."
        vcolor  = colors.HexColor("#7f1d1d")

    story.append(Paragraph("6. Safety Verdict", h1))
    story.append(Paragraph(verdict,
                           ParagraphStyle("verdict", parent=styles["Normal"],
                                          fontSize=12, textColor=vcolor,
                                          fontName="Helvetica-Bold",
                                          spaceBefore=6, spaceAfter=6)))

    doc.build(story)

# ─── Main Application ─────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hospital Radiation Safety Simulator v2 — Advanced")
        self.configure(bg="#060b14")
        self.state("zoomed")

        self.sources      = []
        self.persons      = []
        self.obstacles    = []
        self.field        = None
        self.room_w       = 20.0
        self.room_h       = 15.0
        self.dose_history = []   # list of (time, max_dose)
        self._monitor_running = False
        self._anim = None

        self._style()
        self._build_ui()

    def _style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("Dark.TCombobox",
                    fieldbackground="#1e293b", background="#1e293b",
                    foreground="#e2e8f0", selectbackground="#1e293b",
                    selectforeground="#38bdf8", arrowcolor="#38bdf8")
        s.configure("TNotebook",         background="#060b14", borderwidth=0)
        s.configure("TNotebook.Tab",
                    background="#0f172a", foreground="#64748b",
                    padding=[14, 6], font=("Courier New", 9, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", "#1e3a5f")],
              foreground=[("selected", "#38bdf8")])

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg="#060b14")
        hdr.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(hdr, text="☢  HOSPITAL RADIATION SAFETY SIMULATOR  v2",
                 font=("Courier New", 18, "bold"),
                 fg="#38bdf8", bg="#060b14").pack(side="left")
        tk.Label(hdr, text="Physics Engine: ISL + Beer-Lambert | ICRP Compliant",
                 font=("Courier New", 9), fg="#334155", bg="#060b14").pack(side="right")

        tk.Frame(self, bg="#1e3a5f", height=1).pack(fill="x", padx=16)

        body = tk.Frame(self, bg="#060b14")
        body.pack(fill="both", expand=True, padx=16, pady=10)

        # Left panel — tabs
        left = tk.Frame(body, bg="#060b14", width=330)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)

        tab1 = tk.Frame(nb, bg="#0f172a")
        tab2 = tk.Frame(nb, bg="#0f172a")
        tab3 = tk.Frame(nb, bg="#0f172a")
        tab4 = tk.Frame(nb, bg="#0f172a")
        nb.add(tab1, text=" Sources ")
        nb.add(tab2, text=" People ")
        nb.add(tab3, text=" Obstacles ")
        nb.add(tab4, text=" Actions ")

        self._build_sources_tab(tab1)
        self._build_people_tab(tab2)
        self._build_obstacles_tab(tab3)
        self._build_actions_tab(tab4)

        # Right panel — notebook for views
        right = tk.Frame(body, bg="#060b14")
        right.pack(side="left", fill="both", expand=True)

        view_nb = ttk.Notebook(right)
        view_nb.pack(fill="both", expand=True)

        self.tab_2d   = tk.Frame(view_nb, bg="#060b14")
        self.tab_3d   = tk.Frame(view_nb, bg="#060b14")
        self.tab_hist = tk.Frame(view_nb, bg="#060b14")
        view_nb.add(self.tab_2d,   text=" 2D Heatmap ")
        view_nb.add(self.tab_3d,   text=" 3D View ")
        view_nb.add(self.tab_hist, text=" Dose History ")

        self._build_2d_view(self.tab_2d)
        self._build_3d_view(self.tab_3d)
        self._build_history_view(self.tab_hist)

        # Status + alert bar
        bot = tk.Frame(self, bg="#0f172a")
        bot.pack(fill="x", padx=16, pady=(4, 8))
        self.status_var = tk.StringVar(value="Ready. Add sources and run simulation.")
        self.alert_var  = tk.StringVar(value="")
        tk.Label(bot, textvariable=self.status_var,
                 font=("Courier New", 8), fg="#64748b", bg="#0f172a",
                 anchor="w").pack(side="left", padx=8)
        self.alert_lbl = tk.Label(bot, textvariable=self.alert_var,
                                  font=("Courier New", 9, "bold"),
                                  fg="#ef4444", bg="#0f172a", anchor="e")
        self.alert_lbl.pack(side="right", padx=8)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Courier New", 8, "bold"),
                          fg="#38bdf8", bg="#0f172a",
                          bd=1, relief="solid", labelanchor="nw")
        f.pack(fill="x", pady=4, padx=6, ipady=4, ipadx=4)
        return f

    def _lrow(self, parent, label):
        r = tk.Frame(parent, bg="#0f172a")
        r.pack(fill="x", pady=2, padx=4)
        tk.Label(r, text=label, font=("Courier New", 8),
                 fg="#94a3b8", bg="#0f172a", width=15, anchor="w").pack(side="left")
        return r

    def _entry(self, parent, default="", width=10):
        e = tk.Entry(parent, font=("Courier New", 9),
                     bg="#1e293b", fg="#f1f5f9",
                     insertbackground="#38bdf8",
                     relief="flat", bd=4, width=width)
        e.insert(0, default)
        return e

    def _combo(self, parent, values, default=0, width=16):
        c = ttk.Combobox(parent, values=values, state="readonly",
                         font=("Courier New", 9),
                         style="Dark.TCombobox", width=width)
        c.current(default)
        return c

    def _btn(self, parent, text, cmd, fg="#38bdf8", full=False):
        b = tk.Button(parent, text=text, command=cmd,
                      font=("Courier New", 9, "bold"),
                      bg="#1e3a5f", fg=fg,
                      activebackground="#2563eb", activeforeground="#fff",
                      relief="flat", bd=0, padx=8, pady=5, cursor="hand2")
        if full:
            b.pack(fill="x", padx=6, pady=2)
        return b

    # ── Sources Tab ───────────────────────────────────────────────────────────

    def _build_sources_tab(self, parent):
        s1 = self._section(parent, "ROOM")
        r = self._lrow(s1, "Preset")
        self.preset_cb = self._combo(r, list(ROOM_PRESETS.keys()))
        self.preset_cb.pack(side="left")
        self.preset_cb.bind("<<ComboboxSelected>>", self._on_preset)

        r2 = self._lrow(s1, "Width (m)")
        self.room_w_e = self._entry(r2, "20", 8); self.room_w_e.pack(side="left")
        r3 = self._lrow(s1, "Height (m)")
        self.room_h_e = self._entry(r3, "15", 8); self.room_h_e.pack(side="left")

        s2 = self._section(parent, "ADD SOURCE")
        r = self._lrow(s2, "X Position")
        self.src_x = self._entry(r, "5"); self.src_x.pack(side="left")
        r = self._lrow(s2, "Y Position")
        self.src_y = self._entry(r, "5"); self.src_y.pack(side="left")
        r = self._lrow(s2, "Power Sv/hr")
        self.src_p = self._entry(r, "1.0"); self.src_p.pack(side="left")
        r = self._lrow(s2, "Type")
        self.src_type = self._combo(r, list(RADIATION_TYPES.keys()))
        self.src_type.pack(side="left")
        r = self._lrow(s2, "Shield Mat.")
        self.src_mat = self._combo(r, list(SHIELDING_MATERIALS.keys()))
        self.src_mat.pack(side="left")
        r = self._lrow(s2, "Thickness cm")
        self.src_thk = self._entry(r, "5"); self.src_thk.pack(side="left")

        bf = tk.Frame(s2, bg="#0f172a"); bf.pack(fill="x", padx=4, pady=4)
        self._btn(bf, "+ Add", self._add_src).pack(side="left", padx=2)
        self._btn(bf, "✕ Clear", self._clear_src, fg="#ef4444").pack(side="left", padx=2)

        s3 = self._section(parent, "ACTIVE SOURCES")
        self.src_lb = tk.Listbox(s3, font=("Courier New", 7),
                                 bg="#080c18", fg="#94a3b8",
                                 selectbackground="#1e3a5f",
                                 relief="flat", height=6, activestyle="none")
        self.src_lb.pack(fill="x", padx=4, pady=2)
        self._btn(s3, "Remove Selected", self._rm_src, fg="#f97316").pack(padx=4, pady=2, anchor="w")

    # ── People Tab ────────────────────────────────────────────────────────────

    def _build_people_tab(self, parent):
        s = self._section(parent, "ADD PERSON")
        r = self._lrow(s, "Name")
        self.per_name = self._entry(r, "Worker 1"); self.per_name.pack(side="left")
        r = self._lrow(s, "Role")
        self.per_role = self._combo(r, list(PERSONS.keys())); self.per_role.pack(side="left")
        r = self._lrow(s, "X Position")
        self.per_x = self._entry(r, "10"); self.per_x.pack(side="left")
        r = self._lrow(s, "Y Position")
        self.per_y = self._entry(r, "10"); self.per_y.pack(side="left")

        bf = tk.Frame(s, bg="#0f172a"); bf.pack(fill="x", padx=4, pady=4)
        self._btn(bf, "+ Add Person", self._add_person).pack(side="left", padx=2)
        self._btn(bf, "✕ Clear All", self._clear_persons, fg="#ef4444").pack(side="left", padx=2)

        s2 = self._section(parent, "PERSONNEL LIST")
        self.per_lb = tk.Listbox(s2, font=("Courier New", 7),
                                 bg="#080c18", fg="#94a3b8",
                                 selectbackground="#1e3a5f",
                                 relief="flat", height=8, activestyle="none")
        self.per_lb.pack(fill="x", padx=4, pady=2)
        self._btn(s2, "Remove Selected", self._rm_person, fg="#f97316").pack(padx=4, pady=2, anchor="w")

        s3 = self._section(parent, "DOSE SUMMARY")
        self.dose_txt = tk.Text(s3, font=("Courier New", 7),
                                bg="#080c18", fg="#94a3b8",
                                relief="flat", height=8, state="disabled")
        self.dose_txt.pack(fill="x", padx=4, pady=2)

    # ── Obstacles Tab ─────────────────────────────────────────────────────────

    def _build_obstacles_tab(self, parent):
        s = self._section(parent, "ADD WALL / OBSTACLE")
        r = self._lrow(s, "X (m)"); self.obs_x = self._entry(r, "8"); self.obs_x.pack(side="left")
        r = self._lrow(s, "Y (m)"); self.obs_y = self._entry(r, "0"); self.obs_y.pack(side="left")
        r = self._lrow(s, "Width (m)"); self.obs_w = self._entry(r, "1"); self.obs_w.pack(side="left")
        r = self._lrow(s, "Height (m)"); self.obs_h = self._entry(r, "10"); self.obs_h.pack(side="left")
        r = self._lrow(s, "Material")
        self.obs_mat = self._combo(r, list(SHIELDING_MATERIALS.keys()), default=1)
        self.obs_mat.pack(side="left")
        r = self._lrow(s, "Thickness cm"); self.obs_thk = self._entry(r, "20"); self.obs_thk.pack(side="left")

        bf = tk.Frame(s, bg="#0f172a"); bf.pack(fill="x", padx=4, pady=4)
        self._btn(bf, "+ Add Wall", self._add_obs).pack(side="left", padx=2)
        self._btn(bf, "✕ Clear", self._clear_obs, fg="#ef4444").pack(side="left", padx=2)

        s2 = self._section(parent, "OBSTACLES LIST")
        self.obs_lb = tk.Listbox(s2, font=("Courier New", 7),
                                 bg="#080c18", fg="#94a3b8",
                                 selectbackground="#1e3a5f",
                                 relief="flat", height=10, activestyle="none")
        self.obs_lb.pack(fill="x", padx=4, pady=2)
        self._btn(s2, "Remove Selected", self._rm_obs, fg="#f97316").pack(padx=4, pady=2, anchor="w")

    # ── Actions Tab ───────────────────────────────────────────────────────────

    def _build_actions_tab(self, parent):
        s = self._section(parent, "SIMULATION")
        self._btn(s, "▶  RUN SIMULATION",    self._run,         fg="#38bdf8",  full=True)
        self._btn(s, "◉  ANIMATE SPREAD",    self._animate,     fg="#a78bfa",  full=True)
        self._btn(s, "⬡  3D VIEW",           self._update_3d,   fg="#34d399",  full=True)
        self._btn(s, "✦  FIND SAFEST SPOT",  self._find_safe,   fg="#34d399",  full=True)

        s2 = self._section(parent, "MONITORING")
        self._btn(s2, "◉  START MONITORING", self._start_monitor, fg="#fbbf24", full=True)
        self._btn(s2, "◻  STOP MONITORING",  self._stop_monitor,  fg="#64748b", full=True)

        s3 = self._section(parent, "EXPORT")
        self._btn(s3, "📋  GENERATE REPORT",  self._gen_report,   fg="#fbbf24",  full=True)
        self._btn(s3, "📄  EXPORT PDF",        self._export_pdf,   fg="#f472b6",  full=True)
        self._btn(s3, "🖼  SAVE HEATMAP PNG",  self._save_png,     fg="#60a5fa",  full=True)

        s4 = self._section(parent, "ALERT THRESHOLD")
        r = self._lrow(s4, "Alert mSv/hr")
        self.alert_thresh = self._entry(r, "2.0"); self.alert_thresh.pack(side="left")

    # ── 2D View ───────────────────────────────────────────────────────────────

    def _build_2d_view(self, parent):
        self.fig2d = plt.Figure(figsize=(9, 6), facecolor="#060b14")
        self.ax2d  = self.fig2d.add_subplot(111)
        self._init_2d()
        self.canvas2d = FigureCanvasTkAgg(self.fig2d, master=parent)
        self.canvas2d.get_tk_widget().pack(fill="both", expand=True)
        self.canvas2d.draw()

    def _init_2d(self):
        ax = self.ax2d
        ax.set_facecolor("#080c18")
        ax.set_title("Radiation Field — awaiting simulation",
                     color="#334155", fontsize=11, fontfamily="monospace")
        ax.tick_params(colors="#334155")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.set_xlabel("X (m)", color="#475569", fontsize=9, fontfamily="monospace")
        ax.set_ylabel("Y (m)", color="#475569", fontsize=9, fontfamily="monospace")

    # ── 3D View ───────────────────────────────────────────────────────────────

    def _build_3d_view(self, parent):
        self.fig3d = plt.Figure(figsize=(9, 6), facecolor="#060b14")
        self.ax3d  = self.fig3d.add_subplot(111, projection="3d")
        self.ax3d.set_facecolor("#060b14")
        self.ax3d.set_title("3D Dose Surface — run simulation first",
                            color="#334155", fontsize=11, fontfamily="monospace")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=parent)
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)
        self.canvas3d.draw()

    # ── History View ──────────────────────────────────────────────────────────

    def _build_history_view(self, parent):
        self.fig_hist = plt.Figure(figsize=(9, 6), facecolor="#060b14")
        self.ax_hist  = self.fig_hist.add_subplot(111)
        self.ax_hist.set_facecolor("#080c18")
        self.ax_hist.set_title("Dose History — start monitoring",
                               color="#334155", fontsize=11, fontfamily="monospace")
        self.ax_hist.tick_params(colors="#334155")
        for sp in self.ax_hist.spines.values(): sp.set_edgecolor("#1e3a5f")
        self.canvas_hist = FigureCanvasTkAgg(self.fig_hist, master=parent)
        self.canvas_hist.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_hist.draw()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_preset(self, _=None):
        p = self.preset_cb.get()
        w, h = ROOM_PRESETS[p]
        if w:
            self.room_w_e.delete(0,"end"); self.room_w_e.insert(0, str(w))
            self.room_h_e.delete(0,"end"); self.room_h_e.insert(0, str(h))

    def _parse_room(self):
        self.room_w = float(self.room_w_e.get())
        self.room_h = float(self.room_h_e.get())

    # Sources
    def _add_src(self):
        try:
            src = {"x": float(self.src_x.get()), "y": float(self.src_y.get()),
                   "power": float(self.src_p.get()),
                   "type": self.src_type.get(),
                   "material": self.src_mat.get(),
                   "thickness": float(self.src_thk.get())}
        except ValueError:
            messagebox.showerror("Error", "Invalid input values."); return
        self.sources.append(src)
        sym = RADIATION_TYPES[src["type"]]["symbol"]
        self.src_lb.insert("end",
            f"  {sym} {src['type']}  ({src['x']},{src['y']})  "
            f"{src['power']}Sv/hr  [{src['material']} {src['thickness']}cm]")
        self.src_lb.itemconfig("end", fg=RADIATION_TYPES[src["type"]]["color"])
        self.status_var.set(f"Source added: {src['type']} at ({src['x']},{src['y']})")

    def _clear_src(self):
        self.sources.clear(); self.src_lb.delete(0,"end"); self.field = None

    def _rm_src(self):
        s = self.src_lb.curselection()
        if s: self.src_lb.delete(s[0]); self.sources.pop(s[0]); self.field = None

    # People
    def _add_person(self):
        try:
            p = {"name": self.per_name.get(), "role": self.per_role.get(),
                 "x": float(self.per_x.get()), "y": float(self.per_y.get())}
        except ValueError:
            messagebox.showerror("Error", "Invalid position."); return
        self.persons.append(p)
        col = PERSONS[p["role"]]["color"]
        self.per_lb.insert("end", f"  {p['name']} ({p['role']})  @({p['x']},{p['y']})")
        self.per_lb.itemconfig("end", fg=col)

    def _clear_persons(self):
        self.persons.clear(); self.per_lb.delete(0,"end")

    def _rm_person(self):
        s = self.per_lb.curselection()
        if s: self.per_lb.delete(s[0]); self.persons.pop(s[0])

    # Obstacles
    def _add_obs(self):
        try:
            obs = {"x": float(self.obs_x.get()), "y": float(self.obs_y.get()),
                   "w": float(self.obs_w.get()), "h": float(self.obs_h.get()),
                   "material": self.obs_mat.get(),
                   "thickness": float(self.obs_thk.get())}
        except ValueError:
            messagebox.showerror("Error", "Invalid obstacle values."); return
        self.obstacles.append(obs)
        self.obs_lb.insert("end",
            f"  Wall @ ({obs['x']},{obs['y']})  {obs['w']}x{obs['h']}m"
            f"  [{obs['material']} {obs['thickness']}cm]")
        self.obs_lb.itemconfig("end", fg=SHIELDING_MATERIALS[obs["material"]]["color"])

    def _clear_obs(self):
        self.obstacles.clear(); self.obs_lb.delete(0,"end"); self.field = None

    def _rm_obs(self):
        s = self.obs_lb.curselection()
        if s: self.obs_lb.delete(s[0]); self.obstacles.pop(s[0]); self.field = None

    # ── Simulation ────────────────────────────────────────────────────────────

    def _run(self):
        if not self.sources:
            messagebox.showwarning("No Sources", "Add at least one radiation source."); return
        self._parse_room()
        self.status_var.set("Computing field…"); self.update()
        self.field, xs, ys = compute_field(self.sources, self.room_w, self.room_h, self.obstacles)

        # Record history
        ts = len(self.dose_history)
        self.dose_history.append((ts, self.field.max()))

        self._draw_2d(self.field)
        self._check_alerts(self.field.max())
        self._update_dose_summary()
        self.status_var.set(
            f"Done. Peak: {self.field.max():.3f} mSv/hr | "
            f"Mean: {self.field.mean():.3f} mSv/hr | Sources: {len(self.sources)}")

    def _draw_2d(self, field, safe_pt=None, step=None):
        ax = self.ax2d
        ax.cla()
        ax.set_facecolor("#080c18")

        title = "Radiation Dose Field [mSv/hr]"
        if step is not None:
            title += f"  — spread t={step}"
        if safe_pt:
            title += "  |  ✦ Safest spot marked"

        vmax = np.percentile(field, 98) if field.max() > 0 else 1
        im = ax.imshow(field, extent=[0, self.room_w, 0, self.room_h],
                       origin="lower", cmap=HEATMAP_CMAP,
                       aspect="auto", interpolation="bilinear",
                       norm=plt.Normalize(vmin=0, vmax=vmax))

        # Contours
        try:
            lvls = np.percentile(field[field>0], [25,50,75,90])
            ax.contour(np.linspace(0,self.room_w,GRID),
                       np.linspace(0,self.room_h,GRID),
                       field, levels=lvls, colors=["#fff"], alpha=0.12, linewidths=0.5)
        except: pass

        # Obstacles
        for obs in self.obstacles:
            col = SHIELDING_MATERIALS[obs["material"]]["color"]
            ax.add_patch(patches.Rectangle(
                (obs["x"], obs["y"]), obs["w"], obs["h"],
                linewidth=1.5, edgecolor=col,
                facecolor=col, alpha=0.45, zorder=4))
            ax.text(obs["x"]+obs["w"]/2, obs["y"]+obs["h"]/2,
                    obs["material"][:4], color="#fff",
                    fontsize=6, ha="center", va="center",
                    fontfamily="monospace", zorder=5)

        # Sources
        for src in self.sources:
            col = RADIATION_TYPES[src["type"]]["color"]
            sym = RADIATION_TYPES[src["type"]]["symbol"]
            ax.plot(src["x"], src["y"], "o", color=col,
                    markersize=11, markeredgecolor="#fff",
                    markeredgewidth=1.2, zorder=6)
            ax.annotate(f" {sym} {src['power']}Sv",
                        xy=(src["x"], src["y"]),
                        fontsize=7, color=col, fontfamily="monospace",
                        xytext=(5,4), textcoords="offset points", zorder=7)

        # Personnel
        for p in self.persons:
            col  = PERSONS[p["role"]]["color"]
            icon = PERSONS[p["role"]]["icon"]
            ax.plot(p["x"], p["y"], "s", color=col,
                    markersize=10, markeredgecolor="#fff",
                    markeredgewidth=1.0, zorder=6)
            ax.text(p["x"], p["y"], icon, color="#fff",
                    fontsize=6, ha="center", va="center",
                    fontweight="bold", zorder=7)
            ax.annotate(f" {p['name']}",
                        xy=(p["x"], p["y"]),
                        fontsize=6.5, color=col, fontfamily="monospace",
                        xytext=(6,3), textcoords="offset points", zorder=7)

        # Safest spot
        if safe_pt:
            sx, sy, sd = safe_pt
            ax.plot(sx, sy, "*", color="#34d399", markersize=18,
                    markeredgecolor="#fff", markeredgewidth=1.0, zorder=8)
            ax.annotate(f"  Safest\n  {sd:.4f} mSv/hr",
                        xy=(sx,sy), fontsize=7.5, color="#34d399",
                        fontfamily="monospace",
                        xytext=(8,4), textcoords="offset points", zorder=9)

        # Room border
        ax.add_patch(patches.Rectangle((0,0), self.room_w, self.room_h,
                                       linewidth=2, edgecolor="#1e3a5f",
                                       facecolor="none", zorder=3))

        try:
            cb = self.fig2d.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cb.set_label("mSv/hr", color="#94a3b8", fontsize=8)
            cb.ax.yaxis.set_tick_params(color="#475569")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=7)
        except: pass

        ax.set_xlim(0, self.room_w); ax.set_ylim(0, self.room_h)
        ax.set_title(title, color="#94a3b8", fontsize=10, fontfamily="monospace")
        ax.set_xlabel("X (m)", color="#475569", fontsize=9, fontfamily="monospace")
        ax.set_ylabel("Y (m)", color="#475569", fontsize=9, fontfamily="monospace")
        ax.tick_params(colors="#334155")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#0f172a", linewidth=0.4, linestyle="--")
        self.canvas2d.draw()

    # ── 3D View ───────────────────────────────────────────────────────────────

    def _update_3d(self):
        if self.field is None:
            messagebox.showinfo("Run First", "Run simulation first."); return
        ax = self.ax3d; ax.cla()
        ax.set_facecolor("#060b14")
        self.fig3d.set_facecolor("#060b14")

        step = max(1, GRID // 50)
        xs   = np.linspace(0, self.room_w, GRID)[::step]
        ys   = np.linspace(0, self.room_h, GRID)[::step]
        X, Y = np.meshgrid(xs, ys)
        Z    = self.field[::step, ::step]

        surf = ax.plot_surface(X, Y, Z, cmap=HEATMAP_CMAP, alpha=0.92,
                               linewidth=0, antialiased=True)
        self.fig3d.colorbar(surf, ax=ax, shrink=0.5, aspect=10,
                            label="mSv/hr").ax.yaxis.label.set_color("#94a3b8")

        ax.set_xlabel("X (m)", color="#475569", labelpad=8)
        ax.set_ylabel("Y (m)", color="#475569", labelpad=8)
        ax.set_zlabel("Dose (mSv/hr)", color="#475569", labelpad=8)
        ax.set_title("3D Radiation Dose Surface",
                     color="#94a3b8", fontsize=11, fontfamily="monospace")
        ax.tick_params(colors="#334155", labelsize=7)
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.grid(True, color="#1e293b", linewidth=0.4)
        self.canvas3d.draw()
        self.status_var.set("3D view updated.")

    # ── Animation ─────────────────────────────────────────────────────────────

    def _animate(self):
        if not self.sources:
            messagebox.showwarning("No Sources", "Add sources first."); return
        self._parse_room()
        self.status_var.set("Animating radiation spread…")

        STEPS = 30
        fields = []
        for t in range(1, STEPS+1):
            fake_sources = [dict(s, power=s["power"] * (t/STEPS)) for s in self.sources]
            f, _, _ = compute_field(fake_sources, self.room_w, self.room_h, self.obstacles)
            fields.append(f)

        step_idx = [0]
        def _step():
            if step_idx[0] < STEPS:
                self._draw_2d(fields[step_idx[0]], step=step_idx[0]+1)
                step_idx[0] += 1
                self.after(80, _step)
            else:
                self.field = fields[-1]
                self.status_var.set("Animation complete.")

        _step()

    # ── Safest spot ───────────────────────────────────────────────────────────

    def _find_safe(self):
        if self.field is None:
            messagebox.showinfo("Run First", "Run simulation first."); return
        sx, sy, sd = safest_spot(self.field, self.room_w, self.room_h)
        self._draw_2d(self.field, safe_pt=(sx, sy, sd))
        self.status_var.set(f"Safest: ({sx:.2f}m, {sy:.2f}m)  {sd:.6f} mSv/hr")

    # ── Dose Summary ──────────────────────────────────────────────────────────

    def _update_dose_summary(self):
        if self.field is None or not self.persons: return
        self.dose_txt.config(state="normal")
        self.dose_txt.delete("1.0","end")
        for p in self.persons:
            d   = person_dose(self.field, p, self.room_w, self.room_h)
            ann = d * 8760
            lim = PERSONS[p["role"]]["dose_limit"]
            pct = (ann / lim) * 100
            st  = "SAFE" if pct < 25 else ("WARN" if pct < 75 else "DANGER")
            self.dose_txt.insert("end",
                f"{p['name']} ({p['role']})\n"
                f"  Dose: {d:.4f} mSv/hr\n"
                f"  Annual: {ann:.2f} mSv/yr ({pct:.1f}% of limit)\n"
                f"  Status: {st}\n\n")
        self.dose_txt.config(state="disabled")

    # ── Real-time Monitoring ──────────────────────────────────────────────────

    def _start_monitor(self):
        if not self.sources:
            messagebox.showwarning("No Sources","Add sources first."); return
        self._monitor_running = True
        self.status_var.set("Real-time monitoring active…")
        self._monitor_loop()

    def _stop_monitor(self):
        self._monitor_running = False
        self.alert_var.set("")
        self.status_var.set("Monitoring stopped.")

    def _monitor_loop(self):
        if not self._monitor_running: return
        self._parse_room()
        self.field, _, _ = compute_field(
            self.sources, self.room_w, self.room_h, self.obstacles)
        peak = self.field.max()
        ts   = len(self.dose_history)
        self.dose_history.append((ts, peak))
        self._check_alerts(peak)
        self._update_history_graph()
        self._update_dose_summary()
        self.after(1500, self._monitor_loop)

    def _check_alerts(self, peak):
        try: thresh = float(self.alert_thresh.get())
        except: thresh = ALERT_THRESHOLD
        if peak > thresh:
            self.alert_var.set(f"🚨 ALERT: {peak:.3f} mSv/hr exceeds {thresh} mSv/hr threshold!")
            self.alert_lbl.config(fg="#ef4444")
        else:
            self.alert_var.set(f"✅ Safe: {peak:.3f} mSv/hr  (threshold: {thresh} mSv/hr)")
            self.alert_lbl.config(fg="#34d399")

    # ── Dose History Graph ────────────────────────────────────────────────────

    def _update_history_graph(self):
        if not self.dose_history: return
        ax = self.ax_hist; ax.cla()
        ax.set_facecolor("#080c18")
        ts   = [h[0] for h in self.dose_history]
        vals = [h[1] for h in self.dose_history]
        ax.plot(ts, vals, color="#38bdf8", linewidth=1.8, marker="o",
                markersize=3, alpha=0.9)
        ax.fill_between(ts, vals, alpha=0.15, color="#38bdf8")
        try:
            thresh = float(self.alert_thresh.get())
            ax.axhline(thresh, color="#ef4444", linewidth=1,
                       linestyle="--", alpha=0.7, label=f"Alert {thresh} mSv/hr")
            ax.legend(facecolor="#1e293b", edgecolor="#334155",
                      labelcolor="#94a3b8", fontsize=8)
        except: pass
        ax.set_title("Real-time Dose History",
                     color="#94a3b8", fontsize=11, fontfamily="monospace")
        ax.set_xlabel("Sample", color="#475569", fontsize=9, fontfamily="monospace")
        ax.set_ylabel("Peak Dose (mSv/hr)", color="#475569", fontsize=9, fontfamily="monospace")
        ax.tick_params(colors="#334155")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a5f")
        ax.grid(color="#0f172a", linewidth=0.4, linestyle="--")
        self.canvas_hist.draw()

    # ── Report & Export ───────────────────────────────────────────────────────

    def _gen_report(self):
        if self.field is None:
            messagebox.showinfo("Run First","Run simulation first."); return
        sx, sy, sd = safest_spot(self.field, self.room_w, self.room_h)
        annual = sd * 8760
        pct    = (annual / 20) * 100
        verdict = ("SAFE" if pct < 10 else "CAUTION" if pct < 75 else "DANGER")

        win = tk.Toplevel(self)
        win.title("Simulation Report"); win.configure(bg="#060b14"); win.geometry("700x580")
        tk.Label(win, text="☢  RADIATION SAFETY REPORT",
                 font=("Courier New", 13,"bold"), fg="#38bdf8", bg="#060b14").pack(pady=10)
        txt = tk.Text(win, font=("Courier New", 9), bg="#0f172a", fg="#cbd5e1",
                      relief="flat", bd=10, wrap="word")
        txt.pack(fill="both", expand=True, padx=14, pady=(0,12))

        lines = [
            "=" * 60,
            "  HOSPITAL RADIATION SAFETY SIMULATION REPORT",
            f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
            "=" * 60,
            f"\n  Room       : {self.room_w} m x {self.room_h} m",
            f"  Sources    : {len(self.sources)}",
            f"  Personnel  : {len(self.persons)}",
            f"  Obstacles  : {len(self.obstacles)}",
            "\n" + "-"*60,
            "\n  DOSE STATISTICS",
            f"  Peak dose  : {self.field.max():.4f} mSv/hr",
            f"  Mean dose  : {self.field.mean():.4f} mSv/hr",
            f"  Min dose   : {self.field.min():.6f} mSv/hr",
            f"  Safest spot: ({sx:.2f} m, {sy:.2f} m)",
            f"  Dose there : {sd:.6f} mSv/hr",
            f"  Annual est : {annual:.2f} mSv/yr",
            f"  % of limit : {pct:.1f}%",
            f"\n  VERDICT    : {verdict}",
            "\n" + "-"*60,
        ]
        if self.persons and self.field is not None:
            lines.append("\n  PERSONNEL ASSESSMENT")
            for p in self.persons:
                d   = person_dose(self.field, p, self.room_w, self.room_h)
                ann = d * 8760
                lim = PERSONS[p["role"]]["dose_limit"]
                lines.append(
                    f"  {p['name']:16s} ({p['role']:12s})"
                    f"  {d:.4f} mSv/hr  |  {ann:.1f}/{lim} mSv/yr")
        lines += ["\n" + "="*60,
                  "  ICRP Limits: Workers 20 mSv/yr | Public 1 mSv/yr",
                  "=" * 60]
        txt.insert("1.0", "\n".join(lines))
        txt.config(state="disabled")

    def _export_pdf(self):
        if not PDF_AVAILABLE:
            messagebox.showerror("Missing Library",
                "reportlab not installed.\nRun: pip install reportlab"); return
        if self.field is None:
            messagebox.showinfo("Run First","Run simulation first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Report","*.pdf")],
            title="Save PDF Report As")
        if not path: return
        try:
            export_pdf(path, self.sources, self.persons, self.field,
                       self.room_w, self.room_h,
                       self.obstacles, self.dose_history, self.fig2d)
            self.status_var.set(f"PDF saved: {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"PDF report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"PDF export failed:\n{e}")

    def _save_png(self):
        if self.field is None:
            messagebox.showinfo("Run First","Run simulation first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image","*.png")],
            title="Save Heatmap As")
        if path:
            self.fig2d.savefig(path, dpi=180, bbox_inches="tight",
                               facecolor=self.fig2d.get_facecolor())
            self.status_var.set(f"Heatmap saved: {os.path.basename(path)}")
            messagebox.showinfo("Saved", f"Heatmap saved to:\n{path}")

# ─── Entry ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()

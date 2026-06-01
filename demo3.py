"""
Three-Body Problem  —  Interactive Simulator
Darrus Loamayer · Rafi Vidianta  |  run:  python demo.py
"""
import sys, os, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d import proj3d as _p3d  # used for screen projection

import simulator

# ── Palette ───────────────────────────────────────────────────────────────────
BG        = '#080C14'
BG_PANEL  = '#0D1421'
BG_CARD   = '#141E2E'
BG_INPUT  = '#111826'
SEP       = '#1F2D42'
ACCENT    = '#8896A8'
ACCENT_LT = '#AAB4C2'
TEXT      = '#D4DAE5'
TEXT_DIM  = '#5A6677'
BODY_C    = ['#5B9CF6', '#F5A623', '#4ECDC4']
HOVER_BG  = '#1C2B3F'


class ThreeBodySimulator:
    TRAIL = 350

    def __init__(self, root):
        self.root = root
        self.root.title("Three-Body Problem  —  Interactive Simulator")
        self.root.configure(bg=BG)
        self.root.geometry("1560x960")
        self.root.minsize(1100, 680)

        self.times = self.states = self.masses_run = self.en_errors = None
        self.frame_idx   = 0
        self.running     = False
        self._after_id   = None
        self._computing  = False
        self._has_result = False

        # screen-space positions of the three body dots (updated each draw)
        self._dot_screen : dict[int, tuple[int,int]] = {}

        self._axis_dynamic  = True
        self._static_center = [0., 0., 0.]
        self._static_range  = 5.0

        self._tooltip_win   = None
        self._tooltip_label = None

        self.show_energy = tk.BooleanVar(value=True)
        self.integrator  = tk.StringVar(value='RK45')
        self.dt_var      = tk.StringVar(value='0.005')
        self.t_end_var   = tk.StringVar(value='20.0')
        self.soft_var    = tk.StringVar(value='0.0')
        self.speed_var   = tk.IntVar(value=3)
        self.entries: dict = {}

        self._build_ui()
        self._load_figure8()

    # ══════════════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.sidebar = tk.Frame(self.root, bg=BG_PANEL, width=400)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        self.canvas_area = tk.Frame(self.root, bg=BG)
        self.canvas_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._build_sidebar()
        self._build_canvas()

    def _build_sidebar(self):
        p = self.sidebar
        hf = tk.Frame(p, bg=BG_CARD)
        hf.pack(fill=tk.X)
        tk.Label(hf, text="THREE-BODY PROBLEM", bg=BG_CARD, fg=TEXT,
                 font=('Consolas',13,'bold'), pady=10).pack()
        tk.Label(hf, text="Interactive Simulator  ·  Darrus Loamayer",
                 bg=BG_CARD, fg=TEXT_DIM, font=('Consolas',8), pady=4).pack()

        self._line(p); self._section(p, "PRESETS")
        pf = tk.Frame(p, bg=BG_PANEL)
        pf.pack(fill=tk.X, padx=10, pady=(0,8))
        for lbl, cmd, clr in [("Figure-8", self._load_figure8, BODY_C[0]),
                               ("Pythagorean", self._load_pythagorean, BODY_C[1]),
                               ("Kozai-Lidov", self._load_kozai, BODY_C[2])]:
            tk.Button(pf, text=lbl, command=cmd, bg=BG_CARD, fg=clr,
                      activebackground=clr, activeforeground=BG,
                      relief=tk.FLAT, padx=6, pady=8,
                      font=('Consolas',9,'bold'), cursor='hand2', bd=0
                      ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self._line(p); self._section(p, "INITIAL CONDITIONS")
        outer = tk.Frame(p, bg=BG_PANEL)
        outer.pack(fill=tk.BOTH, expand=True, padx=6)
        sc = tk.Canvas(outer, bg=BG_PANEL, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient='vertical', command=sc.yview)
        self.ic_frame = tk.Frame(sc, bg=BG_PANEL)
        self.ic_frame.bind('<Configure>',
            lambda e: sc.configure(scrollregion=sc.bbox('all')))
        sc.create_window((0,0), window=self.ic_frame, anchor='nw')
        sc.configure(yscrollcommand=sb.set)
        sc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        sc.bind_all('<MouseWheel>',
            lambda e: sc.yview_scroll(-1*(e.delta//120), 'units'))
        for b in range(3):
            self._build_body_block(self.ic_frame, b)

        self._line(p); self._section(p, "PARAMETERS")
        pf2 = tk.Frame(p, bg=BG_PANEL)
        pf2.pack(fill=tk.X, padx=12, pady=2)
        for r,(lbl,var,opts) in enumerate([
            ("Integrator", self.integrator, ['RK45','RK4','Verlet']),
            ("dt",         self.dt_var,     None),
            ("t_end",      self.t_end_var,  None),
            ("softening",  self.soft_var,   None)]):
            tk.Label(pf2, text=lbl, bg=BG_PANEL, fg=TEXT_DIM,
                     font=('Consolas',8), anchor='e'
                     ).grid(row=r, column=0, sticky='e', padx=(0,8), pady=3)
            w = (ttk.Combobox(pf2, textvariable=var, values=opts,
                              state='readonly', width=13, font=('Consolas',9))
                 if opts else
                 tk.Entry(pf2, textvariable=var, width=15,
                          bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
                          relief=tk.FLAT, font=('Consolas',9)))
            w.grid(row=r, column=1, sticky='ew', pady=3)
        pf2.columnconfigure(1, weight=1)

        sf = tk.Frame(p, bg=BG_PANEL); sf.pack(fill=tk.X, padx=12, pady=(4,2))
        tk.Label(sf, text="speed", bg=BG_PANEL, fg=TEXT_DIM,
                 font=('Consolas',8)).pack(side=tk.LEFT)
        tk.Scale(sf, variable=self.speed_var, from_=1, to=100,
                 orient=tk.HORIZONTAL, length=200, bg=BG_PANEL, fg=TEXT,
                 troughcolor=BG_INPUT, highlightthickness=0,
                 font=('Consolas',8), showvalue=True
                 ).pack(side=tk.LEFT, padx=8)

        tk.Checkbutton(p, text="Show energy panel",
                       variable=self.show_energy, bg=BG_PANEL, fg=TEXT,
                       selectcolor=BG_CARD, activebackground=BG_PANEL,
                       font=('Consolas',9),
                       command=self._toggle_energy).pack(pady=4)

        self._line(p); self._section(p, "AXIS MODE")
        af = tk.Frame(p, bg=BG_PANEL)
        af.pack(fill=tk.X, padx=10, pady=(0,4))
        self._dyn_btn = tk.Button(af, text="● Dynamic",
                                   command=self._set_dynamic,
                                   bg=ACCENT, fg=BG, relief=tk.FLAT,
                                   font=('Consolas',9,'bold'), pady=5,
                                   cursor='hand2')
        self._dyn_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._sta_btn = tk.Button(af, text="○ Static",
                                   command=self._set_static,
                                   bg=BG_CARD, fg=ACCENT_LT,
                                   relief=tk.FLAT, font=('Consolas',9),
                                   pady=5, cursor='hand2')
        self._sta_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self._zoom_frame = tk.Frame(p, bg=BG_PANEL)
        for lbl, cmd in [("  +  Zoom In", self._zoom_in),
                         ("  −  Zoom Out", self._zoom_out)]:
            tk.Button(self._zoom_frame, text=lbl, command=cmd,
                      bg=BG_CARD, fg=TEXT_DIM, relief=tk.FLAT,
                      font=('Consolas',9), pady=4, cursor='hand2'
                      ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        self._line(p)
        bf = tk.Frame(p, bg=BG_PANEL); bf.pack(fill=tk.X, padx=10, pady=8)
        self.run_btn = tk.Button(bf, text="⟳ RUN", command=self._run,
                                  bg=ACCENT, fg=BG,
                                  activebackground=ACCENT_LT,
                                  activeforeground=BG, relief=tk.FLAT,
                                  font=('Consolas',10,'bold'), pady=10,
                                  cursor='hand2')
        self.run_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.play_btn = tk.Button(bf, text="▶ PLAY", command=self._play,
                                   bg=BG_CARD, fg='#4ADE80',
                                   activebackground='#4ADE80',
                                   activeforeground=BG, relief=tk.FLAT,
                                   font=('Consolas',10,'bold'), pady=10,
                                   cursor='hand2', state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(bf, text="■ STOP", command=self._stop_pause,
                  bg=BG_CARD, fg='#EF4444',
                  activebackground='#EF4444', activeforeground=BG,
                  relief=tk.FLAT, font=('Consolas',10,'bold'), pady=10,
                  cursor='hand2').pack(side=tk.LEFT, expand=True,
                                       fill=tk.X, padx=2)
        self.status_var = tk.StringVar(
            value="Select a preset or enter values, then click RUN")
        tk.Label(p, textvariable=self.status_var, bg=BG_PANEL, fg=TEXT_DIM,
                 font=('Consolas',7), wraplength=360,
                 justify=tk.LEFT).pack(padx=10, pady=4)

    def _build_body_block(self, parent, b):
        clr = BODY_C[b]
        frm = tk.Frame(parent, bg=BG_PANEL)
        frm.pack(fill=tk.X, padx=4, pady=6)
        tk.Label(frm, text=f"Body {b}", bg=BG_PANEL, fg=clr,
                 font=('Consolas',10,'bold')).grid(
            row=0, column=0, columnspan=6, sticky='w', padx=2, pady=(2,4))
        tk.Label(frm, text="mass", bg=BG_PANEL, fg=TEXT_DIM,
                 font=('Consolas',8)).grid(row=1, column=0,
                                             sticky='e', padx=(2,4))
        e = self._entry(frm, clr, width=18)
        e.grid(row=1, column=1, columnspan=5, sticky='ew', padx=2, pady=2)
        self.entries[(b,'mass')] = e; self._bind_preview(e)
        for c, lbl in enumerate(['x','y','z']):
            tk.Label(frm, text=lbl, bg=BG_PANEL, fg=TEXT_DIM,
                     font=('Consolas',8)).grid(
                row=2, column=c*2, sticky='e', padx=(6,2))
            e = self._entry(frm, clr, width=8)
            e.grid(row=2, column=c*2+1, sticky='ew', padx=2, pady=2)
            self.entries[(b,lbl)] = e; self._bind_preview(e)
        for c, lbl in enumerate(['vx','vy','vz']):
            tk.Label(frm, text=lbl, bg=BG_PANEL, fg=TEXT_DIM,
                     font=('Consolas',8)).grid(
                row=3, column=c*2, sticky='e', padx=(6,2))
            e = self._entry(frm, clr, width=8)
            e.grid(row=3, column=c*2+1, sticky='ew', padx=2, pady=2)
            self.entries[(b,lbl)] = e; self._bind_preview(e)
        for col in range(6): frm.columnconfigure(col, weight=1)
        tk.Frame(parent, bg=SEP, height=1).pack(fill=tk.X, padx=4)

    def _bind_preview(self, w):
        w.bind('<KeyRelease>', lambda e: self.root.after(80, self._live_preview))
        w.bind('<FocusOut>', lambda e: self._live_preview())

    def _entry(self, parent, fg, width=10):
        return tk.Entry(parent, width=width, bg=BG_INPUT, fg=fg,
                        insertbackground=fg, relief=tk.FLAT,
                        font=('Consolas',9))

    def _section(self, p, lbl):
        tk.Label(p, text=lbl, bg=BG_PANEL, fg=ACCENT,
                 font=('Consolas',8,'bold'), pady=5).pack()

    def _line(self, p):
        tk.Frame(p, bg=SEP, height=1).pack(fill=tk.X, pady=2)

    # ── canvas ────────────────────────────────────────────────────────────────

    def _build_canvas(self):
        for w in self.canvas_area.winfo_children(): w.destroy()
        self.fig = plt.figure(facecolor=BG)
        if self.show_energy.get():
            self.ax3d = self.fig.add_axes([0.02,0.30,0.96,0.68],
                                           projection='3d')
            self.ax_e = self.fig.add_axes([0.08,0.04,0.88,0.20])
            self._style_energy_ax()
        else:
            self.ax3d = self.fig.add_subplot(111, projection='3d')
            self.ax_e = None
        self._style_3d_ax()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_area)
        self.mpl_canvas.draw()
        wgt = self.mpl_canvas.get_tk_widget()
        wgt.pack(fill=tk.BOTH, expand=True)
        wgt.bind('<Motion>', self._on_hover)
        wgt.bind('<Leave>',  lambda e: self._hide_tooltip())

    def _style_3d_ax(self):
        ax = self.ax3d; ax.set_facecolor(BG)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False; pane.set_edgecolor(SEP)
        ax.tick_params(colors=TEXT_DIM, labelsize=6)
        for a in ('xaxis','yaxis','zaxis'):
            getattr(ax, a).label.set_color(TEXT_DIM)
        ax.grid(True, color=SEP, linewidth=0.35)

    def _style_energy_ax(self):
        ax = self.ax_e; ax.set_facecolor(BG_CARD)
        for s in ax.spines.values(): s.set_color(SEP)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        ax.set_xlabel('Time', color=TEXT_DIM, fontsize=7)
        ax.set_ylabel('|ΔE/E₀|', color=TEXT_DIM, fontsize=7)

    def _toggle_energy(self):
        self._cancel_after(); self._build_canvas()
        if self._has_result: self._start_animation()
        else: self._live_preview()

    # ── axis mode ─────────────────────────────────────────────────────────────

    def _set_dynamic(self):
        self._axis_dynamic = True
        self._dyn_btn.config(bg=ACCENT, fg=BG, font=('Consolas',9,'bold'))
        self._sta_btn.config(bg=BG_CARD, fg=ACCENT_LT, font=('Consolas',9))
        self._zoom_frame.pack_forget()

    def _set_static(self):
        self._axis_dynamic = False
        try:
            xl=self.ax3d.get_xlim(); yl=self.ax3d.get_ylim()
            zl=self.ax3d.get_zlim()
            self._static_center = [(xl[0]+xl[1])/2,(yl[0]+yl[1])/2,
                                    (zl[0]+zl[1])/2]
            self._static_range = max((xl[1]-xl[0])/2,(yl[1]-yl[0])/2,
                                      (zl[1]-zl[0])/2)
        except Exception: pass
        self._sta_btn.config(bg=ACCENT, fg=BG, font=('Consolas',9,'bold'))
        self._dyn_btn.config(bg=BG_CARD, fg=ACCENT_LT, font=('Consolas',9))
        self._zoom_frame.pack(fill=tk.X, padx=10, pady=(0,4))

    def _zoom_in(self):  self._static_range = max(0.5, self._static_range*0.75)
    def _zoom_out(self): self._static_range *= 1.35

    def _apply_axis_limits(self):
        if self._axis_dynamic: return
        cx,cy,cz = self._static_center; r = self._static_range
        self.ax3d.set_xlim(cx-r,cx+r); self.ax3d.set_ylim(cy-r,cy+r)
        self.ax3d.set_zlim(cz-r,cz+r)

    # ══════════════════════════════════════════════════════════════════════════
    # SCREEN-POSITION HELPER
    # Stores the pixel position of each body dot right after drawing it.
    # This is the only reliable way — proj3d is called here, not on hover.
    # ══════════════════════════════════════════════════════════════════════════

    def _store_dot_positions(self, positions):
        """
        Project world-space body positions to tkinter-widget pixel coords
        and store in self._dot_screen.  Called immediately after drawing.
        """
        self._dot_screen.clear()
        try:
            widget = self.mpl_canvas.get_tk_widget()
            wh = widget.winfo_height()
            ww = widget.winfo_width()
            fh = self.fig.get_figheight() * self.fig.dpi
            fw = self.fig.get_figwidth()  * self.fig.dpi
            sx = ww / fw if fw > 0 else 1.0
            sy = wh / fh if fh > 0 else 1.0
            proj = self.ax3d.get_proj()
            for i, pos in enumerate(positions):
                x2d, y2d, _ = _p3d.proj_transform(
                    pos[0], pos[1], pos[2], proj)
                xd, yd = self.ax3d.transData.transform((x2d, y2d))
                col = int(xd * sx)
                row = int(wh - yd * sy)
                self._dot_screen[i] = (col, row)
        except Exception:
            self._dot_screen.clear()

    # ══════════════════════════════════════════════════════════════════════════
    # HOVER TOOLTIP
    # ══════════════════════════════════════════════════════════════════════════

    def _on_hover(self, event):
        if not self._dot_screen:
            self._hide_tooltip(); return

        # find closest body to mouse
        pos, vel, masses = self._current_pos_vel_mass()
        if pos is None:
            self._hide_tooltip(); return

        mx, my = event.x, event.y
        threshold = 28
        closest = -1; best = threshold + 1
        for i, (col, row) in self._dot_screen.items():
            d = np.hypot(mx - col, my - row)
            if d < best:
                best = d; closest = i

        if closest == -1:
            self._hide_tooltip(); return

        i = closest
        speed = np.linalg.norm(vel[i])
        px,py,pz = pos[i]; vx,vy,vz = vel[i]
        info = (f"  Body {i}  (m = {masses[i]:.4g})  \n"
                f"  x = {px:+.4f}   y = {py:+.4f}   z = {pz:+.4f}  \n"
                f"  vx = {vx:+.4f}  vy = {vy:+.4f}  vz = {vz:+.4f}  \n"
                f"  speed = {speed:.4f}  ")
        self._show_tooltip(info, event)

    def _current_pos_vel_mass(self):
        if self._has_result and self.states is not None:
            k = self.frame_idx; n = self.states.shape[0]
            traj = self.states[:,:9].reshape(n,3,3)
            velr = self.states[:,9:].reshape(n,3,3)
            return traj[k], velr[k], self.masses_run
        try:
            return self._read()
        except (ValueError, KeyError):
            return None, None, None

    def _show_tooltip(self, text, event):
        if self._tooltip_win is None:
            self._tooltip_win = tk.Toplevel(self.root)
            self._tooltip_win.overrideredirect(True)
            self._tooltip_win.configure(bg=HOVER_BG)
            self._tooltip_label = tk.Label(
                self._tooltip_win, text='', bg=HOVER_BG, fg=TEXT,
                font=('Consolas',9), justify=tk.LEFT, padx=10, pady=6)
            self._tooltip_label.pack()
        self._tooltip_label.config(text=text)
        wx = self.root.winfo_rootx() + event.x + 18
        wy = self.root.winfo_rooty() + event.y - 90
        self._tooltip_win.geometry(f'+{wx}+{wy}')
        self._tooltip_win.deiconify(); self._tooltip_win.lift()

    def _hide_tooltip(self):
        if self._tooltip_win is not None:
            self._tooltip_win.withdraw()

    # ══════════════════════════════════════════════════════════════════════════
    # PRESETS
    # ══════════════════════════════════════════════════════════════════════════

    def _fill(self, positions, velocities, masses,
              integrator='RK45', dt='0.005', t_end='20.0', soft='0.0'):
        for b in range(3):
            self.entries[(b,'mass')].delete(0,tk.END)
            self.entries[(b,'mass')].insert(0,f'{masses[b]:.8g}')
            for c,f in enumerate(['x','y','z']):
                self.entries[(b,f)].delete(0,tk.END)
                self.entries[(b,f)].insert(0,f'{positions[b,c]:.10g}')
            for c,f in enumerate(['vx','vy','vz']):
                self.entries[(b,f)].delete(0,tk.END)
                self.entries[(b,f)].insert(0,f'{velocities[b,c]:.10g}')
        self.integrator.set(integrator); self.dt_var.set(dt)
        self.t_end_var.set(t_end); self.soft_var.set(soft)

    def _read(self):
        pos=np.zeros((3,3)); vel=np.zeros((3,3)); m=np.zeros(3)
        for b in range(3):
            m[b] = float(self.entries[(b,'mass')].get())
            for c,f in enumerate(['x','y','z']):
                pos[b,c] = float(self.entries[(b,f)].get())
            for c,f in enumerate(['vx','vy','vz']):
                vel[b,c] = float(self.entries[(b,f)].get())
        return pos, vel, m

    def _load_figure8(self):
        pos,vel,m,_ = simulator.figure_8_initial_conditions()
        self._fill(pos,vel,m,'RK45','0.005','20.0','0.0')
        self.status_var.set("Preset: Figure-8  (Chenciner & Montgomery, 2000)")
        self.root.after(100, self._live_preview)

    def _load_pythagorean(self):
        pos,vel,m = simulator.pythagorean_initial_conditions()
        self._fill(pos,vel,m,'RK45','0.001','70.0','0.0')
        self.status_var.set("Preset: Pythagorean  (Burrau, 1913)")
        self.root.after(100, self._live_preview)

    def _load_kozai(self):
        pos,vel,m = simulator.kozai_lidov_initial_conditions()
        self._fill(pos,vel,m,'RK45','0.01','500.0','0.0')
        self.status_var.set("Preset: Kozai-Lidov  ·  hierarchical triple")
        self.root.after(100, self._live_preview)

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE PREVIEW
    # ══════════════════════════════════════════════════════════════════════════

    def _live_preview(self):
        if self.running: return
        try: pos, vel, masses = self._read()
        except (ValueError, KeyError): return
        self.ax3d.cla(); self._style_3d_ax()
        for i in range(3):
            c = BODY_C[i]
            self.ax3d.scatter(*pos[i], color=c, s=120,
                               edgecolors='#CDD3DC', linewidths=0.5,
                               zorder=8,
                               label=f'Body {i}  m={masses[i]:.3g}')
            v=vel[i]; vl=np.linalg.norm(v)
            if vl > 1e-10:
                self.ax3d.quiver(*pos[i], *(v/vl*0.4), color=c,
                                  alpha=0.7, linewidth=1.2,
                                  arrow_length_ratio=0.3)
        self.ax3d.set_xlabel('x'); self.ax3d.set_ylabel('y')
        self.ax3d.set_zlabel('z')
        self.ax3d.set_title("Initial positions — preview",
                             color=TEXT_DIM, fontsize=9, pad=6)
        self.ax3d.legend(loc='upper left', fontsize=7,
                          facecolor=BG_CARD, edgecolor='none',
                          labelcolor=TEXT)
        self._apply_axis_limits()
        self.mpl_canvas.draw()
        # store dot positions AFTER draw (renderer knows where dots are)
        self.root.after(50, lambda: self._store_dot_positions(pos))

    # ══════════════════════════════════════════════════════════════════════════
    # SIMULATION
    # ══════════════════════════════════════════════════════════════════════════

    def _run(self):
        if self._computing: return
        self._cancel_after(); self.running = False; self._has_result = False
        self.run_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.status_var.set("Computing trajectory…  please wait")
        self.root.update()
        threading.Thread(target=self._compute, daemon=True).start()

    def _play(self):
        if self._has_result and not self.running:
            self._start_animation()

    def _stop_pause(self):
        self._cancel_after(); self.running = False
        if self._has_result:
            self.play_btn.config(state=tk.NORMAL)
            self.status_var.set(
                f"Paused at t = {self.times[self.frame_idx]:.3f}"
                f"  — click ▶ PLAY to resume")

    def _cancel_after(self):
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _compute(self):
        self._computing = True
        try:
            pos,vel,masses = self._read()
            dt=float(self.dt_var.get()); t_end=float(self.t_end_var.get())
            soft=float(self.soft_var.get()); integ=self.integrator.get()
            state=simulator.pack(pos,vel); t_span=(0.0,t_end)
            if   integ=='RK4':   times,states=simulator.integrate_rk4(state,masses,t_span,dt,softening=soft)
            elif integ=='Verlet': times,states=simulator.integrate_verlet(state,masses,t_span,dt,softening=soft)
            else:                 times,states=simulator.integrate_rk45(state,masses,t_span,dt_output=dt,softening=soft)
            n=states.shape[0]; energies=np.zeros(n)
            for k in range(n):
                p_k,v_k=simulator.unpack(states[k])
                E,_,_=simulator.compute_energy(p_k,v_k,masses,softening=soft)
                energies[k]=E
            E0=energies[0]
            en_err=np.abs(energies-E0)/(np.abs(E0)+1e-30)
            self.times=times; self.states=states
            self.masses_run=masses; self.en_errors=en_err
            self._has_result=True
            self.root.after(0,self._done,len(times)-1,integ,float(en_err.max()))
        except Exception as ex:
            self.root.after(0,lambda:messagebox.showerror("Simulation error",str(ex)))
            self.root.after(0,lambda:self.status_var.set(f"Error: {ex}"))
        finally:
            self._computing=False
            self.root.after(0,lambda:self.run_btn.config(state=tk.NORMAL))

    def _done(self, n_steps, integ, max_err):
        self.status_var.set(
            f"{n_steps} steps  ·  {integ}  ·  max |ΔE/E₀| = {max_err:.2e}")
        self.frame_idx=0; self.play_btn.config(state=tk.NORMAL)
        self._start_animation()

    # ══════════════════════════════════════════════════════════════════════════
    # ANIMATION LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def _start_animation(self):
        self.running=True; self._step()

    def _step(self):
        if not self.running or self.states is None: return
        n=self.states.shape[0]; k=self.frame_idx
        traj=self.states[:,:9].reshape(n,3,3)
        self.ax3d.cla(); self._style_3d_ax()
        trail_start=max(0,k-self.TRAIL)
        current_pos = traj[k]
        for i in range(3):
            c=BODY_C[i]; lbl=f'Body {i}  m={self.masses_run[i]:.3g}'
            self.ax3d.plot(traj[trail_start:k+1,i,0],
                           traj[trail_start:k+1,i,1],
                           traj[trail_start:k+1,i,2],
                           color=c,linewidth=0.9,alpha=0.70,label=lbl)
            self.ax3d.scatter(*current_pos[i], color=c, s=60,
                               edgecolors='#CDD3DC',linewidths=0.4,zorder=6)
        self.ax3d.set_xlabel('x'); self.ax3d.set_ylabel('y')
        self.ax3d.set_zlabel('z')
        self.ax3d.set_title(f't = {self.times[k]:.3f}',
                             color=TEXT,fontsize=10,pad=6)
        self.ax3d.legend(loc='upper left',fontsize=7,
                          facecolor=BG_CARD,edgecolor='none',labelcolor=TEXT)
        self._apply_axis_limits()
        if self.ax_e is not None:
            self.ax_e.cla(); self._style_energy_ax()
            errs=self.en_errors[:k+1]
            if np.any(errs>0):
                self.ax_e.semilogy(self.times[:k+1],errs+1e-20,
                                    color=ACCENT_LT,linewidth=0.9)
        self.mpl_canvas.draw()
        # store dot positions after each frame draw
        self._store_dot_positions(current_pos)
        skip=max(1,self.speed_var.get())
        self.frame_idx=(k+skip)%n
        self._after_id=self.root.after(16,self._step)


def main():
    root = tk.Tk()
    ThreeBodySimulator(root)
    root.mainloop()


if __name__ == '__main__':
    main()

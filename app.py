"""
app.py - Real-Time AI Object Detection Demo
============================================

A Tkinter desktop application that streams webcam footage through a
YOLOv8 model and displays annotated results in real time.

Installation:
    pip install ultralytics opencv-python pillow

Usage:
    python app.py
"""

import queue
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from detector import ObjectDetector

# ── Layout constants ───────────────────────────────────────────────────────────
_VIDEO_W = 800          # Default display width  (pixels)
_VIDEO_H = 600          # Default display height (pixels)
_CTRL_W  = 230          # Fixed width of the right-hand control panel
_POLL_MS = 15           # GUI frame-poll interval  (~67 fps upper-bound)
_Q_SIZE  = 2            # Keep queue tiny to minimise display latency


# ── Colour scheme (Catppuccin-inspired dark palette) ──────────────────────────
_C = {
    "bg":       "#1e1e2e",
    "surface":  "#181825",
    "overlay":  "#313244",
    "text":     "#cdd6f4",
    "subtext":  "#6c7086",
    "green":    "#a6e3a1",
    "red":      "#f38ba8",
    "blue":     "#89b4fa",
    "teal":     "#89dceb",
    "yellow":   "#f9e2af",
}


class App:
    """
    Main application window.

    Threading model
    ---------------
    - **Main thread**   : Tkinter event loop + ``_consume_frames()`` polling.
    - **Capture thread**: ``_capture_loop()`` — reads webcam, runs YOLO,
                          pushes (rgb_frame, detections, fps) into a small Queue.
    The Queue has ``maxsize=_Q_SIZE`` so the capture thread automatically drops
    stale frames when the GUI can't keep up.
    """

    MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Real-Time AI Object Detection Demo")
        self.root.configure(bg=_C["bg"])
        self.root.minsize(900, 640)

        # Application state
        self._running = False
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._queue: queue.Queue = queue.Queue(maxsize=_Q_SIZE)
        self._current_frame: np.ndarray | None = None  # latest annotated BGR frame (for screenshots)

        self._detector = ObjectDetector()

        self._build_ui()
        self._schedule_poll()       # start the GUI frame consumer

    # ══════════════════════════════════════════════════════════════════════════
    # UI construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self) -> None:
        """Construct every widget once at start-up."""

        # ── Title bar ──────────────────────────────────────────────────────────
        tk.Label(
            self.root,
            text="Real-Time AI Object Detection Demo",
            font=("Helvetica", 17, "bold"),
            fg=_C["text"], bg=_C["bg"], pady=10,
        ).pack(fill=tk.X)

        # ── Body: video | controls ─────────────────────────────────────────────
        body = tk.Frame(self.root, bg=_C["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        self._build_video_panel(body)
        self._build_control_panel(body)

        # ── Status bar ─────────────────────────────────────────────────────────
        self._statusbar = tk.Label(
            self.root,
            text="Ready — select a model and press Start Camera.",
            anchor=tk.W, padx=8,
            font=("Helvetica", 9),
            fg=_C["subtext"], bg=_C["surface"],
        )
        self._statusbar.pack(fill=tk.X, side=tk.BOTTOM)

    # ── Video panel ────────────────────────────────────────────────────────────

    def _build_video_panel(self, parent: tk.Frame) -> None:
        container = tk.Frame(parent, bg=_C["overlay"], bd=2, relief=tk.SUNKEN)
        container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._video_lbl = tk.Label(
            container,
            text="Camera feed will appear here",
            font=("Helvetica", 13),
            fg=_C["subtext"], bg="#000000",
            width=_VIDEO_W, height=_VIDEO_H,
        )
        self._video_lbl.pack(fill=tk.BOTH, expand=True)

    # ── Control panel ──────────────────────────────────────────────────────────

    def _build_control_panel(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=_C["surface"], width=_CTRL_W)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        panel.pack_propagate(False)

        self._build_status_section(panel)
        self._build_model_section(panel)
        self._build_conf_section(panel)
        self._build_button_section(panel)
        self._build_log_section(panel)

    def _build_status_section(self, parent: tk.Frame) -> None:
        frm = self._labeled_frame(parent, "Status")
        frm.pack(fill=tk.X, padx=8, pady=(12, 5))

        self._status_dot = tk.Label(frm, text="●", font=("Helvetica", 20),
                                    fg=_C["red"], bg=_C["surface"])
        self._status_dot.pack()

        self._status_txt = tk.Label(frm, text="Stopped",
                                    font=("Helvetica", 11, "bold"),
                                    fg=_C["red"], bg=_C["surface"])
        self._status_txt.pack()

        self._fps_lbl = tk.Label(frm, text="FPS: --",
                                 font=("Courier", 10), fg=_C["green"], bg=_C["surface"])
        self._fps_lbl.pack(pady=(3, 0))

        self._obj_lbl = tk.Label(frm, text="Objects: 0",
                                 font=("Courier", 10), fg=_C["teal"], bg=_C["surface"])
        self._obj_lbl.pack()

    def _build_model_section(self, parent: tk.Frame) -> None:
        frm = self._labeled_frame(parent, "Model")
        frm.pack(fill=tk.X, padx=8, pady=5)

        # Style the combobox for the dark theme
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                         fieldbackground=_C["overlay"],
                         background=_C["overlay"],
                         foreground=_C["text"],
                         selectbackground=_C["overlay"],
                         selectforeground=_C["text"],
                         arrowcolor=_C["text"])
        style.map("TCombobox", fieldbackground=[("readonly", _C["overlay"])])

        self._model_var = tk.StringVar(value="yolov8n")
        ttk.Combobox(frm, textvariable=self._model_var,
                     values=self.MODELS, state="readonly", width=17).pack()

        tk.Label(frm, text="n=fastest  x=most accurate",
                 font=("Helvetica", 7), fg=_C["subtext"], bg=_C["surface"]).pack()

    def _build_conf_section(self, parent: tk.Frame) -> None:
        frm = self._labeled_frame(parent, "Confidence Threshold")
        frm.pack(fill=tk.X, padx=8, pady=5)

        self._conf_lbl = tk.Label(frm, text="0.50",
                                  font=("Courier", 10), fg=_C["text"], bg=_C["surface"])
        self._conf_lbl.pack()

        self._conf_var = tk.DoubleVar(value=0.50)
        tk.Scale(
            frm,
            from_=0.05, to=0.95, resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self._conf_var,
            command=self._on_conf_change,
            bg=_C["surface"], fg=_C["text"],
            troughcolor=_C["overlay"],
            activebackground=_C["blue"],
            highlightthickness=0, bd=0,
            length=200,
        ).pack()

    def _build_button_section(self, parent: tk.Frame) -> None:
        frm = self._labeled_frame(parent, "Controls")
        frm.pack(fill=tk.X, padx=8, pady=5)

        _btn: dict[str, Any] = dict(font=("Helvetica", 11, "bold"), bd=0,
                                   relief=tk.FLAT, cursor="hand2", width=19, pady=7)

        self._start_btn = tk.Button(
            frm, text="  Start Camera",
            bg=_C["green"], fg=_C["surface"],
            command=self.start_camera, **_btn)
        self._start_btn.pack(pady=(2, 5))

        self._stop_btn = tk.Button(
            frm, text="  Stop Camera",
            bg=_C["red"], fg=_C["surface"],
            command=self.stop_camera, state=tk.DISABLED, **_btn)
        self._stop_btn.pack(pady=(0, 5))

        self._shot_btn = tk.Button(
            frm, text="  Save Screenshot",
            bg=_C["blue"], fg=_C["surface"],
            command=self.take_screenshot, state=tk.DISABLED, **_btn)
        self._shot_btn.pack()

    def _build_log_section(self, parent: tk.Frame) -> None:
        frm = self._labeled_frame(parent, "Detection Log")
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=(5, 8))

        sb = tk.Scrollbar(frm)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._log = tk.Text(
            frm,
            bg=_C["overlay"], fg=_C["text"],
            font=("Courier", 8), insertbackground=_C["text"],
            yscrollcommand=sb.set,
            state=tk.DISABLED, wrap=tk.WORD, relief=tk.FLAT,
        )
        self._log.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self._log.yview)

    # ══════════════════════════════════════════════════════════════════════════
    # Event handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_conf_change(self, value: str) -> None:
        threshold = float(value)
        self._conf_lbl.config(text=f"{threshold:.2f}")
        self._detector.set_conf_threshold(threshold)

    # ══════════════════════════════════════════════════════════════════════════
    # Camera lifecycle
    # ══════════════════════════════════════════════════════════════════════════

    def start_camera(self) -> None:
        """Load the selected model, open the webcam, and begin detection."""
        if self._running:
            return

        model_name = self._model_var.get()
        self._set_status(text=f"Loading {model_name}…", color=_C["yellow"])
        self._statusbar.config(text=f"Downloading / loading {model_name} — please wait…")
        self.root.update_idletasks()

        # Load model (may download .pt on first use)
        try:
            self._detector.load_model(model_name)
        except Exception as exc:
            messagebox.showerror("Model Error", f"Could not load model:\n{exc}")
            self._set_status(text="Stopped", color=_C["red"])
            self._statusbar.config(text="Model load failed.")
            return

        # Open webcam
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Cannot open webcam (device index 0).\n"
                                 "Make sure no other application is using it.")
            self._cap = None
            self._set_status(text="Stopped", color=_C["red"])
            return

        # Request a sensible resolution; the driver may not honour it exactly
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _VIDEO_W)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _VIDEO_H)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # flush stale frames fast

        self._running = True
        self._set_status(text="Camera Running", color=_C["green"])
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._shot_btn.config(state=tk.NORMAL)
        self._statusbar.config(text=f"Running  |  model: {model_name}")

        # Spawn background worker
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_camera(self) -> None:
        """Signal the capture thread to stop and release the webcam."""
        self._running = False

        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

        if self._cap:
            self._cap.release()
            self._cap = None

        # Reset video panel to placeholder
        self._video_lbl.config(
            image="",
            text="Camera feed will appear here",
            fg=_C["subtext"], font=("Helvetica", 13),
        )

        self._set_status(text="Stopped", color=_C["red"])
        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._shot_btn.config(state=tk.DISABLED)
        self._fps_lbl.config(text="FPS: --")
        self._obj_lbl.config(text="Objects: 0")
        self._statusbar.config(text="Camera stopped.")

    def take_screenshot(self) -> None:
        """Write the latest annotated frame to a PNG file."""
        if self._current_frame is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"detection_{ts}.png"
        try:
            cv2.imwrite(fname, self._current_frame)
            self._statusbar.config(text=f"Screenshot saved: {fname}")
        except Exception as exc:
            messagebox.showerror("Screenshot Error", f"Could not save file:\n{exc}")

    # ══════════════════════════════════════════════════════════════════════════
    # Background capture thread
    # ══════════════════════════════════════════════════════════════════════════

    def _capture_loop(self) -> None:
        """
        Runs entirely in the capture thread.

        Reads frames from the webcam, runs YOLO inference, and enqueues
        ``(rgb_frame, detections, fps)`` tuples for the GUI to consume.
        Old frames are dropped silently when the queue is full so the
        display stays as fresh as the hardware allows.
        """
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break

            ok, frame = self._cap.read()
            if not ok:
                break

            # Detection happens here, off the main thread
            annotated, detections, fps = self._detector.detect(frame)

            # Keep a BGR copy for the screenshot feature
            self._current_frame = annotated

            # BGR → RGB for PIL/Tkinter
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Resize to the display panel dimensions
            rgb = cv2.resize(rgb, (_VIDEO_W, _VIDEO_H), interpolation=cv2.INTER_LINEAR)

            # Non-blocking enqueue — drop stale frame if queue is full
            try:
                self._queue.put_nowait((rgb, detections, fps))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait((rgb, detections, fps))
                except queue.Full:
                    pass  # give up for this frame

    # ══════════════════════════════════════════════════════════════════════════
    # GUI frame consumer (runs on main thread via after())
    # ══════════════════════════════════════════════════════════════════════════

    def _schedule_poll(self) -> None:
        """Kick off the recurring GUI update loop."""
        self._consume_frames()

    def _consume_frames(self) -> None:
        """
        Called every ``_POLL_MS`` ms on the main thread.

        Drains the latest frame from the queue, converts it to a
        ``PhotoImage``, and updates the video label plus stats widgets.
        """
        try:
            rgb_frame, detections, fps = self._queue.get_nowait()

            img   = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=img)

            # PhotoImage must be held by a Python variable; otherwise the
            # garbage collector deletes it before Tkinter can render it.
            self._video_lbl.config(image=photo, text="")
            self._video_lbl._photo_ref = photo  # type: ignore[attr-defined]

            self._fps_lbl.config(text=f"FPS: {fps:5.1f}")
            self._obj_lbl.config(text=f"Objects: {len(detections)}")

            if detections:
                self._append_log(detections)

        except queue.Empty:
            pass  # Nothing new this tick — that's fine

        # Reschedule unconditionally so the loop never stops
        self.root.after(_POLL_MS, self._consume_frames)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _set_status(self, *, text: str, color: str) -> None:
        self._status_dot.config(fg=color)
        self._status_txt.config(text=text, fg=color)

    def _append_log(self, detections: list[tuple]) -> None:
        """Add the current detection batch to the scrollable log."""
        ts    = datetime.now().strftime("%H:%M:%S")
        lines = [f"[{ts}]"]
        for label, conf, _ in detections:
            lines.append(f"  {label}: {conf:.2%}")
        entry = "\n".join(lines) + "\n"

        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, entry)
        self._log.see(tk.END)

        # Prune old entries to prevent unbounded memory growth
        if int(self._log.index("end-1c").split(".")[0]) > 300:
            self._log.delete("1.0", "100.0")

        self._log.config(state=tk.DISABLED)

    @staticmethod
    def _labeled_frame(parent: tk.Widget, title: str) -> tk.LabelFrame:
        return tk.LabelFrame(
            parent, text=title,
            bg=_C["surface"], fg=_C["text"],
            font=("Helvetica", 9, "bold"),
            padx=6, pady=6,
        )

    def on_close(self) -> None:
        """Clean-up handler wired to the WM_DELETE_WINDOW protocol."""
        self.stop_camera()
        self.root.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    application = App(root)
    root.protocol("WM_DELETE_WINDOW", application.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

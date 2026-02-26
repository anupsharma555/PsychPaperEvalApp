from __future__ import annotations

import tkinter as tk

from desktop_legacy.models import BackendStatus

PANEL_BG = "#ffffff"
PANEL_BORDER = "#c7d1dc"
HEADER_BG = "#dbe7f5"


class StatusBar:
    def __init__(self, parent: tk.Widget) -> None:
        self.frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=PANEL_BORDER)
        tk.Label(
            self.frame,
            text="System Status",
            bg=HEADER_BG,
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(fill=tk.X)

        row = tk.Frame(self.frame, bg=PANEL_BG, padx=8, pady=8)
        row.pack(fill=tk.X)
        row.grid_columnconfigure(0, weight=1, uniform="status")
        row.grid_columnconfigure(1, weight=1, uniform="status")
        row.grid_columnconfigure(2, weight=1, uniform="status")
        row.grid_columnconfigure(3, weight=1, uniform="status")

        self.backend_label = self._make_badge(row, 0, "Backend: Starting")
        self.model_label = self._make_badge(row, 1, "Models: Checking")
        self.processing_label = self._make_badge(row, 2, "Processing: Unknown")
        self.inflight_label = self._make_badge(row, 3, "Inflight: 0")

    def _make_badge(self, row: tk.Frame, column: int, text: str) -> tk.Label:
        container = tk.Frame(row, bg="#f3f7fc", bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground="#d7e0ea", padx=8, pady=6)
        container.grid(row=0, column=column, sticky="nsew", padx=4)
        label = tk.Label(container, text=text, anchor=tk.W, justify=tk.LEFT, bg="#f3f7fc", fg="#1f2933")
        label.pack(fill=tk.X)
        return label

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def set_backend_unavailable(self) -> None:
        self.backend_label.config(text="Backend: Unavailable")
        self.processing_label.config(text="Processing: Unknown")
        self.inflight_label.config(text="Inflight: 0")

    def update(self, status: BackendStatus) -> None:
        backend_text = "Ready" if status.backend_ready else "Unavailable"
        self.backend_label.config(text=f"Backend: {backend_text}")
        models_text = "Ready" if status.model_exists and status.mmproj_exists else "Missing"
        self.model_label.config(text=f"Models: {models_text}")
        if status.processing_paused:
            proc = "Paused"
        elif status.processing_running:
            proc = "Running"
        else:
            proc = "Stopped"
        self.processing_label.config(text=f"Processing: {proc}")
        self.inflight_label.config(text=f"Inflight: {status.inflight}/{max(status.worker_capacity, 1)}")

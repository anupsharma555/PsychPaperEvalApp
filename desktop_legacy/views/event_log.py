from __future__ import annotations

import time
import tkinter as tk

PANEL_BG = "#ffffff"
PANEL_BORDER = "#c7d1dc"
HEADER_BG = "#dbe7f5"


class EventLogBar:
    def __init__(self, parent: tk.Widget) -> None:
        self.frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=PANEL_BORDER)
        tk.Label(
            self.frame,
            text="Event Log",
            bg=HEADER_BG,
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=6,
            font=("Helvetica", 11, "bold"),
        ).pack(fill=tk.X)
        self.var = tk.StringVar(value="Ready.")
        self.label = tk.Label(
            self.frame,
            textvariable=self.var,
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=1800,
            bg=PANEL_BG,
            fg="#1f2933",
            padx=10,
            pady=7,
        )
        self.label.pack(fill=tk.X)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def push(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.var.set(f"[{ts}] {message}")

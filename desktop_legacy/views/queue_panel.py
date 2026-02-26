from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

from desktop_legacy.models import JobRow

PANEL_BG = "#ffffff"
PANEL_BORDER = "#c7d1dc"
HEADER_BG = "#dbe7f5"


class QueuePanel:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_select_job: Callable[[Optional[int]], None],
    ) -> None:
        self.on_select_job = on_select_job
        self.frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=PANEL_BORDER)
        tk.Label(
            self.frame,
            text="Active Work",
            bg=HEADER_BG,
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=7,
            font=("Helvetica", 12, "bold"),
        ).pack(fill=tk.X)

        body = tk.Frame(self.frame, bg=PANEL_BG, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(1, weight=3)
        body.grid_rowconfigure(3, weight=2)

        tk.Label(
            body,
            text="Queue (sorted by most recently updated)",
            anchor=tk.W,
            bg=PANEL_BG,
            fg="#1f2933",
        ).grid(row=0, column=0, sticky="ew")

        queue_frame = tk.Frame(body, bd=1, relief=tk.SOLID, bg="#ffffff")
        queue_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        queue_frame.grid_columnconfigure(0, weight=1)
        queue_frame.grid_rowconfigure(0, weight=1)
        self.listbox = tk.Listbox(
            queue_frame,
            selectbackground="#c7dbea",
            relief=tk.FLAT,
            borderwidth=0,
            bg="#ffffff",
            activestyle="none",
        )
        self.listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = tk.Scrollbar(queue_frame, orient=tk.VERTICAL, command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        tk.Label(body, text="Selected Job Details", anchor=tk.W, bg=PANEL_BG, fg="#1f2933").grid(row=2, column=0, sticky="ew")
        self.detail = tk.Text(body, height=8, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, bg="#ffffff")
        self.detail.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        self.detail.configure(state=tk.DISABLED)

        self._items: list[JobRow] = []

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def update_jobs(self, rows: list[JobRow]) -> None:
        self._items = rows
        self.listbox.delete(0, tk.END)
        for row in rows:
            text = (
                f"Job {row.job_id} | Doc {row.document_id} | {row.source_kind.upper()} | "
                f"{row.status} | {int(row.progress * 100)}%"
            )
            self.listbox.insert(tk.END, text)

    def set_details(self, text: str) -> None:
        self.detail.configure(state=tk.NORMAL)
        self.detail.delete("1.0", tk.END)
        self.detail.insert("1.0", text)
        self.detail.configure(state=tk.DISABLED)

    def _on_select(self, _event) -> None:
        selected = self.listbox.curselection()
        if not selected:
            self.on_select_job(None)
            return
        idx = selected[0]
        if idx < 0 or idx >= len(self._items):
            self.on_select_job(None)
            return
        self.on_select_job(self._items[idx].job_id)

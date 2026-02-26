from __future__ import annotations

import json
import tkinter as tk
from tkinter import messagebox
import webbrowser
from typing import Callable, Optional

from desktop_legacy.models import DesktopReport

PANEL_BG = "#ffffff"
PANEL_BORDER = "#c7d1dc"
HEADER_BG = "#dbe7f5"


class ReportPanel:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_load_latest: Callable[[], None],
        on_export: Callable[[], Optional[str]],
    ) -> None:
        self.on_load_latest = on_load_latest
        self.on_export = on_export

        self.frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=PANEL_BORDER)
        tk.Label(
            self.frame,
            text="Report",
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
        body.grid_rowconfigure(3, weight=1)

        tk.Label(
            body,
            text="Review modality summaries and discrepancies for the selected completed job.",
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=520,
            fg="#1f2933",
            bg=PANEL_BG,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))

        button_row = tk.Frame(body, bg=PANEL_BG)
        button_row.grid(row=1, column=0, sticky="ew")
        self.load_btn = tk.Button(button_row, text="Load Latest Report", command=self.on_load_latest, state=tk.DISABLED)
        self.load_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.export_btn = tk.Button(button_row, text="Export Report JSON", command=self._handle_export, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT)

        self.badge_var = tk.StringVar(value="Report: Not Ready")
        tk.Label(body, textvariable=self.badge_var, anchor=tk.W, fg="#0d47a1", bg=PANEL_BG).grid(row=2, column=0, sticky="ew", pady=(8, 4))

        self.text = tk.Text(body, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1, bg="#ffffff")
        self.text.grid(row=3, column=0, sticky="nsew")
        self.text.configure(state=tk.DISABLED)

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def set_load_enabled(self, enabled: bool) -> None:
        self.load_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_report_status(self, status: str) -> None:
        self.badge_var.set(f"Report: {status}")

    def render_summary(self, report: DesktopReport, full_payload: Optional[dict] = None) -> None:
        lines = []
        summary_json = None
        if full_payload is not None:
            summary_json = full_payload.get("summary_json")
            if summary_json is None and full_payload.get("summary"):
                try:
                    summary_json = json.loads(full_payload.get("summary") or "{}")
                except Exception:
                    summary_json = None
        lines.append(f"Summary Version: {report.summary_version}")
        lines.append(f"Status: {report.report_status}")
        if report.overall_confidence is not None:
            lines.append(f"Overall Confidence: {int(float(report.overall_confidence) * 100)}%")
        lines.append("")
        if report.executive_summary:
            lines.append("Executive Summary")
            lines.append(report.executive_summary)
            lines.append("")
        if isinstance(summary_json, dict):
            strengths = summary_json.get("methods_strengths") or summary_json.get("strengths") or []
            weaknesses = summary_json.get("methods_weaknesses") or summary_json.get("weaknesses") or []
            if isinstance(strengths, list) and strengths:
                lines.append("Strengths")
                for item in strengths[:8]:
                    lines.append(f"  • {item}")
                lines.append("")
            if isinstance(weaknesses, list) and weaknesses:
                lines.append("Weaknesses")
                for item in weaknesses[:8]:
                    lines.append(f"  • {item}")
                lines.append("")
        lines.append("Modality Cards")
        for card in report.modality_cards:
            lines.append(f"- {card.modality.upper()}: {card.finding_count} findings")
            for item in card.highlights[:3]:
                lines.append(f"  • {item}")
            for gap in card.coverage_gaps[:2]:
                lines.append(f"  ! {gap}")
        lines.append("")
        lines.append(f"Discrepancies: {report.discrepancy_count}")
        if full_payload is not None:
            lines.append("")
            lines.append("Raw Summary JSON (truncated)")
            summary = summary_json
            if summary is None and full_payload.get("summary"):
                summary = {"raw_summary": full_payload.get("summary")}
            if summary is not None:
                lines.append(json.dumps(summary, indent=2)[:4000])
        self._set_text("\n".join(lines))

    def clear(self) -> None:
        self._set_text("Report not loaded.")

    def _set_text(self, content: str) -> None:
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.text.configure(state=tk.DISABLED)

    def _handle_export(self) -> None:
        url = self.on_export()
        if not url:
            messagebox.showerror("Export unavailable", "Report export URL is not available.")
            return
        webbrowser.open(url, new=2)

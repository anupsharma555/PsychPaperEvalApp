from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from typing import Callable

from desktop_legacy.workflow import WorkflowStep

PANEL_BG = "#ffffff"
PANEL_BORDER = "#c7d1dc"
HEADER_BG = "#dbe7f5"
SECTION_BG = "#f6f9fd"
TEXT_MAIN = "#1f2933"
TEXT_MUTED = "#455a64"


class SourceStepView:
    def __init__(
        self,
        parent: tk.Widget,
        *,
        on_validate: Callable[[], None],
        on_submit: Callable[[], None],
    ) -> None:
        self.on_validate = on_validate
        self.on_submit = on_submit

        self.frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground=PANEL_BORDER)
        tk.Label(
            self.frame,
            text="Workflow",
            bg=HEADER_BG,
            fg="#12304a",
            anchor=tk.W,
            padx=10,
            pady=7,
            font=("Helvetica", 12, "bold"),
        ).pack(fill=tk.X)

        body = tk.Frame(self.frame, bg=PANEL_BG, padx=10, pady=10)
        body.pack(fill=tk.BOTH, expand=True)

        self.controls_enabled = True
        self.submit_enabled = False

        self.next_step_var = tk.StringVar(
            value="Step 1: choose source type and provide URL/DOI or upload a main PDF."
        )
        guidance = tk.Label(
            body,
            textvariable=self.next_step_var,
            justify=tk.LEFT,
            anchor=tk.W,
            wraplength=340,
            fg=TEXT_MAIN,
            bg="#edf4fb",
            bd=1,
            relief=tk.SOLID,
            padx=8,
            pady=6,
        )
        guidance.pack(fill=tk.X, pady=(0, 10))

        progress = self._make_section(body, "Progress", fill=False)
        self.step_labels: dict[int, tk.Label] = {}
        for step, name in [
            (WorkflowStep.SELECT_SOURCE, "1. Select Source"),
            (WorkflowStep.VALIDATE_INPUT, "2. Validate Input"),
            (WorkflowStep.SUBMIT_JOB, "3. Submit Job"),
            (WorkflowStep.MONITOR, "4. Monitor"),
            (WorkflowStep.REVIEW, "5. Review Report"),
        ]:
            row = tk.Label(progress, text=f"{name} - Pending", anchor=tk.W, fg=TEXT_MAIN, bg=SECTION_BG)
            row.pack(fill=tk.X, pady=(0, 3))
            self.step_labels[step] = row

        source = self._make_section(body, "Step 1: Source Input", fill=False)
        self.source_mode = tk.StringVar(value="url")
        tk.Label(source, text="Source Type", fg=TEXT_MAIN, bg=SECTION_BG, anchor=tk.W).pack(fill=tk.X)
        self.url_radio = tk.Radiobutton(
            source,
            text="From URL / DOI",
            value="url",
            variable=self.source_mode,
            anchor=tk.W,
            highlightthickness=0,
            command=self._on_mode_changed,
            bg=SECTION_BG,
            fg=TEXT_MAIN,
            activebackground=SECTION_BG,
            selectcolor=PANEL_BG,
        )
        self.url_radio.pack(fill=tk.X)
        self.upload_radio = tk.Radiobutton(
            source,
            text="From Upload",
            value="upload",
            variable=self.source_mode,
            anchor=tk.W,
            highlightthickness=0,
            command=self._on_mode_changed,
            bg=SECTION_BG,
            fg=TEXT_MAIN,
            activebackground=SECTION_BG,
            selectcolor=PANEL_BG,
        )
        self.upload_radio.pack(fill=tk.X)

        tk.Label(source, text="URL", anchor=tk.W, bg=SECTION_BG, fg=TEXT_MAIN).pack(fill=tk.X, pady=(6, 0))
        self.url_entry = tk.Entry(source, relief=tk.SOLID, bd=1)
        self.url_entry.pack(fill=tk.X, pady=(0, 4))

        tk.Label(source, text="DOI", anchor=tk.W, bg=SECTION_BG, fg=TEXT_MAIN).pack(fill=tk.X)
        self.doi_entry = tk.Entry(source, relief=tk.SOLID, bd=1)
        self.doi_entry.pack(fill=tk.X, pady=(0, 8))

        self.main_file_var = tk.StringVar(value="Main PDF: not selected")
        tk.Label(source, textvariable=self.main_file_var, anchor=tk.W, fg=TEXT_MUTED, bg=SECTION_BG).pack(fill=tk.X)
        self.main_btn = tk.Button(source, text="Choose Main PDF", command=self._choose_main_file)
        self.main_btn.pack(fill=tk.X, pady=(3, 3))

        self.supp_file_var = tk.StringVar(value="Supplements: none")
        tk.Label(source, textvariable=self.supp_file_var, anchor=tk.W, fg=TEXT_MUTED, bg=SECTION_BG).pack(fill=tk.X)
        self.supp_btn = tk.Button(source, text="Choose Supplements", command=self._choose_supp_files)
        self.supp_btn.pack(fill=tk.X, pady=(3, 0))

        validate = self._make_section(body, "Step 2: Validate Input", fill=False)
        self.validation_var = tk.StringVar(value="Click Validate after selecting your source input.")
        self.validation_label = tk.Label(
            validate,
            textvariable=self.validation_var,
            fg=TEXT_MUTED,
            anchor=tk.W,
            wraplength=320,
            justify=tk.LEFT,
            bg=SECTION_BG,
        )
        self.validation_label.pack(fill=tk.X, pady=(0, 8))
        self.validate_btn = tk.Button(validate, text="Validate", command=self.on_validate)
        self.validate_btn.pack(fill=tk.X)

        submit = self._make_section(body, "Step 3: Submit", fill=False)
        self.submit_hint_var = tk.StringVar(value="Submit Analysis is enabled after validation succeeds.")
        self.submit_hint_label = tk.Label(
            submit,
            textvariable=self.submit_hint_var,
            fg=TEXT_MUTED,
            anchor=tk.W,
            wraplength=320,
            justify=tk.LEFT,
            bg=SECTION_BG,
        )
        self.submit_hint_label.pack(fill=tk.X, pady=(0, 8))
        self.submit_btn = tk.Button(submit, text="Submit Analysis", command=self.on_submit, state=tk.DISABLED)
        self.submit_btn.pack(fill=tk.X)

        self.main_file: Path | None = None
        self.supp_files: list[Path] = []
        self._refresh_mode_state()

    def _make_section(self, parent: tk.Widget, title: str, *, fill: bool) -> tk.Frame:
        outer = tk.Frame(parent, bg=SECTION_BG, bd=1, relief=tk.SOLID, highlightthickness=1, highlightbackground="#d8e2ed")
        outer.pack(fill=tk.BOTH if fill else tk.X, expand=fill, pady=(0, 10))
        tk.Label(
            outer,
            text=title,
            bg="#eaf1f8",
            fg="#243b53",
            anchor=tk.W,
            padx=8,
            pady=4,
            font=("Helvetica", 10, "bold"),
        ).pack(fill=tk.X)
        body = tk.Frame(outer, bg=SECTION_BG, padx=8, pady=8)
        body.pack(fill=tk.BOTH, expand=True)
        return body

    def pack(self, **kwargs) -> None:
        self.frame.pack(**kwargs)

    def _choose_main_file(self) -> None:
        path = filedialog.askopenfilename(title="Choose main PDF", filetypes=[("PDF", "*.pdf"), ("All files", "*.*")])
        if not path:
            return
        self.main_file = Path(path)
        self.main_file_var.set(f"Main PDF: {self.main_file.name}")
        self.set_next_step_hint("Step 2: click Validate to check your upload inputs.")

    def _choose_supp_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Choose supplement files",
            filetypes=[("All supported", "*.pdf *.zip *.csv *.tsv *.xlsx *.xls *.txt *.png *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")],
        )
        if not files:
            self.supp_files = []
            self.supp_file_var.set("Supplements: none")
            return
        self.supp_files = [Path(item) for item in files]
        self.supp_file_var.set(f"Supplements: {len(self.supp_files)} selected")
        self.set_next_step_hint("Supplements selected. Next: click Validate.")

    def get_source_payload(self) -> dict:
        return {
            "source_mode": self.source_mode.get(),
            "url": self.url_entry.get().strip(),
            "doi": self.doi_entry.get().strip(),
            "main_file": str(self.main_file) if self.main_file else None,
            "supp_files": [str(item) for item in self.supp_files],
        }

    def set_validation_message(self, message: str, *, is_error: bool) -> None:
        self.validation_var.set(message)
        self.validation_label.config(fg="#8b0000" if is_error else "#1b5e20")
        if is_error:
            self.submit_hint_var.set("Fix validation issues before submitting.")
        else:
            self.submit_hint_var.set("Validation passed. Continue with Submit Analysis.")

    def set_submit_enabled(self, enabled: bool) -> None:
        self.submit_enabled = enabled
        self._refresh_mode_state()

    def set_controls_enabled(self, enabled: bool) -> None:
        self.controls_enabled = enabled
        self._refresh_mode_state()

    def set_step_states(self, current_step: int) -> None:
        for step, label in self.step_labels.items():
            if current_step > step:
                state = "Done"
                color = "#1b5e20"
            elif current_step == step:
                state = "Active"
                color = "#0d47a1"
            else:
                state = "Pending"
                color = TEXT_MAIN
            base = label.cget("text").split(" - ")[0]
            label.config(text=f"{base} - {state}", fg=color)

    def set_next_step_hint(self, text: str) -> None:
        self.next_step_var.set(text)

    def _on_mode_changed(self) -> None:
        self._refresh_mode_state()
        if self.source_mode.get() == "upload":
            self.set_next_step_hint("Step 1: choose Main PDF (required). Then click Validate.")
        else:
            self.set_next_step_hint("Step 1: enter URL or DOI, then click Validate.")

    def _refresh_mode_state(self) -> None:
        state = tk.NORMAL if self.controls_enabled else tk.DISABLED
        self.url_radio.config(state=state)
        self.upload_radio.config(state=state)
        self.main_btn.config(state=state)
        self.supp_btn.config(state=state)
        self.validate_btn.config(state=state)

        if self.source_mode.get() == "url" and self.controls_enabled:
            self.url_entry.config(state=tk.NORMAL)
            self.doi_entry.config(state=tk.NORMAL)
        else:
            self.url_entry.config(state=tk.DISABLED)
            self.doi_entry.config(state=tk.DISABLED)

        submit_state = tk.NORMAL if self.controls_enabled and self.submit_enabled else tk.DISABLED
        self.submit_btn.config(state=submit_state)

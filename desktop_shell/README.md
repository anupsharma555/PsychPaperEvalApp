# PaperEval Desktop Shell (Tauri)

This is the primary desktop shell for PaperEval V7.

## Build

1. `make desktop-env`
2. `make desktop-build`
3. Launch `/Users/anup/gitProjects/language-models-psychiatry/PsychPaperEvalApp/PaperEval.app`

## Notes

- The shell supervises the backend sidecar and exposes runtime commands to the web UI.
- The old Tk launcher is deprecated and moved under `desktop_legacy/`.

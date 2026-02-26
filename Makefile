ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
APP_NAME ?= PaperEval
DESKTOP_UI := $(ROOT)/desktop_ui
DESKTOP_SHELL := $(ROOT)/desktop_shell

.PHONY: desktop-env desktop-build desktop-smoke desktop-clean

desktop-env:
	@command -v node >/dev/null || (echo "Node.js is required" && exit 1)
	@command -v npm >/dev/null || (echo "npm is required" && exit 1)
	@command -v cargo >/dev/null || (echo "cargo is required (install rustup)" && exit 1)
	@command -v rustc >/dev/null || (echo "rustc is required (install rustup)" && exit 1)
	npm --prefix $(DESKTOP_UI) install
	npm --prefix $(DESKTOP_SHELL) install

desktop-build: desktop-env
	python3 $(ROOT)/scripts/build_macos_app.py --name $(APP_NAME) --root $(ROOT)

desktop-smoke:
	python3 $(ROOT)/scripts/desktop_smoke.py --root $(ROOT) --app-name $(APP_NAME)

desktop-clean:
	rm -rf $(DESKTOP_UI)/dist
	rm -rf $(DESKTOP_SHELL)/src-tauri/target
	rm -rf $(ROOT)/dist/$(APP_NAME).app

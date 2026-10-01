# Top-level entry points. The real work is in scripts/ and in the Cargo
# workspace under host/. Run `make help` for the list.

.DEFAULT_GOAL := help
HOST := host
# phi-pld links a llama.cpp build (headers here, libraries in
# build-native/bin) and needs libclang; it joins the build where the
# headers are found, and is skipped with a note where not.
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp
export LLAMA_CPP_DIR
PLD := $(if $(wildcard $(LLAMA_CPP_DIR)/include/llama.h),--workspace,)
PLD_NOTE = $(if $(PLD),,@echo "phi-pld and phi-stream skipped: no $(LLAMA_CPP_DIR)/include/llama.h (set LLAMA_CPP_DIR)")

.PHONY: help build test fmt clippy docs-check layout-check mvex-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

build: ## Build the host workspace (libphi512.so, phi-vpu, the translator, the encoder; phi-pld and phi-stream where llama.cpp is found) and libggml_phi.so from host/asm
	cd $(HOST) && cargo build $(PLD) && cargo build --release $(PLD)
	$(PLD_NOTE)
	host/asm/ggml-phi/build.sh --install

test: ## Run host tests that do not need the card
	cd $(HOST) && cargo test $(PLD)

fmt: ## Check formatting
	cd $(HOST) && cargo fmt --all -- --check

clippy: ## Lint
	cd $(HOST) && cargo clippy $(PLD) --all-targets -- -D warnings

docs-check: ## Enforce sibling .md files, the no-dash rule and relative links
	scripts/check-docs.sh

layout-check: ## Compare the C protocol layout with the Rust and assembly constants, and ggml's headers with ggml_layout.inc (tcc; the ggml check is skipped without the headers)
	tcc -run tools/vpu-layout-check.c
	@if [ -f $(LLAMA_CPP_DIR)/ggml/include/ggml.h ]; then \
		tcc -I$(LLAMA_CPP_DIR)/ggml/include -I$(LLAMA_CPP_DIR)/ggml/src -run tools/ggml-layout-check.c host/asm/ggml-phi/ggml_layout.inc; \
	else echo "ggml layout check skipped: no $(LLAMA_CPP_DIR)/ggml/include/ggml.h (set LLAMA_CPP_DIR)"; fi

mvex-check: ## The knc-mvex copy matches the stack's (skipped when the stack is not found)
	scripts/mvex-sync.sh

check: docs-check fmt clippy build test layout-check mvex-check ## Everything CI would run

clean: ## Remove build outputs
	cd $(HOST) && cargo clean
	rm -rf host/asm/out

# Top-level entry points. The real work is in scripts/ and in the Cargo
# workspace under host/. Run `make help` for the list.

.DEFAULT_GOAL := help
HOST := host
# phi-pld links a llama.cpp build (headers here, libraries in
# build-native/bin, or LLAMA_BUILD_DIR) and needs libclang; it joins the
# build where both the headers and libllama.so are found, and is skipped
# with a note naming the missing one where not (headers alone made its
# build script stop the whole build).
LLAMA_CPP_DIR ?= $(HOME)/llama.cpp
LLAMA_BUILD_DIR ?= $(LLAMA_CPP_DIR)/build-native/bin
export LLAMA_CPP_DIR LLAMA_BUILD_DIR
PLD := $(if $(and $(wildcard $(LLAMA_CPP_DIR)/include/llama.h),$(wildcard $(LLAMA_BUILD_DIR)/libllama.so)),--workspace,)
PLD_NOTE = $(if $(PLD),,@echo "phi-pld skipped: no $(if $(wildcard $(LLAMA_CPP_DIR)/include/llama.h),$(LLAMA_BUILD_DIR)/libllama.so (set LLAMA_BUILD_DIR),$(LLAMA_CPP_DIR)/include/llama.h (set LLAMA_CPP_DIR))")

.PHONY: help build test fmt clippy docs-check layout-check mvex-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

build: ## Build the host workspace (libphi512.so, phi-vpu, the translator, the encoder; phi-pld where llama.cpp is found) and libggml_phi.so from host/asm
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

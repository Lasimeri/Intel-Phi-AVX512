# Top-level entry points. The real work is in scripts/ and in the Cargo
# workspace under host/. Run `make help` for the list.

.DEFAULT_GOAL := help
HOST := host

.PHONY: help build test fmt clippy docs-check layout-check check clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

build: ## Build the host workspace (libphi512.so, libggml_phi.so, phi-vpu, the translator, the encoder)
	cd $(HOST) && cargo build && cargo build --release

test: ## Run host tests that do not need the card
	cd $(HOST) && cargo test

fmt: ## Check formatting
	cd $(HOST) && cargo fmt --all -- --check

clippy: ## Lint
	cd $(HOST) && cargo clippy --all-targets -- -D warnings

docs-check: ## Enforce sibling .md files, the no-dash rule and relative links
	scripts/check-docs.sh

layout-check: ## Compare the C protocol layout with the Rust constants (tcc)
	tcc -run tools/vpu-layout-check.c

check: docs-check fmt clippy build test layout-check ## Everything CI would run

clean: ## Remove build outputs
	cd $(HOST) && cargo clean

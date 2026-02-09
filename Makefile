PREFIX ?= $(HOME)/.local
BINDIR = $(PREFIX)/bin
FEATURES ?= voice

.PHONY: build install uninstall clean

build:
	cargo build --release --features $(FEATURES)

install: build
	@mkdir -p $(BINDIR)
	cp target/release/nanoclaw $(BINDIR)/
	@# Copy shared libs needed by voice feature next to binary
	@if echo "$(FEATURES)" | grep -q voice; then \
		for lib in target/release/lib{sherpa-onnx-c-api,onnxruntime,onnxruntime_providers_shared}.so; do \
			[ -f "$$lib" ] && cp "$$lib" $(BINDIR)/; \
		done; \
		echo "Installed nanoclaw + voice libs to $(BINDIR)"; \
	else \
		echo "Installed nanoclaw to $(BINDIR)"; \
	fi

uninstall:
	rm -f $(BINDIR)/nanoclaw
	rm -f $(BINDIR)/libsherpa-onnx-c-api.so
	rm -f $(BINDIR)/libonnxruntime.so
	rm -f $(BINDIR)/libonnxruntime_providers_shared.so

clean:
	cargo clean

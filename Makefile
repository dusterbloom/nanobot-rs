PREFIX ?= $(HOME)/.local
BINDIR = $(PREFIX)/bin
FEATURES ?= voice

.PHONY: build install uninstall clean

build:
	cargo build --release --features $(FEATURES)

install: build
	@mkdir -p $(BINDIR)
	cp target/release/nanobot $(BINDIR)/
	@# Copy shared libs needed by voice feature next to binary
	@if echo "$(FEATURES)" | grep -q voice; then \
		if [ "$$(uname)" = "Darwin" ]; then \
			for lib in target/release/lib{sherpa-onnx-c-api,sherpa-onnx-cxx-api,onnxruntime}.dylib \
			           target/release/libonnxruntime.1.17.1.dylib; do \
				[ -f "$$lib" ] && cp "$$lib" $(BINDIR)/; \
			done; \
			echo "Installed nanobot + voice libs (macOS) to $(BINDIR)"; \
		else \
			for lib in target/release/lib{sherpa-onnx-c-api,onnxruntime,onnxruntime_providers_shared}.so; do \
				[ -f "$$lib" ] && cp "$$lib" $(BINDIR)/; \
			done; \
			echo "Installed nanobot + voice libs (Linux) to $(BINDIR)"; \
		fi; \
	else \
		echo "Installed nanobot to $(BINDIR)"; \
	fi

uninstall:
	rm -f $(BINDIR)/nanobot
	rm -f $(BINDIR)/libsherpa-onnx-c-api.so
	rm -f $(BINDIR)/libsherpa-onnx-c-api.dylib
	rm -f $(BINDIR)/libsherpa-onnx-cxx-api.dylib
	rm -f $(BINDIR)/libonnxruntime.so
	rm -f $(BINDIR)/libonnxruntime.dylib
	rm -f $(BINDIR)/libonnxruntime.1.17.1.dylib
	rm -f $(BINDIR)/libonnxruntime_providers_shared.so

clean:
	cargo clean

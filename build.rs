fn main() {
    // When the voice feature is enabled, sherpa-rs links against libsherpa-onnx-c-api.so
    // which gets copied to the target dir at build time. Set rpath so the binary can find
    // it at runtime relative to the executable ($ORIGIN).
    #[cfg(feature = "voice")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
    }
}

/// Copy src to dst, but skip if they are hardlinked (same inode) to avoid
/// truncating the source when sherpa-rs-sys creates hardlinks from cache to target.
#[cfg(feature = "voice")]
fn safe_copy(src: &std::path::Path, dst: &std::path::Path) {
    use std::os::unix::fs::MetadataExt;
    if let (Ok(sm), Ok(dm)) = (src.metadata(), dst.metadata()) {
        if sm.ino() == dm.ino() && sm.dev() == dm.dev() {
            return;
        }
    }
    let _ = std::fs::copy(src, dst);
}

fn main() {
    // Accelerate framework is needed unconditionally on macOS for cpu_gemm (SGEMM).
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // When the voice feature is enabled, sherpa-rs links against libsherpa-onnx-c-api.so/.dylib
    // which gets copied to the target dir at build time. Set rpath so the binary can find
    // it at runtime relative to the executable ($ORIGIN on Linux, @executable_path on macOS)
    // and also at ~/.local/lib.
    #[cfg(feature = "voice")]
    {
        #[cfg(target_os = "macos")]
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
        }
        #[cfg(not(target_os = "macos"))]
        {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");
        }

        // libtorch: find PyTorch's lib directory and add rpath so the binary
        // can locate libtorch_cpu.dylib etc. at runtime without DYLD_LIBRARY_PATH.
        if let Ok(output) = std::process::Command::new("python3")
            .args([
                "-c",
                "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))",
            ])
            .output()
        {
            let torch_lib = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !torch_lib.is_empty() && std::path::Path::new(&torch_lib).exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{torch_lib}");
                // Also copy libtorch dylibs next to the binary so it works
                // even when the venv is not present.
                if let Ok(out_dir) = std::env::var("OUT_DIR") {
                    let mut target_dir = std::path::PathBuf::from(&out_dir);
                    while target_dir
                        .file_name()
                        .map_or(false, |f| f != "release" && f != "debug")
                    {
                        if !target_dir.pop() {
                            break;
                        }
                    }
                    let torch_lib_path = std::path::Path::new(&torch_lib);
                    for name in [
                        "libtorch.dylib",
                        "libtorch_cpu.dylib",
                        "libtorch_global_deps.dylib",
                        "libc10.dylib",
                    ] {
                        let src = torch_lib_path.join(name);
                        if src.exists() {
                            safe_copy(&src, &target_dir.join(name));
                        }
                    }
                    let home = std::env::var("HOME").unwrap_or_default();
                    let local_lib = std::path::PathBuf::from(&home).join(".local/lib");
                    let _ = std::fs::create_dir_all(&local_lib);
                    for name in [
                        "libtorch.dylib",
                        "libtorch_cpu.dylib",
                        "libtorch_global_deps.dylib",
                        "libc10.dylib",
                    ] {
                        let src = torch_lib_path.join(name);
                        if src.exists() {
                            safe_copy(&src, &local_lib.join(name));
                        }
                    }
                }
            }
        }

        // Copy sherpa-onnx shared libraries to the target directory so they're
        // next to the binary after build. Also copy to ~/.local/lib for the
        // installed binary.
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            // Walk up from OUT_DIR to find the profile dir (target/release or target/debug).
            let mut target_dir = std::path::PathBuf::from(&out_dir);
            while target_dir
                .file_name()
                .map_or(false, |f| f != "release" && f != "debug")
            {
                if !target_dir.pop() {
                    break;
                }
            }

            // Find the sherpa-onnx libs in the cache.
            let home = std::env::var("HOME").unwrap_or_default();
            let sherpa_cache = std::path::PathBuf::from(&home).join(".cache/sherpa-rs");
            if sherpa_cache.exists() {
                // On macOS the libs are .dylib; on Linux they are .so.
                #[cfg(target_os = "macos")]
                let libs = [
                    "libsherpa-onnx-c-api.dylib",
                    "libsherpa-onnx-cxx-api.dylib",
                    "libonnxruntime.dylib",
                    "libonnxruntime.1.17.1.dylib",
                ];
                #[cfg(not(target_os = "macos"))]
                let libs = [
                    "libsherpa-onnx-c-api.so",
                    "libonnxruntime.so",
                    "libsherpa-onnx-cxx-api.so",
                ];
                let local_lib = std::path::PathBuf::from(&home).join(".local/lib");
                let _ = std::fs::create_dir_all(&local_lib);

                for entry in walkdir(&sherpa_cache) {
                    let name = match entry.file_name() {
                        Some(n) => n.to_string_lossy().to_string(),
                        None => continue,
                    };
                    if libs.contains(&name.as_str()) {
                        // Copy to target dir (for cargo run).
                        safe_copy(&entry, &target_dir.join(&name));
                        // Copy to ~/.local/lib (for installed binary).
                        safe_copy(&entry, &local_lib.join(&name));
                    }
                }
            }
        }
    }
}

/// Simple recursive directory walker (no extra deps).
#[allow(dead_code)]
fn walkdir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(walkdir(&path));
            } else {
                results.push(path);
            }
        }
    }
    results
}

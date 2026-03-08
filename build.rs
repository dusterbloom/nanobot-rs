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

    // ANE (Apple Neural Engine) bridge — compile Obj-C dylib when `ane` feature is active.
    #[cfg(feature = "ane")]
    {
        #[cfg(not(target_os = "macos"))]
        compile_error!("The `ane` feature requires macOS with Apple Silicon");

        #[cfg(target_os = "macos")]
        {
            let bridge_dir = std::path::Path::new("bridge/ane");
            println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.h");
            println!("cargo:rerun-if-changed=bridge/ane/ane_bridge.m");

            let status = std::process::Command::new("make")
                .arg("-C")
                .arg(bridge_dir)
                .status()
                .expect("Failed to run make for ANE bridge");
            assert!(status.success(), "ANE bridge compilation failed");

            let bridge_abs =
                std::fs::canonicalize(bridge_dir).expect("Failed to resolve bridge/ane path");
            println!("cargo:rustc-link-search=native={}", bridge_abs.display());
            println!("cargo:rustc-link-lib=dylib=ane_bridge");

            // Set install name on the dylib so it can be found at runtime
            // relative to the executable, and also add rpath for the bridge dir.
            let dylib_src = bridge_abs.join("libane_bridge.dylib");
            let _ = std::process::Command::new("install_name_tool")
                .args([
                    "-id",
                    "@rpath/libane_bridge.dylib",
                    dylib_src.to_str().unwrap(),
                ])
                .status();
            println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", bridge_abs.display());

            // Copy dylib next to output binaries so @rpath finds it.
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
                // Copy to target/debug (or release) and target/debug/deps
                let _ = std::fs::copy(&dylib_src, target_dir.join("libane_bridge.dylib"));
                let _ = std::fs::copy(&dylib_src, target_dir.join("deps/libane_bridge.dylib"));
            }
        }
    }
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

            // Copy espeak-ng-data to ~/.local/share/espeak-ng-data/ so Kokoro TTS
            // works when the binary is installed outside the build tree.
            // espeak-rs-sys bakes the build-time path which breaks after `cp`.
            let build_dir = target_dir.join("build");
            if build_dir.exists() {
                for entry in walkdir(&build_dir) {
                    if entry.ends_with("share/espeak-ng-data/phontab") {
                        if let Some(espeak_data_dir) = entry.parent() {
                            let local_share =
                                std::path::PathBuf::from(&home).join(".local/share/espeak-ng-data");
                            copy_dir_recursive(espeak_data_dir, &local_share);
                            break;
                        }
                    }
                }
            }
        }
    }
}

/// Recursively copy a directory tree.
#[allow(dead_code)]
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) {
    #[cfg(feature = "voice")]
    let _ = std::fs::create_dir_all(dst);
    if let Ok(entries) = std::fs::read_dir(src) {
        for entry in entries.flatten() {
            let path = entry.path();
            let dest = dst.join(entry.file_name());
            if path.is_dir() {
                copy_dir_recursive(&path, &dest);
            } else {
                let _ = std::fs::copy(&path, &dest);
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

/// Copy src to dst, but skip if they are hardlinked (same inode) to avoid
/// truncating the source when sherpa-rs-sys creates hardlinks from cache to target.
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
    // When the voice feature is enabled, sherpa-rs links against libsherpa-onnx-c-api.so
    // which gets copied to the target dir at build time. Set rpath so the binary can find
    // it at runtime relative to the executable ($ORIGIN) and also at ~/.local/lib.
    #[cfg(feature = "voice")]
    {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../lib");

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
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) {
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

//! FFI bridge to Apple Neural Engine via private APIs.
//!
//! Wraps the Objective-C `ane_bridge.{h,m}` dylib into safe Rust types.
//! The bridge compiles MIL programs into ANE kernels, evaluates them,
//! and manages IOSurface-backed input/output tensors.

use std::ffi::{c_char, c_int, c_void, CString};
use std::ptr;

// ---------------------------------------------------------------------------
// Raw FFI declarations (mirrors ane_bridge.h)
// ---------------------------------------------------------------------------

/// Opaque handle returned by ane_bridge_compile*.
#[repr(C)]
pub(crate) struct ANEKernelHandle {
    _opaque: [u8; 0],
}

#[link(name = "ane_bridge")]
extern "C" {
    fn ane_bridge_init() -> c_int;

    fn ane_bridge_compile(
        mil_text: *const c_char,
        mil_len: usize,
        weight_data: *const u8,
        weight_len: usize,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_compile_multi_weights(
        mil_text: *const c_char,
        mil_len: usize,
        weight_names: *const *const c_char,
        weight_datas: *const *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle;

    fn ane_bridge_eval(kernel: *mut ANEKernelHandle) -> bool;

    fn ane_bridge_write_input(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        data: *const c_void,
        bytes: usize,
    );

    fn ane_bridge_write_input_region(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        offset: usize,
        data: *const c_void,
        bytes: usize,
    );

    fn ane_bridge_write_input_strided(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        dst_offset: usize,
        dst_stride: usize,
        src: *const c_void,
        src_stride: usize,
        chunk_bytes: usize,
        n_chunks: c_int,
    );

    fn ane_bridge_read_output(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        data: *mut c_void,
        bytes: usize,
    );

    fn ane_bridge_free(kernel: *mut ANEKernelHandle);

    fn ane_bridge_get_compile_count() -> c_int;
    fn ane_bridge_reset_compile_count();

    fn ane_bridge_build_weight_blob(
        src: *const f32,
        rows: c_int,
        cols: c_int,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_build_weight_blob_transposed(
        src: *const f32,
        rows: c_int,
        cols: c_int,
        out_len: *mut usize,
    ) -> *mut u8;

    fn ane_bridge_free_blob(ptr: *mut c_void);
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Initialize the ANE runtime. Must be called once before any compilation.
pub fn ane_init() -> Result<(), String> {
    let rc = unsafe { ane_bridge_init() };
    if rc == 0 {
        Ok(())
    } else {
        Err("ane_bridge_init failed (ANE unavailable or private framework missing)".into())
    }
}

/// Current global kernel compile count.
pub fn compile_count() -> i32 {
    unsafe { ane_bridge_get_compile_count() }
}

/// Reset the global compile counter.
pub fn reset_compile_count() {
    unsafe { ane_bridge_reset_compile_count() }
}

/// Convert f32 weights into ANE blob format (128-byte header + fp16 data).
pub fn build_weight_blob(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    let mut out_len: usize = 0;
    let ptr = unsafe {
        ane_bridge_build_weight_blob(weights.as_ptr(), rows as c_int, cols as c_int, &mut out_len)
    };
    assert!(!ptr.is_null(), "ane_bridge_build_weight_blob returned NULL");
    let blob = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
    unsafe { ane_bridge_free_blob(ptr as *mut c_void) };
    blob
}

/// Convert f32 weights into ANE blob format with transposition.
pub fn build_weight_blob_transposed(weights: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(weights.len(), rows * cols, "weight dimensions mismatch");
    let mut out_len: usize = 0;
    let ptr = unsafe {
        ane_bridge_build_weight_blob_transposed(
            weights.as_ptr(),
            rows as c_int,
            cols as c_int,
            &mut out_len,
        )
    };
    assert!(
        !ptr.is_null(),
        "ane_bridge_build_weight_blob_transposed returned NULL"
    );
    let blob = unsafe { std::slice::from_raw_parts(ptr, out_len) }.to_vec();
    unsafe { ane_bridge_free_blob(ptr as *mut c_void) };
    blob
}

/// RAII wrapper around a compiled ANE kernel.
///
/// IOSurface handles are thread-bound, so `AneKernel` is `!Send + !Sync`.
pub struct AneKernel {
    handle: *mut ANEKernelHandle,
    _not_send_sync: std::marker::PhantomData<*mut ()>,
}

impl AneKernel {
    /// Compile a MIL program into an ANE kernel.
    ///
    /// - `mil_text`: UTF-8 MIL program source
    /// - `weights`: optional raw weight blob (ANE format with 128-byte header)
    /// - `input_sizes`: byte sizes of each input tensor
    /// - `output_sizes`: byte sizes of each output tensor
    pub fn compile(
        mil_text: &str,
        weights: Option<&[u8]>,
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self, String> {
        let (w_ptr, w_len) = match weights {
            Some(w) => (w.as_ptr(), w.len()),
            None => (ptr::null(), 0),
        };
        let handle = unsafe {
            ane_bridge_compile(
                mil_text.as_ptr() as *const c_char,
                mil_text.len(),
                w_ptr,
                w_len,
                input_sizes.len() as c_int,
                input_sizes.as_ptr(),
                output_sizes.len() as c_int,
                output_sizes.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err("ANE compilation failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Compile a MIL program with multiple named weight files.
    pub fn compile_multi_weights(
        mil_text: &str,
        weight_names: &[&str],
        weight_datas: &[&[u8]],
        input_sizes: &[usize],
        output_sizes: &[usize],
    ) -> Result<Self, String> {
        assert_eq!(weight_names.len(), weight_datas.len());
        let c_names: Vec<CString> = weight_names
            .iter()
            .map(|n| CString::new(*n).expect("weight name contains null byte"))
            .collect();
        let name_ptrs: Vec<*const c_char> = c_names.iter().map(|c| c.as_ptr()).collect();
        let data_ptrs: Vec<*const u8> = weight_datas.iter().map(|d| d.as_ptr()).collect();
        let data_lens: Vec<usize> = weight_datas.iter().map(|d| d.len()).collect();

        let handle = unsafe {
            ane_bridge_compile_multi_weights(
                mil_text.as_ptr() as *const c_char,
                mil_text.len(),
                name_ptrs.as_ptr(),
                data_ptrs.as_ptr(),
                data_lens.as_ptr(),
                weight_names.len() as c_int,
                input_sizes.len() as c_int,
                input_sizes.as_ptr(),
                output_sizes.len() as c_int,
                output_sizes.as_ptr(),
            )
        };
        if handle.is_null() {
            return Err("ANE multi-weight compilation failed".into());
        }
        Ok(Self {
            handle,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Execute the kernel on ANE hardware.
    pub fn eval(&self) -> Result<(), String> {
        let ok = unsafe { ane_bridge_eval(self.handle) };
        if ok {
            Ok(())
        } else {
            Err("ANE eval failed".into())
        }
    }

    /// Write data to input tensor at `idx` (full buffer).
    pub fn write_input(&self, idx: usize, data: &[u8]) {
        unsafe {
            ane_bridge_write_input(
                self.handle,
                idx as c_int,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
    }

    /// Write data to a region of input tensor at `idx`.
    ///
    /// Only bytes `[offset..offset+data.len())` in the IOSurface are modified.
    /// Use this to patch activations into a pre-populated weight buffer without
    /// re-copying static weights — reduces CPU memory bandwidth from ~13MB to
    /// ~512KB per layer dispatch.
    pub fn write_input_region(&self, idx: usize, offset: usize, data: &[u8]) {
        unsafe {
            ane_bridge_write_input_region(
                self.handle,
                idx as c_int,
                offset,
                data.as_ptr() as *const c_void,
                data.len(),
            );
        }
    }

    /// Write scattered chunks from `src` into input tensor at `idx`.
    ///
    /// Copies `n_chunks` blocks of `chunk_bytes` from `src` (at `src_stride`
    /// intervals) into the IOSurface starting at `dst_offset` (at `dst_stride`
    /// intervals). All writes happen under a single IOSurface lock.
    ///
    /// Use this to patch interleaved activation rows (e.g. dim rows of seq×4
    /// bytes each) without re-copying the static weight columns.
    pub fn write_input_strided(
        &self,
        idx: usize,
        dst_offset: usize,
        dst_stride: usize,
        src: &[u8],
        src_stride: usize,
        chunk_bytes: usize,
        n_chunks: usize,
    ) {
        unsafe {
            ane_bridge_write_input_strided(
                self.handle,
                idx as c_int,
                dst_offset,
                dst_stride,
                src.as_ptr() as *const c_void,
                src_stride,
                chunk_bytes,
                n_chunks as c_int,
            );
        }
    }

    /// Read data from output tensor at `idx` into `buf`.
    pub fn read_output(&self, idx: usize, buf: &mut [u8]) {
        unsafe {
            ane_bridge_read_output(
                self.handle,
                idx as c_int,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
            );
        }
    }
}

impl Drop for AneKernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { ane_bridge_free(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Smoke tests — require actual ANE hardware (Apple Silicon Mac)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal MIL program: fp32 → fp16 → fp32 round-trip (cast-only identity).
    /// Uses the actual ANE MIL IR format with buildInfo metadata.
    /// Shape [1, 64, 1, 64] — ANE has a minimum tensor size requirement.
    const CAST_IDENTITY_MIL: &str = concat!(
        "program(1.3)\n",
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, ",
        "{\"coremlc-version\", \"3505.4.1\"}, ",
        "{\"coremltools-component-milinternal\", \"\"}, ",
        "{\"coremltools-version\", \"9.0\"}})]\n",
        "{\n",
        "    func main<ios18>(tensor<fp32, [1, 64, 1, 64]> x) {\n",
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n",
        "        tensor<fp16, [1, 64, 1, 64]> xh = cast(dtype = to16, x = x)",
        "[name = string(\"cin\")];\n",
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n",
        "        tensor<fp32, [1, 64, 1, 64]> y = cast(dtype = to32, x = xh)",
        "[name = string(\"cout\")];\n",
        "    } -> (y);\n",
        "}\n",
    );

    /// Number of elements in the test tensor: 1 * 64 * 1 * 64.
    const N: usize = 64 * 64;

    #[test]
    fn smoke_init_compile_eval() {
        // 1. Initialize ANE runtime
        ane_init().expect("ane_init failed — is this Apple Silicon?");

        // 2. Compile the cast-identity MIL (no weights needed)
        let tensor_bytes = N * 4; // fp32
        let input_sizes = [tensor_bytes];
        let output_sizes = [tensor_bytes];

        let kernel = AneKernel::compile(CAST_IDENTITY_MIL, None, &input_sizes, &output_sizes)
            .expect("compile failed");

        // 3. Write fp32 input: values 0.0 .. N as f32
        let input_f32: Vec<f32> = (0..N).map(|i| (i % 100) as f32).collect();
        let input_bytes: Vec<u8> = input_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input(0, &input_bytes);

        // 4. Evaluate on ANE
        kernel.eval().expect("eval failed");

        // 5. Read output and verify round-trip
        let mut output_bytes = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut output_bytes);
        let output_f32: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // fp32→fp16→fp32 should preserve small integer values exactly
        assert_eq!(
            input_f32, output_f32,
            "Cast identity kernel should preserve values through fp32→fp16→fp32"
        );

        // 6. Verify compile count incremented
        assert!(compile_count() >= 1, "compile count should be >= 1");

        // 7. Drop is implicit — verifies cleanup doesn't crash
    }

    /// Test write_input_region: partial write then eval should preserve the patched region.
    #[test]
    fn smoke_write_input_region() {
        ane_init().expect("ane_init failed");
        let tensor_bytes = N * 4;
        let kernel = AneKernel::compile(CAST_IDENTITY_MIL, None, &[tensor_bytes], &[tensor_bytes])
            .expect("compile failed");

        // Write zeros to the full input first
        let zeros = vec![0u8; tensor_bytes];
        kernel.write_input(0, &zeros);

        // Patch the first 64 elements (one row of the [1,64,1,64] tensor)
        let patch: Vec<f32> = (0..64).map(|i| (i + 1) as f32).collect();
        let patch_bytes: Vec<u8> = patch.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input_region(0, 0, &patch_bytes);

        kernel.eval().expect("eval failed");

        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // First 64 elements should be our patch (1..=64), rest should be 0
        for i in 0..64 {
            assert_eq!(out_f32[i], (i + 1) as f32, "patched element {i}");
        }
        for i in 64..N {
            assert_eq!(out_f32[i], 0.0, "unpatched element {i}");
        }
    }

    /// Test write_input_strided: scattered writes under single lock.
    #[test]
    fn smoke_write_input_strided() {
        ane_init().expect("ane_init failed");
        let tensor_bytes = N * 4; // 64*64*4 = 16384 bytes
        let kernel = AneKernel::compile(CAST_IDENTITY_MIL, None, &[tensor_bytes], &[tensor_bytes])
            .expect("compile failed");

        // Write zeros first
        kernel.write_input(0, &vec![0u8; tensor_bytes]);

        // Use strided write to set the first 4 bytes of each 64-element row.
        // Tensor is [1, 64, 1, 64] = 64 rows of 64 floats each.
        // Write one f32 per row: chunk_bytes=4, n_chunks=64,
        // dst_stride=64*4=256, src_stride=4 (contiguous source)
        let src: Vec<f32> = (0..64).map(|i| (i + 10) as f32).collect();
        let src_bytes: Vec<u8> = src.iter().flat_map(|f| f.to_le_bytes()).collect();
        kernel.write_input_strided(
            0,      // idx
            0,      // dst_offset
            256,    // dst_stride (64 floats * 4 bytes)
            &src_bytes,
            4,      // src_stride (contiguous f32s)
            4,      // chunk_bytes (one f32)
            64,     // n_chunks (64 rows)
        );

        kernel.eval().expect("eval failed");

        let mut out = vec![0u8; tensor_bytes];
        kernel.read_output(0, &mut out);
        let out_f32: Vec<f32> = out
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // First element of each row should be our value, rest should be 0
        for row in 0..64 {
            assert_eq!(
                out_f32[row * 64], (row + 10) as f32,
                "row {row} first element"
            );
            for col in 1..64 {
                assert_eq!(out_f32[row * 64 + col], 0.0, "row {row} col {col}");
            }
        }
    }
}

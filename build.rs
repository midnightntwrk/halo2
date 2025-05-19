// Copyright Supranational LLC
// Modifications Copyright Input Output
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf};

#[cfg(any(
    all(feature = "nvcc_sm_86", feature = "nvcc_sm_80"),
    all(feature = "nvcc_sm_86", feature = "nvcc_sm_90"),
    all(feature = "nvcc_sm_80", feature = "nvcc_sm_90")
))]
compile_error!("Please select only one feature: nvcc_sm_86, nvcc_sm_80, or nvcc_sm_90.");

#[cfg(feature = "nvcc_sm_80")]
const NVCC_CONFIG: (&str, &str) = ("sm_80", "arch=compute_70,code=sm_70");
#[cfg(feature = "nvcc_sm_86")]
const NVCC_CONFIG: (&str, &str) = ("sm_86", "arch=compute_80,code=sm_80");
#[cfg(feature = "nvcc_sm_90")]
const NVCC_CONFIG: (&str, &str) = ("sm_90", "arch=compute_90,code=sm_90");

fn main() {
    let curve = "FEATURE_BLS12_381";

    // account for cross-compilation [by examining environment variable]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Set CC environment variable to choose an alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    let mut cc = cc::Build::new();

    let c_src_dir = PathBuf::from("src");
    let files = vec![c_src_dir.join("lib.c")];
    let mut cc_opt = None;

    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc_opt = Some("__BLST_PORTABLE__");
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc_opt = Some("__ADX__");
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) =>
        {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx") {
                println!("Enabling ADX because it was detected on the host");
                cc_opt = Some("__ADX__");
            }
        }
        (true, true) => panic!("Cannot compile with both `portable` and `force-adx` features"),
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.opt_level(2);
    }
    if let Some(def) = cc_opt {
        cc.define(def, None);
    }
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        cc.include(include);
    }
    cc.files(&files).compile("blstrs_cuda");

    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        return;
    }
    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };

    let (nvcc_arch, nvcc_gencode) = NVCC_CONFIG;

    if nvcc.is_ok() {
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true);
        nvcc.flag(&format!("-arch={}", nvcc_arch));
        nvcc.flag("-gencode").flag(nvcc_gencode);
        nvcc.flag("-lineinfo"); 
        nvcc.flag("-t0");

        // If compiling in debug mode, add the -G flag and reduce optimizations.
        if cfg!(debug_assertions) {
            println!("Compiling nvcc in debug mode: adding -G and -O0 flags");
            nvcc.flag("-lineinfo"); 
            nvcc.flag("-Xptxas").flag("-v");
            nvcc.flag("-G");
            nvcc.flag("-O0");
        } else {
            // For release mode, set the optimization level
            nvcc.flag("-O3");
            nvcc.flag("--use_fast_math");
        }

        #[cfg(not(target_env = "msvc"))]
        nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
        nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
        nvcc.define(curve, None);
        if let Some(def) = cc_opt {
            nvcc.define(def, None);
        }
        if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
            nvcc.include(include);
        }
        if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
            //nvcc.include(include);
            nvcc.include(&include); 
            let gpu_t_path = PathBuf::from(&include).join("util/gpu_t.cu");

            if gpu_t_path.exists() {
                nvcc.file(gpu_t_path);
            }
        }
        nvcc.file("src/cuda/io_cuda.cu").compile("io_cuda");

        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
    }
}

use anyhow::{anyhow, bail, Context, Result};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cuh");
    println!("cargo:rerun-if-changed=src/prefill_paged_attn.cu");
    println!("cargo:rerun-if-changed=src/prefill_paged_attn_opt.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/sort.cu");
    println!("cargo:rerun-if-changed=src/update_kvscales.cu");
    println!("cargo:rerun-if-changed=src/mask.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm.cu");
    println!("cargo:rerun-if-changed=src/moe_gemv.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_wmma.cu");
    println!("cargo:rerun-if-changed=src/moe_gemm_gguf.cu");
    println!("cargo:rerun-if-changed=src/moe_gguf_small_m.cu");
    println!("cargo:rerun-if-changed=src/moe_wmma_gguf.cu");
    println!("cargo:rerun-if-changed=src/gpu_sampling.cuh");
    println!("cargo:rerun-if-changed=src/gpu_sampling.cu");
    println!("cargo:rerun-if-changed=src/fused_rope.cu");
    println!("cargo:rerun-if-changed=src/fp8_matmul.cu");
    println!("cargo:rerun-if-changed=src/fp8_gemm_cutlass.cu");
    println!("cargo:rerun-if-changed=src/fp8_moe_cutlass.cu");

    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CUTLASS_DIR");
    println!("cargo:rerun-if-env-changed=CARGO_HOME");
    println!("cargo:rerun-if-env-changed=HOME");

    let marlin_disabled = std::env::var("CARGO_FEATURE_NO_MARLIN").is_ok();
    let fp8_kvcache_disabled = std::env::var("CARGO_FEATURE_NO_FP8_KVCACHE").is_ok();

    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or_default());

    let mut builder = bindgen_cuda::Builder::default()
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math");

    let compute_cap = compute_capability().unwrap_or(80);

    println!("cargo:info=compute capability: {}", compute_cap);

    if compute_cap < 80 {
        builder = builder.arg("-DNO_BF16_KERNEL");
        builder = builder.arg("-DNO_MARLIN_KERNEL");
        builder = builder.arg("-DNO_HARDWARE_FP8");
    }

    if compute_cap >= 121 {
        builder = builder.arg("--gpu-architecture=sm_121a");
        builder = builder.arg("-DENABLE_SM120a"); // 121 compatible with 120 features
    } else if compute_cap >= 120 {
        builder = builder.arg("--gpu-architecture=sm_120a");
        builder = builder.arg("-DENABLE_SM120");
    } else if compute_cap >= 100 {
        builder = builder.arg("--gpu-architecture=sm_100a");
        builder = builder.arg("-DENABLE_SM100");
    } else if compute_cap >= 90 {
        builder = builder.arg("--gpu-architecture=sm_90a");
    }

    if marlin_disabled {
        builder = builder.arg("-DNO_MARLIN_KERNEL");
    }

    if fp8_kvcache_disabled {
        builder = builder.arg("-DNO_FP8_KVCACHE");
    }

    // CUTLASS resolution:
    // - If CUTLASS_DIR set, use it (must be non-empty and contain include/)
    // - Else scan cargo git checkouts for candle-flash-attn/cutlass
    if std::env::var("CARGO_FEATURE_CUTLASS").is_ok() {
        let cutlass_dir = resolve_cutlass_dir_by_scanning_checkouts()?;
        let cutlass_dir_str = cutlass_dir.to_string_lossy();

        let include_root = Box::leak(format!("-I{cutlass_dir_str}").into_boxed_str());
        let include_main = Box::leak(format!("-I{cutlass_dir_str}/include").into_boxed_str());
        let include_tools =
            Box::leak(format!("-I{cutlass_dir_str}/tools/util/include").into_boxed_str());

        builder = builder
            .arg("-DUSE_CUTLASS")
            .arg(include_root)
            .arg(include_main)
            .arg(include_tools);

        println!("cargo:info=Using CUTLASS at {}", cutlass_dir.display());
    }

    // Target handling
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC").arg("-std=c++17");
    }

    println!("cargo:info={builder:?}");

    builder.build_lib(build_dir.join("libpagedattention.a"));

    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");
    // println!("cargo:rustc-link-lib=dylib=stdc++");

    Ok(())
}

fn compute_capability() -> Result<usize> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv"])
        .output();

    if let Ok(out) = out {
        let output = String::from_utf8(out.stdout).context("nvidia-smi output was not utf8")?;
        let line = output
            .lines()
            .nth(1)
            .ok_or_else(|| anyhow!("unexpected nvidia-smi output:\n{output}"))?;
        let cap = line
            .trim()
            .parse::<f32>()
            .context("failed to parse compute_cap")?;
        return Ok((cap * 10.0) as usize);
    }

    if let Ok(var) = std::env::var("CUDA_COMPUTE_CAP") {
        let v = var
            .parse::<usize>()
            .context("CUDA_COMPUTE_CAP must be an integer")?;
        return Ok(if v >= 100 { v } else { v * 10 });
    }

    Err(anyhow!(
        "failed to run nvidia-smi; set CUDA_COMPUTE_CAP env var instead"
    ))
}

fn resolve_cutlass_dir_by_scanning_checkouts() -> Result<PathBuf> {
    // 1) Explicit override
    if let Ok(env) = std::env::var("CUTLASS_DIR") {
        let p = PathBuf::from(env);
        ensure_cutlass_dir_valid(&p, None)?;
        return Ok(p);
    }

    // 2) Scan cargo git checkouts
    let checkouts = cargo_git_checkouts_dir()?;
    if !checkouts.is_dir() {
        bail!(
            "cargo git checkouts directory not found: `{}` (set CUTLASS_DIR to override)",
            checkouts.display()
        );
    }

    // We will:
    // - collect candidate candle-flash-attn dirs (for helpful error messages)
    // - pick the "best" cutlass dir among those that have non-empty cutlass/include
    let mut flash_attn_dirs: Vec<PathBuf> = Vec::new();
    let mut best: Option<(PathBuf, SystemTime)> = None;

    let repo_dirs = fs::read_dir(&checkouts)
        .with_context(|| format!("failed to read `{}`", checkouts.display()))?;

    for repo in repo_dirs {
        let repo = repo?;
        if !repo.file_type()?.is_dir() {
            continue;
        }

        let repo_path = repo.path();
        let repo_name = repo_path.file_name().and_then(OsStr::to_str).unwrap_or("");

        // Prefer candle-* but still allow other names (some forks rename).
        let candle_pref = repo_name.starts_with("candle-");

        // Under each repo dir, there are revision subdirs like dfa48cd/
        let rev_dirs = match fs::read_dir(&repo_path) {
            Ok(it) => it,
            Err(_) => continue,
        };

        for rev in rev_dirs {
            let rev = rev?;
            if !rev.file_type()?.is_dir() {
                continue;
            }
            let rev_path = rev.path();

            let flash_attn = rev_path.join("candle-flash-attn");
            if flash_attn.is_dir() {
                flash_attn_dirs.push(flash_attn.clone());

                let cutlass = flash_attn.join("cutlass");
                let include = cutlass.join("include");

                // Only accept if cutlass is valid and include exists and is non-empty
                if cutlass.is_dir() && include.is_dir() && dir_non_empty(&include) {
                    // Score by mtime of candle-flash-attn/Cargo.toml if present; else by flash_attn dir mtime.
                    let score_time = mtime_best_effort(&flash_attn.join("Cargo.toml"))
                        .or_else(|| mtime_best_effort(&flash_attn))
                        .unwrap_or(SystemTime::UNIX_EPOCH);

                    // Prefer candle-* repos by boosting score slightly (implemented by choosing candle-* first when times equal)
                    match &best {
                        None => best = Some((cutlass, score_time)),
                        Some((best_path, best_time)) => {
                            let better_time = score_time > *best_time;
                            let equal_time = score_time == *best_time;

                            if better_time {
                                best = Some((cutlass, score_time));
                            } else if equal_time {
                                // Tie-break: prefer candle-* repo name
                                let best_repo_name = best_path
                                    .ancestors()
                                    .nth(2) // .../checkouts/<repo>/<rev>/...
                                    .and_then(|p| p.file_name())
                                    .and_then(OsStr::to_str)
                                    .unwrap_or("");
                                let best_is_candle = best_repo_name.starts_with("candle-");
                                if candle_pref && !best_is_candle {
                                    best = Some((cutlass, score_time));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some((cutlass_dir, _)) = best {
        // Final sanity for root cutlass dir (not just include)
        ensure_cutlass_dir_valid(&cutlass_dir, None)?;
        return Ok(cutlass_dir);
    }

    // If we found candle-flash-attn dirs but cutlass missing/empty, print explicit submodule instructions.
    if !flash_attn_dirs.is_empty() {
        // Deduplicate for readability
        flash_attn_dirs.sort();
        flash_attn_dirs.dedup();

        // Provide up to a few paths; these are concrete and actionable.
        let mut msg = String::new();
        msg.push_str(
            "Found `candle-flash-attn` checkouts, but no usable `cutlass` submodule was found.\n\n",
        );
        msg.push_str("You likely need to initialize the CUTLASS submodule in the candle checkout used by Cargo.\n\n");
        msg.push_str("Run the following in ONE of these directories (choose the one you intend to build against):\n");

        for (i, p) in flash_attn_dirs.iter().take(5).enumerate() {
            msg.push_str(&format!(
                "  [{}] cd {}\n      git submodule update --init --recursive\n",
                i + 1,
                p.display()
            ));
        }

        msg.push_str(
            "\nThen rebuild. Alternatively, set CUTLASS_DIR to an external CUTLASS checkout.\n",
        );
        bail!(msg);
    }

    bail!(
        "Could not find any candle checkout containing `candle-flash-attn` under `{}`.\n\
Set CUTLASS_DIR to an external CUTLASS checkout, or ensure the candle git dependency is present in Cargo's git checkouts cache.",
        checkouts.display()
    );
}

fn ensure_cutlass_dir_valid(
    cutlass_dir: &Path,
    candle_flash_attn_dir: Option<&Path>,
) -> Result<()> {
    let include = cutlass_dir.join("include");
    if cutlass_dir.is_dir() && include.is_dir() && dir_non_empty(&include) {
        return Ok(());
    }

    if let Some(flash_attn) = candle_flash_attn_dir {
        bail!(
            "CUTLASS directory missing/invalid: `{}`\n\
\n\
To fetch CUTLASS, run:\n\
  cd {}\n\
  git submodule update --init --recursive\n\
\n\
Then rebuild. Alternatively, set CUTLASS_DIR to an external CUTLASS checkout.",
            cutlass_dir.display(),
            flash_attn.display()
        );
    }

    bail!(
        "CUTLASS directory missing/invalid: `{}` (expected non-empty `{}`)\n\
Set CUTLASS_DIR to a valid CUTLASS checkout.",
        cutlass_dir.display(),
        include.display()
    );
}

fn cargo_git_checkouts_dir() -> Result<PathBuf> {
    if let Ok(ch) = std::env::var("CARGO_HOME") {
        return Ok(PathBuf::from(ch).join("git").join("checkouts"));
    }
    let home = std::env::var("HOME").context("HOME not set; set CARGO_HOME or HOME")?;
    Ok(PathBuf::from(home)
        .join(".cargo")
        .join("git")
        .join("checkouts"))
}

fn dir_non_empty(p: &Path) -> bool {
    if !p.is_dir() {
        return false;
    }
    fs::read_dir(p).ok().and_then(|mut it| it.next()).is_some()
}

fn mtime_best_effort(p: &Path) -> Option<SystemTime> {
    fs::metadata(p).ok().and_then(|m| m.modified().ok())
}

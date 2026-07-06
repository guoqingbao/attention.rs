fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if cfg!(feature = "flashattn") {
        println!("cargo:rustc-link-lib=dylib=flashkernels");
    }
}

use std::env;
use std::path::PathBuf;
use std::process::exit;

fn main() {
    println!("cargo::rerun-if-env-changed=EVALFILE");
    let path = env::var_os("EVALFILE")
        .filter(|s| !s.is_empty())
        .map_or(PathBuf::from("nnue.bin"), PathBuf::from);
    match path.canonicalize() {
        Ok(path) => println!("cargo::rustc-env=NNUE_PATH={}", path.display()),
        Err(e) => {
            eprintln!(
                "Failed to find NNUE file at path `{}`: {}",
                path.display(),
                e
            );
            exit(1);
        }
    }
}

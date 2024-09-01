use futhark_bindgen::Backend;

fn main() {
    println!("cargo:rerun-if-env-changed=FUTHARK_BACKEND");
    for f in std::fs::read_dir("src").unwrap() {
        let f = f.unwrap().path();
        if f.extension() == Some("fut".as_ref()) {
            println!("cargo:rerun-if-changed={}", f.display());
        }
    }
    futhark_bindgen::build(
        Backend::from_env().unwrap_or(Backend::C),
        "src/trainer.fut",
        "trainer.rs",
    );
}

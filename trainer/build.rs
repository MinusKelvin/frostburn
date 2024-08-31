use futhark_bindgen::Backend;

fn main() {
    println!("cargo:rerun-if-env-changed=FUTHARK_BACKEND");
    futhark_bindgen::build(
        Backend::from_env().unwrap_or(Backend::C),
        "src/trainer.fut",
        "trainer.rs",
    );
}

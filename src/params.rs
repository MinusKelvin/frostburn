macro_rules! params {
    ($($name:ident: $min:literal..=$max:literal = $default:literal;)*) => {
        $(
            #[cfg(not(feature = "tunable"))]
            pub fn $name() -> i16 {
                $default
            }

            #[cfg(feature = "tunable")]
            pub fn $name() -> i16 {
                atomics::$name.load(std::sync::atomic::Ordering::Relaxed)
            }
        )*

        #[cfg(feature = "tunable")]
        mod atomics {
            use std::sync::atomic::AtomicI16;
            $(
                pub(super) static $name: AtomicI16 = AtomicI16::new($default);
            )*
        }

        #[cfg(feature = "tunable")]
        pub struct Tunable {
            pub name: &'static str,
            pub atomic: &'static std::sync::atomic::AtomicI16,
            pub min: i16,
            pub max: i16,
            pub default: i16,
        }

        #[cfg(feature = "tunable")]
        pub static TUNABLES: &[Tunable] = &[$(
            Tunable {
                name: stringify!($name),
                atomic: &atomics::$name,
                min: $min,
                max: $max,
                default: $default,
            },
        )*];
    };
}

params! {
    rfp_margin: 0..=200 = 57;
    rfp_max_depth: 1..=10 = 8;
    nmp_min_depth: 1..=10 = 2;
    nmp_divisor: 1..=500 = 171;
    nmp_depth: 0..=500 = 43;
    nmp_constant: 0..=1000 = 622;
    lmr_base: -150..=150 = 24;
    lmr_factor: 0..=200 = 77;
    lmr_history: 1..=32000 = 4956;
    lmr_history_max: 1..=10 = 4;
    lmp_a: 0..=1024 = 9;
    lmp_b: -1024..=1024 = 0;
    lmp_c: -1024..=1024 = 83;
    asp_initial: 1..=1000 = 30;
    asp_widening: 1..=1000 = 100;
    tm_soft_limit: 2..=1000 = 30;
}

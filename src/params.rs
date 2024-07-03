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
    rfp_margin: 0..=200 = 63;
    rfp_max_depth: 1..=10 = 9;
    nmp_min_depth: 1..=10 = 3;
    nmp_divisor: 1..=500 = 140;
    nmp_depth: 0..=500 = 32;
    nmp_constant: 0..=1000 = 575;
    lmr_base: -150..=150 = 50;
    lmr_factor: 0..=200 = 98;
    lmr_history: 1..=32000 = 5622;
    lmr_history_max: 1..=10 = 3;
    lmp_a: 0..=1024 = 7;
    lmp_b: -1024..=1024 = 0;
    lmp_c: -1024..=1024 = 47;
    asp_initial: 1..=1000 = 23;
    asp_widening: 1..=1000 = 110;
    tm_soft_limit: 2..=1000 = 29;
    razor_max_depth: 1..=10 = 3;
    razor_base: 0..=1000 = 200;
    razor_margin: 0..=1000 = 50;
}

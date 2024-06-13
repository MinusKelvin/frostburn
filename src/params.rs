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
    rfp_margin: 0..=200 = 50;
    rfp_max_depth: 1..=10 = 4;
    nmp_min_depth: 1..=10 = 1;
    nmp_divisor: 1..=500 = 150;
    nmp_depth: 0..=500 = 50;
    nmp_constant: 0..=1000 = 300;
}

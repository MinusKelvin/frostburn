use core::fmt::Display;
use core::ops::{Add, Neg, Sub};

use bytemuck::{Pod, TransparentWrapper, Zeroable};

#[derive(Copy, Clone, Debug, TransparentWrapper, Pod, Zeroable, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Eval(i16);

const MAX_NONMATE: i16 = 29_000;

impl Eval {
    pub fn cp(v: i16) -> Self {
        Eval(v)
    }

    pub fn mated(ply: usize) -> Self {
        Eval(ply as i16 - 30_000)
    }

    pub fn mating(ply: usize) -> Self {
        Eval(30_000 - ply as i16)
    }

    pub fn is_mate(self) -> bool {
        self.0 < -MAX_NONMATE || self.0 > MAX_NONMATE
    }

    pub fn losing(self) -> bool {
        self.0 < -MAX_NONMATE
    }

    pub fn clamp_nonmate(self) -> Self {
        Eval(self.0.clamp(-MAX_NONMATE, MAX_NONMATE))
    }

    pub fn sub_time(self, ply: usize) -> Eval {
        if self.0 < -MAX_NONMATE {
            Eval(self.0 - ply as i16)
        } else if self.0 > MAX_NONMATE {
            Eval(self.0 + ply as i16)
        } else {
            self
        }
    }

    pub fn add_time(self, ply: usize) -> Eval {
        if self.0 < -MAX_NONMATE {
            Eval(self.0 + ply as i16)
        } else if self.0 > MAX_NONMATE {
            Eval(self.0 - ply as i16)
        } else {
            self
        }
    }
}

impl Display for Eval {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            if self.0 < -MAX_NONMATE {
                write!(f, "  M#-{:<2}", (self.0 + 30_000 + 1) / 2)
            } else if self.0 > MAX_NONMATE {
                write!(f, "   M#{:<2}", (30_000 - self.0 + 1) / 2)
            } else {
                write!(f, "{:>+7.02}", self.0 as f64 / 100.0)
            }
        } else {
            if self.0 < -MAX_NONMATE {
                write!(f, "mate -{}", (self.0 + 30_000 + 1) / 2)
            } else if self.0 > MAX_NONMATE {
                write!(f, "mate {}", (30_000 - self.0 + 1) / 2)
            } else {
                write!(f, "cp {}", self.0)
            }
        }
    }
}

impl Neg for Eval {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Eval(-self.0)
    }
}

impl Add<i16> for Eval {
    type Output = Eval;

    fn add(self, rhs: i16) -> Self::Output {
        Eval(self.0 + rhs)
    }
}

impl Sub<i16> for Eval {
    type Output = Eval;

    fn sub(self, rhs: i16) -> Self::Output {
        Eval(self.0 - rhs)
    }
}

impl Sub<Eval> for Eval {
    type Output = i32;

    fn sub(self, rhs: Eval) -> Self::Output {
        self.0 as i32 - rhs.0 as i32
    }
}

impl Add<i32> for Eval {
    type Output = Eval;

    fn add(self, rhs: i32) -> Self::Output {
        Eval((self.0 as i32 + rhs).clamp(-30_000, 30_000) as i16)
    }
}

impl Sub<i32> for Eval {
    type Output = Eval;

    fn sub(self, rhs: i32) -> Self::Output {
        Eval((self.0 as i32 - rhs).clamp(-30_000, 30_000) as i16)
    }
}

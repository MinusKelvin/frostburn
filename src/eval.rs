use std::fmt::Display;
use std::ops::{Add, Neg, Sub};

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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 < -MAX_NONMATE {
            write!(f, "mate {}", (self.0 + 30_000) / 2 - 1)
        } else if self.0 > MAX_NONMATE {
            write!(f, "mate {}", (30_000 - self.0) / 2 + 1)
        } else {
            write!(f, "cp {}", self.0)
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
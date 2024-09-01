import "nn"

type Model = {
    l1: Dense[2][1],
    l2: Dense[1+1][1],
}

def init (v: f32): Model = {
    l1 = init v,
    l2 = init v,
}

def map_model (f: f32 -> f32) (a: Model): *Model = {
    l1 = map_dense f a.l1,
    l2 = map_dense f a.l2,
}

def map2_model (f: f32 -> f32 -> f32) (a: Model) (b: Model): *Model = {
    l1 = map2_dense f a.l1 b.l1,
    l2 = map2_dense f a.l2 b.l2,
}

def map3_model (f: f32 -> f32 -> f32 -> f32) (a: Model) (b: Model) (c: Model): *Model = {
    l1 = map3_dense f a.l1 b.l1 c.l1,
    l2 = map3_dense f a.l2 b.l2 c.l2,
}

def model [b] (w: Model) (x: VecBatch [b][2] Model): VecBatch [b][1] Model =
    let first = dense w.l1 (\d (g: Model) -> g with l1 = map2_dense (+) g.l1 d) x in
    let alt_x = rep (rep 1) `subv` x in
    let second = dense w.l1 (\d (g: Model) -> g with l1 = map2_dense (+) g.l1 d) alt_x in
    cat first second |>
    crelu |>
    dense w.l2 (\d (g: Model) -> g with l2 = map2_dense (+) g.l2 d) |>
    sigmoid

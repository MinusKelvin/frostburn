import "nn"

type Model = {
    ft: Linear[768][512],
    l2: Linear[512+512][1],
}

def init (v: f32): Model = {
    ft = init v,
    l2 = init v,
}

def map_model (f: f32 -> f32) (a: Model): *Model = {
    ft = map_linear f a.ft,
    l2 = map_linear f a.l2,
}

def map2_model (f: f32 -> f32 -> f32) (a: Model) (b: Model): *Model = {
    ft = map2_linear f a.ft b.ft,
    l2 = map2_linear f a.l2 b.l2,
}

def map3_model (f: f32 -> f32 -> f32 -> f32) (a: Model) (b: Model) (c: Model): *Model = {
    ft = map3_linear f a.ft b.ft c.ft,
    l2 = map3_linear f a.l2 b.l2 c.l2,
}

def model [b] (w: Model) (stm: VecBatch [b][768] Model) (nstm: VecBatch [b][768] Model): VecBatch [b][1] Model =
    let stm = linear w.ft (\d (g: Model) -> g with ft = map2_linear (+) g.ft d) stm in
    let nstm = linear w.ft (\d (g: Model) -> g with ft = map2_linear (+) g.ft d) nstm in
    cat stm nstm |>
    screlu |>
    linear w.l2 (\d (g: Model) -> g with l2 = map2_linear (+) g.l2 d) |>
    sigmoid

def clip (w: Model): Model =
    let clip x = x |> f32.max (-127.0/64) |> f32.min (127/64) in
    {
        ft = w.ft,
        l2 = {
            weights = map_2d clip w.l2.weights,
            bias = w.l2.bias
        }
    }

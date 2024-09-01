import "nn"
import "model"

type AdamOptions = {
    b1: f32,
    b2: f32,
    eps: f32,
    lr: f32,
    decay: f32
}

type AdamState = {
    m: Model,
    v: Model,
    b1_t: f32,
    b2_t: f32,
}

entry adam_init: AdamState = {
    m = init 0,
    v = init 0,
    b1_t = 1,
    b2_t = 1,
}

def adam
    ({b1, b2, eps, lr, decay}: AdamOptions)
    (state: AdamState)
    (weights: Model)
    (grad: Model)
: (Model, AdamState) =
    let b1_t = b1 * state.b1_t in
    let b2_t = b2 * state.b2_t in
    let update_m m g = b1 * m + (1 - b1) * g in
    let update_v v g = b2 * v + (1 - b2) * g * g in
    let update_w w m v =
        let m' = m / (1 - b1_t) in
        let v' = v / (1 - b2_t) in
        w - lr * m' / (f32.sqrt v' + eps)
    in
    let m = map2_model update_m state.m grad in
    let v = map2_model update_v state.v grad in
    let weights = map3_model update_w weights m v in
    (weights, { m, v, b1_t, b2_t })

entry infer (weights: Model) (x: [][2]f32): [][1]f32 =
    let y = model weights (make_vecbatch x) in
    y.x

entry step [b]
    (options: AdamOptions)
    (weights: Model)
    (state: AdamState)
    (x: [b][2]f32)
    (target: [b][1]f32)
: (f32, Model, AdamState) =
    let loss = model weights (make_vecbatch x) |> mse target in
    let grad = backwards (init <| options.decay) loss in
    let (weights, state) = adam options state weights grad in
    (loss.x[0][0], weights, state)

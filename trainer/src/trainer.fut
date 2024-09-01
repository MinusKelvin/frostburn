import "nn"
import "model"

entry infer (weights: Model) (x: [][2]f32): [][1]f32 =
    let y = model weights (make_vecbatch x) in
    y.x

entry step [b] (weights: Model) (x: [b][2]f32) (target: [b][1]f32): (f32, Model) =
    let loss = model weights (make_vecbatch x) |> mse target in
    let grad = backwards (init 0) loss in
    let updated = map_weights (-) weights grad in
    (loss.x[0][0], updated)

entry grad (weights: Model) (x: [1][2]f32): (f32, Model) =
    let loss = model weights (make_vecbatch x) in
    (loss.x[0][0], backwards (init 0) loss)

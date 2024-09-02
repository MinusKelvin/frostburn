type^ VecBatch [b][n] 'grads = {
    x: [b][n]f32,
    backward: [b][n]f32 -> grads -> grads
}

def make_vecbatch [b][n] 'grads (x: [b][n]f32): VecBatch [b][n] grads =
    { x, backward = \_ (g: grads): grads -> g }

def backwards 'grads (g: grads) (x: VecBatch [][] grads): grads =
    x.backward (rep (rep 1)) g

-- Parameter mapping ops

def map_2d 'a 'b (f: a -> b) (x: [][]a): *[][]b =
    map (map f) x

def map2_2d 'a 'b 'c (f: a -> b -> c) (x: [][]a) (y: [][]b): *[][]c =
    map2 (map2 f) x y

def map3_2d 'a 'b 'c 'd (f: a -> b -> c -> d) (x: [][]a) (y: [][]b) (z: [][]c): *[][]d =
    map3 (map3 f) x y z

-- Matrix and vector math

local def dot (x: []f32) (y: []f32) =
    map2 (*) x y |> reduce (+) 0

local def outerprod [n][m] (x: [m]f32) (y: [n]f32): [n][m]f32 =
    map (\y -> map (* y) x) y

local def matvecmul [n][m] (a: [n][m]f32) (x: [m]f32): [n]f32 =
    map (dot x) a

local def vecmatmul [n][m] (x: [n]f32) (a: [n][m]f32): [m]f32 =
    map (dot x) (transpose a)

local def matmul [n][m][p] (a: [n][m]f32) (b: [m][p]f32): [n][p]f32 =
    map (\ar -> map (\bc -> dot ar bc) (transpose b)) a

def vsub 'grads (x: VecBatch [][] grads) (nograd: [][]f32) =
    x with x = map2_2d (-) x.x nograd

def subv 'grads (nograd: [][]f32) (x: VecBatch [][] grads) =
    let y = map2_2d (-) nograd x.x in
    {
        x = y,
        backward = \dLdy grads ->
            let dLdx = map_2d f32.neg dLdy in
            x.backward dLdx grads
    }

def cat [b][n][m] 'grads (x1: VecBatch [b][n] grads) (x2: VecBatch [b][m] grads): VecBatch [b][n+m] grads =
    let y = transpose x1.x ++ transpose x2.x |> transpose in
    {
        x = y,
        backward = \dLdy grads ->
            let (dLdx1_t, dLdx2_t) = transpose dLdy |> split in
            x1.backward (transpose dLdx1_t) (x2.backward (transpose dLdx2_t) grads)
    }

-- Activation functions

def crelu 'grads (x: VecBatch [][] grads): VecBatch [][] grads =
    let y = map_2d (f32.max 0 >-> f32.min 1) x.x in
    {
        x = y,
        backward = \dLdy grads ->
            let dLdx = map2_2d (\x dLdy -> if x >= 0 && x <= 1 then dLdy else 0) x.x dLdy in
            x.backward dLdx grads
    }

def sq 'grads (x: VecBatch [][] grads): VecBatch [][] grads =
    let y = map_2d (\x -> x * x) x.x in
    {
        x = y,
        backward = \dLdy grads ->
            let dLdx = map2_2d (\x dLdy -> 2 * x * dLdy) x.x dLdy in
            x.backward dLdx grads
    }

def screlu 'grads = crelu >-> sq

def sigmoid 'grads (x: VecBatch [][] grads): VecBatch [][] grads =
    let f (x: f32) = 1 / (1 + f32.exp (-x)) in
    let y = map_2d f x.x in
    let dydx = map_2d (\y -> y * (1 - y)) y in
    {
        x = y,
        backward = \dLdy grads ->
            let dLdx = map2_2d (*) dLdy dydx in
            x.backward dLdx grads
    }

-- Linear layer

type Linear [i][o] = {
    weights: [o][i]f32,
    bias: [o]f32
}

def init [i][o] (v: f32): *Linear [i][o] = {
    weights = rep (rep v),
    bias = rep v
}

def map_linear (f: f32 -> f32) (a: Linear [][]): *Linear [][] = {
    weights = map_2d f a.weights,
    bias = map f a.bias
}

def map2_linear (f: f32 -> f32 -> f32) (a: Linear [][]) (b: Linear [][]): *Linear [][] = {
    weights = map2_2d f a.weights b.weights,
    bias = map2 f a.bias b.bias
}

def map3_linear (f: f32 -> f32 -> f32 -> f32) (a: Linear [][]) (b: Linear [][]) (c: Linear [][]): *Linear [][] = {
    weights = map3_2d f a.weights b.weights c.weights,
    bias = map3 f a.bias b.bias c.bias
}

def linear [i][o] 'grads
    ({weights, bias}: Linear[i][o])
    (update: Linear[i][o] -> grads -> grads)
    (x: VecBatch [][i] grads)
: VecBatch [][o] grads =
    let y = map (matvecmul weights >-> map2 (+) bias) x.x in
    {
        x = y,
        backward = \dLdy grads ->
            let dLdx = map (\dLdy -> vecmatmul dLdy weights) dLdy in
            let dLdw = map (\dLdy -> map (dot dLdy) (transpose x.x)) (transpose dLdy) in
            let dLdb = map (f32.sum) (transpose dLdy) in
            x.backward dLdx (update { weights = dLdw, bias = dLdb } grads)
    }

def feature_transformer [i][o] 'grads
    ({weights, bias}: Linear[i][o])
    (update: Linear[i][o] -> grads -> grads)
    (feats: [][32]i64)
: VecBatch [][2*o] grads =
    let sparse_dot ws is = reduce (+) 0 (map (\i -> if i >= 0 then ws[i] else 0) is) in
    let y = map (\feats -> flatten [
        map2 (\ws b -> b + sparse_dot ws feats) weights bias,
        map2 (\ws b -> b + sparse_dot ws (map (^0b1_111_000) feats)) weights bias,
    ]
    ) feats in
    {
        x = y,
        backward = \dLdy grads ->
            let dLdy: [][2][o]f32 = map unflatten dLdy in
            let (oi, ii, v) = map2 (\dLdy feats ->
                map3 (\i dLdy_stm dLdy_nstm ->
                    map (\f -> [(i, f, dLdy_stm), (i, f ^ 0b1_111_000, dLdy_nstm)]) feats |> flatten
                ) (indices dLdy[0]) dLdy[0] dLdy[1] |> flatten
            ) dLdy feats |> flatten |> filter (\(_, ii, _) -> ii >= 0) |> unzip3 in
            let dLdw = reduce_by_index_2d (rep (rep 0)) (+) 0 (zip oi ii) v in
            let dLdb_stm = map (f32.sum) (transpose dLdy[:,0,:]) in
            let dLdb_nstm = map (f32.sum) (transpose dLdy[:,1,:]) in
            let dLdb = map2 (+) dLdb_stm dLdb_nstm in
            update { weights = dLdw, bias = dLdb } grads
    }

-- Loss functions

def mean [b] 'grads (x: VecBatch [b][1] grads): VecBatch [1][1] grads =
    let n = f32.i64 b in
    let mean = f32.sum (flatten x.x) / n in
    {
        x = [[mean]],
        backward = \dLdy grads ->
            let dLdx = rep [dLdy[0][0] / n] in
            x.backward dLdx grads
    }

def mse 'grads (target: [][1]f32): VecBatch [][1] grads -> VecBatch [1][1] grads =
    (`vsub` target) >-> sq >-> mean

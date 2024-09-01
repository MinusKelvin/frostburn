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

-- Math

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

-- Dense layer

type Dense [i][o] 't = {
    weights: [o][i]t,
    bias: [o]t
}
type DenseWeights [i][o] = Dense [i][o] f32

def init [i][o] 't (v: t): *Dense [i][o] t = {
    weights = rep (rep v),
    bias = rep v
}

def map_dense 'a 'b 'c (f: a -> b -> c) (a: Dense [][] a) (b: Dense [][] b): *Dense [][] c = {
    weights = map2_2d f a.weights b.weights,
    bias = map2 f a.bias b.bias
}

def dense [i][o] 'grads
    ({weights, bias}: Dense[i][o] f32)
    (update: Dense[i][o] f32 -> grads -> grads)
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

-- Loss functions

def vsub 'grads (x: VecBatch [][] grads) (nograd: [][]f32) =
    x with x = map2_2d (-) x.x nograd

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

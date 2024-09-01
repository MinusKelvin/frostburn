import "nn"

-- type Weights = {
--     ft_w: [512][768]f32,
--     ft_b: [512]f32,
--     l1_w: [1][512]f32,
--     l1_b: [1]f32,
-- }

-- def model [b] (w: Weights): VecBatch [b][768] Weights -> VecBatch [b][1] Weights =
--     linear w.ft_w (\(g: Weights) d -> g with ft_w = d) >->
--     bias   w.ft_b (\(g: Weights) d -> g with ft_b = d) >->
--     screlu >->
--     linear w.l1_w (\(g: Weights) d -> g with l1_w = d) >->
--     bias   w.l1_b (\(g: Weights) d -> g with l1_b = d) >->
--     sigmoid

type Model = {
    l1: Dense[2][4],
    l2: Dense[4][1],
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

def model [b] (w: Model): VecBatch [b][2] Model -> VecBatch [b][1] Model =
    dense w.l1 (\d (g: Model) -> g with l1 = map2_dense (+) g.l1 d) >->
    crelu >->
    dense w.l2 (\d (g: Model) -> g with l2 = map2_dense (+) g.l2 d) >->
    sigmoid

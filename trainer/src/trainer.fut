def dotprod [n] (x: [n]i64) (y: [n]i64): i64 =
    reduce (+) 0 (map2 (*) x y)

def matmul [n][m][p] (x: [n][m]i64) (y: [m][p]i64): [n][p]i64 =
    map (\xr -> map (\yc -> dotprod xr yc) (transpose y)) x

def to_rowvec [n] (x: [n]i64): [1][n]i64 = unflatten (x :> [1*n]i64)

entry sum_stuff [n] (x: [n]i64): i64 =
    let
        x = to_rowvec x
    in
        reduce (+) 0 (flatten (matmul (transpose x) x))

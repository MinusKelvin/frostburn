use rand::{thread_rng, Rng};
use trainer::{AdamOptions, ArrayF32D1, ArrayF32D2, Context, Dense, Model};

type Result<T, E = trainer::Error> = std::result::Result<T, E>;

#[allow(unused)]
mod trainer {
    include!(concat!(env!("OUT_DIR"), "/trainer.rs"));
}

fn random_2d(ctx: &Context, d1: usize, d2: usize) -> Result<ArrayF32D2> {
    let data: Vec<_> = (0..d1 * d2)
        .map(|_| thread_rng().gen_range(-0.5..0.5))
        .collect();
    ArrayF32D2::new(ctx, [d1 as i64, d2 as i64], data)
}

fn random_1d(ctx: &Context, d1: usize) -> Result<ArrayF32D1> {
    let data: Vec<_> = (0..d1).map(|_| thread_rng().gen_range(-0.1..0.1)).collect();
    ArrayF32D1::new(ctx, [d1 as i64], data)
}

fn random_dense(ctx: &Context, d1: usize, d2: usize) -> Result<Dense> {
    Dense::new(ctx, &random_1d(ctx, d2)?, &random_2d(ctx, d2, d1)?)
}

fn main() {
    let ctx = Context::new().unwrap();
    run(&ctx).unwrap_or_else(|e| {
        panic!(
            "{} ({e})",
            ctx.get_error().as_deref().map_or("<no error>", str::trim)
        )
    });
}

fn run(ctx: &Context) -> Result<()> {
    let options = AdamOptions::new(&ctx, 0.9, 0.999, 1.0e-8, 0.001)?;

    let mut weights = Model::new(&ctx, random_dense(&ctx, 2, 4)?, random_dense(&ctx, 4, 1)?)?;
    let mut state = ctx.adam_init()?;

    #[rustfmt::skip]
    let inputs = ArrayF32D2::new(&ctx, [4, 2], [
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ])?;
    #[rustfmt::skip]
    let targets = ArrayF32D2::new(&ctx, [4, 1], [
        0.0,
        1.0,
        1.0,
        0.0,
    ])?;

    for i in 0..=10000 {
        if i % 1000 == 0 {
            let predictions = ctx.infer(&weights, &inputs)?;

            print_result(&inputs, &predictions, &targets)?;

            println!("l1:");
            print_matrix(&weights.get_l1()?)?;
            println!("l2:");
            print_matrix(&weights.get_l2()?)?;
        }

        let (loss, w, s) = ctx.step(&options, &weights, &state, &inputs, &targets)?;

        if i % 100 == 0 {
            println!("{loss}");
        }

        weights = w;
        state = s;
    }

    Ok(())
}

fn print_result(
    inputs: &ArrayF32D2<'_>,
    predictions: &ArrayF32D2<'_>,
    targets: &ArrayF32D2<'_>,
) -> Result<()> {
    let [items, inlen] = inputs.shape;
    let items = items as usize;
    let inlen = inlen as usize;

    let inputs = inputs.get()?;
    let predictions = predictions.get()?;
    let targets = targets.get()?;

    for y in 0..items {
        for x in 0..inlen {
            print!("{:6.3} ", inputs[y * inlen + x]);
        }
        println!("-> {:6.3} ({:6.3})", predictions[y], targets[y]);
    }

    Ok(())
}

fn print_matrix(arr: &Dense) -> Result<()> {
    let weights = arr.get_weights()?;
    let bias = arr.get_bias()?;
    let [out, inp] = weights.shape;
    let out = out as usize;
    let inp = inp as usize;
    let weights = weights.get()?;
    let bias = bias.get()?;
    for y in 0..out {
        for x in 0..inp {
            print!("{:6.3} ", weights[y * inp + x]);
        }
        println!(" | {:6.3}", bias[y]);
    }
    Ok(())
}

use std::io::Write;
use std::time::Instant;

use dataload::BATCH_SIZE;
use rand::{thread_rng, Rng};
use trainer::{AdamOptions, ArrayF32D1, ArrayF32D2, ArrayI64D2, Context, Linear, Model};

mod dataload;

#[allow(unused)]
mod trainer {
    include!(concat!(env!("OUT_DIR"), "/trainer.rs"));
}

type Result<T, E = trainer::Error> = std::result::Result<T, E>;

const WEIGHT_DECAY: f32 = 1e-6;
const TRAIN_STEPS: usize = 500_000;

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
    let batches = dataload::spawn_data_loader();

    let options = AdamOptions::new(&ctx, 0.9, 0.999, WEIGHT_DECAY, 1e-8, 0.001)?;

    let mut weights = Model::new(
        &ctx,
        init_linear(&ctx, 768, 512)?,
        init_linear(&ctx, 1024, 1)?,
    )?;
    let mut state = ctx.adam_init()?;
    let mut recent_losses = vec![];

    let start = Instant::now();

    for (i, batch) in batches.into_iter().take(TRAIN_STEPS).enumerate() {
        let stm_input = ArrayI64D2::new(ctx, [BATCH_SIZE as i64, 32], batch.stm.as_flattened())?;
        let targets = ArrayF32D2::new(ctx, [BATCH_SIZE as i64, 1], batch.targets)?;

        let t = Instant::now();
        let (loss, w, s) = ctx.step(
            &options,
            &weights,
            &state,
            &stm_input,
            &targets,
        )?;
        println!("{:.3?}", t.elapsed());

        weights = w;
        state = s;
        recent_losses.push(loss);

        if recent_losses.len() == 100 {
            let iter = i + 1;
            let loss = recent_losses.drain(..).sum::<f32>() / 100.0;
            let dur = start.elapsed().as_secs_f64();
            let speed = (iter * BATCH_SIZE) as f64 / dur;
            let secs = dur as i32 % 60;
            let mins = dur as i32 / 60;
            eprint!("\r{iter:>8}/{TRAIN_STEPS}   {speed:>5.0} pos/s   loss: {loss:.6}   time: {mins:2}:{secs:02}    ");
            std::io::stderr().flush().unwrap();
        }
    }

    Ok(())
}

fn init_linear(ctx: &Context, inputs: usize, outputs: usize) -> Result<Linear> {
    let bound = (inputs as f32).sqrt().recip();

    let weights = (0..inputs * outputs)
        .map(|_| thread_rng().gen_range(-bound..=bound))
        .collect::<Vec<_>>();

    let bias = (0..outputs)
        .map(|_| thread_rng().gen_range(-bound..=bound))
        .collect::<Vec<_>>();

    let weights = ArrayF32D2::new(ctx, [outputs as i64, inputs as i64], &weights)?;

    let bias = ArrayF32D1::new(ctx, [outputs as i64], &bias)?;

    Linear::new(ctx, &bias, &weights)
}

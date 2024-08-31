use trainer::Context;

#[allow(unused)]
mod trainer {
    include!(concat!(env!("OUT_DIR"), "/trainer.rs"));
}

fn main() {
    let ctx = Context::new().unwrap();

    let mut iota = vec![];
    for i in 0..10000 {
        iota.push(i + 1);
    }
    let arr = trainer::ArrayI64D1::new(&ctx, [iota.len() as i64], iota).unwrap();

    let t = std::time::Instant::now();
    let result = ctx.sum_stuff(&arr).unwrap();
    println!("{result} {:.2?}", t.elapsed());
}

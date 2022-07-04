use rand::Rng;
use std::fs::OpenOptions;
use std::io::prelude::*;

fn main() {
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("./examples/stress_log.csv")
        .unwrap();
    let mut rng = rand::thread_rng();

    let runs = 10_000_000;

    for i in 0..runs {
        if i % 1_000 == 0 {
            let percent = 100 * i / runs;
            println!("Processed {} out of {} ({} %)", i, runs, percent);
        }
        let gain: f64 = rng.gen_range(0.0..100.0);
        let offset: f64 = rng.gen_range(0.0..100.0);
        let power: f64 = rng.gen_range(1.0..5.0);
        let base: f64 = rng.gen_range(0.0..100.0);
        let min_n = 1;
        let max_n = 100;

        type Func = Box<dyn Fn(f64) -> f64>;

        let functions: Vec<(Func, big_o::Name)> = vec![
            (Box::new(move |x| 0.0 * x + offset), big_o::Name::Constant),
            (
                Box::new(move |x| gain * x.ln() + offset),
                big_o::Name::Logarithmic,
            ),
            (Box::new(move |x| gain * x + offset), big_o::Name::Linear),
            (
                Box::new(move |x| gain * x * x.ln() + offset),
                big_o::Name::Linearithmic,
            ),
            (
                Box::new(move |x| gain * x.powi(2) + offset),
                big_o::Name::Quadratic,
            ),
            (
                Box::new(move |x| gain * x.powi(3) + offset),
                big_o::Name::Cubic,
            ),
            (
                Box::new(move |x| gain * x.powf(power)),
                big_o::Name::Polynomial,
            ),
            (
                Box::new(move |x| gain * base.powf(x)),
                big_o::Name::Exponential,
            ),
        ];

        for (f, name) in functions {
            let data: Vec<(f64, f64)> = (min_n..max_n)
                .map(|i| i as f64)
                .map(|x| (x, f(x)))
                .collect();
            let (complexity, _all) = big_o::infer_complexity(data).unwrap();

            let (a, b) = match name {
                big_o::Name::Constant => (0.0, offset),
                big_o::Name::Polynomial => (gain, power),
                big_o::Name::Exponential => (gain, base),
                _ => (gain, offset),
            };

            let is_ok = name == complexity.name;
            if !is_ok {
                let line = format!("{},{:?},{},{},{:?}", is_ok, name, a, b, complexity.name);
                if let Err(e) = writeln!(file, "{}", line) {
                    eprintln!("Couldn't write to file: {}", e);
                }
            }
        }
    }
}

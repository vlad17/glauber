//! Samples a low-degree connected simple graph and writes it out to a file.

use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fs::File;
use std::io::BufWriter;
use std::io::{Write};
use std::path::PathBuf;
use std::time::Instant;
use std::u32;

use rand::Rng;
use rand_pcg::Lcg64Xsh32;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde_json::json;
use structopt::StructOpt;



/// Generate a connected simple graph with the provided average degree.
#[derive(Debug, StructOpt)]
#[structopt(name = "sample", about = "Sample a connected graph.")]
struct Opt {
    /// Output path prefix for the graph.
    #[structopt(long)]
    out: PathBuf,

    /// Average degree less two (the true average degree is actually larger
    /// than this by about two, because we'd like to ensure the graph is
    /// connected). If this is set too large to be feasible, then graph will
    /// just end up being connected.
    #[structopt(long)]
    degree: usize,

    /// Number of vertices.
    #[structopt(long)]
    nvertices: usize,

    /// Random sampling seed
    #[structopt(long)]
    seed: u64,
}

fn main() {
    let opt = Opt::from_args();
    let n = opt.nvertices;

    // To start with, our graph includes edges from vertex
    // i to i+1 for all i to ensure that it's connected.
    let edges: HashSet<Edge> = (1..n)
        .map(|v| {
            let v: u32 = v.try_into().unwrap();
            fromtup(v - 1, v)
        })
        .collect();

    let mut rng = Lcg64Xsh32::new(0xcafef00dd15ea5e5, opt.seed);
    let to_sample = n * opt.degree;
    let sample_start = Instant::now();
    let additional =
        sample_with_replacement(&mut rng, &edges, c2(n.try_into().unwrap()), to_sample);

    let m = additional.len() + edges.len();
    println!(
        "{}",
        json!({
            "nvertices": n,
            "nedges": m,
            "sample_duration": format!("{:.0?}", Instant::now().duration_since(sample_start))
        })
    );

    let indexing_start = Instant::now();
    let mut neighbors: HashMap<_, Vec<_>> = (0..n).map(|v| (v as u32, Vec::new())).collect();
    for (from, to) in edges.into_iter().chain(additional.into_iter()).map(totup) {
        neighbors.get_mut(&from).expect("vertex").push(to);
    }
    println!(
        "{}",
        json!({
            "indexing_duration": format!("{:.0?}", Instant::now().duration_since(indexing_start))
        })
    );

    let lines_per_file = 10000;
    let nfiles = (n + lines_per_file - 1) / lines_per_file;

    let write_graph_start = Instant::now();
    (0..nfiles).into_par_iter().for_each(|file_ix| {
        let lo = file_ix * n / nfiles;
        let hi = (file_ix + 1) * n / nfiles;
        let hi = hi.min(n);

        let mut fname = opt.out.file_name().expect("file name").to_owned();
        fname.push(format!(".{}", file_ix));

        let new_path = opt.out.with_file_name(fname);
        let file = File::create(&new_path).expect("write file");
        let mut writer = BufWriter::new(file);

        for v in lo..hi {
            let v: u32 = v.try_into().unwrap();
            write!(writer, "{}", v).expect("write src");
            for nbr in &neighbors[&v] {
                write!(writer, " {}", nbr).expect("write dest");
            }
            write!(writer, "\n").expect("newline");
        }
    });

    println!(
        "{}",
        json!({
            "nfiles": nfiles,
            "write_duration": format!("{:.0?}", Instant::now().duration_since(write_graph_start))
        })
    );
}

// To sample with replacement from the set of edges over a simple graph
// over n vertices without incurring the memory overhead of fully generating
// all (n choose 2) edges, we must use a specialized sampler [1].
//
// However, the sampler requires a total order over the items we're drawing a
// combination of, which we achieve by isomorphism to a contiguous integer range.
//
// We modify the sampler to accept a prefix of always-included edges
// which ensures the resulting graph is connected.
//
// [1]: https://stackoverflow.com/a/2394292/1779853
// [2]: https://vladfeinberg.com/2020/03/07/subset-isomorphism.html

type Edge = u64;

/// Samples from the universe `[n] - exclude` with replacement for `k` variables.
fn sample_with_replacement<R: Rng>(
    rng: &mut R,
    exclude: &HashSet<Edge>,
    n: Edge,
    k: usize,
) -> HashSet<Edge> {
    // https://stackoverflow.com/a/2394292/1779853
    assert!(
        exclude.iter().all(|&e| e < n),
        "{:?}",
        exclude
            .iter()
            .filter(|e| **e >= n)
            .copied()
            .collect::<Vec<_>>()
    );
    assert!(
        n >= (k + exclude.len()).try_into().unwrap(),
        "n {} > k {} + exclude.len() {}",
        n,
        k,
        exclude.len()
    );

    let mut start = n;
    for _ in 0..k {
        start -= 1;
        while exclude.contains(&start) {
            start -= 1;
        }
    }
    assert!((start..n).filter(|i| !exclude.contains(i)).count() == k);

    let mut ret = HashSet::new();
    for i in start..n {
        if exclude.contains(&i) {
            continue;
        }
        let j = loop {
            let j = rng.gen_range(0..=i);
            if !exclude.contains(&j) {
                break j;
            }
        };
        if ret.contains(&j) {
            ret.insert(i);
        } else {
            ret.insert(j);
        }
    }

    assert!(ret.is_disjoint(exclude));
    assert!(ret.len() == k);
    ret
}

fn c2(n: Edge) -> Edge {
    n * (n - 1) / 2
}

fn fromtup(i: u32, j: u32) -> Edge {
    assert!(i < j);
    let j = j - i - 1;
    let diagonal = i + j;
    c2((diagonal + 1).into()) + Edge::from(i)
}

fn totup(x: Edge) -> (u32, u32) {
    let larger = (8 * x + 1) as f64;
    let diagonal = ((larger.sqrt() as u64) + 1) / 2 - 1;
    let i = x - c2(diagonal + 1);
    let j = diagonal - i;
    let j = j + i + 1;
    (i.try_into().unwrap(), j.try_into().unwrap())
}

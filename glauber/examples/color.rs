//! Performs greedy coloring over a provided graph stored in simple
//! graph format across sharded input files.

use std::collections::HashMap;
use std::convert::TryInto;

use std::path::PathBuf;
use std::time::Instant;
use std::u32;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_json::json;
use structopt::StructOpt;

use glauber::graph::Graph;
use glauber::{color, graphio, Scanner, SummaryStats};

/// Reads simplified graph format files.
///
/// Computes the number of colors used with a greedy coloring scheme
/// with a max-degree sorting.
///
/// Saves the sampled Glauber colorings as a plain ascii text row of integers
/// with the first number in the row being the current step count.
/// I.e., the `out` file will look like:
///
/// ```
/// 0 0 1 2 3 4
/// 100 3 2 1 3 4
/// 200 3 2 2 2 4
/// ```
///
/// where the above corresponds to vertices `0-4` being initialized with colors
/// `0-4` at step 0, respectively, then changing colors for steps `100` and `200`.
///
/// `out_times` the contains a single line of the elapsed seconds corresponding
/// to each row of the `out` file, i.e., `0.0\n23.1\n46.5\n` would be viable for the above
/// example.
#[derive(Debug, StructOpt)]
#[structopt(name = "color", about = "Sample a uniform graph coloring.")]
struct Opt {
    /// Simple graph format graph files in deduplicated adjacency list form.
    #[structopt(long)]
    graph: Vec<PathBuf>,

    /// Total number of Glauber samples.
    #[structopt(long)]
    nsamples: usize,

    /// Number of steps to take between coloring observations
    #[structopt(long)]
    frequency: usize,

    /// Out file for captured colorings
    #[structopt(long)]
    out: PathBuf,

    /// Out file for the time elapsed at each sample.
    #[structopt(long)]
    out_times: PathBuf,

    /// Random seed
    #[structopt(long)]
    seed: usize,
}

fn main() {
    let opt = Opt::from_args();

    let load_graph_start = Instant::now();
    let graph_scanner = Scanner::new(opt.graph, b' ');
    let graph = graphio::read(&graph_scanner);
    println!(
        "{}",
        json!({
            "load_graph_duration":
                format!("{:.0?}", Instant::now().duration_since(load_graph_start))
        })
    );

    let max_degree = (0..graph.nvertices())
        .into_iter()
        .map(|v| graph.degree(v.try_into().unwrap()))
        .max()
        .expect("nonempty");

    println!(
        "{}",
        json!({
            "nvertices": graph.nvertices(),
            "nedges": graph.nedges(),
            "max_degree": max_degree,
        })
    );

    let ncolors = 2 * max_degree + 1;
    let ncolors: u32 = ncolors.try_into().unwrap();
    let colors_start = Instant::now();
    let colors = color::glauber(
        &graph,
        ncolors,
        opt.nsamples,
        opt.frequency,
        &opt.out,
        &opt.out_times,
        opt.seed,
    );
    let remap = color::remap(ncolors, &colors);
    println!(
        "{}",
        json!({
            "ncolors": ncolors,
            "color_cardinalities": compute_color_cardinalities(&colors, &remap),
            "colors_duration": format!("{:.0?}", Instant::now().duration_since(colors_start)),
        })
    );

    assert!(check_proper_coloring(&graph, &colors));
}

/// Returns a set of summary statistics over the cardinality (number of features)
/// mapping to each color column.
fn compute_color_cardinalities(colors: &[u32], remap: &[u32]) -> HashMap<String, f64> {
    let cards = remap.iter().copied().enumerate().fold(
        HashMap::new(),
        |mut acc: HashMap<u32, u32>, (feature, remap_val)| {
            let entry = acc.entry(colors[feature]).or_default();
            *entry = (*entry).max(remap_val);
            acc
        },
    );
    SummaryStats::from(cards.values().map(|x| *x as f64)).to_map()
}

fn check_proper_coloring(graph: &Graph, colors: &[u32]) -> bool {
    (0..graph.nvertices()).into_par_iter().all(|v| {
        for &nbr in graph.neighbors(v as u32) {
            let nbr = nbr as usize;
            if colors[v] == colors[nbr] {
                return false;
            }
        }
        true
    })
}

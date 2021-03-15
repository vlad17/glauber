//! Simple graph format reader.

use std::convert::TryInto;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::iter;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

use rayon::iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde_json::json;

use crate::{graph::Graph, simsvm, Scanner};

/// Reads a single file behind a scanner into an in-memory graph.
pub fn read(scanner: &Scanner) -> Graph {
    let nvertices = 1 + scanner
        .fold(
            |_| 0,
            |m, line| {
                let line = simsvm::parse(line);
                let target: u32 = line.target();
                line.max().unwrap_or(0).max(target).max(m)
            },
        )
        .max()
        .unwrap_or(0) as usize;

    // TODO: this gets confusing if duplicate edges are present: maybe worth
    // deduping?

    // if you *really* want this to crank then swap out the atomics for sharded owners
    // and use mpsc queues to pass around increment/store messages
    let (offsets, mut edges, offset_time, edge_time) = {
        let mut atomic_offsets: Vec<_> = iter::repeat_with(|| AtomicUsize::new(0))
            .take(nvertices + 1)
            .collect();
        let offset_start = Instant::now();
        scanner
            .fold(
                |_| (),
                |_, line| {
                    let line = simsvm::parse(line);
                    let target: u32 = line.target();
                    for neighbor in line {
                        atomic_offsets[1 + neighbor as usize].fetch_add(1, Ordering::Relaxed);
                        atomic_offsets[1 + target as usize].fetch_add(1, Ordering::Relaxed);
                    }
                },
            )
            .collect::<()>();
        let offset_time = format!("{:.0?}", Instant::now().duration_since(offset_start));

        let mut cumsum = 0;
        for offset in atomic_offsets.iter_mut() {
            cumsum += *offset.get_mut();
            *offset.get_mut() = cumsum;
        }
        let offsets: Vec<usize> = atomic_offsets
            .iter_mut()
            .map(|offset| *offset.get_mut())
            .collect();

        // technically this is double the number of edges
        let nedges = offsets[offsets.len() - 1];
        let atomic_edges: Vec<_> = iter::repeat_with(|| AtomicU32::new(0))
            .take(nedges)
            .collect();
        let edge_start = Instant::now();
        scanner
            .fold(
                |_| (),
                |_, line| {
                    let line = simsvm::parse(line);
                    let target = line.target();
                    for neighbor in line {
                        let target_ix =
                            atomic_offsets[target as usize].fetch_add(1, Ordering::Relaxed);
                        let neighbor_ix =
                            atomic_offsets[neighbor as usize].fetch_add(1, Ordering::Relaxed);
                        atomic_edges[target_ix].store(neighbor, Ordering::Relaxed);
                        atomic_edges[neighbor_ix].store(target, Ordering::Relaxed);
                    }
                },
            )
            .collect::<()>();
        let edge_time = format!("{:.0?}", Instant::now().duration_since(edge_start));
        (
            offsets,
            atomic_edges
                .into_iter()
                .map(|a| a.into_inner())
                .collect::<Vec<_>>(),
            offset_time,
            edge_time,
        )
    };

    let (slice_build_time, sort_time) = {
        // fight the borrow checker
        let slice_build_start = Instant::now();
        let mut head_and_tail = edges.split_at_mut(0);
        let mut neighbor_lists = Vec::with_capacity(offsets.len() - 1);
        for s in offsets.windows(2) {
            let next_chunk = (s[1] - s[0]) as usize;
            head_and_tail = head_and_tail.1.split_at_mut(next_chunk);
            neighbor_lists.push(head_and_tail.0);
        }
        let slice_build_time = format!("{:.0?}", Instant::now().duration_since(slice_build_start));
        let sort_start = Instant::now();
        neighbor_lists
            .par_iter_mut()
            .for_each(|s| s.sort_unstable());
        let sort_time = format!("{:.0?}", Instant::now().duration_since(sort_start));
        (slice_build_time, sort_time)
    };

    println!(
        "{}",
        json!({
            "sort_time": sort_time,
            "edge_time": edge_time,
            "offset_time": offset_time,
            "slice_build_time": slice_build_time,
        })
    );

    Graph::new(offsets, edges)
}

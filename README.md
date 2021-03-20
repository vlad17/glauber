# Glauber Dynamics Experiment

This repository contains some Rust code for simulating Glauber dynamics. In particular, given a fixed simple graph G, how may one sample a random, uniform proper coloring of this graph?

If we allow enough colors, then coloring is clearly feasible, since `chi(G) <= Delta(G) + 1`, where `chi` is the chromatic number and `Delta` is the maximum degree.

That said, for `n` vertices, a proper `k`-coloring is some `n`-length array of integers in `[k]`. Simply enumerating all such arrays isn't easy, so sampling is nontrivial.

## Available routines

Run with `--help` for details.

| command | description |
| --- | --- | 
| `cargo run sample` | sample a low-average-degree graph |
| `cargo run color` | color a graph, emit diagnostics |

```
cargo run --example sample -- --out test --degree 5 --nvertices 100 --seed 1234
cargo run --example color -- --graph test.0 --nsamples 1000000 --frequency 100000 --out colors.txt --out-times times.txt 
```

eval: use the same starting point intentionally, then ask what’s the earthmover distance (allowing arbitrary color permutation) between the starting point and the final point: if it’s truly random then the distance will increase: average distance is quality.
solve via assignment problem
(this is just hungarian)

## .graph format

A simple graph processing and coloring library for a custom format.

Consider a simple text graph format where each line conforms to the following:

```
<source> <dest0> <dest1> <dest2>...
```

This identifies edges between the first node, `<source>` and all subsequent `<dest*>` nodes.

These must be `uint32`-sized integers written in plain ascii text, space delimited, with no other funny stuff in the files. The edges going the other way are not encoded. This creates a simple adjacency list format. For instance, the file

```
3 4 5
5 4
```

Defines a simple graph over 6 vertices, integers 0 to 5 inclusive, with a triangle subgraph between 3, 4, 5.
Note the range is compactified, so singleton vertices are implicitly generated.

This repository contains some utilities that process such simple graph files.

Parallelism is determined at the file level, where a `.graph` file can be split up vertically and all splits are handled on different processors for I/O. For this reason a useful operation for "load balancing" inputs:

```
cat input*.graph | split --numeric-suffixes --line-bytes 1M - balanced.graph.
```

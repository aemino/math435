pub mod simplex;
use crate::simplex::{faces, SimplicialComplex};
use petgraph::graph::{DiGraph, NodeIndex};
use rand::{seq::IteratorRandom, thread_rng};
use std::collections::{HashMap, HashSet};
fn main() {
    let mut rng = rand::thread_rng();
    let mut graph: DiGraph<u32, ()> = DiGraph::new();
    let num_nodes = 30;
    let num_edges = 1000;
    for i in 0..num_nodes {
        graph.add_node(i as u32);
    }
    let mut simplicial_complex = SimplicialComplex::new((0..num_nodes).collect());
    let mut so_far: HashSet<Vec<usize>> = HashSet::new();
    // let samples = vec![
    //     vec![1, 2],
    //     vec![2, 3],
    //     vec![1, 3],
    //     vec![3, 5],
    //     vec![5, 7],
    //     vec![1, 7],
    // ];
    for i in 0..num_edges {
        let choices: Vec<usize> = (0..num_nodes).collect();
        let sample: Vec<usize> = choices
            .iter()
            .choose_multiple(&mut rng, 2)
            .into_iter()
            .cloned()
            .collect();
        // let sample: Vec<usize> = samples[i].iter().cloned().collect();
        if so_far.contains(&vec![sample[1], sample[0]]) || so_far.contains(&sample) {
            continue;
        } else {
            so_far.insert(sample.clone());
        }
        simplicial_complex.update(sample.clone());

        graph.add_edge(NodeIndex::new(sample[0]), NodeIndex::new(sample[1]), ());
        let lengths: Vec<usize> = simplicial_complex
            .simplices
            .iter()
            .map(|simplex| simplex.len())
            .collect();

        if i % 100 == 0 {
            println!("{:?}", lengths);
            let bettis = simplicial_complex.betti_numbers();

            println!("{:?}", bettis);
        }
    }
}

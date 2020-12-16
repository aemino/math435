use connectome_model::{
    sim::Simulation,
    simplex::{faces, SimplicialComplex},
};
use petgraph::graph::{DiGraph, NodeIndex};
use rand::{rngs::ThreadRng, seq::IteratorRandom, thread_rng, Rng};
use std::collections::{HashMap, HashSet};

const NUM_NODES: u32 = 6;

fn main() {
    let rng = rand::thread_rng();

    let mut simulation = Simulation::<ThreadRng>::new(1., 0.5, 0.01, 5, rng);
    let mut rng = rand::thread_rng();
    simulation.init_uniform(1, NUM_NODES);
    let mut simplicial_complex = SimplicialComplex::new((0..NUM_NODES.pow(3) as usize).collect());
    let mut vec: HashSet<(usize, usize)> = HashSet::new();
    let mut i = 0;
    loop {
        let step_result = simulation.step(&[rng.gen_range(0, NUM_NODES.pow(3) as usize)]);
        for (in_node, out_node) in step_result.removed_edges {
            
            // println!("{:?} {}", vec, vec.len());
            assert!(vec.remove(&(in_node, out_node)));
            simplicial_complex.remove(vec![in_node, out_node]);
        }
        for (in_node, out_node) in step_result.added_edges {
            // println!("{:?} {}", vec, vec.len());
            vec.insert((in_node, out_node));
            simplicial_complex.add(vec![in_node, out_node]);
        }

        i += 1;

        if i % 10 == 0 {
            
            let lengths: Vec<usize> = simplicial_complex
                .simplices
                .iter()
                .map(|simplex| simplex.len())
                .collect();
            let betti_numbers = simplicial_complex.betti_numbers();
            // let betti_numbers = vec![0];
            println!(
                "simplex sizes: {:?}\n\n betti numbers: {:?}\n\n",
                lengths, betti_numbers
            );
            if betti_numbers.contains(&-1) {
                println!("{:?}", simplicial_complex.simplex_indices);
            }
        }
    }
}

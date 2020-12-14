use std::collections::HashMap;

use csv::Writer;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use petgraph::{
    self,
    dot::{Config, Dot},
    graph::DiGraph,
    graph::NodeIndex,
    EdgeDirection,
};
use rand::prelude::*;
use rand_distr::InverseGaussian;
use rayon::prelude::*;

struct Simulation<R, D> {
    rng: R,
    fitness_dist: D,
    temperature: f64,
    graph: DiGraph<(f64, f64), ()>,
}

impl<R, D> Simulation<R, D>
where
    R: Rng,
    D: Distribution<f64>,
{
    fn init(rng: R, fitness_dist: D, temperature: f64) -> Self {
        let mut sim = Self {
            rng,
            fitness_dist,
            temperature,
            graph: DiGraph::new(),
        };

        let node_a = sim.add_sampled_node();
        let node_b = sim.add_sampled_node();
        let node_c = sim.add_sampled_node();

        sim.graph.add_edge(node_a, node_b, ());
        sim.graph.add_edge(node_b, node_c, ());
        sim.graph.add_edge(node_c, node_a, ());

        sim
    }

    fn sample_node_properties(&mut self) -> (f64, f64) {
        let fitness = self.fitness_dist.sample(&mut self.rng);

        // The fitness distribution is expected to include only positive values.
        // `ln_1p` is used to ensure that the energy level is also positive.
        let energy_level = self.temperature * fitness.ln_1p();

        (fitness, energy_level)
    }

    fn add_sampled_node(&mut self) -> NodeIndex<u32> {
        let props = self.sample_node_properties();

        self.graph.add_node(props)
    }

    fn try_add_node(&mut self) -> Option<NodeIndex<u32>> {
        let mut attach_weights = HashMap::with_capacity(self.graph.node_count());

        for node in self.graph.node_indices() {
            let (_, energy_level) = self.graph.node_weight(node).unwrap();
            let degree = self.graph.neighbors_undirected(node).count() as f64;

            attach_weights.insert(node, energy_level * degree);
        }

        let attach_weights_sum: f64 = attach_weights.values().sum();

        let new_node = self.add_sampled_node();

        let mut degree: usize = 0;

        for (node, weight) in attach_weights {
            if attach_weights_sum > 0. && !self.rng.gen_bool(weight / attach_weights_sum) {
                continue;
            }

            self.graph.add_edge(new_node, node, ());
            degree += 1;
        }

        if degree < 1 {
            self.graph.remove_node(new_node);
            return None;
        }

        Some(new_node)
    }

    fn step(&mut self) {
        let _new_node = loop {
            if let Some(node) = self.try_add_node() {
                break node;
            }
        };
    }

    fn graph(&self) -> &DiGraph<(f64, f64), ()> {
        &self.graph
    }
}

fn main() {
    const NUM_STEPS: u64 = 10000;
    const NUM_RUNS: u64 = 1000;
    const INITIAL_TEMPERATURE: f64 = 1.0;

    let mut csv = Writer::from_path("out/10k_1e1l.csv").unwrap();
    csv.write_record(&["id", "run", "in_degree", "fitness"]).unwrap();

    let pb = ProgressBar::new(NUM_RUNS).with_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar}] {pos}/{len} ({per_sec}, eta {eta})",
    ));

    (0..NUM_RUNS)
        .into_par_iter()
        .progress_with(pb)
        .flat_map_iter(|run| {
            let fitness_dist = InverseGaussian::new(1.0, 10.0).unwrap();

            let mut simulation = Simulation::init(thread_rng(), fitness_dist, INITIAL_TEMPERATURE);

            for _ in 0..NUM_STEPS {
                simulation.step();
            }

            simulation.graph().node_indices().map(move |node| {
                [
                    node.index().to_string(),
                    run.to_string(),
                    simulation
                        .graph()
                        .neighbors_directed(node, EdgeDirection::Incoming)
                        .count()
                        .to_string(),
                    simulation.graph().node_weight(node).unwrap().0.to_string(),
                ]
            })
        })
        .collect::<Vec<_>>()
        .iter()
        .for_each(|record| csv.write_record(record).unwrap());
}

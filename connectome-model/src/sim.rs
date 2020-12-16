use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
};

use nalgebra::{distance_squared, Point3};
use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph, visit::EdgeRef, EdgeDirection};
use rand::Rng;

pub struct NodeWeight {
    pub position: Point3<f64>,
    pub last_active: Option<usize>,
}

impl NodeWeight {
    pub fn is_active(&self, timestep: usize) -> bool {
        match self.last_active {
            Some(last_active) => last_active == timestep,
            None => false,
        }
    }

    pub fn set_active(&mut self, timestep: usize) {
        self.last_active = Some(timestep);
    }
}

#[derive(Default)]
pub struct EdgeWeight {
    pub myelination: usize,
    pub activation_queue: BinaryHeap<Reverse<usize>>,
}

impl EdgeWeight {
    pub fn myelination_prob(&self, max: usize) -> f64 {
        (max - self.myelination) as f64 / (max + self.myelination) as f64
    }
}

pub struct StepResult {
    pub removed_edges: Vec<(usize, usize)>,
    pub added_edges: Vec<(usize, usize)>,
}

pub struct Simulation<R: Rng> {
    pub timestep: usize,
    pub connectivity_rate: f64,
    pub myelination_rate: f64,
    pub decay_rate: f64,
    pub max_myelination: usize,
    pub graph: StableDiGraph<NodeWeight, EdgeWeight>,
    pub rng: R,
}

impl<R> Simulation<R>
where
    R: Rng,
{
    pub fn new(
        connectivity_rate: f64,
        myelination_rate: f64,
        decay_rate: f64,
        max_myelination: usize,
        rng: R,
    ) -> Self {
        Self {
            timestep: Default::default(),
            connectivity_rate,
            myelination_rate,
            decay_rate,
            max_myelination,
            graph: StableDiGraph::new(),
            rng,
        }
    }

    /// Initializes nodes in a uniform grid, spaced `dist` units apart in each
    /// direction, with `n^3` total nodes.
    pub fn init_uniform(&mut self, dist: u32, n: u32) {
        let dist = dist as f64;
        let min = (n - 1) as f64 * dist * 0.5;

        for xs in 0..n {
            let x = xs as f64 * dist - min;

            for ys in 0..n {
                let y = ys as f64 * dist - min;

                for zs in 0..n {
                    let z = zs as f64 * dist - min;

                    self.graph.add_node(NodeWeight {
                        position: Point3::new(x, y, z),
                        last_active: None,
                    });
                }
            }
        }
    }

    /// Steps the simulation forward by a single timestep.
    pub fn step(&mut self, activations: &[usize]) -> StepResult {
        let next_timestep = self.timestep + 1;

        let mut pending_removed_edges = Vec::new();
        let mut pending_activations = activations
            .iter()
            .map(|&id| NodeIndex::new(id))
            .collect::<HashSet<_>>();

        for id in self.graph.edge_indices().collect::<Vec<_>>() {
            let edge = &mut self.graph[id];

            // Compute the myelination probability with the max + 1. This
            // ensures that the probability doesn't reach zero, with the side
            // effect of decreasing overall decay probability.
            let decay_prob = edge.myelination_prob(self.max_myelination + 1) * self.decay_rate;

            if self.rng.gen_bool(decay_prob) {
                self.graph.remove_edge(id);

                pending_removed_edges.push(self.graph.edge_endpoints(id).unwrap());
                continue;
            }

            if !edge
                .activation_queue
                .peek()
                .map_or(false, |t| t.0 == next_timestep)
            {
                // The outgoing node is not scheduled to be activated in the next timestep.
                continue;
            }

            edge.activation_queue.pop();

            let (_, target_id) = self.graph.edge_endpoints(id).unwrap();
            pending_activations.insert(target_id);
        }

        let mut pending_added_edges = Vec::new();

        for &target_id in &pending_activations {
            let target_node = &self.graph[target_id];

            for source_id in self.graph.node_indices() {
                if target_id == source_id {
                    continue;
                }

                // An edge already exists between these nodes; don't bother trying to compute attachment.
                if self
                    .graph
                    .find_edge_undirected(source_id, target_id)
                    .is_some()
                {
                    continue;
                }

                let source_node = &self.graph[source_id];

                if let Some(last_active) = source_node.last_active {
                    let delta_timestep = (next_timestep - last_active) as f64;
                    let distance = distance_squared(&target_node.position, &source_node.position);
                    let attachment_prob =
                        self.connectivity_rate * (delta_timestep.exp() * distance).recip();

                    if self.rng.gen_bool(attachment_prob) {
                        pending_added_edges.push((source_id, target_id));
                    }
                }
            }
        }

        self.timestep = next_timestep;

        for (source_id, target_id) in &pending_added_edges {
            self.graph
                .add_edge(*source_id, *target_id, EdgeWeight::default());
        }

        for &id in &pending_activations {
            let node = &mut self.graph[id];
            node.set_active(self.timestep);

            for edge_id in self
                .graph
                .edges_directed(id, EdgeDirection::Outgoing)
                .map(|edge_ref| edge_ref.id())
                .collect::<Vec<_>>()
            {
                let edge = &mut self.graph[edge_id];
                edge.activation_queue.push(Reverse(
                    self.timestep + 1 + (self.max_myelination - edge.myelination),
                ));

                if edge.myelination >= self.max_myelination {
                    continue;
                }

                let myelination_prob =
                    edge.myelination_prob(self.max_myelination) * self.myelination_rate;

                if self.rng.gen_bool(myelination_prob) {
                    edge.myelination += 1;
                }
            }
        }

        StepResult {
            removed_edges: pending_removed_edges
                .iter()
                .map(|(a, b)| (a.index(), b.index()))
                .collect(),
            added_edges: pending_added_edges
                .iter()
                .map(|(a, b)| (a.index(), b.index()))
                .collect(),
        }
    }
}

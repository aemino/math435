use std::{cmp::Reverse, collections::BinaryHeap};

use nalgebra::{distance_squared, Point3};
use petgraph::{graph::DiGraph, visit::EdgeRef, EdgeDirection};
use rand::Rng;

pub struct NodeWeight {
    pub position: Point3<f64>,
    pub last_active: usize,
}

impl NodeWeight {
    pub fn is_active(&self, timestep: usize) -> bool {
        timestep == self.last_active
    }

    pub fn set_active(&mut self, timestep: usize) {
        self.last_active = timestep;
    }
}

#[derive(Default)]
pub struct EdgeWeight {
    pub myelination: usize,
    pub activation_queue: BinaryHeap<Reverse<usize>>,
}

pub struct StepResult {
    pub added_edges: Vec<(usize, usize)>,
    pub removed_edges: Vec<(usize, usize)>,
}

pub struct Simulation<R: Rng> {
    pub timestep: usize,
    pub temperature: f64,
    pub max_myelination: usize,
    pub graph: DiGraph<NodeWeight, EdgeWeight>,
    pub rng: R,
}

impl<R> Simulation<R>
where
    R: Rng,
{
    pub fn new(temperature: f64, max_myelination: usize, rng: R) -> Self {
        Self {
            timestep: Default::default(),
            temperature,
            max_myelination,
            graph: DiGraph::new(),
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
                        last_active: self.timestep,
                    });
                }
            }
        }
    }

    /// Steps the simulation forward by a single timestep.
    pub fn step(&mut self) -> StepResult {
        let next_timestep = self.timestep + 1;

        let mut pending_activations = Vec::new();

        for id in self.graph.edge_indices() {
            let edge = &mut self.graph[id];

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
            pending_activations.push(target_id);
        }

        let mut pending_new_edges = Vec::new();

        for &target_id in &pending_activations {
            let target_node = &self.graph[target_id];

            for source_id in self.graph.node_indices() {
                if target_id == source_id {
                    continue;
                }

                // An edge already exists between these nodes; don't bother trying to compute attachment.
                if self.graph.edges_connecting(source_id, target_id).count() > 0 {
                    continue;
                }

                let source_node = &self.graph[source_id];

                let delta_timestep = (next_timestep - source_node.last_active) as f64;
                let distance = distance_squared(&target_node.position, &source_node.position);
                let attachment_prob = self.temperature * (delta_timestep.exp() * distance).recip();

                if self.rng.gen_bool(attachment_prob) {
                    pending_new_edges.push((source_id, target_id));
                }
            }
        }

        self.timestep = next_timestep;

        for (source_id, target_id) in &pending_new_edges {
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
                edge.activation_queue.push(Reverse(self.timestep + 1));
            }
        }

        StepResult {
            added_edges: pending_new_edges
                .iter()
                .map(|(a, b)| (a.index(), b.index()))
                .collect(),
            removed_edges: vec![],
        }
    }
}

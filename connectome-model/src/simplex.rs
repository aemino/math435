use bimap::BiHashMap;
use nalgebra::{Dynamic, Matrix, VecStorage};
use std::collections::{HashMap, HashSet};

type GenericMatrix = Matrix<u64, Dynamic, Dynamic, VecStorage<u64, Dynamic, Dynamic>>;
pub struct SimplicialComplex {
    pub simplices: Vec<HashMap<Vec<usize>, HashSet<usize>>>,
    pub simplex_indices: Vec<BiHashMap<usize, Vec<usize>>>,
    pub boundary_matrices: Vec<GenericMatrix>,
}

impl SimplicialComplex {
    pub fn new(vertices: Vec<usize>) -> Self {
        SimplicialComplex {
            simplices: vec![vertices
                .iter()
                .map(|&v| (vec![v], HashSet::new()))
                .collect()],
            simplex_indices: vec![],
            boundary_matrices: vec![],
        }
    }

    pub fn faces(simplex: &Vec<usize>) -> Vec<Vec<usize>> {
        let mut faces: Vec<Vec<usize>> = Vec::new();
        for i in 0..simplex.len() {
            let mut sub_simplex: Vec<usize> = simplex[..i].iter().cloned().collect();
            sub_simplex.extend(simplex[i + 1..].iter());
            faces.push(sub_simplex);
        }
        faces
    }

    fn add_row(&mut self, mat_index: usize) {
        let n_rows = self.boundary_matrices[mat_index].nrows() + 1;
        self.boundary_matrices[mat_index].resize_vertically_mut(n_rows, 0);
    }

    fn add_column(&mut self, mat_index: usize, indices: Vec<usize>) {
        let n_cols = self.boundary_matrices[mat_index].ncols() + 1;
        self.boundary_matrices[mat_index].resize_horizontally_mut(n_cols, 0);
        for index in indices {
            self.boundary_matrices[mat_index][(index, n_cols - 1)] = 1;
        }
    }

    /// Return betti numbers, 1 and onward.
    pub fn betti_numbers(&self) -> Vec<usize> {
        let mut betti_numbers: Vec<usize> = vec![0];
        for (i, matrix) in self.boundary_matrices.iter().enumerate() {
            betti_numbers.push(matrix.ncols() - rank(matrix));
            betti_numbers[i] += betti_numbers[i + 1] - matrix.ncols();
        }
        betti_numbers.remove(0);
        betti_numbers
    }

    pub fn update(&mut self, simplex: Vec<usize>) {
        if self.simplices.len() < simplex.len() - 1 {
            self.simplices.push(HashMap::new());
            self.simplex_indices.push(BiHashMap::new());
        }
        if simplex.len() == 2 {
            if self.simplices[0]
                .get(&vec![simplex[0]])
                .unwrap_or(&HashSet::new())
                .contains(&simplex[1])
            {
                return;
            }
        }
        let mut column_indices: Vec<usize> = Vec::new();
        for (i, face) in SimplicialComplex::faces(&simplex).into_iter().enumerate() {
            self.simplices[simplex.len() - 2]
                .entry(face.clone())
                .or_insert(HashSet::new())
                .insert(simplex[i]);
            if simplex.len() > 2 {
                let index = self.simplex_indices[simplex.len() - 3].len();

                if !self.simplex_indices[simplex.len() - 3].contains_right(&face) {
                    column_indices.push(index);
                    self.simplex_indices[simplex.len() - 3].insert(index, face);
                    self.add_row(simplex.len() - 3);
                } else {
                    column_indices.push(
                        *self.simplex_indices[simplex.len() - 3]
                            .get_by_right(&face)
                            .unwrap(),
                    );
                }
            }
        }
        if simplex.len() > 2 {
            self.add_column(simplex.len() - 3, column_indices);
        }

        let prefix_simplex: Vec<usize> = simplex[1..].iter().cloned().collect();
        let suffix_simplex: Vec<usize> = simplex[..simplex.len() - 1].iter().cloned().collect();

        // let simplex_2: Vec<usize> = simplex[..simplex.len() - 1].iter().cloned().collect();
        let empty = HashSet::new();
        let options: HashSet<usize> = (&self.simplices[simplex.len() - 2])
            .get(&prefix_simplex)
            .unwrap_or(&empty)
            .intersection(
                (&self.simplices[simplex.len() - 2])
                    .get(&suffix_simplex)
                    .unwrap_or(&empty),
            )
            .into_iter()
            .cloned()
            .collect();

        for &node in &options {
            let mut super_simplex: Vec<usize> = Vec::new();
            for (i, &n) in simplex.iter().enumerate() {
                if (&self.simplices[1]).contains_key(&vec![node, n]) {
                    super_simplex.push(node);
                    super_simplex.extend(simplex[i..].iter());

                    break;
                }
                super_simplex.push(n);
            }
            if super_simplex.len() == 3 {
                if ((&self.simplices[1]).contains_key(&vec![super_simplex[0], super_simplex[1]])
                    && (&self.simplices[1]).contains_key(&vec![super_simplex[1], super_simplex[2]])
                    && (&self.simplices[1]).contains_key(&vec![super_simplex[2], super_simplex[0]]))
                    || ((&self.simplices[1])
                        .contains_key(&vec![super_simplex[1], super_simplex[0]])
                        && (&self.simplices[1])
                            .contains_key(&vec![super_simplex[2], super_simplex[1]])
                        && (&self.simplices[1])
                            .contains_key(&vec![super_simplex[0], super_simplex[2]]))
                {
                    continue;
                }
            }
            self.update(super_simplex);
        }

        if options.len() == 0 {
            let index = self.simplex_indices[simplex.len() - 2].len();
            self.simplex_indices[simplex.len() - 2].insert(index, simplex.clone());
            self.simplices[simplex.len() - 2].insert(simplex, HashSet::new());
        }
    }

    pub fn delete(&mut self, simplex: Vec<usize>) {
        if simplex.len() == 2 {
            self.simplices[simplex.len() - 1]
                .get_mut(&vec![simplex[0]])
                .unwrap()
                .remove(&simplex[1]);
            self.simplices[simplex.len() - 1]
                .get_mut(&vec![simplex[1]])
                .unwrap()
                .remove(&simplex[0]);
        }
        let &simplex_row = self.simplex_indices[simplex.len() - 2]
            .get_by_right(&simplex)
            .unwrap();
        let super_simplex_indices: Vec<usize> = self.boundary_matrices[simplex.len() - 2]
            .row(simplex_row)
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| if e == 1 { Some(i) } else { None })
            .collect();
        self.boundary_matrices[simplex.len() - 2] = self.boundary_matrices[simplex.len() - 2]
            .clone()
            .remove_row(simplex_row)
            .remove_columns_at(&super_simplex_indices);
        for i in super_simplex_indices {
            let super_simplex = self.simplex_indices[simplex.len() - 1]
                .get_by_left(&i)
                .unwrap()
                .clone();
            self.delete(super_simplex.clone());
            self.simplices[simplex.len() - 1].remove_entry(&super_simplex);
        }
    }
}

/// Get rank of matrix with finite field of order 2.
pub fn rank(mat: &GenericMatrix) -> usize {
    let mut mat_dup = mat.clone();
    let mut mat_rank = 0;
    for x in 0..mat_dup.ncols() {
        for i in x..mat_dup.nrows() {
            for j in 0..mat_dup.ncols() {
                if mat_dup[(i, j)] == 1 {
                    mat_dup.swap_rows(x, i);
                    mat_dup.swap_columns(x, j);
                }
            }
        }

        if mat_dup[(x, x)] == 0 {
            return mat_rank;
        }

        for i in x + 1..mat_dup.nrows() {
            if mat_dup[(i, x)] == 1 {
                for j in 0..mat_dup.ncols() {
                    mat_dup[(i, j)] = (mat_dup[(i, j)] + mat_dup[(x, j)]) % 2;
                }
            }
        }

        for i in x + 1..mat_dup.nrows() {
            if mat_dup[(i, x)] == 1 {
                for j in 0..mat_dup.ncols() {
                    mat_dup[(i, j)] = (mat_dup[(i, j)] + mat_dup[(i, x)]) % 2;
                }
            }
        }

        mat_rank += 1
    }
    mat_rank
}

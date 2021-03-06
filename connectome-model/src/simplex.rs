use bimap::BiHashMap;
use nalgebra::{Dynamic, Matrix, VecStorage};
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;


type GenericMatrix = Matrix<u64, Dynamic, Dynamic, VecStorage<u64, Dynamic, Dynamic>>;
pub struct SimplicialComplex {
    pub simplices: Vec<HashMap<Vec<usize>, HashSet<usize>>>,
    pub simplex_indices: Vec<BiHashMap<usize, Vec<usize>>>,
    pub boundary_matrices: Vec<GenericMatrix>,
}

impl SimplicialComplex {
    pub fn new(vertices: Vec<usize>) -> Self {
        SimplicialComplex {
            simplices: vec![HashMap::new()],
            simplex_indices: vec![BiHashMap::new()],
            boundary_matrices: vec![GenericMatrix::from_iterator(1, 1, vec![0u64])],
        }
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
    pub fn betti_numbers(&self) -> Vec<i64> {
        let mut betti_numbers: Vec<i64> = vec![0];
        for (i, matrix) in self.boundary_matrices.iter().enumerate() {
            let rank = rank(matrix);

            betti_numbers.push(matrix.ncols() as i64 - 1 - rank as i64);
            betti_numbers[i] -= rank as i64;
        }
        betti_numbers.remove(0);
        betti_numbers.remove(betti_numbers.len() - 1);

        betti_numbers
    }

    pub fn add(&mut self, simplex: Vec<usize>) {
        
        if self.simplices.len() < simplex.len() + 1 {
            self.simplices.push(HashMap::new());
            self.simplex_indices.push(BiHashMap::new());
            self.boundary_matrices
                .push(GenericMatrix::from_iterator(1, 1, vec![0u64]));
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

        for (i, face) in faces(&simplex).into_iter().enumerate() {
            self.simplices[simplex.len() - 2]
                .entry(face.clone())
                .or_insert(HashSet::new())
                .insert(simplex[i]);
            // Add one to the index because of the dummy element in the matrix to allow for the addition of rows and columns.
            let index = self.simplex_indices[simplex.len() - 2].len() + 1;

            if !self.simplex_indices[simplex.len() - 2].contains_right(&face) {
                column_indices.push(index);
                self.simplex_indices[simplex.len() - 2].insert(index, face);
                self.add_row(simplex.len() - 2);
            } else {
                column_indices.push(
                    *self.simplex_indices[simplex.len() - 2]
                        .get_by_right(&face)
                        .unwrap(),
                );
            }
        }
        self.add_column(simplex.len() - 2, column_indices);

        let prefix_simplex: Vec<usize> = simplex[1..].iter().cloned().collect();

        let empty = HashSet::new();
        let mut options: HashSet<usize> = &empty
            | (&self.simplices[simplex.len() - 2])
                .get(&prefix_simplex)
                .unwrap_or(&empty);
        for (_, face) in faces(&simplex).into_iter().enumerate() {
            options = &options
                & (&self.simplices[simplex.len() - 2])
                    .get(&face)
                    .unwrap_or(&empty);
        }
        let mut option_count =0;
        for &node in &options {
            let mut super_simplex: Vec<usize> = Vec::new();
            let mut pushed = false;
            for (i, &n) in simplex.iter().enumerate() {
                if (&self.simplices[1]).contains_key(&vec![node, n]) {
                    super_simplex.push(node);
                    super_simplex.extend(simplex[i..].iter());
                    pushed = true;
                    break;
                }
                super_simplex.push(n);
            }
            if !pushed {
                super_simplex.push(node);
            }
            if simplex.len() == 2 {
                let edge_map = &self.simplices[1];
                if edge_map.contains_key(&vec![simplex[1], node])
                    && edge_map.contains_key(&vec![node, simplex[0]])
                {
                    continue;
                }
            }
            option_count+=1;
            self.add(super_simplex);
        }
        // if there is nothing above it, so it won't be added backwards.
        if option_count == 0 {
            // Add one to the index because of the dummy element in the matrix to allow for the addition of rows and columns.
            let index = self.simplex_indices[simplex.len() - 1].len() + 1;
            if !self.simplex_indices[simplex.len() - 1].contains_right(&simplex) {
                self.simplex_indices[simplex.len() - 1].insert(index, simplex.clone());
                self.add_row(simplex.len() - 1);
                self.simplices[simplex.len() - 1].insert(simplex.clone(), HashSet::new());
            }
        }
    }
    pub fn update_simplex_indices(&mut self, index: usize) {
        let mut old_key = 0;
        let mut indices: Vec<usize> = self.simplex_indices[index].left_values().into_iter().cloned().collect();
        indices.sort();
        for key in indices {
            
            if key !=old_key+1{
                let new = self.simplex_indices[index].remove_by_left(&key).unwrap().1;
                self.simplex_indices[index].insert(old_key+1, new);
            }
            old_key+=1
        }
    }

    pub fn remove(&mut self, simplex: Vec<usize>) {
        if simplex.len() == 2 {
            assert!(self.simplices[0]
                .get_mut(&vec![simplex[0]])
                .unwrap()
                .remove(&simplex[1]));
            assert!(self.simplices[0]
                .get_mut(&vec![simplex[1]])
                .unwrap()
                .remove(&simplex[0]));
            

            let &simplex_row = self.simplex_indices[0].get_by_right(&vec![simplex[0]]).unwrap();
            let &simplex_2_row = self.simplex_indices[0].get_by_right(&vec![simplex[1]]).unwrap();
            let mut super_simplex_indices: HashSet<usize> = self.boundary_matrices[0]
                .row(simplex_row)
                .iter()
                .enumerate()
                .filter_map(|(i, &e)| if e == 1 { Some(i) } else { None })
                .collect();

            super_simplex_indices = &super_simplex_indices & &self.boundary_matrices[0]
                .row(simplex_2_row)
                .iter()
                .enumerate()
                .filter_map(|(i, &e)| if e == 1 { Some(i) } else { None })
                .collect();
            assert!(super_simplex_indices.len()==1);

            self.boundary_matrices[0] = self.boundary_matrices[0]
                    .clone().remove_column(*super_simplex_indices.iter().next().unwrap());
            
            
        }

        assert!(self.simplices[simplex.len()-1].remove_entry(&simplex).is_some());


        let &simplex_row = self.simplex_indices[simplex.len() - 1].get_by_right(&simplex).unwrap();
        let super_simplex_indices: Vec<usize> = self.boundary_matrices[simplex.len() - 1]
            .row(simplex_row)
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| if e == 1 { Some(i) } else { None })
            .collect();
        
        for &i in &super_simplex_indices {

            let sub_simplex_indices: Vec<Vec<usize>> = self.boundary_matrices[simplex.len() - 1]
                .column(i)
                .iter()
                .enumerate()
                .filter_map(|(i, &e)| if e == 1 { Some(i) } else { None }).take(3).map(|j| self.simplex_indices[simplex.len()-1].get_by_left(&j).unwrap().clone())
                .collect();
            
            
            let super_simplex = combine_simplices(&sub_simplex_indices);
            self.remove(super_simplex.clone());
            // self.simplices[simplex.len()].remove_entry(&super_simplex);
        }
        self.boundary_matrices[simplex.len() - 1] = self.boundary_matrices[simplex.len() - 1]
            .clone().remove_row(simplex_row)
            .remove_columns_at(&super_simplex_indices);
        assert!(self.simplex_indices[simplex.len() - 1].remove_by_right(&simplex).is_some());
        self.update_simplex_indices(simplex.len() - 1);
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

pub fn combine_simplices(sub_simplices: &Vec<Vec<usize>>)-> Vec<usize> {
    let mut elements: HashSet<usize> = sub_simplices[0].iter().cloned().collect();
    elements = &elements | &sub_simplices[1].iter().cloned().collect();
    let mut partial: HashMap<(usize, usize), bool> = HashMap::new();
    for &element in &elements{
        for &element_2 in &elements {
            for sub_simplex in sub_simplices{
                if sub_simplex.contains(&element)&&sub_simplex.contains(&element_2){
                    partial.insert((element, element_2), sub_simplex.iter().position(|&r| r == element).unwrap()<sub_simplex.iter().position(|&r| r == element_2).unwrap());
                }
            }
            
        }
    }
    let mut simplex: Vec<usize> = elements.into_iter().collect();
    simplex.sort_by(|&a, &b|if partial[&(a, b)] {Ordering::Less} else {Ordering::Greater} );
    simplex

    
}
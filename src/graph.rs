use super::*;
use anyhow::{Context, Result};

pub(crate) struct UndirectedGraph<T>
where
    T: Clone,
{
    data: Vec<T>,
    size: usize,
}

impl<T> UndirectedGraph<T>
where
    T: Clone,
{
    pub fn new(size: usize, initial: T) -> Result<Self> {
        // FIXME this needs too much memory if size is large
        //dbg!("Before graph alloc size", size);
        let mut data = Vec::new();
        data.try_reserve(size * size).with_context(|| {
            format!(
                "Tried to allocate {size}*{size} (total {total}) bytes for graph",
                size = size,
                total = (size * size)
            )
        })?;
        for _ in 0..(size * size) {
            data.push(initial.clone());
        }
        //dbg!("after graph alloc size", size);
        Ok(UndirectedGraph { data, size })
    }

    pub fn len(&self) -> usize {
        self.size
    }

    #[allow(unused)]
    pub fn get(&self, i: usize, j: usize) -> &T {
        if i < j {
            &self.data[i * self.size + j]
        } else {
            &self.data[j * self.size + i]
        }
    }
    #[allow(unused)]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        if i < j {
            &mut self.data[i * self.size + j]
        } else {
            &mut self.data[j * self.size + i]
        }
    }
    #[allow(unused)]
    pub fn set_single(&mut self, i: usize, j: usize, val: T) {
        self.data[i * self.size + j] = val.clone();
    }
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        self.data[i * self.size + j] = val.clone();
        self.data[j * self.size + i] = val;
    }

    pub fn values(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.size)
            .flat_map(|i| (0..self.size).map(move |j| (i, j)))
            .map(|(i, j)| (i, j, &self.data[i * self.size + j]))
    }

    pub fn pretty_print(&self, fmt: &dyn Fn(&T) -> String) -> String {
        let mut strs: Vec<Vec<String>> = (0..self.size)
            .map(|i| {
                (0..self.size)
                    .map(|j| fmt(&self.data[i * self.size + j]))
                    .collect()
            })
            .collect();
        let max = strs
            .iter()
            .flat_map(|row| row.iter())
            .map(|el| el.len())
            .max()
            .unwrap_or(1);

        strs.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|val| {
                *val = format!("{0:>1$}", val, max);
            })
        });

        strs.into_iter()
            .map(|row| row.join(" "))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

pub(crate) struct DirectedGraph<T>
where
    T: Clone,
{
    data: Vec<T>,
    size: usize,
}

impl<T> DirectedGraph<T>
where
    T: Clone,
{
    pub fn new(size: usize, initial: T) -> Self {
        DirectedGraph {
            data: vec![initial; size * size],
            size,
        }
    }

    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.size
    }

    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.size + j]
    }
    #[allow(unused)]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.size + j]
    }
    pub fn set(&mut self, i: usize, j: usize, val: T) {
        self.data[i * self.size + j] = val;
    }

    pub fn values(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.size)
            .flat_map(|i| (0..self.size).map(move |j| (i, j)))
            .map(|(i, j)| (i, j, &self.data[i * self.size + j]))
    }

    pub fn pretty_print(&self, fmt: &dyn Fn(&T) -> String) -> String {
        let mut strs: Vec<Vec<String>> = (0..self.size)
            .map(|i| {
                (0..self.size)
                    .map(|j| fmt(&self.data[i * self.size + j]))
                    .collect()
            })
            .collect();
        let max = strs
            .iter()
            .flat_map(|row| row.iter())
            .map(|el| el.len())
            .max()
            .unwrap_or(1);

        strs.par_iter_mut().for_each(|row| {
            row.par_iter_mut().for_each(|val| {
                *val = format!("{0:>1$}", val, max);
            })
        });

        strs.into_iter()
            .map(|row| row.join(" "))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

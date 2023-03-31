use std::cmp::*;

struct TopNVec<T, F>
    where F: FnMut(&T, &T) -> Ordering,
{
    data: Vec<T>,
    size: usize,
    cmp_fn: F,
}

impl<T, F: FnMut(&T, &T) -> Ordering> TopNVec<T, F> {
    fn new(size: usize, cmp_fn: F) -> Self {
        TopNVec {
            data: vec![],
            size, cmp_fn 
            }
    }
    fn add(&mut self, el: T) {
        self.data.push(el);
        //self.data.sort_by(self.cmp_fn);
        self.data.truncate(self.size);
    }
}


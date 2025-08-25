#![allow(unused_imports)]
//! Kosaraju Algorithm to find Strongly Connected Components
//! https://en.wikipedia.org/wiki/Kosaraju%27s_algorithm
use super::*;
use crate::graph::{DirectedGraph, DirectedGraphTrait};
use indicatif::ProgressDrawTarget;

use std::collections::{BTreeMap, HashSet};

pub(crate) fn kosaraju(
    g: &impl DirectedGraphTrait,
    calc_components_bar: &ProgressBar,
) -> BTreeMap<i64, i64> {
    kosaraju_it(g, calc_components_bar)
}

/// Iterative Koaraju algorithm
pub(crate) fn kosaraju_it(
    g: &impl DirectedGraphTrait,
    calc_components_bar: &ProgressBar,
) -> BTreeMap<i64, i64> {
    let mut visited_vertexes: HashSet<i64> = HashSet::new();
    let mut l = Vec::with_capacity(g.num_vertexes());

    // Stack of things to do. ($VERTEX_ID, $HAS_BEEN_VISTED).
    // HAS_BEEN_VISTED = false : we haven't processed all the children yet
    //                 = true : we have finished processing the children
    // To do *post-order* visiting (in an iterative manner), we first push the
    let mut stack = Vec::new();

    for v in g.vertexes() {
        calc_components_bar.inc(1);

        stack.truncate(0);
        stack.push((v, false));

        while let Some((curr, children_added)) = stack.pop() {
            if !children_added {
                if visited_vertexes.contains(&curr) {
                    continue;
                }
                visited_vertexes.insert(curr);
                stack.push((curr, true));

                stack.extend(
                    g.out_neighbours(curr)
                        .filter(|v| !visited_vertexes.contains(v))
                        .map(|v| (v, false)),
                );
            } else {
                l.push(curr);
            }
        }
    }
    l.reverse();
    //assert_eq!(
    //    l.len(),
    //    g.num_vertexes(),
    //    "Unequal lengths for {}",
    //    g.vertexes().next().unwrap()
    //);
    drop(stack);

    let mut components = BTreeMap::new();

    // Second DFS
    // going in other direction
    // doing a pre-order visit, so much simplier.
    // there will be *loads* of single vertex components, we have to store them to ensure we only
    // visit them once
    let mut stack = Vec::new();
    for start in l.into_iter() {
        calc_components_bar.inc(1);
        let root = start;

        stack.truncate(0);
        stack.push(start);
        while let Some(curr) = stack.pop() {
            if let std::collections::btree_map::Entry::Vacant(e) = components.entry(curr) {
                e.insert(root);
                stack.extend(g.in_neighbours(curr))
            }
        }
    }

    // We don't need the single-vertex components.
    components.retain(|k, v| k != v);

    components
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (DirectedGraph<(), ()>, ProgressBar) {
        let g = DirectedGraph::new();
        let bar = ProgressBar::new(1000);
        bar.set_draw_target(ProgressDrawTarget::hidden());

        (g, bar)
    }
    #[test]
    fn test1() {
        let (mut g, bar) = setup();
        g.add_edge_chain(&[1, 2]);

        let _res = kosaraju(&g, &bar);
        //dbg!(res);
    }
}

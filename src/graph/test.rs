#![allow(warnings)]
use super::*;

mod simple_line {
    use super::*;

    #[test]
    fn simple1() {
        let mut g: DirectedGraph<(), ()> = DirectedGraph::new();
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        assert!(g.is_simple_line());
        let l = g.pop_simple_line();
        assert!(l.is_some());
        assert!(g.is_empty());
    }
}

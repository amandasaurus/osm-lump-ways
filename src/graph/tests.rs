use super::*;

#[test]
fn test1() {
    let mut g = DirectedGraph2::new();
    g.add_edge(1, 100);
    assert_eq!(g.edges_iter().collect::<Vec<_>>(), vec![(1, 100)]);
    g.add_edge(1, 100);
    assert_eq!(g.edges_iter().collect::<Vec<_>>(), vec![(1, 100)]);
    g.add_edge(1, 2);
    assert_eq!(g.edges_iter().collect::<Vec<_>>(), vec![(1, 2), (1, 100)]);
    g.add_edge_chain(&[2, 3, 4, 10, 6]);
    assert_eq!(
        g.edges_iter().collect::<Vec<_>>(),
        vec![(1, 2), (1, 100), (2, 3), (3, 4), (4, 10), (10, 6)]
    );
}

mod intermediatevertexstore {
    use super::*;

    #[test]
    fn test1() {
        let mut s = IntermediateVertexStore::<i64>::new();
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![]);
        s.add_next(&(0, 1), &[]);
        assert!(s.get(&(0, 1)).is_none());
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![]);
        s.add_next(&(0, 2), &[1]);
        assert_eq!(s.get(&(0, 2)), Some(&[1_i64] as &[i64]));
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            vec![(&(0, 2), &[1_i64] as &[i64]),]
        );

        s.add_next(&(1, 100), &[2, 3, 4, 5, 6]);
        assert_eq!(s.get(&(1, 100)), Some(&[2_i64, 3, 4, 5, 6] as &[i64]));
        s.add_next(&(100, 110), &[101, 102, 103]);
        assert_eq!(s.get(&(100, 110)), Some(&[101_i64, 102, 103] as &[i64]));
        dbg!("here");
        assert_eq!(
            s.iter().collect::<Vec<_>>(),
            vec![
                (&(0, 2), &[1_i64] as &[i64]),
                (&(1, 100), &[2, 3, 4, 5, 6]),
                (&(100, 110), &[101, 102, 103]),
            ]
        );
    }
}

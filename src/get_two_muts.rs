pub fn get_two_muts<T>(arr: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i <= arr.len());
    assert!(j <= arr.len());
    assert!(i != j);
    assert!(i < j);

    let (left, right) = arr.split_at_mut(i + 1);
    assert!(!left.is_empty());
    assert!(!right.is_empty());

    (left.get_mut(i).unwrap(), right.get_mut(j - i - 1).unwrap())
}

#[cfg(test)]
mod test {
    use super::*;

    macro_rules! test_all {
        ( $name:ident, $arr: expr, $i:expr, $j:expr, $expected_output1:expr, $expected_output2:expr ) => {
            #[test]
            fn $name() {
                let mut arr = $arr;
                let (x, y) = get_two_muts(&mut arr, $i, $j);

                assert_eq!(*x, $expected_output1);
                assert_eq!(*y, $expected_output2);
            }
        };
    }

    macro_rules! test_fails {
        ( $name:ident, $arr: expr, $i:expr, $j:expr ) => {
            #[test]
            #[should_panic]
            fn $name() {
                let mut arr = $arr;
                get_two_muts(&mut arr, $i, $j);
            }
        };
    }

    test_all!(test1, ['a', 'b', 'c', 'd'], 0, 1, 'a', 'b');
    test_fails!(test2, ['a', 'b'], 10, 10);
    test_fails!(test3, ['a', 'b', 'c', 'd'], 10, 1);
    test_fails!(test4, ['a', 'b', 'c', 'd'], 0, 0);
    test_fails!(test5, ['a', 'b', 'c', 'd'], 1, 0);
    test_all!(test6, ['a', 'b', 'c', 'd'], 0, 2, 'a', 'c');
    test_all!(test7, ['a', 'b', 'c', 'd'], 2, 3, 'c', 'd');
}

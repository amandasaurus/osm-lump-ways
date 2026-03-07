pub fn min_max<T: PartialOrd>(a: T, b: T) -> (T, T) {
    if a < b { (a, b) } else { (b, a) }
}

/// Round this float to this many places after the decimal point.
/// Used to reduce size of output geojson file
#[must_use] 
pub fn round(f: &f64, places: u8) -> f64 {
    let places: f64 = 10_u64.pow(places as u32) as f64;
    (f * places).round() / places
}

/// Round this float to be a whole number multiple of base.
#[must_use] 
pub fn round_mult(f: &f64, base: f64) -> i64 {
    ((f / base).round() * base) as i64
}

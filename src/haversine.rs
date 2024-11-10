use ordered_float::OrderedFloat;

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    if lat1 == lat2 && lon1 == lon2 {
        return 0.;
    }
    let d_lat: f64 = (lat2 - lat1).to_radians();
    let d_lon: f64 = (lon2 - lon1).to_radians();
    let lat1: f64 = lat2.to_radians();
    let lat2: f64 = lat2.to_radians();

    let a: f64 = ((d_lat / 2.0).sin()) * ((d_lat / 2.0).sin())
        + ((d_lon / 2.0).sin()) * ((d_lon / 2.0).sin()) * (lat1.cos()) * (lat2.cos());
    let c: f64 = 2.0 * ((a.sqrt()).atan2((1.0 - a).sqrt()));

    c * 6371008.8
}

pub fn haversine_m_arr(lat_lon1: &[f64], lat_lon2: &[f64]) -> f64 {
    haversine_m(lat_lon1[0], lat_lon1[1], lat_lon2[0], lat_lon2[1])
}

#[allow(dead_code)]
pub fn haversine_m_fpair(lat_lon1: (f64, f64), lat_lon2: (f64, f64)) -> f64 {
    haversine_m(lat_lon1.0, lat_lon1.1, lat_lon2.0, lat_lon2.1)
}

#[allow(dead_code)]
pub(crate) fn haversine_m_fpair_ord(
    lat_lon1: (f64, f64),
    lat_lon2: (f64, f64),
) -> OrderedFloat<f64> {
    OrderedFloat(haversine_m_fpair(lat_lon1, lat_lon2))
}

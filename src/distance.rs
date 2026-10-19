use geo::Point;
use geo::{Distance, Geodesic};
use ordered_float::OrderedFloat;

#[must_use]
pub fn distance_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    Geodesic.distance(Point::new(lon1, lat1), Point::new(lon2, lat2))
}

#[must_use]
pub fn distance_m_arr(lat_lon1: &[f64], lat_lon2: &[f64]) -> f64 {
    distance_m(lat_lon1[0], lat_lon1[1], lat_lon2[0], lat_lon2[1])
}

#[must_use]
pub fn distance_m_fpair(lat_lon1: (f64, f64), lat_lon2: (f64, f64)) -> f64 {
    distance_m(lat_lon1.0, lat_lon1.1, lat_lon2.0, lat_lon2.1)
}

pub(crate) fn distance_m_fpair_ord(
    lat_lon1: (f64, f64),
    lat_lon2: (f64, f64),
) -> OrderedFloat<f64> {
    OrderedFloat(distance_m_fpair(lat_lon1, lat_lon2))
}

#[allow(dead_code)]
pub(crate) fn distance_m_fpair_ord_ord(
    lat_lon1: (OrderedFloat<f64>, OrderedFloat<f64>),
    lat_lon2: (OrderedFloat<f64>, OrderedFloat<f64>),
) -> OrderedFloat<f64> {
    OrderedFloat(distance_m(
        *lat_lon1.0,
        *lat_lon1.1,
        *lat_lon2.0,
        *lat_lon2.1,
    ))
}

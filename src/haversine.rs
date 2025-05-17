use geo::{Haversine, Distance};
use geo::Point;
use ordered_float::OrderedFloat;

pub fn haversine_m(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
	Haversine.distance(Point::new(lat1, lon1), Point::new(lat2, lon2))
}

pub fn haversine_m_arr(lat_lon1: &[f64], lat_lon2: &[f64]) -> f64 {
    haversine_m(lat_lon1[0], lat_lon1[1], lat_lon2[0], lat_lon2[1])
}

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

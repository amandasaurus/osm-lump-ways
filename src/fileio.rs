#[derive(PartialEq, Eq, Debug)]
pub(crate) enum OutputFormat {
    GeoJSON,
    GeoJSONSeq,
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) enum OutputGeometryType {
    MultiLineString,
    LineString,
    MultiPoint,
    Point,
}

impl OutputGeometryType {
    pub(crate) fn bytes(&self) -> &'static [u8] {
        match self {
            OutputGeometryType::MultiLineString => b"MultiLineString",
            OutputGeometryType::LineString => b"LineString",
            OutputGeometryType::MultiPoint => b"MultiPoint",
            OutputGeometryType::Point => b"Point",
        }
    }
}

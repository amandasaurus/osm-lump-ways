use anyhow::Result;
use std::io::Write;
use std::iter::once;

/// Some sort of geometry type that can print out it's own coordinates
pub trait Geometry {
    fn type_name(&self) -> &[u8];
    fn write_coords(&self, f: &mut impl Write) -> Result<()>;
}

impl Geometry for Vec<Vec<(f64, f64)>> {
    fn type_name(&self) -> &[u8] {
        b"MultiLineString"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_multilinestring_coords(f, self)
    }
}

impl Geometry for (f64, f64) {
    fn type_name(&self) -> &[u8] {
        b"Point"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_point_coords(f, self)
    }
}

impl Geometry for ((f64, f64), (f64, f64)) {
    fn type_name(&self) -> &[u8] {
        b"LineString"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_linestring_coords(f, once(self.0).chain(once(self.1)))
    }
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) enum OutputFormat {
    GeoJSON,
    GeoJSONSeq,
}

#[allow(clippy::type_complexity)]
/// Write a geojson featurecollection, but manually construct it
pub(crate) fn write_geojson_features_directly<G>(
    features: impl Iterator<Item = (serde_json::Value, G)>,
    mut f: &mut impl Write,
    output_format: &OutputFormat,
) -> Result<usize>
where
    G: Geometry,
{
    let mut num_written = 0;
    let mut features = features.peekable();

    if output_format == &OutputFormat::GeoJSON {
        f.write_all(b"{\"type\":\"FeatureCollection\", \"features\": [\n")?;
    }
    if features.peek().is_some() {
        let feature_0 = features.next().unwrap();
        num_written += write_geojson_feature_directly(&mut f, &feature_0, output_format)?;
        for feature in features {
            if output_format == &OutputFormat::GeoJSON {
                f.write_all(b",\n")?;
            }
            num_written += write_geojson_feature_directly(&mut f, &feature, output_format)?;
        }
    }
    if output_format == &OutputFormat::GeoJSON {
        f.write_all(b"\n]}")?;
    }

    Ok(num_written)
}

fn write_geojson_feature_directly<G>(
    mut f: &mut impl Write,
    feature: &(serde_json::Value, G),
    output_format: &OutputFormat,
) -> Result<usize>
where
    G: Geometry,
{
    let mut num_written = 0;
    if output_format == &OutputFormat::GeoJSONSeq {
        f.write_all(b"\x1E")?;
    }
    f.write_all(b"{\"properties\":")?;
    serde_json::to_writer(&mut f, &feature.0)?;
    f.write_all(b", \"geometry\": {\"type\":\"")?;
    f.write_all(feature.1.type_name())?;
    f.write_all(b"\", \"coordinates\": ")?;
    feature.1.write_coords(&mut f)?;

    f.write_all(b"}, \"type\": \"Feature\"}")?;
    if output_format == &OutputFormat::GeoJSONSeq {
        f.write_all(b"\x0A")?;
    }
    num_written += 1;

    Ok(num_written)
}

fn write_multilinestring_coords(f: &mut impl Write, coords: &[Vec<(f64, f64)>]) -> Result<()> {
    f.write_all(b"[")?;
    for (i, linestring) in coords.iter().enumerate() {
        if i != 0 {
            f.write_all(b",")?;
        }
        f.write_all(b"[")?;
        for (j, j_coords) in linestring.iter().enumerate() {
            if j != 0 {
                f.write_all(b",")?;
            }
            write!(f, "[{:.6}, {:.6}]", j_coords.0, j_coords.1)?;
        }
        f.write_all(b"]")?;
    }
    f.write_all(b"]")?;
    Ok(())
}

fn write_point_coords(f: &mut impl Write, coords: &(f64, f64)) -> Result<()> {
    f.write_all(b"[")?;
    write!(f, "{:.6}, {:.6}", coords.0, coords.1)?;
    f.write_all(b"]")?;
    Ok(())
}

fn write_linestring_coords(
    f: &mut impl Write,
    coords: impl Iterator<Item = (f64, f64)>,
) -> Result<()> {
    f.write_all(b"[")?;
    for (j, j_coords) in coords.enumerate() {
        if j != 0 {
            f.write_all(b",")?;
        }
        write!(f, "[{:.6}, {:.6}]", j_coords.0, j_coords.1)?;
    }
    f.write_all(b"]")?;
    Ok(())
}

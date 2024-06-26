use anyhow::Result;
use std::io::Write;
use std::iter::once;

/// Some sort of geometry type that can print out it's own coordinates
pub trait Geometry {
    fn type_name(&self) -> &[u8];
    fn write_coords(&self, f: &mut impl Write) -> Result<()>;
    fn write_wkt(&self, buf: &mut Vec<u8>);
}

impl Geometry for Vec<Vec<(f64, f64)>> {
    fn type_name(&self) -> &[u8] {
        b"MultiLineString"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_multilinestring_coords(f, self)
    }

    fn write_wkt(&self, _buf: &mut Vec<u8>) {
        unimplemented!("WKT not implemented for this yet");
    }
}

impl Geometry for Vec<(f64, f64)> {
    fn type_name(&self) -> &[u8] {
        b"LineString"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_linestring_coords(f, self.iter().copied())
    }

    fn write_wkt(&self, _buf: &mut Vec<u8>) {
        unimplemented!("WKT not implemented for this yet");
    }
}

impl Geometry for (f64, f64) {
    fn type_name(&self) -> &[u8] {
        b"Point"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_point_coords(f, self)
    }
    fn write_wkt(&self, buf: &mut Vec<u8>) {
        buf.extend(format!("POINT({:.8} {:.8})", self.0, self.1).bytes());
    }
}

impl Geometry for &(f64, f64) {
    fn type_name(&self) -> &[u8] {
        b"Point"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_point_coords(f, self)
    }

    fn write_wkt(&self, buf: &mut Vec<u8>) {
        buf.extend(format!("POINT({:.8} {:.8})", self.0, self.1).bytes());
    }
}

impl Geometry for ((f64, f64), (f64, f64)) {
    fn type_name(&self) -> &[u8] {
        b"LineString"
    }
    fn write_coords(&self, f: &mut impl Write) -> Result<()> {
        write_linestring_coords(f, once(self.0).chain(once(self.1)))
    }
    fn write_wkt(&self, buf: &mut Vec<u8>) {
        buf.extend(
            format!(
                "LINESTRING({:.8} {:.8},{:.8} {:.8})",
                self.0 .0, self.0 .1, self.1 .0, self.1 .1
            )
            .bytes(),
        );
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
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

#[allow(dead_code)]
pub(crate) fn write_csv_features_directly<G>(
    features: impl Iterator<Item = (serde_json::Value, G)>,
    f: &mut impl Write,
    columns: &[String],
) -> Result<usize>
where
    G: Geometry,
{
    let mut num_written = 0;

    let mut wtr = csv::Writer::from_writer(f);

    for col in columns {
        wtr.write_field(col)?;
    }
    wtr.write_field("geom")?;
    wtr.write_record(None::<&[u8]>)?;

    let mut buf = Vec::new();
    for (props, geom) in features {
        for col in columns {
            wtr.write_field(&props[&col].to_string())?;
        }
        buf.clear();
        geom.write_wkt(&mut buf);
        wtr.write_field(&buf)?;

        wtr.write_record(None::<&[u8]>)?;
        num_written += 1;
    }

    Ok(num_written)
}

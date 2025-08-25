use anyhow::Result;
use std::io::Write;
use std::iter::once;
use std::path::Path;

/// Some sort of geometry type that can print out it's own coordinates
pub trait GeometryOutput {
    fn type_name(&self) -> &[u8];
    fn write_coords(&self, f: &mut impl Write) -> Result<()>;
    fn write_wkt(&self, buf: &mut Vec<u8>);

    fn write_geojson(&self, f: &mut impl Write) -> Result<()> {
        f.write_all(b"{\"type\":\"")?;
        f.write_all(self.type_name())?;
        f.write_all(b"\", \"coordinates\": ")?;
        self.write_coords(f)?;
        f.write_all(b"}")?;
        Ok(())
    }
}


impl GeometryOutput for Vec<Vec<(f64, f64)>> {
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

impl GeometryOutput for Vec<(f64, f64)> {
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

impl GeometryOutput for (f64, f64) {
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

impl GeometryOutput for &(f64, f64) {
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

impl GeometryOutput for ((f64, f64), (f64, f64)) {
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
                self.0.0, self.0.1, self.1.0, self.1.1
            )
            .bytes(),
        );
    }
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum OutputFormat {
    GeoJSON,
    GeoJSONSeq,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum OutputGeometryFormat {
    WKT,
    GeoJSON,
}

pub fn format_for_filename(f: &Path) -> OutputFormat {
    if f.extension().unwrap() == "geojsons" {
        OutputFormat::GeoJSONSeq
    } else if f.extension().unwrap() == "geojson" {
        OutputFormat::GeoJSON
    } else {
        unimplemented!("Unsupported file extension {:?}", f.extension().unwrap())
    }
}

#[allow(clippy::type_complexity)]
/// Write a geojson featurecollection, but manually construct it
pub fn write_geojson_features_directly<G>(
    features: impl Iterator<Item = (serde_json::Value, G)>,
    mut f: &mut impl Write,
    output_format: &OutputFormat,
) -> Result<usize>
where
    G: GeometryOutput,
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

pub fn write_geojson_feature_directly<G>(
    mut f: &mut impl Write,
    feature: &(serde_json::Value, G),
    output_format: &OutputFormat,
) -> Result<usize>
where
    G: GeometryOutput,
{
    let mut num_written = 0;
    if output_format == &OutputFormat::GeoJSONSeq {
        f.write_all(b"\x1E")?;
    }
    f.write_all(b"{\"properties\":")?;
    #[allow(clippy::needless_borrows_for_generic_args)]
    serde_json::to_writer(&mut f, &feature.0)?;
    f.write_all(b", \"geometry\": ")?;
    feature.1.write_geojson(&mut f)?;

    f.write_all(b", \"type\": \"Feature\"}")?;
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
pub fn write_csv_features_directly<G>(
    features: impl Iterator<Item = (serde_json::Value, G)>,
    f: &mut impl Write,
    output_geom_format: impl Into<OutputGeometryFormat>,
) -> Result<usize>
where
    G: GeometryOutput,
{
    let output_geom_format = output_geom_format.into();
    let mut headers: Vec<String> = Vec::with_capacity(10);

    let mut num_written = 0;

    let mut wtr = csv::WriterBuilder::new().from_writer(f);

    let mut buf = Vec::new();
    for (props, geom) in features {
        // Write the headers on the first run
        if headers.is_empty() {
            // NB no sorting here, the results look OK...
            headers = props
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>();
            for col in headers.iter() {
                wtr.write_field(col)?;
            }
            wtr.write_field("geom")?;
            wtr.write_record(None::<&[u8]>)?;
        }

        for col in headers.iter() {
            wtr.write_field(props[&col].to_string())?;
        }
        buf.clear();
        match output_geom_format {
            OutputGeometryFormat::WKT => geom.write_wkt(&mut buf),
            OutputGeometryFormat::GeoJSON => geom.write_geojson(&mut buf)?,
        }
        wtr.write_field(&buf)?;
        wtr.write_record(None::<&[u8]>)?;

        num_written += 1;
    }

    Ok(num_written)
}

pub enum GenericGeometry {
	Point((f64, f64)),
	SimpleLineString(((f64, f64), (f64, f64))),
}

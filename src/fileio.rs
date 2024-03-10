use anyhow::Result;
use std::io::Write;

#[derive(PartialEq, Eq, Debug)]
pub(crate) enum OutputFormat {
    GeoJSON,
    GeoJSONSeq,
}

#[derive(PartialEq, Eq, Debug)]
pub(crate) enum OutputGeometryType {
    MultiLineString,
    LineString,
    #[allow(dead_code)]
    MultiPoint,
    #[allow(dead_code)]
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

#[allow(clippy::type_complexity)]
/// Write a geojson featurecollection, but manually construct it
pub(crate) fn write_geojson_features_directly(
    features: impl Iterator<Item = (serde_json::Value, Vec<Vec<(f64, f64)>>)>,
    mut f: &mut impl Write,
    output_format: &OutputFormat,
    output_geometry_type: &OutputGeometryType,
) -> Result<usize> {
    let mut num_written = 0;
    let mut features = features.peekable();

    if output_format == &OutputFormat::GeoJSON {
        f.write_all(b"{\"type\":\"FeatureCollection\", \"features\": [\n")?;
    }
    if features.peek().is_some() {
        let feature_0 = features.next().unwrap();
        num_written += write_geojson_feature_directly(
            &mut f,
            &feature_0,
            output_format,
            output_geometry_type,
        )?;
        for feature in features {
            if output_format == &OutputFormat::GeoJSON {
                f.write_all(b",\n")?;
            }
            num_written += write_geojson_feature_directly(
                &mut f,
                &feature,
                output_format,
                output_geometry_type,
            )?;
        }
    }
    if output_format == &OutputFormat::GeoJSON {
        f.write_all(b"\n]}")?;
    }

    Ok(num_written)
}

fn write_geojson_feature_directly(
    mut f: &mut impl Write,
    feature: &(serde_json::Value, Vec<Vec<(f64, f64)>>),
    output_format: &OutputFormat,
    output_geometry_type: &OutputGeometryType,
) -> Result<usize> {
    let mut num_written = 0;
    if output_format == &OutputFormat::GeoJSONSeq {
        f.write_all(b"\x1E")?;
    }
    f.write_all(b"{\"properties\":")?;
    serde_json::to_writer(&mut f, &feature.0)?;
    f.write_all(b", \"geometry\": {\"type\":\"")?;
    f.write_all(output_geometry_type.bytes())?;
    f.write_all(b"\", \"coordinates\": ")?;
    write_coords(&mut f, &feature.1, output_geometry_type)?;

    f.write_all(b"}, \"type\": \"Feature\"}")?;
    if output_format == &OutputFormat::GeoJSONSeq {
        f.write_all(b"\x0A")?;
    }
    num_written += 1;

    Ok(num_written)
}

fn write_coords(
    f: &mut impl Write,
    coords: &[Vec<(f64, f64)>],
    output_geometry_type: &OutputGeometryType,
) -> Result<()> {
    match output_geometry_type {
        OutputGeometryType::MultiLineString => write_multilinestring_coords(f, coords),
        OutputGeometryType::LineString => write_linestring_coords(f, coords),
        OutputGeometryType::Point => write_point_coords(f, coords),
        _ => todo!(),
    }
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

fn write_point_coords(f: &mut impl Write, coords: &[Vec<(f64, f64)>]) -> Result<()> {
    f.write_all(b"[")?;
    write!(f, "{:.6}, {:.6}", coords[0][0].0, coords[0][0].1)?;
    f.write_all(b"]")?;
    Ok(())
}

pub fn format_duration_human(duration: &std::time::Duration) -> String {
    let sec_f = duration.as_secs_f32();
    if sec_f < 60. {
        let msec = (sec_f * 1000.).round() as u64;
        if sec_f > 0. && msec == 0 {
            "<1ms".to_string()
        } else if msec > 0 && duration.as_secs_f32() < 1. {
            format!("{}ms", msec)
        } else {
            format!("{:>3.1}s", sec_f)
        }
    } else {
        let sec = sec_f.round() as u64;
        let (min, sec) = (sec / 60, sec % 60);
        if min < 60 {
            format!("{}m{:02}s", min, sec)
        } else {
            let (hr, min) = (min / 60, min % 60);
            if hr < 24 {
                format!("{}h{:02}m{:02}s", hr, min, sec)
            } else {
                let (day, hr) = (hr / 24, hr % 24);
                format!("{}d{:02}h{:02}m{:02}s", day, hr, min, sec)
            }
        }
    }
}

fn write_linestring_coords(f: &mut impl Write, coords: &[Vec<(f64, f64)>]) -> Result<()> {
    f.write_all(b"[")?;
    for (j, j_coords) in coords[0].iter().enumerate() {
        if j != 0 {
            f.write_all(b",")?;
        }
        write!(f, "[{:.6}, {:.6}]", j_coords.0, j_coords.1)?;
    }
    f.write_all(b"]")?;
    Ok(())
}

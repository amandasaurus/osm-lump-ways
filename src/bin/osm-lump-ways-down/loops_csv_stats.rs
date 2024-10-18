//! Generating CSV loops stats
use anyhow::Result;
use log::info;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

pub(crate) fn init(csv_stats_file: &PathBuf) -> csv::Writer<BufWriter<File>> {
    info!("Writing CSV stats to file {csv_stats_file:?}");
    if !csv_stats_file.exists() {
        let mut wtr = csv::Writer::from_writer(std::fs::File::create(csv_stats_file).unwrap());
        wtr.write_record(["timestamp", "iso_datetime", "area", "metric", "value"])
            .unwrap();
        wtr.flush().unwrap();
        drop(wtr);
    }
    csv::Writer::from_writer(std::io::BufWriter::new(
        std::fs::File::options()
            .append(true)
            .open(csv_stats_file)
            .unwrap(),
    ))
}
pub(crate) fn write_boundary(
    csv_stats: &mut csv::Writer<BufWriter<File>>,
    boundary: &str,
    latest_timestamp: i64,
    latest_timestamp_iso: &str,
    count: u64,
    len: f64,
) -> Result<()> {
    csv_stats.write_record(&[
        latest_timestamp.to_string(),
        latest_timestamp_iso.to_string(),
        boundary.to_string(),
        "loops_count".to_string(),
        count.to_string(),
    ])?;
    csv_stats.write_record(&[
        latest_timestamp.to_string(),
        latest_timestamp_iso.to_string(),
        boundary.to_string(),
        "loops_length_m".to_string(),
        format!("{:.1}", len),
    ])?;
    Ok(())
}

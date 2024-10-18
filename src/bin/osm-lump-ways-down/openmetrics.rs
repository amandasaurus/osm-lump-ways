//! Generating Prometheus/OpenMetrics file for loops stats
use anyhow::Result;
use log::info;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

pub(crate) fn init(metrics_path: &PathBuf) -> BufWriter<File> {
    info!("Writing metrics to file {metrics_path:?}");
    let mut metrics = std::io::BufWriter::new(std::fs::File::create(metrics_path).unwrap());
    writeln!(
        metrics,
        "# HELP waterwaymap_loops_count number of cycles/loops in this area"
    )
    .unwrap();
    writeln!(metrics, "# TYPE waterwaymap_loops_count gauge").unwrap();
    writeln!(
        metrics,
        "# HELP waterwaymap_loops_length_m Length of all loops (in metres) in this area)"
    )
    .unwrap();
    writeln!(metrics, "# TYPE waterwaymap_loops_length_m gauge").unwrap();

    metrics
}

pub(crate) fn write_boundary(
    metrics: &mut impl Write,
    boundary: &str,
    latest_timestamp: i64,
    count: u64,
    len: f64,
) -> Result<()> {
    writeln!(
        metrics,
        "waterwaymap_loops_count{{area=\"{}\"}} {} {}",
        boundary, count, latest_timestamp
    )?;
    writeln!(
        metrics,
        "waterwaymap_loops_length_m{{area=\"{}\"}} {} {}",
        boundary, len, latest_timestamp
    )?;

    Ok(())
}

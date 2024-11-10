//! Generating CSV ends stats
use crate::cli_args;
use crate::round;
use crate::NodeIdPosition;
use anyhow::Result;
use log::{info, warn};
use num_format::{Locale, ToFormattedString};
use ordered_float::OrderedFloat;
use smallvec::SmallVec;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

pub(crate) fn init(
    csv_stats_file: &PathBuf,
    args: &cli_args::Args,
) -> csv::Writer<BufWriter<File>> {
    info!("Writing CSV ends stats to file {csv_stats_file:?}");
    let mut headers = vec![
        "timestamp",
        "iso_datetime",
        "upstream_m",
        "upstream_m_rank",
        "nid",
        "lat",
        "lng",
    ];
    headers.extend(args.ends_tag.iter().map(|s| s.as_str()));

    if !csv_stats_file.exists() {
        let mut wtr = csv::Writer::from_writer(std::fs::File::create(csv_stats_file).unwrap());
        wtr.write_record(&headers).unwrap();
        wtr.flush().unwrap();
        drop(wtr);
    } else {
        // confirm the headers are the same
        let mut rdr = csv::Reader::from_reader(std::fs::File::open(csv_stats_file).unwrap());
        let first_row = rdr.headers().unwrap();
        if first_row != headers {
            warn!("Differnet headers. Expected {:?} got {:?}. Are you using a different set (or order) of --ends-tag ? Continuing anyway, and writing the columns we expect.", headers, first_row);
        }
    }
    if args.ends_tag.is_empty() {
        warn!("The ends CSV file only makes sense with the --ends-tag arguments. Since you have specified no end tags, nothing will be written to the ends CSV file");
    }
    csv::Writer::from_writer(std::io::BufWriter::new(
        std::fs::File::options()
            .append(true)
            .open(csv_stats_file)
            .unwrap(),
    ))
}
pub(crate) fn write_ends<'a>(
    csv: &mut csv::Writer<BufWriter<File>>,
    end_points_w_meta: impl Iterator<
        Item = (
            &'a i64,
            &'a SmallVec<[bool; 2]>,
            &'a SmallVec<[Option<std::string::String>; 1]>,
            &'a f64,
        ),
    >,
    args: &cli_args::Args,
    nodeid_pos: &impl NodeIdPosition,
    latest_timestamp: i64,
    latest_timestamp_iso: &str,
) -> Result<()> {
    if args.ends_tag.is_empty() {
        return Ok(());
    }
    let mut end_points_w_meta = end_points_w_meta
        .map(|(nid, _mbms, end_tags, len)| (nid, end_tags, len))
        .filter(|(_nid, end_tags, _len)| end_tags.iter().any(|t| t.is_some()))
        .filter(|(_nid, _end_tags, len)| {
            args.ends_csv_min_length_m.map_or(true, |min| **len >= min)
        })
        .collect::<Vec<_>>();
    end_points_w_meta.sort_by_key(|(_nid, _end_tags, len)| -OrderedFloat(**len));

    if let Some(only_largest_n) = args.ends_csv_only_largest_n {
        end_points_w_meta.truncate(only_largest_n);
    }
    let mut num_written: usize = 0;

    for (uptream_rank, (nid, end_tags, len)) in end_points_w_meta.into_iter().enumerate() {
        let pos = nodeid_pos.get(nid).unwrap();
        csv.write_field(latest_timestamp.to_string())?;
        csv.write_field(latest_timestamp_iso)?;
        csv.write_field(round(len, 1).to_string())?;
        csv.write_field((uptream_rank + 1).to_string())?;
        csv.write_field(nid.to_string())?;
        csv.write_field(round(&pos.1, 7).to_string())?;
        csv.write_field(round(&pos.0, 7).to_string())?;
        for end_tag in end_tags {
            csv.write_field(end_tag.as_ref().map_or("" as &str, |s| s.as_str()))?;
        }
        csv.write_record(None::<&[u8]>)?;
        num_written += 1;
    }

    info!(
        "Wrote {} entries to the ends CSV file at {}",
        num_written.to_formatted_string(&Locale::en),
        args.ends_csv_file.as_ref().unwrap().display()
    );

    Ok(())
}

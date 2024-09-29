use anyhow::Result;
use clap::Parser;
use get_size::GetSize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressIterator, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
#[allow(unused_imports)]
use log::{
    debug, error, info, log, trace, warn,
    Level::{Debug, Trace},
};
use osmio::prelude::*;
use osmio::OSMObjBase;
use rayon::prelude::*;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use std::sync::atomic::{AtomicI64, Ordering as atomic_Ordering};
use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};
use smallvec::SmallVec;

use country_boundaries::{CountryBoundaries, LatLon, BOUNDARIES_ODBL_360X180};
use ordered_float::OrderedFloat;

mod cli_args;

use graph::DirectedGraphTrait;
use haversine::haversine_m;
use nodeid_position::NodeIdPosition;
use osm_lump_ways::graph;
use osm_lump_ways::haversine;
use osm_lump_ways::nodeid_position;
use osm_lump_ways::tagfilter;

use fileio::{write_csv_features_directly, write_geojson_features_directly};
use osm_lump_ways::fileio;

use osm_lump_ways::formatting;

macro_rules! info_memory_used {
    () => {};
}
//    () => {
//        info!(
//            "{}:L{} Total memory used: {}",
//            file!(),
//            line!(),
//            memory_stats::memory_stats()
//                .unwrap()
//                .physical_mem
//                .to_formatted_string(&Locale::en)
//        );
//    };
//}

fn main() -> Result<()> {
    let args = cli_args::Args::parse();

    let logger = env_logger::Builder::new()
        .filter_level(args.verbose.log_level_filter())
        .build();
    let progress_bars = indicatif::MultiProgress::new();
    LogWrapper::new(progress_bars.clone(), logger)
        .try_init()
        .unwrap();
    let show_progress_bars = args.verbose.log_level_filter() >= log::Level::Info;
    if !show_progress_bars {
        progress_bars.set_draw_target(ProgressDrawTarget::hidden());
    }

    let global_start = Instant::now();
    info!(
        "Welcome to osm-lump-ways-down v{}. Source code: <{}>. Have fun! :)",
        std::env!("CARGO_PKG_VERSION"),
        std::env!("CARGO_PKG_REPOSITORY"),
    );

    let file_reading_style =
                ProgressStyle::with_template(
        "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {bytes:>7}/{total_bytes:7} {per_sec:>12} {msg}",
            ).unwrap();
    let input_fp = std::fs::File::open(&args.input_filename)?;
    let input_bar = progress_bars.add(
        ProgressBar::new(input_fp.metadata()?.len())
            .with_message("Reading input file")
            .with_style(file_reading_style.clone()),
    );
    let rdr = input_bar.wrap_read(input_fp);

    if !args.input_filename.is_file() {
        error!(
            "Input file ( {} ) is not a file we can read",
            args.input_filename.display()
        );
        anyhow::bail!(
            "Input file ( {} ) is not a file",
            args.input_filename.display()
        );
    }

    anyhow::ensure!(
        args.ends.is_some()
            || args.loops.is_some()
            || args.upstreams.is_some()
            || args.grouped_ends.is_some(),
        "Nothing to do. You need to specifiy one of --ends/--loops/--upstreams/etc."
    );

    if args.grouped_ends.is_some()
        && !(args.upstream_output_biggest_end || args.upstream_assign_end_by_tag.is_some())
    {
        error!("If you have upstreams which are grouped by ends, you must specificy one of --upstream-output-biggest-end or --upstream-assign-end-by-tag TAG");
        anyhow::bail!("If you have upstreams which are grouped by ends, you must specificy one of --upstream-output-biggest-end or --upstream-assign-end-by-tag TAG");
    }

    info!("Input file: {:?}", &args.input_filename);
    if args.tag_filter.is_empty() {
        if let Some(ref tff) = args.tag_filter_func {
            info!("Tag filter function in operation: {:?}", tff);
        } else {
            info!("No tag filtering in operation. All ways in the file will be used.");
        }
    } else {
        info!("Tag filter(s) in operation: {:?}", args.tag_filter);
    }
    // Attempt to speed up reading, by replacing this Vec with a SmallVec
    let tag_filter: SmallVec<[tagfilter::TagFilter; 3]> = args.tag_filter.clone().into();
    if std::env::var("OSM_LUMP_WAYS_FINISH_AFTER_READ").is_ok() {
        warn!("Programme will exit after reading & parsing input");
    }

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {pos:>7}/{len:7} {per_sec:>12} {msg}",
    )
    .unwrap();
    let obj_reader = progress_bars.add(ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("           {human_pos} ways read {per_sec:>20}").unwrap(),
    ));
    let ways_added = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} ways collected so far for later processing",
            )
            .unwrap(),
        ),
    );
    let nodes_added = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} nodes collected so far for later processing",
            )
            .unwrap(),
        ),
    );

    let mut metrics = args.openmetrics.as_ref().map(|metrics_path| {
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
    });

    let mut csv_stats = args.csv_stats_file.as_ref().map(|csv_stats_file| {
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
    });

    let boundaries = CountryBoundaries::from_reader(BOUNDARIES_ODBL_360X180)?;

    let g = graph::DirectedGraph2::new();
    let g = Arc::new(Mutex::new(g));

    // Stores the tagvalue group for each segment.
    // OSM tag vaules are strings, but we don't need to store the strings. We give each string a
    // unique id (i32) and store that. We don't need to know what the tagvalue for 2 segments is,
    // we only need to know if they are the same or not.
    // FIXME replace this with a sorted vec, rather than hashmap
    let mut nid_pair_to_endtag_group: HashMap<(i64, i64), i32> = HashMap::new();

    if let Some(ref upstream_assign_end_by_tag) = args.upstream_assign_end_by_tag {
        // read all the ways to get a list of which nid pairs are in which name tag
        let rdr = std::fs::File::open(&args.input_filename)?;
        let mut reader = osmio::stringpbf::PBFReader::new(rdr);

        // Produces a HashMap<tagvalue: String, Vec<(nid1,nid2)>> each unique tagvalue.
        let tagvalues_to_edges = reader
            .ways()
            .par_bridge()
            .inspect(|_| obj_reader.inc(1))
            .filter(|w| tagfilter::obj_pass_filters(w, &tag_filter, &args.tag_filter_func))
            .filter(|w| w.has_tag(upstream_assign_end_by_tag))
            .inspect(|_| ways_added.inc(1))
            .fold(
                || HashMap::new() as HashMap<String, HashSet<(i64, i64)>>,
                |mut seen_tagvalues, w| {
                    let seen = seen_tagvalues
                        .entry(w.tag(upstream_assign_end_by_tag).unwrap().to_string())
                        .or_default();
                    for pair in w.nodes().windows(2) {
                        seen.insert((pair[0], pair[1]));
                    }
                    seen_tagvalues
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                let mut pairs_in_this_value;
                for (tag_value, mut pair_set) in b.into_iter() {
                    pairs_in_this_value = a.entry(tag_value).or_default();
                    pairs_in_this_value.extend(pair_set.drain());
                }
                a
            });
        assert!(tagvalues_to_edges.len() < i32::MAX as usize);

        let total_num_pairs = tagvalues_to_edges
            .par_iter()
            .map(|(_tagvalue, pairs)| pairs.len())
            .sum();
        nid_pair_to_endtag_group.reserve(total_num_pairs);

        info!(
            "Have following {} unique '{}' tags in {} node pairs",
            tagvalues_to_edges.len().to_formatted_string(&Locale::en),
            upstream_assign_end_by_tag,
            total_num_pairs.to_formatted_string(&Locale::en),
        );
        for (_tagvalue, pairs) in tagvalues_to_edges.into_iter() {
            let curr_id = nid_pair_to_endtag_group.len();
            for pair in pairs.into_iter() {
                nid_pair_to_endtag_group.insert(pair, curr_id as i32);
            }
        }
        info!(
            "Total size of the '{}' lookup: {} bytes",
            upstream_assign_end_by_tag,
            nid_pair_to_endtag_group
                .get_size()
                .to_formatted_string(&Locale::en)
        );
    }

    // first step, get all the cycles
    let latest_timestamp = AtomicI64::new(0);
    let start_reading_ways = Instant::now();
    let mut reader = osmio::stringpbf::PBFReader::new(rdr);
    reader
        .ways()
        .par_bridge()
        .inspect(|_| obj_reader.inc(1))
        .filter(|w| tagfilter::obj_pass_filters(w, &tag_filter, &args.tag_filter_func))
        .inspect(|_| ways_added.inc(1))
        // TODO support grouping by tag value
        .for_each_with(g.clone(), |g, w| {
            assert!(w.id() > 0, "This file has a way id < 0. negative ids are not supported in this tool Use osmium sort & osmium renumber to convert this file and run again.");
            // add the nodes from w to this graph
            let mut g = g.lock().unwrap();
            nodes_added.inc(w.nodes().len() as u64);
            g.add_edge_chain(w.nodes());
            if let Some(t) = w.timestamp().as_ref().map(|t| t.to_epoch_number()) {
                latest_timestamp.fetch_max(t, atomic_Ordering::SeqCst);
            }
        });
    let way_reading_duration = start_reading_ways.elapsed();
    info!(
        "Finished reading. {} ways, and {} nodes, read in {}, {} ways/sec",
        ways_added.position().to_formatted_string(&Locale::en),
        nodes_added.position().to_formatted_string(&Locale::en),
        formatting::format_duration(way_reading_duration),
        ((ways_added.position() as f64 / (way_reading_duration.as_secs_f64())) as u64)
            .to_formatted_string(&Locale::en),
    );
    info_memory_used!();
    let latest_timestamp = latest_timestamp.into_inner();
    let latest_timestamp_iso =
        osmio::TimestampFormat::EpochNunber(latest_timestamp).to_iso_string();
    info!(
        "Latest timestamp is {} / {}",
        latest_timestamp, latest_timestamp_iso
    );
    obj_reader.finish_and_clear();
    ways_added.finish_and_clear();
    nodes_added.finish_and_clear();
    input_bar.finish_and_clear();
    progress_bars.remove(&input_bar);
    let g: graph::DirectedGraph2 = Arc::try_unwrap(g).unwrap().into_inner().unwrap();

    info!(
        "All data has been loaded in {}. Started processing...",
        formatting::format_duration(global_start.elapsed())
    );
    info_memory_used!();

    if g.is_empty() {
        info!("No ways in the file matched your filters. Nothing to do");
        return Ok(());
    }

    let num_vertexes = g.num_vertexes();

    info!(
        "Size of the connected graph: {} nodes",
        num_vertexes.to_formatted_string(&Locale::en)
    );

    // We used to use a graph that only stored out neighbours, which meant we needed to keep track
    // of which vertexes don't have outgoing nodes, so that we can remember
    // to save their position.
    // But not we use something that stores directly both

    info!("Splitting this large graph into many smaller non-connected graphs...");
    let splitting_graphs_bar = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Splitting into smaller graphs")
            .with_style(style.clone()),
    );
    // Split into separtely connected graphs.
    let graphs = g
        .into_disconnected_graphs()
        .inspect(|g| splitting_graphs_bar.inc(g.num_vertexes() as u64))
        .collect::<Vec<_>>();
    splitting_graphs_bar.finish_and_clear();
    info!(
        "Split the graph into {} different disconnected graphs",
        graphs.len().to_formatted_string(&Locale::en)
    );
    info_memory_used!();

    let calc_components_bar = progress_bars.add(
        ProgressBar::new((num_vertexes * 2) as u64)
            .with_message("Looking for cycles")
            .with_style(style.clone()),
    );
    let graphs_processed = progress_bars.add(
        ProgressBar::new(graphs.len() as u64)
            .with_message("Graphs processed")
            .with_style(style.clone()),
    );
    info_memory_used!();

    let cycles: Vec<Vec<[i64; 2]>> = graphs
        .into_par_iter()
        .update(|g| {
            let old_len = g.num_vertexes();
            g.remove_vertexes_with_in_xor_out();
            calc_components_bar.inc(2 * (old_len - g.num_vertexes()) as u64);
        })
        .flat_map(|g| {
            graphs_processed.inc(1);
            g.strongly_connected_components(&calc_components_bar)
        })
        .collect();

    info!(
        "Found {} cycles, comprised of {} nodes",
        cycles.len().to_formatted_string(&Locale::en),
        cycles
            .par_iter()
            .map(|c| c.len())
            .sum::<usize>()
            .to_formatted_string(&Locale::en),
    );
    info_memory_used!();

    calc_components_bar.finish_and_clear();
    graphs_processed.finish_and_clear();

    if !cycles.is_empty() {
        let mut wanted_nodeids: HashSet<i64> =
            HashSet::with_capacity(cycles.par_iter().map(|cycle| cycle.len()).sum());
        for cycle in cycles.iter() {
            wanted_nodeids.extend(cycle.iter().flat_map(|seg| seg.iter()));
        }

        let setting_node_pos = progress_bars.add(
            ProgressBar::new(wanted_nodeids.len() as u64)
                .with_message("Reading file to save node locations for the cycles")
                .with_style(style.clone()),
        );
        let mut nodeid_pos = nodeid_position::default();
        read_node_positions(
            &args.input_filename,
            |nid| wanted_nodeids.contains(&nid),
            &setting_node_pos,
            &mut nodeid_pos,
        )?;
        setting_node_pos.finish_and_clear();
        info_memory_used!();

        #[allow(clippy::type_complexity)]
        let cycles_output: Vec<(serde_json::Value, Vec<Vec<(f64, f64)>>)> = cycles
            .par_iter()
            .map(|cycle| {
                let coords = node_group_to_lines(cycle.as_slice(), &nodeid_pos);
                // WTF do I have lat & lng mixed up??
                let mut these_boundaries =
                    boundaries.ids(LatLon::new(coords[0][0].1, coords[0][0].0)?);
                these_boundaries.sort_unstable_by_key(|s| -(s.len() as isize));
                if these_boundaries.is_empty() {
                    these_boundaries.push("unknown_area");
                }
                let all_nodes: BTreeSet<_> = cycle.iter().flat_map(|seg| seg.iter()).collect();
                let mut props = serde_json::json!({
                    "root_nid": cycle.iter().flat_map(|seg| seg.iter()).min().unwrap(),
                    "num_nodes": cycle.len(),
                    "length_m": round(&node_group_to_length_m(cycle.as_slice(), &nodeid_pos), 1),
                    "nodes": all_nodes.into_iter()
                                .map(|nid| format!("n{}", nid))
                                .collect::<Vec<_>>()
                                .join(","),
                });

                for (i, boundary) in these_boundaries.iter().enumerate() {
                    props[format!("area_{}", i)] = boundary.to_string().into();
                }
                props["areas_s"] = format!(",{},", these_boundaries.join(",")).into();
                props["areas"] = these_boundaries.into();

                Ok((props, coords))
            })
            .collect::<Result<_>>()?;
        info_memory_used!();

        if csv_stats.is_some() || metrics.is_some() {
            let mut per_boundary: BTreeMap<&str, (u64, f64)> = BTreeMap::new();
            for cycle in cycles_output.iter() {
                let boundaries = cycle.0["areas"].as_array().unwrap();
                let this_len = multilinestring_length(&cycle.1);
                per_boundary.entry("planet").or_default().0 += 1;
                per_boundary.entry("planet").or_default().1 += this_len;
                for boundary in boundaries.iter().filter_map(|s| s.as_str()) {
                    per_boundary.entry(boundary).or_default().0 += 1;
                    per_boundary.entry(boundary).or_default().1 += this_len;
                }
            }
            for (boundary, (count, len)) in per_boundary.into_iter() {
                if let Some(ref mut csv_stats) = csv_stats {
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
                }
                if let Some(ref mut metrics) = metrics {
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
                }
            }

            if let Some(ref mut csv_stats) = csv_stats {
                csv_stats.flush()?;
                info!(
                    "Statistics have been written to file {}",
                    args.csv_stats_file.as_ref().unwrap().display()
                );
            }
        }

        if let Some(ref loops_filename) = args.loops {
            let mut f = BufWriter::new(File::create(loops_filename)?);
            let num_written = write_geojson_features_directly(
                cycles_output.into_iter(),
                &mut f,
                &fileio::format_for_filename(loops_filename),
            )?;

            info!(
                "Wrote {} features to output file {}",
                num_written.to_formatted_string(&Locale::en),
                loops_filename.to_str().unwrap(),
            );
        }
    }

    if args.ends.is_none() && args.upstreams.is_none() && args.grouped_ends.is_none() {
        // nothing else to do
        return Ok(());
    }

    info_memory_used!();
    let mut node_id_replaces: HashMap<i64, i64> =
        HashMap::with_capacity(cycles.par_iter().map(|c| c.len() - 1).sum());

    let mut min_nodeid;
    for cycle in cycles {
        min_nodeid = *cycle.iter().flat_map(|seg| seg.iter()).min().unwrap();
        for nid in cycle.iter().flat_map(|seg| seg.iter()) {
            if *nid != min_nodeid {
                node_id_replaces.insert(*nid, min_nodeid);
            }
        }
    }
    info!(
        "{} nodes will be replaced as part of the cycle-removal",
        node_id_replaces.len().to_formatted_string(&Locale::en)
    );

    info_memory_used!();
    let started_reread = Instant::now();
    info!("Re-reading file to generate upstreams etc");
    let mut g = graph::DirectedGraph2::new();
    read_with_node_replacements(
        &args.input_filename,
        &tag_filter,
        &args.tag_filter_func,
        &|nid| *node_id_replaces.get(&nid).unwrap_or(&nid),
        &progress_bars,
        &mut g,
    )?;
    info!("Re-read all nodes with replacements in {}", formatting::format_duration(started_reread.elapsed()));

    // TODO do we need to sort topologically? Why not just calc lengths from upstreams
    let sorting_nodes_bar = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Sorting nodes topologically")
            .with_style(style.clone()),
    );
    let started_sorting = Instant::now();
    info!("Sorting all vertexes topologically...");
    //// TODO this graph (g) can be split into disconnected components
    let orig_num_vertexes = g.num_vertexes();
    let topologically_sorted_nodes = g.into_vertexes_topologically_sorted(&sorting_nodes_bar);
    sorting_nodes_bar.finish_and_clear();
    info!(
        "All {} nodes have been sorted topographically in {}. Size of sorted nodes: {} bytes = {}",
        topologically_sorted_nodes
            .len()
            .to_formatted_string(&Locale::en),
        formatting::format_duration(started_sorting.elapsed()),
        topologically_sorted_nodes.get_size(),
        topologically_sorted_nodes
            .get_size()
            .to_formatted_string(&Locale::en),
    );

    assert_eq!(orig_num_vertexes, topologically_sorted_nodes.len());
    info!("Re-reading the file again to build the graph");
    let mut g = graph::DirectedGraph2::new();
    read_with_node_replacements(
        &args.input_filename,
        &tag_filter,
        &args.tag_filter_func,
        &|nid| *node_id_replaces.get(&nid).unwrap_or(&nid),
        &progress_bars,
        &mut g,
    )?;
    info!("Re-read nodes",);

    let setting_node_pos = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Reading file to save node locations")
            .with_style(style.clone()),
    );
    let mut nodeid_pos = nodeid_position::default();
    read_node_positions(
        &args.input_filename,
        // Previosuly, the g.contains_vertex would be false for nodes without outgoing, so we
        // needed nids_we_need. But now we don't
        // |nid| g.contains_vertex(&nid) || nids_we_need.contains(&nid),
        |nid| g.contains_vertex(&nid),
        &setting_node_pos,
        &mut nodeid_pos,
    )?;
    setting_node_pos.finish_and_clear();
    info_memory_used!();

    // Sorted list of all nids which are an end point
    let end_points: Vec<i64> = collect_into_vec_set_par(g.vertexes_wo_outgoing_jumbled());

    info!(
        "Calculated the {} end points",
        end_points.len().to_formatted_string(&Locale::en)
    );

    // Using index in `end_points` to get whether this end is in this filter
    let mut end_point_memberships: Vec<smallvec::SmallVec<[bool; 2]>> = Vec::new();

    // Upstream value for every end point
    let mut end_point_upstreams: Vec<f64> = vec![0.; end_points.len()];

    let mut ends_membership_filters: SmallVec<[tagfilter::TagFilter; 3]> =
        args.ends_membership.clone().into();
    ends_membership_filters.sort_by_key(|tf| tf.to_string());
    if !ends_membership_filters.is_empty() {
        end_point_memberships.resize(
            end_points.len(),
            smallvec::smallvec![false; ends_membership_filters.len()],
        );
    }

    // Vec, one for each end point (same index as end_points), with the tag values for the ways
    // which go through there.
    let mut end_point_tag_values: Vec<smallvec::SmallVec<[Option<String>; 1]>> = Vec::new();
    if !args.ends_tag.is_empty() {
        end_point_tag_values.resize(
            end_points.len(),
            smallvec::smallvec![None; args.ends_tag.len()],
        );
    }

    // This is an iterator that returns the total upstream length for all nodes.
    // it keeps track of an intermediate value which is the length of items “further downstream”.
    // It's a function because we want to create it many times.
    // return value: (
    //  nid_idx: usize. Index in topologically_sorted_nodes for this node
    //  nid: i64, OSM Node id
    //  upstream_length_m: f64, upstream value at this node
    let upstream_length_iter = || {
        topologically_sorted_nodes.iter().copied().enumerate().scan(
            // a cache of node id and the currently best known upstream value
            // we “push” values onto this from upstream
            HashMap::new() as HashMap<i64, f64>,
            |tmp_upstream_length, (nid_idx, nid)| {
                let curr_upstream = tmp_upstream_length.remove(&nid).unwrap_or(0.);

                // FIXME rather than split the upstream value equally between all out-edges, look
                // at the name tag (cf nid_pair_to_name_group)
                let num_outs = g.out_neighbours(nid).count() as f64;
                let per_downstream = curr_upstream / num_outs;
                let curr_pos = nodeid_pos.get(&nid).unwrap();
                for other in g.out_neighbours(nid) {
                    let other_pos = nodeid_pos.get(&other).unwrap();
                    let this_edge_len =
                        haversine::haversine_m(curr_pos.0, curr_pos.1, other_pos.0, other_pos.1);
                    *tmp_upstream_length.entry(other).or_default() +=
                        per_downstream + this_edge_len;
                }

                Some((nid_idx, nid, curr_upstream))
            },
        )
    };

    let calc_upstream_ends_bar = progress_bars.add(
        ProgressBar::new(end_points.len() as u64)
            .with_message("Calculating upstream value for end points")
            .with_style(style.clone()),
    );
    let calc_all_upstream_bar = progress_bars.add(
        ProgressBar::new(end_points.len() as u64)
            .with_message("Calculating upstream value for all End points")
            .with_style(style.clone()),
    );
    calc_all_upstream_bar.set_length(topologically_sorted_nodes.len() as u64);

    // Calculate all the upstream value for all the end points.
    for (_nid_idx, nid, upstream_length) in calc_all_upstream_bar.wrap_iter(upstream_length_iter())
    {
        if let Ok(idx) = end_points.binary_search(&nid) {
            end_point_upstreams[idx] = upstream_length;
            calc_all_upstream_bar.inc(1);
        }
    }
    calc_all_upstream_bar.finish_and_clear();
    calc_upstream_ends_bar.finish_and_clear();
    info!(
        "Calculated the upstream value for {} nodes",
        topologically_sorted_nodes
            .len()
            .to_formatted_string(&Locale::en)
    );
    info_memory_used!();

    let end_point_memberships = Arc::new(std::sync::RwLock::new(end_point_memberships));
    let end_point_tag_values = Arc::new(std::sync::RwLock::new(end_point_tag_values));

    if !ends_membership_filters.is_empty() || !args.ends_tag.is_empty() {
        info!("Rereading file to add memberships, or tag values, for ends");
        info!(
            "Adding the following {} attributes for each end: {}",
            ends_membership_filters.len(),
            ends_membership_filters
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
        let mut reader = osmio::stringpbf::PBFReader::new(reader);
        reader
            .ways()
            // ↑ all the ways
            .par_bridge()
            //
            // ↓ .. which match at least one end-membership filter, or match the regular tag
            // filter, and have a desired tag
            .filter(|w| {
                tagfilter::obj_pass_filters(w, &ends_membership_filters, &None)
                    || (tagfilter::obj_pass_filters(w, &tag_filter, &args.tag_filter_func)
                        && args.ends_tag.iter().any(|end_tag| w.has_tag(end_tag)))
            })
            //
            // ↓ .. which have at least one node in the end points
            .filter(|w| {
                w.nodes()
                    .iter()
                    .any(|nid| end_points.binary_search(nid).is_ok())
            })
            .for_each_with(
                (end_point_memberships.clone(), end_point_tag_values.clone()),
                |(end_point_memberships, end_point_tag_values), w| {
                    let filter_results = ends_membership_filters
                        .iter()
                        .map(|f| f.filter(&w))
                        .collect::<SmallVec<[bool; 2]>>();
                    for end_point_idx in w
                        .nodes()
                        .iter()
                        .filter_map(|nid| end_points.binary_search(nid).ok())
                    {
                        let mut curr_mbmrs_all = end_point_memberships.write().unwrap();
                        if !curr_mbmrs_all.is_empty() {
                            let curr_mbmrs = curr_mbmrs_all.get_mut(end_point_idx).unwrap();
                            for (new, old) in filter_results.iter().zip(curr_mbmrs.iter_mut()) {
                                *old |= new;
                            }
                        }

                        let mut curr_tags_all = end_point_tag_values.write().unwrap();
                        if !curr_tags_all.is_empty() {
                            let curr_tags = curr_tags_all.get_mut(end_point_idx).unwrap();
                            for (tag_key, tag_value) in
                                args.ends_tag.iter().zip(curr_tags.iter_mut())
                            {
                                if let Some(val) = w.tag(tag_key) {
                                    *tag_value = match tag_value {
                                        None => Some(val.to_string()),
                                        // There are mulitple ways that go through here, so join
                                        // do semicolon style concatination.
                                        Some(old_val) => Some(format!("{};{}", old_val, val)),
                                    }
                                }
                            }
                        }
                    }
                },
            );
    }
    let end_point_memberships = Arc::try_unwrap(end_point_memberships)
        .unwrap()
        .into_inner()
        .unwrap();
    let end_point_tag_values = Arc::try_unwrap(end_point_tag_values)
        .unwrap()
        .into_inner()
        .unwrap();

    if !end_point_memberships.is_empty() {
        // How many have ≥1 true value (versus all default of false)
        let num_nodes_attributed = end_point_memberships
            .par_iter()
            .filter(|membs| membs.par_iter().any(|m| *m))
            .count();
        if num_nodes_attributed == 0 {
            warn!("No end nodes got an end attribute.")
        } else {
            info!(
                "{} of {} ({:.1}%) end points got an attribute for way membership",
                num_nodes_attributed.to_formatted_string(&Locale::en),
                end_points.len().to_formatted_string(&Locale::en),
                ((100. * num_nodes_attributed as f64) / end_points.len() as f64)
            );
        }
    }

    // needed for default values later
    let empty_smallvec_bool = smallvec::smallvec![];
    let empty_smallvec_str = smallvec::smallvec![];

    if let Some(ref ends_filename) = args.ends {
        let end_points_output = end_points
            .iter()
            .zip(
                // If no end_point_memberships's then that vec is empty, so the zip doesn't return
                // anything. using chain(repeat(…)) to always give something
                end_point_memberships
                    .iter()
                    .chain(std::iter::repeat(&empty_smallvec_bool)),
            )
            .zip(
                end_point_tag_values
                    .iter()
                    .chain(std::iter::repeat(&empty_smallvec_str)),
            )
            .zip(end_point_upstreams.iter())
            .filter(|(((_nid, _mbms), _end_tgs), len)| {
                args.min_upstream_m.map_or(true, |min| *len >= &min)
            })
            .map(|(((nid, mbms), end_tags), len)| {
                (nid, mbms, end_tags, len, nodeid_pos.get(nid).unwrap())
            })
            .map(|(nid, mbms, end_tags, len, pos)| {
                // Round the upstream to only output 1 decimal place
                let mut props = serde_json::json!({"upstream_m": round(len, 1), "nid": nid});
                if !ends_membership_filters.is_empty() {
                    for (end_attr_filter, res) in ends_membership_filters.iter().zip(mbms.iter()) {
                        props[format!("is_in:{}", end_attr_filter)] = (*res).into();
                    }
                    props["is_in_count"] = mbms.iter().filter(|m| **m).count().into();
                }
                if !args.ends_tag.is_empty() {
                    for (tag_key, tag_value) in args.ends_tag.iter().zip(end_tags.into_iter()) {
                        props[format!("tag:{}", tag_key)] = tag_value.clone().into();
                    }
                }
                (props, pos)
            });

        let mut f = BufWriter::new(File::create(ends_filename)?);
        let num_written = write_geojson_features_directly(
            end_points_output,
            &mut f,
            &fileio::format_for_filename(ends_filename),
        )?;
        info!(
            "Wrote {} features to output file {}",
            num_written.to_formatted_string(&Locale::en),
            ends_filename.to_str().unwrap()
        );
    }

    assert!(
        end_points.len() < i32::MAX as usize,
        "Too many end nodes (>2³²). We optimize by addressing nodes with a i32"
    );

    // For every node in topologically_sorted_nodes, store the index (in end_points) of the biggest
    // end point that this node flows into.
    // We store the index as a i32 to save space. We assume we will have <2³² end points
    // -1 = no known end point (yet).
    // (biggest end point = end point with the largest upstream value)
    // NB: This is a top level variable (for scoping reasons), but remains empty if we're not doing
    // anything
    // TODO replace this with nonzerou32
    let mut upstream_assigned_end: Vec<i32> = Vec::new();

    if args.upstream_output_biggest_end {
        upstream_assigned_end.resize(topologically_sorted_nodes.len(), -1);

        // this is a cache of values as we walk upstream
        let mut tmp_biggest_end: HashMap<i64, i32> = HashMap::new();

        // Doing topologically_sorted_nodes in reverse, means we are “walking upstream”. We will
        for (nid_idx, &nid) in topologically_sorted_nodes.iter().enumerate().rev() {
            // if this node is an end point then save that
            // otherwise, use the value from the cache
            let this_end_idx = end_points.binary_search(&nid).ok().map(|i| i as i32);
            let curr_biggest = tmp_biggest_end.remove(&nid).or(this_end_idx).unwrap();
            upstream_assigned_end[nid_idx] = curr_biggest;

            for upper in g.in_neighbours(nid) {
                tmp_biggest_end
                    .entry(upper)
                    .and_modify(|prev_biggest_end_idx| {
                        // for all nodes which are one step upstream of this node, check the
                        // previously calcualted best and update if needed.
                        if end_point_upstreams[*prev_biggest_end_idx as usize]
                            < end_point_upstreams[curr_biggest as usize]
                        {
                            *prev_biggest_end_idx = curr_biggest;
                        }
                    })
                    // or just store this end point.
                    .or_insert(curr_biggest);
            }
        }
    } else if args.upstream_assign_end_by_tag.is_some() {
        upstream_assigned_end.resize(topologically_sorted_nodes.len(), -1);

        // this is a cache of values as we walk upstream
        // key: i64 = the node id
        // value: (bool, i32), i32 is the endidx for the best guess. bool=true → this assignment is
        // based on the name, bool=false → assignemnt is based on largest end
        let mut tmp_biggest_end: HashMap<i64, (bool, i32)> = HashMap::new();

        // Doing topologically_sorted_nodes in reverse, means we are “walking upstream”. This
        // ensures we visit an end point before we visit any node upstream of it.
        for (nid_idx, &nid) in topologically_sorted_nodes.iter().enumerate().rev() {
            // if this node is an end point then save that
            // otherwise, use the value from the cache
            let this_end_idx = end_points.binary_search(&nid).ok().map(|i| i as i32);
            let curr_biggest = tmp_biggest_end
                .remove(&nid)
                .map(|(_, i)| i)
                .or(this_end_idx)
                .unwrap();
            upstream_assigned_end[nid_idx] = curr_biggest;

            let downstream_nid_opt = g.out_neighbours(nid).next();
            if g.out_neighbours(nid).count() == 1
                && nid_pair_to_endtag_group.contains_key(&(nid, downstream_nid_opt.unwrap()))
            {
                let name_id = nid_pair_to_endtag_group
                    .get(&(nid, downstream_nid_opt.unwrap()))
                    .unwrap();
                // This node (nid) has 1 outgoing vertex, which has name name_id
                for upper in g.in_neighbours(nid) {
                    if nid_pair_to_endtag_group.get(&(upper, nid)) == Some(name_id) {
                        // The vertex from upper to nid, also has the same name.
                        // assign upper to this end, regardless of which is bigger
                        tmp_biggest_end.insert(upper, (true, curr_biggest));
                    } else {
                        // assign only if this is bigger
                        tmp_biggest_end
                            .entry(upper)
                            .and_modify(|(prev_is_name_based, prev_biggest_end_idx)| {
                                // Only set the value if the previous value wasn't based on names
                                if !*prev_is_name_based {
                                    // for all nodes which are one step upstream of this node, check the
                                    // previously calcualted best and update if needed.
                                    if end_point_upstreams[*prev_biggest_end_idx as usize]
                                        < end_point_upstreams[curr_biggest as usize]
                                    {
                                        *prev_biggest_end_idx = curr_biggest;
                                    }
                                }
                            })
                            // or just store this end point.
                            .or_insert((false, curr_biggest));
                    }
                }
            } else {
                for upper in g.in_neighbours(nid) {
                    tmp_biggest_end
                        .entry(upper)
                        .and_modify(|(_prev_is_name_based, prev_biggest_end_idx)| {
                            // for all nodes which are one step upstream of this node, check the
                            // previously calcualted best and update if needed.
                            if end_point_upstreams[*prev_biggest_end_idx as usize]
                                < end_point_upstreams[curr_biggest as usize]
                            {
                                *prev_biggest_end_idx = curr_biggest;
                            }
                        })
                        // or just store this end point.
                        .or_insert((false, curr_biggest));
                }
            }
        }
    }
    assert!(upstream_assigned_end.par_iter().all(|end| *end >= 0));

    if let Some(ref grouped_ends) = args.grouped_ends {
        do_group_by_ends(
            grouped_ends,
            &g,
            &progress_bars,
            &style,
            &end_points,
            &topologically_sorted_nodes,
            &end_point_upstreams,
            &upstream_assigned_end,
            &nodeid_pos,
            &end_point_tag_values,
            &args.ends_tag,
        )?;
    }

    if let Some(ref upstream_filename) = args.upstreams {
        // we loop over all nodes in topologically_sorted_nodes (which is annotated in
        // upstream_length_iter with the upstream value) and flat_map that into each line segment
        // that goes out of that.
        let lines = upstream_length_iter()
            .filter(|(_from_nid_idx, _from_nid, upstream_m)| {
                args.min_upstream_m.map_or(true, |min| *upstream_m >= min)
            })
            .flat_map(|(from_nid_idx, from_nid, upstream_len)| {
                g.out_neighbours(from_nid)
                    .map(move |to_nid| (from_nid_idx, from_nid, to_nid, upstream_len))
            })
            .map(|(from_nid_idx, from_nid, to_nid, upstream_len)| {
                // Round the upstream to only output 1 decimal place
                let mut props = serde_json::json!({
                    "from_upstream_m": round(&upstream_len, 1),
                });

                for mult in args.upstream_from_upstream_multiple.iter() {
                    props[format!("from_upstream_m_{}", mult)] =
                        round_mult(&upstream_len, *mult).into();
                }

                if upstream_assigned_end.len() >= from_nid_idx {
                    let end_idx: usize = upstream_assigned_end[from_nid_idx] as usize;
                    props["end_upstream_m"] = round(&end_point_upstreams[end_idx], 1).into();
                    props["end_nid"] = end_points[end_idx].into();

                    for (tag_key, tag_value) in args
                        .ends_tag
                        .iter()
                        .zip(end_point_tag_values[end_idx].iter())
                    {
                        if let Some(tag_value) = tag_value {
                            props[format!("end_tag:{}", tag_key)] = tag_value.clone().into();
                        }
                    }
                } else if args.upstream_output_ends_full {
                    todo!();
                    //let ends = upstream_ends_full.get(&from_nid).unwrap();
                    //props["num_ends"] = ends.len().into();
                    //props["ends"] = ends.iter().copied().collect::<Vec<i64>>().into();
                    //let mut ends_strs = vec![",".to_string()];
                    //let mut this_len;
                    //let mut biggest_end = (upstream_length.get(&ends[0]).unwrap(), ends[0]);
                    //for end in ends {
                    //    ends_strs.push(end.to_string());
                    //    ends_strs.push(",".to_string());
                    //    this_len = upstream_length.get(&ends[0]).unwrap();
                    //    if this_len > biggest_end.0 {
                    //        biggest_end = (this_len, *end);
                    //    }
                    //}
                    //props["ends_s"] = ends_strs.join("").into();
                    //props["biggest_end_upstream_m"] = round(biggest_end.0, 1).into();
                    //props["biggest_end_nid"] = biggest_end.1.into();
                }

                (
                    props,
                    (
                        nodeid_pos.get(&from_nid).unwrap(),
                        nodeid_pos.get(&to_nid).unwrap(),
                    ),
                )
            });
        info_memory_used!();

        let writing_upstreams_bar = progress_bars.add(
            ProgressBar::new(g.num_edges() as u64)
                .with_message("Writing upstreams geojson(s) file")
                .with_style(style.clone()),
        );

        let lines = lines.progress_with(writing_upstreams_bar);

        let mut f = std::io::BufWriter::new(std::fs::File::create(upstream_filename)?);

        let num_written;
        if upstream_filename.extension().unwrap() == "geojsons"
            || upstream_filename.extension().unwrap() == "geojson"
        {
            num_written = write_geojson_features_directly(
                lines,
                &mut f,
                &fileio::format_for_filename(upstream_filename),
            )?;
        } else if upstream_filename.extension().unwrap() == "csv" {
            let mut csv_columns = vec!["from_upstream_m".to_string()];
            if args.upstream_output_biggest_end {
                csv_columns.push("biggest_end_nid".to_string());
            }
            for mult in args.upstream_from_upstream_multiple.iter() {
                csv_columns.push(format!("from_upstream_m_{}", mult));
            }
            if !args.ends_tag.is_empty() {
                warn!("Adding the end tag values has not been implemented for CSV files yet.");
            }

            num_written = write_csv_features_directly(lines, &mut f, &csv_columns)?;
        } else {
            anyhow::bail!("Unsupported output format");
        }

        info!(
            "Wrote {} features to output file {}",
            num_written.to_formatted_string(&Locale::en),
            upstream_filename.display(),
        );
    }

    info!(
        "Finished all in {}",
        formatting::format_duration(global_start.elapsed())
    );

    Ok(())
}

fn read_with_node_replacements(
    input_filename: &Path,
    tag_filter: &SmallVec<[tagfilter::TagFilter; 3]>,
    tag_filter_func: &Option<tagfilter::TagFilterFunc>,
    node_id_replaces: &(impl Fn(i64) -> i64 + Sync),
    progress_bars: &MultiProgress,
    graph: &mut impl graph::DirectedGraphTrait,
) -> Result<()> {
    let file_reading_style =
                ProgressStyle::with_template(
        "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {bytes:>7}/{total_bytes:7} {per_sec:>12} {msg}",
            ).unwrap();
    let input_fp = std::fs::File::open(input_filename)?;
    let input_bar = progress_bars.add(
        ProgressBar::new(input_fp.metadata()?.len())
            .with_message("Reading input file")
            .with_style(file_reading_style.clone()),
    );
    let rdr = input_bar.wrap_read(input_fp);

    let mut reader = osmio::stringpbf::PBFReader::new(rdr);

    let obj_reader = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {human_pos} OSM ways read {per_sec:>20}",
            )
            .unwrap(),
        ),
    );
    let ways_added = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} ways collected so far for later processing",
            )
            .unwrap(),
        ),
    );
    let nodes_added = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} nodes collected so far for later processing",
            )
            .unwrap(),
        ),
    );

    let graph = Arc::new(Mutex::new(graph));

    reader
        .ways()
        .par_bridge()
        .inspect(|_| obj_reader.inc(1))
        .filter(|w| tagfilter::obj_pass_filters(w, tag_filter, tag_filter_func))
        .inspect(|_| ways_added.inc(1))
        // TODO support grouping by tag value
        .for_each_with(graph.clone(), |graph, w| {
            let nodes = w.nodes();
            nodes_added.inc(nodes.len() as u64);
            if nodes.par_iter().any(|nid| node_id_replaces(*nid) != *nid) {
                let mut new_nodes = nodes
                    .iter()
                    .map(|nid| node_id_replaces(*nid))
                    .collect::<Vec<i64>>();
                new_nodes.dedup();
                let mut graph = graph.lock().unwrap();
                graph.add_edge_chain(&new_nodes);
            } else {
                // None of these are replacements, so add directly
                // add the nodes from w to this graph
                let mut graph = graph.lock().unwrap();
                graph.add_edge_chain(nodes);
            }
        });
    obj_reader.finish_and_clear();
    ways_added.finish_and_clear();
    nodes_added.finish_and_clear();

    Ok(())
}

fn read_node_positions(
    input_filename: &Path,
    wanted_nodeid_func: impl Fn(i64) -> bool + Sync,
    setting_node_pos: &ProgressBar,
    nodeid_pos: &mut impl NodeIdPosition,
) -> Result<()> {
    let reader = osmio::stringpbf::PBFNodePositionReader::from_filename(input_filename)?;
    let nodeid_pos = Arc::new(Mutex::new(nodeid_pos));
    reader
        .into_iter()
        .par_bridge()
        .filter(|(nid, _pos)| wanted_nodeid_func(*nid))
        .map(|(nid, pos)| (nid, (pos.1.inner(), pos.0.inner()))) // WTF do I have lat & lon
        // mixed up??
        .for_each_with(nodeid_pos.clone(), |nodeid_pos, (nid, pos)| {
            assert!(nid > 0, "This file has a node < 0. negative ids are not supported in this tool Use osmium sort & osmium renumber to convert this file and run again.");
            setting_node_pos.inc(1);
            nodeid_pos.lock().unwrap().insert_i32(nid, pos);
        });

    let nodeid_pos = Arc::try_unwrap(nodeid_pos).unwrap().into_inner().unwrap();
    nodeid_pos.finished_inserting();

    setting_node_pos.finish_and_clear();

    debug!("{}", nodeid_pos.detailed_size());

    Ok(())
}

fn node_group_to_lines(nids: &[[i64; 2]], pos: &impl NodeIdPosition) -> Vec<Vec<(f64, f64)>> {
    nids.iter()
        .map(|seg| {
            let pos0 = pos.get(&seg[0]).unwrap();
            let pos1 = pos.get(&seg[1]).unwrap();
            vec![pos0, pos1]
        })
        .collect()
}

fn node_group_to_length_m(nids: &[[i64; 2]], pos: &impl NodeIdPosition) -> f64 {
    nids.iter()
        .map(|seg| {
            let pos0 = pos.get(&seg[0]).unwrap();
            let pos1 = pos.get(&seg[1]).unwrap();
            haversine_m(pos0.1, pos0.0, pos1.0, pos1.1)
        })
        .sum()
}

fn linestring_length(coords: &[(f64, f64)]) -> f64 {
    coords
        .par_windows(2)
        .map(|pair| haversine_m(pair[0].1, pair[0].0, pair[1].1, pair[1].0))
        .sum::<f64>()
}

fn multilinestring_length(coords: &Vec<Vec<(f64, f64)>>) -> f64 {
    coords.par_iter().map(|c| linestring_length(c)).sum()
}

/// Round this float to this many places after the decimal point.
/// Used to reduce size of output geojson file
fn round(f: &f64, places: u8) -> f64 {
    let places: f64 = 10_u64.pow(places as u32) as f64;
    (f * places).round() / places
}

/// Round this float to be a whole number multiple of base.
fn round_mult(f: &f64, base: i64) -> i64 {
    (f / (base as f64)).round() as i64 * base
}

fn _bbox_area(points: impl Iterator<Item = (f64, f64)>) -> Option<f64> {
    use std::cmp::{max, min};
    points
        .map(|p| (OrderedFloat(p.0), OrderedFloat(p.1)))
        .map(|p| (p.0, p.0, p.1, p.1))
        .reduce(|acc, p| {
            (
                min(acc.0, p.0),
                max(acc.1, p.1),
                min(acc.2, p.2),
                max(acc.3, p.3),
            )
        })
        .map(|bbox| (bbox.1 - bbox.0, bbox.3 - bbox.2))
        .map(|delta| (delta.0.into_inner(), delta.1.into_inner()))
        .map(|delta| delta.0 * delta.1)
}

fn collect_into_vec_set_par<T>(it: impl ParallelIterator<Item = T>) -> Vec<T>
where
    T: Ord + Send,
{
    let mut result: Vec<T> = it.collect();
    result.sort_unstable();
    result.dedup();
    result.shrink_to_fit();
    result
}

#[allow(dead_code)]
fn collect_into_vec_set<T>(it: impl Iterator<Item = T>) -> Vec<T>
where
    T: Ord + Send,
{
    let mut result: Vec<T> = it.collect();
    result.sort_unstable();
    result.dedup();
    result.shrink_to_fit();
    result
}

#[allow(clippy::too_many_arguments)]
fn do_group_by_ends(
    output_filename: &Path,
    g: &graph::DirectedGraph2,
    progress_bars: &MultiProgress,
    style: &ProgressStyle,
    end_points: &[i64],
    topologically_sorted_nodes: &[i64],
    end_point_upstreams: &[f64],
    upstream_assigned_end: &[i32],
    nodeid_pos: &impl NodeIdPosition,
    end_point_tag_values: &[SmallVec<[Option<String>; 1]>],
    ends_tags: &[String],
) -> Result<()> {
    let started = Instant::now();
    let nodes_bar = progress_bars.add(
        ProgressBar::new(topologically_sorted_nodes.len() as u64)
            .with_message("Nodes allocated to an End")
            .with_style(style.clone()),
    );
    let segments_spinner =
        progress_bars.add(ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("       {human_pos} segments output").unwrap(),
        ));

    let mut in_progress_lines: HashMap<i64, SmallVec<[Vec<i64>; 2]>> = HashMap::new();

    anyhow::ensure!(
        !upstream_assigned_end.is_empty(),
        "The upstream_biggest_ends is empty. Has this not been calculated?"
    );

    anyhow::ensure!(
        !upstream_assigned_end.is_empty(),
        "The upstream_biggest_ends is empty. Has this not been calculated?"
    );
    anyhow::ensure!(
        upstream_assigned_end.len() == topologically_sorted_nodes.len(),
        "Lengths not equal, something will be lost"
    );

    let mut nid_end_iter = topologically_sorted_nodes
        .iter()
        .rev()
        .zip(upstream_assigned_end.iter().rev());

    let mut possible_ins: SmallVec<[i64; 5]> = smallvec::smallvec![];
    let mut results_to_pop: SmallVec<[(i64, Vec<i64>); 3]> = smallvec::smallvec![];

    // Iterator that yields (end_node_id, and a path of nids which end in this nid)
    let upstreams_grouped_by_end = std::iter::from_fn(|| {
        // ↓ This code definitly does too many allocations (incl. for Vec's) and could be optimised
        loop {
            if let Some(mut result) = results_to_pop.pop() {
                // we have something to return.
                result.1.reverse();
                return Some(result);
            }
            let (nid, real_end_idx) = match nid_end_iter.next() {
                None => {
                    // finished all the nodes
                    return None;
                }
                Some(x) => x,
            };

            // which in progress lines are there for this node
            let mut lines_to_here = in_progress_lines.remove(nid).unwrap_or_default();
            if *nid == end_points[*real_end_idx as usize] {
                // this node is an end node (and it is it's own end node)

                assert!(lines_to_here.is_empty()); // sanity check

                // Give us something to work with later. This is the start of the lines that start
                // here.
                lines_to_here.push(vec![]);
            }

            // include this point in every line so far
            for line in lines_to_here.iter_mut() {
                line.push(*nid);
            }

            // If >1 line is ending here, end all lines (bar the first one) here
            while lines_to_here.len() > 1 {
                results_to_pop.push((*real_end_idx as i64, lines_to_here.pop().unwrap()));
            }

            // this is the only line we continue with
            let line_to_here = lines_to_here.pop().unwrap();
            assert!(lines_to_here.is_empty());

            possible_ins.clear();
            possible_ins.extend(g.in_neighbours(*nid));
            // for the 2nd+ ups, create new paths that start on this node
            for later_upstream_nodes in possible_ins.iter().skip(1) {
                in_progress_lines
                    .entry(*later_upstream_nodes)
                    .or_default()
                    .push(vec![*nid]);
            }

            if possible_ins.is_empty() {
                // no upstreams, so finish it.
                results_to_pop.push((*real_end_idx as i64, line_to_here));
                continue;
            } else {
                // We have >0 upstreams, we've already done something with the 2nd+, so the first
                // upstream is part of the only line here.
                let nxt = possible_ins[0];
                in_progress_lines.entry(nxt).or_default().push(line_to_here);
            }
        }
    })
    .map(|(end_idx, path)| {
        segments_spinner.inc(1);
        nodes_bar.inc(path.len().saturating_sub(1) as u64);

        let points = path
            .into_iter()
            .map(|nid| nodeid_pos.get(&nid).unwrap())
            .collect::<Vec<_>>();
        let mut props = serde_json::json!({
            "end_nid": end_points[end_idx as usize],
            "end_upstream_m": round(&end_point_upstreams[end_idx as usize], 1),
        });
        if !ends_tags.is_empty() {
            for (tag_key, tag_value) in ends_tags
                .iter()
                .zip(end_point_tag_values[end_idx as usize].iter())
            {
                if let Some(tag_value) = tag_value {
                    props[format!("end_tag:{}", tag_key)] = tag_value.clone().into();
                }
            }
        }
        (props, points)
    });

    let output_format = fileio::format_for_filename(output_filename);
    let mut f = BufWriter::new(File::create(output_filename)?);
    let (send, recv) = std::sync::mpsc::channel();

    std::thread::spawn({
        move || {
            let _total_written =
                write_geojson_features_directly(recv.iter(), &mut f, &output_format).unwrap();
        }
    });
    let mut num_written = 0;
    for val in upstreams_grouped_by_end {
        send.send(val).unwrap();
        num_written += 1;
    }

    let grouping_duration = started.elapsed();
    info!(
        "Wrote {} end-grouped-features to output file {} in {}. {:.3e} nodes/sec",
        num_written.to_formatted_string(&Locale::en),
        output_filename.to_str().unwrap(),
        formatting::format_duration(grouping_duration),
        (topologically_sorted_nodes.len() as f64) / grouping_duration.as_secs_f64(),
    );

    Ok(())
}

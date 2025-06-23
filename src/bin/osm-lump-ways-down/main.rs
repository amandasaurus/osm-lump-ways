#![allow(unused_variables)]
use anyhow::Result;
use clap::Parser;
use get_size::GetSize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressIterator, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
#[allow(unused_imports)]
use log::{
    Level::{Debug, Trace},
    debug, error, info, log, trace, warn,
};
use osmio::OSMObjBase;
use osmio::prelude::*;
use rayon::prelude::*;

use itertools::Itertools;
use std::cmp::min;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

use std::sync::atomic::{AtomicI64, Ordering as atomic_Ordering};
use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};
use smallvec::SmallVec;

use country_boundaries::{BOUNDARIES_ODBL_360X180, CountryBoundaries, LatLon};
use ordered_float::OrderedFloat;

mod cli_args;

use graph::DirectedGraphTrait;
use haversine::haversine_m;
use nodeid_position::NodeIdPosition;
use osm_lump_ways::dij;
use osm_lump_ways::graph;
use osm_lump_ways::haversine;
use osm_lump_ways::inter_store;
use osm_lump_ways::nodeid_position;
use osm_lump_ways::sorted_slice_store::{SortedSliceMap, SortedSliceSet};
use osm_lump_ways::tagfilter;

use fileio::{write_csv_features_directly, write_geojson_features_directly};
use osm_lump_ways::fileio;

use osm_lump_ways::formatting;
use smallvec::smallvec;

use serde_json::json;

mod ends_csv;
mod loops_csv_stats;
mod openmetrics;

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

macro_rules! sort_dedup {
    ($item:expr_2021) => {
        $item.par_sort_unstable();
        $item.dedup();
    };
}

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
    let reader = osmio::stringpbf::PBFReader::new(rdr);

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
            || args.ends_csv_file.is_some()
            || args.loops.is_some()
            || args.loops_csv_stats_file.is_some()
            || args.upstreams.is_some()
            || args.grouped_ends.is_some()
            || args.grouped_waterways.is_some(),
        "Nothing to do. You need to specifiy one of --ends/--loops/--upstreams/etc."
    );

    if (args.grouped_ends.is_some()
        || args.upstreams.is_some()
        || args.ends.is_some()
        || args.ends_csv_file.is_some())
        && !(args.flow_split_equally || args.flow_follows_tag.is_some())
    {
        error!(
            "If you want to output upstreams or ends, you must specificy one of --flow-split-equally or --flow-follows-tag TAG"
        );
        anyhow::bail!(
            "If you want to output upstreams or ends, you must specificy one of --flow-split-equally or --flow-follows-tag TAG"
        );
    }
    if args.ends_csv_file.is_some() && args.ends_tag.is_empty() {
        warn!(
            "The ends CSV file only makes sense with the --ends-tag arguments. Since you have specified no end tags, nothing will be written to the ends CSV file"
        );
    }

    info!("Input file: {:?}", &args.input_filename);
    if args.tag_filter.is_empty() {
        match args.tag_filter_func {
            Some(ref tff) => {
                info!("Tag filter function in operation: {:?}", tff);
            }
            _ => {
                info!("No tag filtering in operation. All ways in the file will be used.");
            }
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

    let mut loops_metrics = args.loops_openmetrics.as_ref().map(openmetrics::init);
    let mut loops_csv_stats = args
        .loops_csv_stats_file
        .as_ref()
        .map(loops_csv_stats::init);
    let mut ends_csv = args
        .ends_csv_file
        .as_ref()
        .map(|f| ends_csv::init(f, &args));

    let boundaries = CountryBoundaries::from_reader(BOUNDARIES_ODBL_360X180)?;

    // how many vertexes are there per node id? (which do we need to keep)
    let nids_in_ne2_ways = do_read_nids_in_ne2_ways(
        reader,
        &tag_filter,
        &args.tag_filter_func,
        &input_bar,
        &progress_bars,
    )?;

    let g = graph::DirectedGraph2::new();
    let g = Arc::new(Mutex::new(g));

    // Stores the tagvalue group for each segment.
    // OSM tag vaules are strings, but we don't need to store the strings. We give each string a
    // unique id (u32) and store that. We don't need to know what the tagvalue for 2 segments is,
    // we only need to know if they are the same or not.
    let mut nid_pair_to_tagid: HashMap<(i64, i64), u32> = HashMap::new();

    // first step, get all the cycles
    let latest_timestamp = AtomicI64::new(0);
    let start_reading_ways = Instant::now();
    let input_fp = std::fs::File::open(&args.input_filename)?;
    let input_bar = progress_bars.add(
        ProgressBar::new(input_fp.metadata()?.len())
            .with_message("Reading input file")
            .with_style(file_reading_style.clone()),
    );
    let rdr = input_bar.wrap_read(input_fp);
    let mut reader = osmio::stringpbf::PBFReader::new(rdr);
    let inter_store = inter_store::InterStore::new();
    //let inter_store: HashMap<(i64, i64), Box<[i64]>> = Default::default();
    let inter_store = Arc::new(Mutex::new(inter_store));
    let tagvalues_to_edges = Arc::new(Mutex::new(
        HashMap::new() as HashMap<String, HashSet<(i64, i64)>>
    ));

    reader
        .ways()
        .par_bridge()
        .inspect(|_| obj_reader.inc(1))
        .filter(|w| tagfilter::obj_pass_filters(w, &tag_filter, &args.tag_filter_func))
        .inspect(|_| ways_added.inc(1))
        // TODO support grouping by tag value
        .for_each_with((g.clone(), inter_store.clone(), tagvalues_to_edges.clone(), Vec::<i64>::new()),
            |(g, inter_store, seen_tagvalues, nodes_buf), w| {
                assert!(w.id() > 0, "This file has a way id < 0. negative ids are not supported in this tool Use osmium sort & osmium renumber to convert this file and run again.");
                // add the nodes from w to this graph
                let mut g = g.lock().unwrap();
                let mut inter_store = inter_store.lock().unwrap();
                let mut seen_tagvalues = seen_tagvalues.lock().unwrap();
                nodes_added.inc(w.nodes().len() as u64);

                // If we're assigning based on tag, get the hashset where it'll be stored
                let mut tagvalues_to_edges = args.flow_follows_tag
                    .as_ref()
                    .and_then(|flow_follows_tag| w.tag(flow_follows_tag))
                    .map(|way_tag_value| seen_tagvalues.entry(way_tag_value.to_string()).or_default());

                // Possibly remove duplicate nodes in a way. IME this happens once in the planet.
                let mut nodes = if w.nodes().windows(2).any(|w| w[0] == w[1]) {
                    warn!("Way {} has repeating nodes. at: {:?} Removing them for this processing", w.id(), w.nodes().windows(2).enumerate().filter(|(_i, w)| w[0] == w[1]).collect::<Vec<_>>());
                    nodes_buf.truncate(0);
                    nodes_buf.extend(w.nodes().iter().copied());
                    nodes_buf.dedup();
                    nodes_buf
                } else {
                    w.nodes()
                };

                // Don't add all the nodes, just the ones we need
                while nodes.len() >= 2 {
                    let i_opt = nodes.iter().skip(1).position(|nid| nids_in_ne2_ways.contains(nid));

                    let mut i = i_opt.unwrap() + 1;
                    // can happen when a river splits and then joins again. try to stop reducing
                    // this little tributary away.
                    while ( g.contains_edge(nodes[0], nodes[i]) || nodes[0] == nodes[i] ) && i > 1 {
                        i -= 1;
                    }
                    // 2 nodes after another with nothing in between? That can happen with someone
                    // double maps a river. But assert a differnet problem, which shows our ability
                    // to contract edges has problems
                    if i > 1 {
                        assert!(!g.contains_edge(nodes[0], nodes[i]), "already existing edge from {} to {} (there are {} nodes in the middle) i={}", nodes[0], nodes[i], nodes.len(), i);
                    }


                    assert!(i != 0);
                    assert!(nodes[0] != nodes[i], "Duplicate nodes in this way={:?} curr nodes={:?} i={}", w, nodes, i);
                    g.add_edge(nodes[0], nodes[i]);

                    if let Some(ref mut tagvalues_to_edges) = tagvalues_to_edges {
                        tagvalues_to_edges.insert((nodes[0], nodes[i]));
                    }

                    inter_store.insert_directed((nodes[0], nodes[i]), &nodes[1..i]);

                    // For the next loop iteration
                    nodes = &nodes[i..];
                }


                //g.add_edge_chain_contractable(w.nodes(), &|nid| nids_in_ne2_ways.binary_search(nid).is_err());
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

    let mut g = Arc::try_unwrap(g).unwrap().into_inner().unwrap();
    let inter_store = Arc::try_unwrap(inter_store).unwrap().into_inner().unwrap();
    let tagvalues_to_edges = Arc::try_unwrap(tagvalues_to_edges)
        .unwrap()
        .into_inner()
        .unwrap();

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
    let mut tag_group_value = Vec::with_capacity(tagvalues_to_edges.len());

    // Convert the HashMap to something we can look up based on an edge. also throw away the
    // unneeded string value.
    if let Some(ref flow_follows_tag) = args.flow_follows_tag {
        assert!(tagvalues_to_edges.len() < u32::MAX as usize);

        let total_num_pairs = tagvalues_to_edges
            .par_iter()
            .map(|(_tagvalue, pairs)| pairs.len())
            .sum();
        nid_pair_to_tagid.reserve(total_num_pairs);

        info!(
            "Have following {} unique '{}' tags in {} node pairs",
            tagvalues_to_edges.len().to_formatted_string(&Locale::en),
            flow_follows_tag,
            total_num_pairs.to_formatted_string(&Locale::en),
        );
        for (tagvalue, pairs) in tagvalues_to_edges.into_iter() {
            let curr_id = tag_group_value.len();
            for pair in pairs.into_iter() {
                nid_pair_to_tagid.insert(pair, curr_id as u32);
            }
            tag_group_value.push(tagvalue);
        }
        info!(
            "Total size of the '{}' lookup: {} bytes",
            flow_follows_tag,
            nid_pair_to_tagid
                .get_size()
                .to_formatted_string(&Locale::en)
        );
    }

    let calc_components_bar = progress_bars.add(
        ProgressBar::new((num_vertexes * 2) as u64)
            .with_message("Looking for cycles")
            .with_style(style.clone()),
    );
    info_memory_used!();

    let cycles: Vec<Vec<[i64; 2]>> = g.strongly_connected_components(&calc_components_bar);
    // expand the intermediate values
    let cycles = cycles
        .into_iter()
        .map(|segs| {
            segs.into_iter()
                .flat_map(|seg| {
                    inter_store
                        .expand_directed(seg[0], seg[1])
                        .tuple_windows()
                        .map(|(a, b)| [a, b])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

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

    if !cycles.is_empty()
        && (args.loops.is_some() || loops_csv_stats.is_some() || loops_metrics.is_some())
    {
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
                });
                if !(!args.loops_incl_nids & args.loops_no_incl_nids) {
                    props["nodes"] = all_nodes
                        .into_iter()
                        .map(|nid| format!("n{}", nid))
                        .collect::<Vec<_>>()
                        .join(",")
                        .into();
                }

                for (i, boundary) in these_boundaries.iter().enumerate() {
                    props[format!("area_{}", i)] = boundary.to_string().into();
                }
                props["areas_s"] = format!(",{},", these_boundaries.join(",")).into();
                props["areas"] = these_boundaries.into();

                Ok((props, coords))
            })
            .collect::<Result<_>>()?;
        info_memory_used!();

        if loops_csv_stats.is_some() || loops_metrics.is_some() {
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
                if let Some(ref mut loops_csv_stats) = loops_csv_stats {
                    loops_csv_stats::write_boundary(
                        loops_csv_stats,
                        boundary,
                        latest_timestamp,
                        &latest_timestamp_iso,
                        count,
                        len,
                    )?;
                }
                if let Some(ref mut loops_metrics) = loops_metrics {
                    openmetrics::write_boundary(
                        loops_metrics,
                        boundary,
                        latest_timestamp,
                        count,
                        len,
                    )?;
                }
            }

            if let Some(ref mut loops_csv_stats) = loops_csv_stats {
                loops_csv_stats.flush()?;
                info!(
                    "Loop statistics have been written to file {}",
                    args.loops_csv_stats_file.as_ref().unwrap().display()
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

    if args.ends.is_none()
        && args.ends_csv_file.is_none()
        && args.upstreams.is_none()
        && args.grouped_ends.is_none()
        && args.grouped_waterways.is_none()
    {
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
    info!("Contracting the graph");
    for (vertex, replacement) in node_id_replaces.iter() {
        g.contract_vertex(vertex, replacement);

        // If this segment is part of a cycle, then just delete it from the tag groups and move
        // on.
        nid_pair_to_tagid.remove(&(*vertex, *replacement));
        nid_pair_to_tagid.remove(&(*replacement, *vertex));
    }

    // convert to memory effecient sorted vec.
    let nid_pair_to_tagid = SortedSliceMap::from_iter(nid_pair_to_tagid.into_iter());
    let tag_group_value = tag_group_value.into_boxed_slice();

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
    let topologically_sorted_nodes = g
        .clone()
        .into_vertexes_topologically_sorted(&sorting_nodes_bar)
        .into_boxed_slice();
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

    let mut nids_we_need = HashSet::with_capacity(g.num_vertexes());
    nids_we_need.extend(g.vertexes());
    nids_we_need.extend(inter_store.all_inter_nids());
    nids_we_need.extend(node_id_replaces.keys());
    nids_we_need.shrink_to_fit();

    let setting_node_pos = progress_bars.add(
        ProgressBar::new(nids_we_need.len() as u64)
            .with_message("Reading file to save node locations")
            .with_style(style.clone()),
    );
    let mut nodeid_pos = nodeid_position::default();
    read_node_positions(
        &args.input_filename,
        |nid| nids_we_need.contains(&nid),
        &setting_node_pos,
        &mut nodeid_pos,
    )?;
    setting_node_pos.finish_and_clear();
    drop(nids_we_need);
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

    // Calculate the upstream for every node and edge.

    // Upstream m value for each node in topologically_sorted_nodes
    let upstream_length: Vec<f64> = vec![0.; topologically_sorted_nodes.len()];
    let mut upstream_length = upstream_length.into_boxed_slice();

    // for an edge in the graph, what's the amount flowing down it. Keys are the from & to nid.
    // Value is upstream_m just at the start of the from nid.
    let mut upstream_per_edge: Vec<((i64, i64), f64)> =
        Vec::with_capacity(topologically_sorted_nodes.len());
    upstream_per_edge.extend(topologically_sorted_nodes.iter().flat_map(|&nid1| {
        g.out_neighbours(nid1)
            .map(move |nid2| ((nid1, nid2), f64::NAN))
    }));
    let mut upstream_per_edge = SortedSliceMap::from_vec(upstream_per_edge);

    let calc_all_upstreams = progress_bars.add(
        ProgressBar::new(topologically_sorted_nodes.len() as u64)
            .with_message("Calculating all upstream values")
            .with_style(style.clone()),
    );
    // a cache of node id and the currently best known upstream value
    // we “push” values onto this from upstream HashMap::new() as HashMap<i64, f64>,
    let mut tmp_upstream_length = HashMap::new() as HashMap<i64, f64>;

    for (nid, upstream_value) in topologically_sorted_nodes
        .iter()
        .copied()
        .zip(upstream_length.iter_mut())
    {
        let curr_upstream = tmp_upstream_length.remove(&nid).unwrap_or(0.);
        *upstream_value = curr_upstream;

        let outs = g.out_neighbours(nid).collect::<SmallVec<[_; 2]>>();

        if outs.len() == 1 {
            // simple case, only 1 out → send it all down there
            let other = outs[0];
            let outgoing_edge_len = inter_store
                .expand_directed(nid, other)
                .map(|nid| nodeid_pos.get(&nid).unwrap())
                .tuple_windows::<(_, _)>()
                .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                .sum::<f64>();

            *tmp_upstream_length.entry(other).or_default() += curr_upstream + outgoing_edge_len;
            *upstream_per_edge.get_mut(&(nid, other)).unwrap() = curr_upstream;
        } else if outs.is_empty() {
            // nothing to do
        } else if outs.len() > 1 {
            // For all the incoming edges, calculate how much goes in from each group
            let inflow_per_group: SmallVec<[(Option<u32>, f64); 2]> = g
                .in_neighbours(nid)
                .map(|in_nid| (nid_pair_to_tagid.get(&(in_nid, nid)), (in_nid, nid)))
                .map(|(group, (prev_nid, this_nid))| {
                    let edge_len = inter_store
                        .expand_directed(prev_nid, this_nid)
                        .map(|nid| nodeid_pos.get(&nid).unwrap())
                        .tuple_windows::<(_, _)>()
                        .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                        .sum::<f64>();
                    let edge_pre_upstream = *upstream_per_edge.get(&(prev_nid, this_nid)).unwrap();
                    (group.copied(), edge_len + edge_pre_upstream)
                })
                .fold(SmallVec::new(), |mut map, (grp, inflow)| {
                    match map.iter_mut().find(|(g, _)| *g == grp) {
                        None => map.push((grp, inflow)),
                        Some((_group, prev_inflow)) => *prev_inflow += inflow,
                    }
                    map
                });

            #[allow(clippy::type_complexity)]
            let outs: SmallVec<[(Option<u32>, (i64, i64)); 2]> = outs
                .iter()
                .map(|&nid2| (nid_pair_to_tagid.get(&(nid, nid2)).copied(), (nid, nid2)))
                .collect();

            let num_outs_per_group: SmallVec<[(Option<u32>, usize); 2]> = outs
                .iter()
                .map(|(group, _nids)| group)
                .fold(SmallVec::new(), |mut map, grp| {
                    match map.iter().position(|(g, _)| g == grp) {
                        None => map.push((*grp, 1)),
                        Some(i) => map[i].1 += 1,
                    }
                    map
                });

            // now update num_outs_per_group with the total
            // Allocate inflow that goes to the same group
            let mut outflow_per_group: SmallVec<[(Option<u32>, f64); 2]> = SmallVec::new();
            for (group, num_outs) in num_outs_per_group.iter() {
                let inflow = inflow_per_group
                    .iter()
                    .position(|(grp, _inflow)| grp == group)
                    .map_or(0., |i| inflow_per_group[i].1);
                outflow_per_group.push((*group, inflow / (*num_outs as f64)));
            }

            // Any inflow groups without an outflow group are allocated equally to all the outflow
            // groups
            for (in_group, inflow) in inflow_per_group.iter() {
                if !outflow_per_group
                    .iter()
                    .any(|(out_group, _)| out_group == in_group)
                {
                    for (_out_group, outflow) in outflow_per_group.iter_mut() {
                        *outflow += inflow / (outs.len() as f64);
                    }
                }
            }

            if *upstream_value
                != inflow_per_group
                    .iter()
                    .map(|(_group, len)| len)
                    .sum::<f64>()
            {
                // TODO fix this
                debug!("upstream ≠ incomign");
            }

            for (out_group, (_this_nid, other)) in outs.into_iter() {
                let outflow = outflow_per_group
                    .iter()
                    .find(|(g, _)| *g == out_group)
                    .unwrap()
                    .1;
                *tmp_upstream_length.entry(other).or_default() += outflow;
                *upstream_per_edge.get_mut(&(nid, other)).unwrap() = outflow;
            }
        }

        calc_all_upstreams.inc(1);
    }
    calc_all_upstreams.finish_and_clear();
    let upstream_per_edge = upstream_per_edge;
    info!(
        "Have calculated the upstream values for {} different edges",
        upstream_per_edge.len().to_formatted_string(&Locale::en)
    );

    let calc_all_upstream_bar = progress_bars.add(
        ProgressBar::new(end_points.len() as u64)
            .with_message("Calculating upstream value for all End points")
            .with_style(style.clone()),
    );
    calc_all_upstream_bar.set_length(topologically_sorted_nodes.len() as u64);

    // Calculate all the upstream value for all the end points.
    for (nid, upstream_length) in calc_all_upstream_bar.wrap_iter(
        topologically_sorted_nodes
            .iter()
            .zip(upstream_length.iter()),
    ) {
        if let Ok(idx) = end_points.binary_search(nid) {
            end_point_upstreams[idx] = *upstream_length;
            calc_all_upstream_bar.inc(1);
        }
    }
    calc_all_upstream_bar.finish_and_clear();
    let end_point_upstreams = end_point_upstreams.into_boxed_slice();
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
        if !ends_membership_filters.is_empty() {
            info!(
                "Adding the following {} attributes for each end: {}",
                ends_membership_filters.len(),
                ends_membership_filters
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        if !args.ends_tag.is_empty() {
            info!("Adding the following tags to each end: {:?}", args.ends_tag,);
        }
        let reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
        let mut reader = osmio::stringpbf::PBFReader::new(reader);
        reader
            .ways()
            // ↑ all the ways
            .par_bridge()
            //
            // ↓ .. which match at least one end-membership filter, or match the regular tag
            // filter, and have a desired tag
            .map(|w| {
                let has_end_member_tags =
                    tagfilter::obj_pass_filters(&w, &ends_membership_filters, &None);
                let has_end_point_tags =
                    tagfilter::obj_pass_filters(&w, &tag_filter, &args.tag_filter_func)
                        && args.ends_tag.iter().any(|end_tag| w.has_tag(end_tag));
                (w, has_end_member_tags, has_end_point_tags)
            })
            .filter(|(_w, has_end_member_tags, has_end_point_tags)| {
                *has_end_member_tags || *has_end_point_tags
            })
            //
            // ↓ .. which have at least one node in the end points
            .filter(|(w, _has_end_member_tags, _has_end_point_tags)| {
                w.nodes()
                    .iter()
                    .any(|nid| end_points.binary_search(nid).is_ok())
            })
            .for_each_with(
                (end_point_memberships.clone(), end_point_tag_values.clone()),
                |(end_point_memberships, end_point_tag_values),
                 (w, has_end_member_tags, has_end_point_tags)| {
                    let filter_results = ends_membership_filters
                        .iter()
                        .map(|f| f.filter(&w))
                        .collect::<SmallVec<[bool; 2]>>();
                    for end_point_idx in w
                        .nodes()
                        .iter()
                        .filter_map(|nid| end_points.binary_search(nid).ok())
                    {
                        if !ends_membership_filters.is_empty() && has_end_member_tags {
                            let mut curr_mbmrs_all = end_point_memberships.write().unwrap();
                            let curr_mbmrs = curr_mbmrs_all.get_mut(end_point_idx).unwrap();
                            for (new, old) in filter_results.iter().zip(curr_mbmrs.iter_mut()) {
                                *old |= new;
                            }
                        }

                        if !args.ends_tag.is_empty() && has_end_point_tags {
                            let mut curr_tags_all = end_point_tag_values.write().unwrap();
                            let curr_tags = curr_tags_all.get_mut(end_point_idx).unwrap();
                            for (tag_key, this_end_tag_value) in
                                args.ends_tag.iter().zip(curr_tags.iter_mut())
                            {
                                if let Some(way_tag_value) = w.tag(tag_key) {
                                    *this_end_tag_value = match this_end_tag_value {
                                        None => Some(way_tag_value.to_string()),
                                        // If 2 ways with the same tag value come here, don't
                                        // duplicate it.
                                        Some(old_end_tag_value)
                                            if old_end_tag_value == way_tag_value =>
                                        {
                                            Some(way_tag_value.to_string())
                                        }
                                        // There are mulitple ways that go through here, so join
                                        // do semicolon style concatination.
                                        Some(old_end_tag_value) => {
                                            Some(format!("{};{}", old_end_tag_value, way_tag_value))
                                        }
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
    let end_points_w_meta = || {
        end_points
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
            .map(|(((nid, mbms), end_tags), len)| (nid, mbms, end_tags, len))
    };

    if let Some(ref ends_filename) = args.ends {
        let end_points_output = end_points_w_meta()
            .filter(|(_nid, _mbms, _end_tgs, len)| {
                args.min_upstream_m.is_none_or(|min| *len >= &min)
            })
            .map(|(nid, mbms, end_tags, len)| {
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
                    for (tag_key, tag_value) in args
                        .ends_tag
                        .iter()
                        .zip(end_tags.into_iter())
                        .filter(|(_k, v)| v.is_some())
                    {
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

    if let Some(ref mut ends_csv) = ends_csv {
        ends_csv::write_ends(
            ends_csv,
            end_points_w_meta(),
            &args,
            &nodeid_pos,
            latest_timestamp,
            &latest_timestamp_iso,
        )?;
    }

    // For every node in topologically_sorted_nodes, store the index (in end_points) of the biggest
    // end point that this node flows into.
    // We store the index as a i32 to save space. We assume we will have <2³² end points
    // -1 = no known end point (yet).
    // (biggest end point = end point with the largest upstream value)
    // TODO replace this with nonzerou32
    let mut upstream_assigned_end: Vec<i32> = Vec::new();

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
    assert!(upstream_assigned_end.par_iter().all(|end| *end >= 0));
    let upstream_assigned_end = upstream_assigned_end.into_boxed_slice();

    let new_progress_bar_func = |total: u64, message: &str| {
        progress_bars.add(
            ProgressBar::new(total)
                .with_message(message.to_string())
                .with_style(style.clone()),
        )
    };

    let tag_group_data_opt = if args.upstreams.is_some() || args.grouped_waterways.is_some() {
        Some(calc_tag_group(
            &topologically_sorted_nodes,
            &nid_pair_to_tagid,
            &tag_group_value,
            &g,
            &upstream_per_edge,
            new_progress_bar_func,
        ))
    } else {
        None
    };

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
            &upstream_per_edge,
            &nodeid_pos,
            &end_point_tag_values,
            &args.ends_tag,
            &inter_store,
            args.grouped_ends_max_upstream_delta,
            args.grouped_ends_max_distance_m,
        )?;
    }

    if let Some(ref upstream_filename) = args.upstreams {
        let (nid_pair_to_taggroupid, tag_group_info) = tag_group_data_opt.as_ref().unwrap();
        do_write_upstreams(
            &args,
            upstream_filename,
            &progress_bars,
            &style,
            &upstream_assigned_end,
            &topologically_sorted_nodes,
            &upstream_per_edge,
            &inter_store,
            &nodeid_pos,
            &end_points,
            &end_point_upstreams,
            &end_point_tag_values,
            &nid_pair_to_tagid,
            nid_pair_to_taggroupid,
            tag_group_info,
            &tag_group_value,
        )?;
    }
    if let Some(ref waterway_grouped_file) = args.grouped_waterways {
        let (nid_pair_to_taggroupid, tag_group_info) = tag_group_data_opt.as_ref().unwrap();
        do_waterway_grouped(
            waterway_grouped_file,
            &g,
            &progress_bars,
            &style,
            &end_points,
            &topologically_sorted_nodes,
            &end_point_upstreams,
            &upstream_assigned_end,
            &upstream_per_edge,
            &nodeid_pos,
            &end_point_tag_values,
            &args.ends_tag,
            &inter_store,
            nid_pair_to_taggroupid,
            tag_group_info,
            &tag_group_value,
            args.min_length_m,
        )?;
    }

    info!(
        "Finished all in {}",
        formatting::format_duration(global_start.elapsed())
    );

    // collect and output geometry

    info!("slán");
    Ok(())
}

fn do_read_nids_in_ne2_ways(
    mut reader: osmio::pbf::PBFReader<indicatif::ProgressBarIter<File>>,
    tag_filter: &SmallVec<[tagfilter::TagFilter; 3]>,
    tag_filter_func: &Option<tagfilter::TagFilterFunc>,
    input_bar: &ProgressBar,
    progress_bars: &MultiProgress,
) -> Result<SortedSliceSet<i64>> {
    // how many vertexes are there per node id? (which do we need to keep)
    info!("About to preform first read of file, to calculate which nids we need to keep");
    let nid2nways = Arc::new(Mutex::new(HashMap::<i64, u8>::new()));
    reader
        .ways()
        .par_bridge()
        .filter(|w| tagfilter::obj_pass_filters(w, tag_filter, tag_filter_func))
        // TODO support grouping by tag value
        .for_each_with(nid2nways.clone(), |nid2nways, w| {
            assert!(w.id() > 0, "This file has a way id < 0. negative ids are not supported in this tool Use osmium sort & osmium renumber to convert this file and run again.");

            let mut nid2nways = nid2nways.lock().unwrap();
            let nids = w.nodes();
            let mut val = nid2nways.get(&nids[0]).unwrap_or(&0).saturating_add(1);
            nid2nways.insert(nids[0], val);
            val = nid2nways.get(nids.last().unwrap()).unwrap_or(&0).saturating_add(1);
            nid2nways.insert(*nids.last().unwrap(), val);
            for n in &nids[1..nids.len()] {
                val = nid2nways.get(n).unwrap_or(&0).saturating_add(2);
                nid2nways.insert(*n, val);
            }

        });
    input_bar.finish();
    progress_bars.remove(input_bar);
    let nid2nways = Arc::try_unwrap(nid2nways).unwrap().into_inner().unwrap();

    let num_nids = nid2nways.len();
    let nids_in_ne2_ways: Vec<i64> = nid2nways
        .into_iter()
        .filter_map(|(nid, nvertexes)| if nvertexes != 2 { Some(nid) } else { None })
        .collect();
    let nids_in_ne2_ways = SortedSliceSet::from_vec(nids_in_ne2_ways);
    info!(
        "There are {} nodes in total, but only {} ({:.1}%) are pillar nodes",
        num_nids.to_formatted_string(&Locale::en),
        nids_in_ne2_ways.len().to_formatted_string(&Locale::en),
        (nids_in_ne2_ways.len() as f64) * 100. / (num_nids as f64),
    );
    // nids_in_ne2_ways is all node ids which have ¬2 number of vertexes. It's about 5% of all
    // nodes. We contract the graph as we build it, and we know we can always contract any nids not
    // in this list. This allows us to keep the in-memory size of the graph small

    Ok(nids_in_ne2_ways)
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
fn round_mult(f: &f64, base: f64) -> i64 {
    ((f / base).round() * base) as i64
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
    sort_dedup!(result);
    result.shrink_to_fit();
    result
}

#[allow(dead_code)]
fn collect_into_vec_set<T>(it: impl Iterator<Item = T>) -> Vec<T>
where
    T: Ord + Send,
{
    let mut result: Vec<T> = it.collect();
    sort_dedup!(result);
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
    upstream_per_edge: &SortedSliceMap<(i64, i64), f64>,
    nodeid_pos: &impl NodeIdPosition,
    end_point_tag_values: &[SmallVec<[Option<String>; 1]>],
    ends_tags: &[String],
    inter_store: &inter_store::InterStore,
    grouped_ends_max_upstream_delta: Option<f64>,
    grouped_ends_max_distance_m: Option<f64>,
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

    // Lines we are drawing
    // key: i64: the last point in the line
    // value: Vec of:
    //          end_idx group
    //          the nid of the previous node on the graph
    //        the points (node ids) expanded by iterstore which are ending here.
    #[allow(clippy::type_complexity)]
    let mut in_progress_lines: HashMap<i64, SmallVec<[(i32, i64, Vec<i64>); 2]>> = HashMap::new();
    // walks upstream
    let mut nid_end_iter = topologically_sorted_nodes
        .iter()
        .rev()
        .zip(upstream_assigned_end.iter().rev());

    // just a little buffer we might need later
    let mut possible_ins: SmallVec<[i64; 5]> = smallvec::smallvec![];

    // Buffer of pending
    // .0 i32 nid of the final end point
    // .1 Vec<i64> nids that make up the path (stored in reverse order, because it's itermediate
    //             and the paths are build by walking upstream).
    #[allow(clippy::type_complexity)]
    let mut results_to_pop: SmallVec<[(i32, Vec<i64>); 3]> = smallvec::smallvec![];

    // Iterator that yields (end_node_id, from_upstream_m, to_upstream_m, and a path of nids which
    // end in this nid) It walks along all the nodes in rev. topological order, and optionally
    // outputs a path when it has completed one.
    let upstreams_grouped_by_end = std::iter::from_fn(|| {
        // This code definitly does too many allocations (incl. for Vec's) and could be optimised

        // We keep walking along the nid_end_iter, and only stop (& return something) when we have
        // a complete line to return.
        loop {
            if let Some(mut result) = results_to_pop.pop() {
                // we have something to return. An iteration of the nid_end_iter might finish 1+
                // lines, so pop them off.
                result.1.reverse();
                return Some(result);
            }
            let (&nid, &this_end_idx) = match nid_end_iter.next() {
                None => {
                    // finished all the nodes
                    return None;
                }
                Some(x) => x,
            };

            // which in progress lines are there for this node
            let mut lines_to_here = in_progress_lines.remove(&nid).unwrap_or_default();
            if nid == end_points[this_end_idx as usize] {
                // this node is an end node (and it is it's own end node)

                assert!(lines_to_here.is_empty()); // sanity check

                // Give us something to work with later. This is the start of the lines that start
                // here.
                lines_to_here.push((this_end_idx, nid, vec![]));
            }

            // include this point in every line so far
            for (_line_end, _last_nid, line_points) in lines_to_here.iter_mut() {
                line_points.push(nid);
            }

            // All the lines that come to here, but are from another end point, we end them here.
            //
            // SmallVec::drain_filter is documentated, but doesn't exist?
            let mut i = 0;
            while i < lines_to_here.len() {
                if lines_to_here[i].0 != this_end_idx {
                    // this line ends here and is for another end, so remove it.
                    let (other_end_idx, _prev_nid, other_points) = lines_to_here.remove(i);
                    results_to_pop.push((other_end_idx, other_points));
                } else {
                    i += 1;
                }
            }

            if let Some(max_distance_m) = grouped_ends_max_distance_m {
                while let Some(i) = lines_to_here
                    .iter()
                    .position(|(_end_idx, _prev_nid, path)| {
                        if path.len() >= 2 {
                            haversine::haversine_m_fpair(
                                nodeid_pos.get(&path[0]).unwrap(),
                                nodeid_pos.get(path.last().unwrap()).unwrap(),
                            ) > max_distance_m
                        } else {
                            false
                        }
                    })
                {
                    let (other_end_idx, _prev_nid, other_points) = lines_to_here.swap_remove(i);
                    results_to_pop.push((other_end_idx, other_points));

                    if lines_to_here.is_empty() {
                        // we've ended this line, so start a new one
                        lines_to_here.push((other_end_idx, nid, vec![nid]));
                    }
                }
            }

            if let Some(max_upstream_delta) = grouped_ends_max_upstream_delta {
                while let Some(i) = lines_to_here
                    .iter()
                    .position(|(_end_idx, _prev_nid, path)| {
                        if path.len() >= 3 {
                            // NB: path is stored in reverse order
                            upstream_per_edge.get(&(path[1], path[0])).unwrap()
                                - upstream_per_edge
                                    .get(&(path[path.len() - 1], path[path.len() - 2]))
                                    .unwrap()
                                > max_upstream_delta
                        } else {
                            false
                        }
                    })
                {
                    let (other_end_idx, _prev_nid, other_points) = lines_to_here.swap_remove(i);
                    results_to_pop.push((other_end_idx, other_points));

                    if lines_to_here.is_empty() {
                        // we've ended this line, so start a new one
                        lines_to_here.push((other_end_idx, nid, vec![nid]));
                    }
                }
            }

            // if we have >1 lines_to_here, how do we decide which to continue onwards for this,
            // and which to end here? This is the index of the one to continue.
            let mut line_to_continue_idx = 0;

            // we choose the incoming edge with the largest flow
            if lines_to_here.len() > 1 {
                line_to_continue_idx = lines_to_here
                    .iter()
                    .map(|(_end_idx, prev_nid, _path)| prev_nid)
                    .map(|&prev_nid| upstream_per_edge.get(&(nid, prev_nid)).unwrap())
                    .enumerate()
                    .max_by_key(|(_idx, total_upstream)| OrderedFloat(**total_upstream))
                    .unwrap()
                    .0;
            }

            // Now all the lines that end here are in the end group, but we need to have only one
            // This happens when 2 lines (which flow eventually to the same end point) come
            // together at this point (while walking upstream). i.e. here is a bifurcation, with
            // nid having >1 outgoing segments.

            assert!(line_to_continue_idx < lines_to_here.len());
            let mut line_to_here = lines_to_here.swap_remove(line_to_continue_idx);

            for other_line_to_here in lines_to_here.drain(..) {
                assert_eq!(other_line_to_here.0, this_end_idx);
                let (end_idx, _prev_nid, points) = other_line_to_here;
                results_to_pop.push((end_idx, points));
            }

            assert!(lines_to_here.is_empty());

            possible_ins.clear();
            possible_ins.extend(g.in_neighbours(nid));

            if possible_ins.is_empty() {
                // no upstreams, so finish it.
                results_to_pop.push((this_end_idx, line_to_here.2));
                continue;
            } else if possible_ins.len() == 1 {
                line_to_here.1 = nid;
                in_progress_lines
                    .entry(possible_ins[0])
                    .or_default()
                    .push(line_to_here);
            } else {
                let line_to_continue_idx = possible_ins
                    .iter()
                    .map(|&next_nid| upstream_per_edge.get(&(next_nid, nid)).unwrap())
                    .enumerate()
                    .max_by_key(|(_idx, total_upstream)| OrderedFloat(**total_upstream))
                    .unwrap()
                    .0;
                let possible_in = possible_ins.swap_remove(line_to_continue_idx);

                // for others, create new paths that start on this node
                for later_upstream_nodes in possible_ins.drain(..) {
                    in_progress_lines
                        .entry(later_upstream_nodes)
                        .or_default()
                        .push((this_end_idx, nid, vec![nid]));
                }

                // We have >0 upstreams, we've already done something with the others, so the first
                // upstream is part of the only line here.
                line_to_here.1 = nid;
                in_progress_lines
                    .entry(possible_in)
                    .or_default()
                    .push(line_to_here);
            }
        }
    })
    .filter(|(_end_idx, path)| path.len() >= 2) // can happen when splitting due to max delta
    .map(|(end_idx, path)| {
        segments_spinner.inc(1);
        nodes_bar.inc(path.len().saturating_sub(1) as u64);

        let from_upstream_m = *upstream_per_edge.get(&(path[0], path[1])).unwrap();
        let to_upstream_m_init = upstream_per_edge
            .get(&(path[path.len() - 2], path[path.len() - 1]))
            .unwrap();
        let to_upstream_m = to_upstream_m_init
            + inter_store
                .expand_directed(path[path.len() - 2], path[path.len() - 1])
                .map(|nid| nodeid_pos.get(&nid).unwrap())
                .tuple_windows::<(_, _)>()
                .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                .sum::<f64>();

        let points = inter_store
            .expand_line_directed(&path)
            .map(|nid| nodeid_pos.get(&nid).expect("Cannot find position for node"))
            .collect::<Vec<_>>();
        let mut props = serde_json::json!({
            "end_nid": end_points[end_idx as usize],
            "end_upstream_m": round(&end_point_upstreams[end_idx as usize], 1),
            "from_upstream_m": round(&from_upstream_m, 1),
            "to_upstream_m": round(&to_upstream_m, 1),
            "avg_upstream_m": round(&((to_upstream_m+from_upstream_m)/2.), 1),
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

#[allow(clippy::too_many_arguments)]
fn do_write_upstreams(
    args: &cli_args::Args,
    upstream_filename: &Path,
    progress_bars: &MultiProgress,
    style: &ProgressStyle,
    upstream_assigned_end: &[i32],
    topologically_sorted_nodes: &[i64],
    upstream_per_edge: &SortedSliceMap<(i64, i64), f64>,
    inter_store: &inter_store::InterStore,
    nodeid_pos: &impl NodeIdPosition,
    end_points: &[i64],
    end_point_upstreams: &[f64],
    end_point_tag_values: &[SmallVec<[Option<String>; 1]>],
    nid_pair_to_tagid: &SortedSliceMap<(i64, i64), u32>,
    nid_pair_to_taggroupid: &SortedSliceMap<(i64, i64), u64>,
    tag_group_info: &[TagGroupInfo],
    tag_group_value: &[String],
) -> Result<()> {
    assert!(
        !upstream_assigned_end.is_empty(),
        "When doing upstreams, we should have assigned each point to an end. Why was this not done?"
    );
    assert_eq!(
        topologically_sorted_nodes.len(),
        upstream_assigned_end.len()
    );

    let upstream_assigned_end_map: SortedSliceMap<i64, i32> = SortedSliceMap::from_iter(
        topologically_sorted_nodes
            .iter()
            .copied()
            .zip(upstream_assigned_end.iter().copied()),
    );

    let writing_upstreams_bar = progress_bars.add(
        ProgressBar::new(upstream_per_edge.len() as u64)
            .with_message("Writing upstreams file")
            .with_style(style.clone()),
    );

    // we loop over all nodes in topologically_sorted_nodes (which is annotated in
    // upstream_length_iter with the upstream value) and flat_map that into each line segment
    // that goes out of that.
    let lines = upstream_per_edge
        .iter()
        .progress_with(writing_upstreams_bar)
        .map(|((from_nid, to_nid), initial_upstream_len)| {
            let end_idx: usize = *upstream_assigned_end_map.get(to_nid).unwrap() as usize;
            let flow_tag_group = nid_pair_to_tagid.get(&(*from_nid, *to_nid)).copied();
            let tag_group_info = nid_pair_to_taggroupid
                .get(&(*from_nid, *to_nid))
                .map(|idx| &tag_group_info[*idx as usize]);
            (
                from_nid,
                to_nid,
                initial_upstream_len,
                end_idx,
                flow_tag_group,
                tag_group_info,
            )
        })
        .flat_map(
            |(from_nid, to_nid, initial_upstream_len, end_idx, flow_tag_group, tag_group_info)| {
                // Expand all the intermediate nodes between these 2, and increase the current
                // total upsteam, and output all that for the next iteration
                inter_store
                    .expand_directed(*from_nid, *to_nid)
                    .map(|nid| (nid, nodeid_pos.get(&nid).unwrap()))
                    .tuple_windows::<(_, _)>()
                    .scan(
                        *initial_upstream_len,
                        move |curr_upstream_len, ((nid1, p1), (nid2, p2))| {
                            let this_len = haversine::haversine_m_fpair(p1, p2);
                            let from_upstream_len = *curr_upstream_len;
                            *curr_upstream_len += this_len;
                            Some((
                                nid1,
                                nid2,
                                p1,
                                p2,
                                from_upstream_len,
                                *curr_upstream_len,
                                end_idx,
                                flow_tag_group,
                                tag_group_info,
                            ))
                        },
                    )
            },
        )
        .filter(
            |(
                _from_nid,
                _to_nid,
                _p1,
                _p2,
                from_upstream_len,
                to_upstream_len,
                _end_idx,
                _flow_tag_group,
                _tag_group_info,
            )| {
                args.upstreams_min_upstream_m
                    .is_none_or(|min| *from_upstream_len >= min || *to_upstream_len >= min)
            },
        )
        .map(
            |(
                _from_nid,
                _to_nid,
                p1,
                p2,
                from_upstream_len,
                to_upstream_len,
                end_idx,
                flow_tag_group,
                tag_group_info,
            )| {
                // Round the upstream to only output 1 decimal place
                let mut props = serde_json::json!({});
                props["nids"] = format!("{},{}", _from_nid, _to_nid).into();
                props["from_upstream_m"] = round(&from_upstream_len, 1).into();
                props["to_upstream_m"] = round(&to_upstream_len, 1).into();
                props["flow_tag_group"] = flow_tag_group.into();
                props["stream_level"] = tag_group_info
                    .map(|tg| tg.stream_level)
                    .map(|sl| if sl == u64::MAX { None } else { Some(sl) })
                    .into();
                props["stream_level_code_str"] =
                    tag_group_info.map(|tg| tg.stream_level_code_str()).into();
                props["stream_level_code"] = tag_group_info
                    .map(|tg| tg.stream_level_code.as_ref())
                    .into();
                props["tag_group_value"] = flow_tag_group
                    .map(|i| tag_group_value[i as usize].as_str())
                    .into();

                for mult in args.upstreams_from_upstream_multiple.iter() {
                    props[format!("from_upstream_m_{}", mult)] =
                        round_mult(&from_upstream_len, *mult).into();
                }
                props["end_upstream_m"] = round(&end_point_upstreams[end_idx], 1).into();
                props["end_nid"] = end_points[end_idx].into();

                if !args.ends_tag.is_empty() {
                    for (tag_key, tag_value) in args
                        .ends_tag
                        .iter()
                        .zip(end_point_tag_values[end_idx].iter())
                    {
                        if let Some(tag_value) = tag_value {
                            props[format!("end_tag:{}", tag_key)] = tag_value.clone().into();
                        }
                    }
                }

                (props, (p1, p2))
            },
        );
    info_memory_used!();

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
        num_written =
            write_csv_features_directly(lines, &mut f, fileio::OutputGeometryFormat::WKT)?;
    } else {
        anyhow::bail!("Unsupported output format");
    }

    info!(
        "Wrote {} features to output file {}",
        num_written.to_formatted_string(&Locale::en),
        upstream_filename.display(),
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct TagGroupInfo {
    /// Index in the tag value
    /// None → This segment doesn't have a tag value
    tagid: Option<u32>,

    min_nid: i64,
    upstream_m: f64,

    unallocated_other_groups: SmallVec<[u64; 1]>,
    branching_distributaries: SmallVec<[u64; 1]>,
    terminal_distributaries: SmallVec<[u64; 1]>,
    sibling_distributaries: SmallVec<[u64; 1]>,
    tributaries: SmallVec<[u64; 1]>,
    parent_channels: SmallVec<[u64; 1]>,
    side_channels: SmallVec<[u64; 1]>,
    parent_rivers: SmallVec<[u64; 1]>,

    /// nids where this taggroup joins another. This is either a tributary or distributary
    confluences: SmallVec<[i64; 2]>,

    /// nids where a waterway starts
    sources: SmallVec<[i64; 1]>,
    /// nids where a waterway ends
    sinks: SmallVec<[i64; 1]>,

    stream_level: u64,
    stream_level_code: SmallVec<[u32; 3]>,

    end_segments: SmallVec<[(i64, i64); 3]>,
}
impl TagGroupInfo {
    fn from_tagid(tagid: Option<u32>) -> Self {
        let mut res = Self::default();
        res.tagid = tagid;
        res
    }
    fn stream_level_code_str(&self) -> String {
        self.stream_level_code
            .iter()
            .map(|x| x.to_string())
            .join(".")
    }

    fn no_stream_level(&self) -> bool {
        self.stream_level == u64::MAX
    }
    fn has_stream_level(&self) -> bool {
        !self.no_stream_level()
    }
}

impl Default for TagGroupInfo {
    fn default() -> Self {
        TagGroupInfo {
            stream_level: u64::MAX,
            unallocated_other_groups: smallvec![],
            branching_distributaries: smallvec![],
            terminal_distributaries: smallvec![],
            sibling_distributaries: smallvec![],
            tributaries: smallvec![],
            confluences: smallvec![],
            parent_channels: smallvec![],
            side_channels: smallvec![],
            parent_rivers: smallvec![],
            sources: smallvec![],
            sinks: smallvec![],
            upstream_m: 0.,
            stream_level_code: smallvec![],
            tagid: None,
            end_segments: smallvec![],
            min_nid: i64::MAX,
        }
    }
}

fn calc_tag_group(
    topologically_sorted_nodes: &[i64],
    nid_pair_to_tagid: &SortedSliceMap<(i64, i64), u32>,
    tag_group_value: &[String],
    g: &graph::DirectedGraph2,
    upstream_per_edge: &SortedSliceMap<(i64, i64), f64>,
    new_progress_bar_func: impl Fn(u64, &str) -> ProgressBar,
) -> (SortedSliceMap<(i64, i64), u64>, Box<[TagGroupInfo]>) {
    let started_calc = Instant::now();
    // Step 1: What are the segments which are the end segment of their taggroup?
    // list of segments which are the end of a group (i.e. there are 0 outgoing segments with the
    // same tagid
    let mut tag_group_ends: Vec<(i64, i64)> = vec![];
    // special list of segments which have zero out neighbours.
    let mut segments_into_nothing: Vec<(i64, i64)> = vec![];

    // calc name end groups
    let mut outgoing_groups: SmallVec<[_; 3]> = smallvec![];
    let mut this_group;
    let get_ends_bar = new_progress_bar_func(g.num_edges() as u64, "Finding the end segments");
    for seg in get_ends_bar.wrap_iter(g.edges_iter()) {
        outgoing_groups.truncate(0);
        outgoing_groups.extend(g.out_edges(seg.1).map(|seg| nid_pair_to_tagid.get(&seg)));
        outgoing_groups.dedup();
        this_group = nid_pair_to_tagid.get(&seg);

        if outgoing_groups.is_empty() {
            segments_into_nothing.push(seg);
            tag_group_ends.push(seg);
        } else if outgoing_groups.contains(&this_group) {
            // there is an outsegment with the same group, so this isn't an end
            continue;
        } else if !outgoing_groups.is_empty() && outgoing_groups.iter().all(|&g| g != this_group) {
            tag_group_ends.push(seg);
        } else {
            unreachable!()
        }
    }
    let tag_group_ends = SortedSliceSet::from_vec(tag_group_ends);
    assert!(tag_group_ends.len() < u64::MAX as usize);
    let mut tag_group_info: Vec<TagGroupInfo> = Vec::with_capacity(tag_group_ends.len());

    // Step 2: Group all the segments based on topological connectivness (yes this is like
    // osm-lump-ways)
    // for each segment, assign it to a group id
    let curr_group_id = 0;
    let mut nid_pair_to_taggroupid: SortedSliceMap<(i64, i64), u64> =
        SortedSliceMap::from_iter(g.edges_iter().map(|seg| (seg, u64::MAX)));
    let mut frontier: VecDeque<_> = VecDeque::new();
    let assign_to_group = new_progress_bar_func(
        nid_pair_to_taggroupid.len() as u64,
        "Assigning each segment to an end",
    );
    for end_segment in tag_group_ends.iter() {
        if nid_pair_to_taggroupid.get(end_segment).unwrap() != &u64::MAX {
            // already assigned to a group
            continue;
        }
        let this_tag_id = nid_pair_to_tagid.get(end_segment);
        let mut this_tag_group = TagGroupInfo::from_tagid(this_tag_id.cloned());
        this_tag_group.end_segments.push(*end_segment);

        let curr_group_id = tag_group_info.len() as u64;
        frontier.truncate(0);
        frontier.push_back(*end_segment);
        while let Some(seg) = frontier.pop_front() {
            if nid_pair_to_tagid.get(&seg) != this_tag_id {
                continue;
            }
            if nid_pair_to_taggroupid.get(&seg).unwrap() != &u64::MAX {
                continue; // already done
            }
            if tag_group_ends.contains(&seg) {
                this_tag_group.end_segments.push(seg);
                this_tag_group.end_segments.dedup();
            }

            // save this group id
            nid_pair_to_taggroupid.set(&seg, curr_group_id);
            assign_to_group.inc(1);

            // extend
            frontier.extend(g.all_connected_edges(&seg));
            this_tag_group.min_nid = min(this_tag_group.min_nid, min(seg.0, seg.1));
        }
        tag_group_info.push(this_tag_group);
    }
    assign_to_group.finish_and_clear();
    tag_group_info.par_iter_mut().for_each(|tg| {
        // minor clean up
        sort_dedup!(tg.end_segments);
        tg.end_segments.shrink_to_fit();
    });
    // For some reason, some segments don't get assigned a taggroupid
    // For Irl, this doens't happen. For Br+Irl, there's 102 segs
    // Hit it with a big hammer, and just loop over the missing and assign them to a matching.
    let mut incomplete_segs = Vec::new();
    let mut possible_taggroupids: SmallVec<[_; 3]> = SmallVec::new();
    incomplete_segs.truncate(0);
    incomplete_segs.extend(
        nid_pair_to_taggroupid
            .iter()
            .filter(|(_seg, group_id)| *group_id == u64::MAX)
            .map(|(seg, _)| seg)
            .copied(),
    );
    while let Some(seg) = incomplete_segs.pop() {
        assert!(!tag_group_ends.contains(&seg));
        assert!(nid_pair_to_tagid.contains_key(&seg));
        let this_tagid = nid_pair_to_tagid.get(&seg).unwrap();
        possible_taggroupids.truncate(0);
        possible_taggroupids.extend(
            g.all_connected_edges(&seg)
                .filter(|seg2| nid_pair_to_tagid.get(seg2) == Some(this_tagid))
                .map(|seg2| nid_pair_to_taggroupid.get(&seg2).copied()),
        );
        sort_dedup!(possible_taggroupids);
        assert_eq!(possible_taggroupids.len(), 1);
        assert!(possible_taggroupids.iter().all(Option::is_some));
        nid_pair_to_taggroupid.set(&seg, possible_taggroupids[0].unwrap());
    }
    assert_eq!(
        nid_pair_to_taggroupid
            .par_iter()
            .filter(|(_seg, group_id)| *group_id == u64::MAX)
            .count(),
        0,
        "Some segments have not been assigned to a tagroup, num segments {}",
        nid_pair_to_taggroupid
            .len()
            .to_formatted_string(&Locale::en)
    );

    let groups_that_flow_into_nothing = segments_into_nothing
        .into_iter()
        .map(|seg| *nid_pair_to_taggroupid.get(&seg).unwrap())
        .collect::<HashSet<u64>>();

    let mut tag_group_info = tag_group_info.into_boxed_slice();

    info!(
        "There are {} different groups of connected named ways",
        tag_group_info.len().to_formatted_string(&Locale::en)
    );

    // calculate combined upstream per group
    for seg in tag_group_ends.iter() {
        let group = nid_pair_to_taggroupid.get(seg).unwrap();
        // TODO need to include last segment?
        if upstream_per_edge.get(seg).is_none() {
            //warn!("No upstream for {:?}", seg);
        }
        tag_group_info[*group as usize].upstream_m += upstream_per_edge.get(seg).unwrap_or(&0.);
    }

    // For every taggroup, calculate the tributaries, distributaries etc.
    for (seg, group_id) in nid_pair_to_taggroupid.iter() {
        let tg = &mut tag_group_info[*group_id as usize];
        if g.num_in_neighbours(seg.0) == 0 {
            tg.sources.push(seg.0);
        }
        if g.num_out_neighbours(seg.1) == 0 {
            tg.sinks.push(seg.1);
        }

        for (other_seg, other_group_id) in g.out_edges(seg.1).filter_map(|seg| {
            nid_pair_to_taggroupid
                .get(&seg)
                .filter(|g| *g != group_id)
                .map(|g| (seg, g))
        }) {
            tg.confluences.push(seg.1);
            tg.unallocated_other_groups.push(*other_group_id);
        }
        for (other_seg, other_group_id) in g.in_edges(seg.0).filter_map(|seg| {
            nid_pair_to_taggroupid
                .get(&seg)
                .filter(|g| *g != group_id)
                .map(|g| (seg, g))
        }) {
            tg.unallocated_other_groups.push(*other_group_id);
            tg.confluences.push(seg.0);
        }
    }

    tag_group_info.par_iter_mut().for_each(|tg| {
        sort_dedup!(tg.unallocated_other_groups);
    });

    #[derive(PartialEq, Debug)]
    enum FlowType {
        In,
        Out,
        Through,
        No,
    }

    let flow_type = |nid: i64, group_id: u64| -> FlowType {
        let has_ins = g
            .in_edges(nid)
            .any(|seg| nid_pair_to_taggroupid.get(&seg) == Some(&group_id));
        let has_outs = g
            .out_edges(nid)
            .any(|seg| nid_pair_to_taggroupid.get(&seg) == Some(&group_id));
        match (has_ins, has_outs) {
            (true, true) => FlowType::Through,
            (true, false) => FlowType::In,
            (false, true) => FlowType::Out,
            (false, false) => FlowType::No,
        }
    };

    let flows_out = |nid: i64, group_id: u64| -> bool { flow_type(nid, group_id) == FlowType::Out };
    let flows_out_or_through = |nid: i64, group_id: u64| -> bool {
        match flow_type(nid, group_id) {
            FlowType::Out | FlowType::Through => true,
            FlowType::In | FlowType::No => false,
        }
    };
    let flows_in = |nid: i64, group_id: u64| -> bool { flow_type(nid, group_id) == FlowType::In };
    let flows_through =
        |nid: i64, group_id: u64| -> bool { flow_type(nid, group_id) == FlowType::Through };
    let flows_through_or_in = |nid: i64, group_id: u64| -> bool {
        match flow_type(nid, group_id) {
            FlowType::Through | FlowType::In => true,
            _ => false,
        }
    };
    let flows_through_or_out = |nid: i64, group_id: u64| -> bool {
        match flow_type(nid, group_id) {
            FlowType::Through | FlowType::Out => true,
            _ => false,
        }
    };

    tag_group_info
        .par_iter_mut()
        .enumerate()
        .filter(|(_taggroupid, tg)| !tg.unallocated_other_groups.is_empty())
        .for_each(|(taggroupid, tg)| {
            let taggroupid = taggroupid as u64;

            let mut confluences: SmallVec<[_; 3]> = smallvec![]; // buffer
            let mut put_back_in: SmallVec<[_; 2]> = smallvec![];

            for other_taggroupid in tg.unallocated_other_groups.drain(..) {
                assert!(other_taggroupid != taggroupid);
                confluences.truncate(0);
                confluences.extend(
                    tg.confluences
                        .iter()
                        .flat_map(|&nid| g.in_edges(nid))
                        .filter(|seg| nid_pair_to_taggroupid.get(seg) == Some(&other_taggroupid))
                        .map(|seg| seg.1),
                );
                confluences.extend(
                    tg.confluences
                        .iter()
                        .flat_map(|&nid| g.out_edges(nid))
                        .filter(|seg| nid_pair_to_taggroupid.get(seg) == Some(&other_taggroupid))
                        .map(|seg| seg.0),
                );
                sort_dedup!(confluences);
                assert!(!confluences.is_empty());

                if confluences.len() >= 2
                    && confluences.iter().any(|nid| {
                        flows_through_or_in(*nid, taggroupid) && flows_out(*nid, other_taggroupid)
                    })
                    && confluences.iter().any(|nid| {
                        flows_through_or_out(*nid, taggroupid) && flows_in(*nid, other_taggroupid)
                    })
                {
                    tg.side_channels.push(other_taggroupid);
                } else if confluences.len() >= 2
                    && confluences.iter().any(|nid| {
                        flows_out(*nid, taggroupid) && flows_through_or_in(*nid, other_taggroupid)
                    })
                    && confluences.iter().any(|nid| {
                        flows_in(*nid, taggroupid) && flows_through_or_out(*nid, other_taggroupid)
                    })
                {
                    tg.parent_channels.push(other_taggroupid);
                } else if confluences
                    .iter()
                    .all(|nid| flows_in(*nid, other_taggroupid))
                {
                    tg.tributaries.push(other_taggroupid);
                } else if confluences.iter().any(|nid| flows_in(*nid, taggroupid)) {
                    tg.terminal_distributaries.push(other_taggroupid)
                } else if confluences
                    .iter()
                    .all(|nid| flows_out(*nid, taggroupid) && flows_through(*nid, other_taggroupid))
                {
                    tg.parent_rivers.push(other_taggroupid)
                } else if confluences
                    .iter()
                    .all(|nid| flows_through(*nid, taggroupid) && flows_out(*nid, other_taggroupid))
                {
                    tg.branching_distributaries.push(other_taggroupid)
                } else if confluences
                    .iter()
                    .any(|nid| flows_out(*nid, taggroupid) && flows_out(*nid, other_taggroupid))
                {
                    tg.sibling_distributaries.push(other_taggroupid)
                } else {
                    put_back_in.push(other_taggroupid);
                    //unreachable!(
                    //    "Unable to allocate. Main: {} other id: {} flows {:?}",
                    //    tg.tagid
                    //        .map_or("(no name tag)", |tagid| &tag_group_value[tagid as usize]),
                    //    other_taggroupid,
                    //    confluences
                    //        .iter()
                    //        .map(|nid| (
                    //            nid,
                    //            flow_type(*nid, taggroupid),
                    //            flow_type(*nid, other_taggroupid)
                    //        ))
                    //        .collect::<Vec<_>>(),
                    //);
                }
            }

            sort_dedup!(put_back_in);
            tg.unallocated_other_groups.extend(put_back_in);
        });

    tag_group_info.par_iter_mut().for_each(|tg| {
        sort_dedup!(tg.unallocated_other_groups);
        sort_dedup!(tg.branching_distributaries);
        sort_dedup!(tg.terminal_distributaries);
        sort_dedup!(tg.sibling_distributaries);
        sort_dedup!(tg.tributaries);
        sort_dedup!(tg.confluences);
        sort_dedup!(tg.parent_channels);
        sort_dedup!(tg.side_channels);
        sort_dedup!(tg.parent_rivers);
        sort_dedup!(tg.sources);
        sort_dedup!(tg.sinks);
    });

    let mut taggroups_into_nothing = tag_group_info
        .par_iter()
        .enumerate()
        .filter_map(|(tg_id, tg)| {
            if !tg.sinks.is_empty() {
                Some(tg_id as u64)
            } else {
                None
            }
        })
        .collect::<Vec<u64>>();
    taggroups_into_nothing.par_sort_unstable_by_key(|gid| {
        OrderedFloat::from(-tag_group_info[*gid as usize].upstream_m)
    });

    // calculate the stream value (ie level) for every group
    let mut frontier: VecDeque<u64> = VecDeque::new();
    for (idx, tgid) in taggroups_into_nothing.drain(..).enumerate() {
        tag_group_info[tgid as usize].stream_level = 0;
        tag_group_info[tgid as usize]
            .stream_level_code
            .push(idx as u32 + 1);
        frontier.push_back(tgid);
    }

    let mut buf = taggroups_into_nothing;

    let mut existing_code: SmallVec<[_; 5]> = smallvec![];
    let mut existing_level;
    while let Some(tgid) = frontier.pop_front() {
        let tgid = tgid as usize;
        assert!(tag_group_info[tgid].has_stream_level());
        assert!(!tag_group_info[tgid].stream_level_code.is_empty());
        buf.truncate(0);
        buf.extend(
            tag_group_info[tgid]
                .confluences
                .iter()
                .flat_map(|&nid| g.in_edges(nid))
                .map(|seg| nid_pair_to_taggroupid.get(&seg).unwrap())
                .filter(|other_tgid| **other_tgid != tgid as u64)
                .filter(|other_tgid| !tag_group_info[**other_tgid as usize].has_stream_level())
                .dedup()
                .copied(),
        );
        sort_dedup!(buf);
        buf.par_sort_unstable_by_key(|gid| {
            OrderedFloat::from(-tag_group_info[*gid as usize].upstream_m)
        });
        existing_level = tag_group_info[tgid].stream_level;
        existing_code.truncate(0);
        existing_code.extend(tag_group_info[tgid].stream_level_code.iter().copied());
        assert_eq!(
            existing_code.len() as u64,
            existing_level + 1,
            "{:?}",
            tag_group_info[tgid]
        );
        for (idx, other_tgid) in buf.drain(..).enumerate() {
            let other_tg = &mut tag_group_info[other_tgid as usize];
            assert!(other_tg.stream_level_code.is_empty());
            other_tg.stream_level = existing_level + 1;
            other_tg.stream_level_code.reserve(existing_code.len() + 1);
            other_tg
                .stream_level_code
                .extend(existing_code.iter().copied());
            other_tg.stream_level_code.push(idx as u32 + 1);
            frontier.push_back(other_tgid);
        }
    }

    assert!(tag_group_info.par_iter().all(|tg| tg.has_stream_level()));
    assert!(
        tag_group_info
            .par_iter()
            .all(|tg| !tg.stream_level_code.is_empty()),
        "unset stream_level_code's {} of {} have no stream_level_code. first: {:?}",
        tag_group_info
            .par_iter()
            .filter(|tg| tg.stream_level_code.is_empty())
            .count(),
        tag_group_info.len(),
        tag_group_info
            .par_iter()
            .find_first(|tg| tg.stream_level_code.is_empty()),
    );

    assert!(
        tag_group_info
            .par_iter()
            .all(|tg| !tg.stream_level_code.is_empty()),
        "There are {} of {} tag groups with empty stream_level_code, first: {:?}",
        tag_group_info
            .par_iter()
            .filter(|tg| tg.stream_level_code.is_empty())
            .count(),
        tag_group_info.len(),
        tag_group_info
            .par_iter()
            .find_first(|tg| tg.stream_level_code.is_empty()),
    );
    assert!(
        tag_group_info
            .par_iter()
            .all(|tg| tg.stream_level_code.len() as u64 == tg.stream_level + 1),
        "There are {} of {} tag groups where stream_level_code.len ≠ stream_level, first: {:?}",
        tag_group_info
            .par_iter()
            .filter(|tg| tg.stream_level_code.len() as u64 != tg.stream_level + 1)
            .count(),
        tag_group_info.len(),
        tag_group_info
            .par_iter()
            .find_first(|tg| tg.stream_level_code.len() as u64 != tg.stream_level + 1),
    );
    info!("The stream level code string has been calculated for every group");

    info!(
        "Finished calculating all tag groups in {}",
        formatting::format_duration(started_calc.elapsed()),
    );

    (nid_pair_to_taggroupid, tag_group_info)
}

#[allow(clippy::too_many_arguments)]
fn do_waterway_grouped(
    output_filename: &Path,
    g: &graph::DirectedGraph2,
    progress_bars: &MultiProgress,
    style: &ProgressStyle,
    end_points: &[i64],
    topologically_sorted_nodes: &[i64],
    end_point_upstreams: &[f64],
    upstream_assigned_end: &[i32],
    upstream_per_edge: &SortedSliceMap<(i64, i64), f64>,
    nodeid_pos: &impl NodeIdPosition,
    end_point_tag_values: &[SmallVec<[Option<String>; 1]>],
    ends_tags: &[String],
    inter_store: &inter_store::InterStore,
    nid_pair_to_taggroupid: &SortedSliceMap<(i64, i64), u64>,
    tag_group_info: &[TagGroupInfo],
    tag_group_value: &[String],
    min_length_m: Option<f64>,
) -> Result<()> {
    let started_do_waterway_grouped = Instant::now();
    let writing_output_bar = progress_bars.add(
        ProgressBar::new(tag_group_info.len() as u64)
            .with_message("Writing waterway groups")
            .with_style(style.clone()),
    );

    let seg_length = |seg: &(i64, i64)| -> f64 {
        inter_store
            .expand_directed(seg.0, seg.1)
            .map(|nid| nodeid_pos.get(&nid).unwrap())
            .tuple_windows::<(_, _)>()
            .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
            .sum::<f64>()
    };
    let seg_to_distrib_json = |seg: &(i64, i64), nid: i64, incl_len: bool| -> serde_json::Value {
        let pos = nodeid_pos.get(&nid).unwrap();
        let extra = if incl_len { seg_length(seg) } else { 0. };
        json!({
            "nid": nid,
            "lat": round(&pos.1, 7), "lon": round(&pos.0, 7),
            "upstream_m": round(&(upstream_per_edge.get(seg).unwrap()+extra), 1),
        })
    };

    let taggroups_with_geom = tag_group_info
        .iter()
        .progress_with(writing_output_bar)
        .enumerate()
        .map(|(taggroupid, tg)| {
            let taggroupid = taggroupid as u64;
            let mut lines: Vec<Vec<(i64, i64)>> = vec![];
            let mut seen_out_nids = HashSet::<i64>::new();
            let mut seen_in_nids = HashSet::<i64>::new();

            let mut incoming_store: SmallVec<[(i64, i64); 2]> = smallvec![];
            let mut end_segments_to_build_from: SmallVec<[(i64, i64); 5]> = smallvec![];
            end_segments_to_build_from.extend(tg.end_segments.iter().copied());
            assert!(tg
                .end_segments
                .par_iter()
                .all(|seg| *nid_pair_to_taggroupid.get(seg).unwrap() == taggroupid));

            while let Some(seg) = end_segments_to_build_from.pop() {
                let mut line = Vec::new();
                let mut seg = seg;
                loop {
                    if lines
                        .par_iter()
                        .any(|line| line.par_iter().any(|&other_seg| seg == other_seg))
                    {
                        break;
                    }
                    line.push(seg);
                    seen_out_nids.insert(seg.0);
                    seen_in_nids.insert(seg.1);
                    incoming_store.truncate(0);
                    incoming_store.extend(g.in_edges(seg.0).filter(
                        |seg2| {
                            nid_pair_to_taggroupid
                                .get(seg2).is_some_and(|&other_taggroupid| other_taggroupid == taggroupid)
                        },
                    ));
                    if incoming_store.is_empty() {
                        break;
                    }
                    if incoming_store.len() > 1 {
                        end_segments_to_build_from.extend(incoming_store.drain(1..));
                    }
                    seg = incoming_store.pop().unwrap();
                }
                if !line.is_empty() {
                    line.reverse();
                    lines.push(line);
                }
            }

            let lines = lines
                .into_par_iter()
                .map(|line| {
                    let mut new_line = Vec::with_capacity(line.len() + 2);
                    new_line.push(line[0].0);
                    new_line.extend(line.into_iter().map(|seg| seg.1));
                    new_line.shrink_to_fit();
                    new_line
                })
                .collect::<Vec<Vec<i64>>>();

            let mut src_nids = seen_out_nids.difference(&seen_in_nids).copied().collect::<Vec<i64>>();
            let mut sink_nids = seen_in_nids.difference(&seen_out_nids).copied().collect::<Vec<i64>>();
            drop(seen_out_nids);
            drop(seen_in_nids);
            sort_dedup!(src_nids);
            sort_dedup!(sink_nids);
            (taggroupid, tg, lines, (src_nids, sink_nids))
        })

        .filter_map(|(taggroupid, tg, lines, src_sink_nids)| {
            let mut props = serde_json::json!({});
            if tg.stream_level < u64::MAX {
                props["stream_level"] = tg.stream_level.into();
            }
            assert!(!tg.stream_level_code.is_empty(), "{:?}", tg);
            if !tg.stream_level_code.is_empty() {
                props["stream_level_code_str"] = tg.stream_level_code_str().into();
                props["stream_level_code"] = tg.stream_level_code.as_ref().into();
            }

            props["tag_group_value"] = tg.tagid.map(|tagid| tag_group_value[tagid as usize].as_str()).into();
            props["taggroupid"] = taggroupid.into();
            props["min_nid"] = tg.min_nid.into();

            let multilinestrings: Vec<_> = lines
                .par_iter()
                .map(|line| {
                    inter_store
                        .expand_line_directed(line)
                        .map(|nid| nodeid_pos.get(&nid).unwrap_or_else(|_| panic!("TagGroupInfo {:?}", tg)))
                        .collect::<Vec<_>>()
                })
                .collect();

            let cum_length_m = multilinestrings.par_iter().map(|line|
                line.iter()
                .tuple_windows::<(_, _)>()
                .par_bridge()
                .map(|(&p1, &p2)| haversine::haversine_m_fpair(p1, p2))
                .sum::<f64>()
            ).sum::<f64>();
            // Round the upstream to only output 1 decimal place
            props["cum_length_m"] = round(&cum_length_m, 1).into();

            if let Some(min_length_m) = min_length_m {
                if cum_length_m < min_length_m {
                    // this will definitely be to small
                    return None;
                }
            }

            let length_m = calc_through_path_length(&lines, inter_store, nodeid_pos, &src_sink_nids.0, &src_sink_nids.1);

            if let Some(min_length_m) = min_length_m {
                if length_m < min_length_m {
                    return None;
                }
            }
            props["length_m"] = round(&length_m, 1).into();

            drop(lines);

            props["side_channels"] = tg.side_channels.iter().copied().collect::<Vec<_>>().into();

            props["branching_distributaries"] = tg.branching_distributaries
                .iter()
                .map(|dist_tg_idx| (dist_tg_idx, &tag_group_info[*dist_tg_idx as usize]))
                .map(|(dist_tg_idx, dist_tg)| {
                    let confluences = tg.confluences
                        .iter().filter(|nid| dist_tg.confluences.contains(nid))
                        .flat_map(|&nid| g.out_edges(nid))
                        .filter(|seg| nid_pair_to_taggroupid.get(seg).unwrap() == dist_tg_idx)
                        .map(|seg| seg_to_distrib_json(&seg, seg.0, false))
                        .collect::<Vec<_>>();
                    assert!(!confluences.is_empty(), "Can't find confluence with a distributary main: {:?} & dist: {:?}", tg, dist_tg);
                    serde_json::json!({
                        "tag_group_value": dist_tg.tagid.map(|t| tag_group_value[t as usize].clone()),
                        "min_nid": dist_tg.min_nid,
                        "stream_level_code": dist_tg.stream_level_code.as_ref(),
                        "dist_tg_idx": dist_tg_idx,
                        "confluences": confluences,
                        "outflow_m": confluences.iter().map(|c| c["upstream_m"].as_f64().unwrap()).sum::<f64>(),
                    })
            })
            .collect::<Vec<_>>().into();
            props["terminal_distributaries"] = tg.terminal_distributaries
                .iter()
                .map(|dist_tg_idx| (dist_tg_idx, &tag_group_info[*dist_tg_idx as usize]))
                .map(|(dist_tg_idx, dist_tg)| {
                    let confluences = tg.confluences
                        .iter().filter(|nid| dist_tg.confluences.contains(nid))
                        .flat_map(|&nid| g.out_edges(nid))
                        .filter(|seg| nid_pair_to_taggroupid.get(seg).unwrap() == dist_tg_idx)
                        .map(|seg| seg_to_distrib_json(&seg, seg.0, false))
                        .collect::<Vec<_>>();
                    assert!(!confluences.is_empty(), "Can't find confluence with a distributary main: {:?} & dist: {:?}", tg, dist_tg);
                    serde_json::json!({
                        "tag_group_value": dist_tg.tagid.map(|t| tag_group_value[t as usize].clone()),
                        "min_nid": dist_tg.min_nid,
                        "stream_level_code": dist_tg.stream_level_code.as_ref(),
                        "dist_tg_idx": dist_tg_idx,
                        "confluences": confluences,
                        "outflow_m": confluences.iter().map(|c| c["upstream_m"].as_f64().unwrap()).sum::<f64>(),
                    })
            })
            .collect::<Vec<_>>().into();
            props["branching_distributaries"].as_array_mut().unwrap().sort_by_key(|e| OrderedFloat(-e["outflow_m"].as_f64().unwrap()));
            props["terminal_distributaries"].as_array_mut().unwrap().sort_by_key(|e| OrderedFloat(-e["outflow_m"].as_f64().unwrap()));
            props["distributaries_sea"] = tg.sinks.iter()
                .flat_map(|&nid| g.in_edges(nid))
                .filter(|seg| nid_pair_to_taggroupid.get(seg).unwrap() == &taggroupid)
                .map(|seg| seg_to_distrib_json(&seg, seg.1, true))
            .collect::<Vec<_>>().into();
            props["distributaries_sea"].as_array_mut().unwrap().sort_by_key(|e| OrderedFloat(-e["upstream_m"].as_f64().unwrap()));

            props["tributaries"] = tg.tributaries
                .par_iter()
                .map(|trib_tg_idx| (trib_tg_idx, &tag_group_info[*trib_tg_idx as usize]))
                //.filter(|(_trib_tg_idx, trib_tg)| trib_tg.tagid.map_or(false, |t| tag_group_value[t as usize] == "Ballylow Brook"))
                .map(|(trib_tg_idx, trib_tg)| {
                let confluences = tg.confluences
                    .iter().filter(|nid| trib_tg.confluences.contains(nid))
                    .flat_map(|&nid| g.in_edges(nid))
                    .filter(|seg| nid_pair_to_taggroupid.get(seg).unwrap() == trib_tg_idx)
                    .map(|seg| seg_to_distrib_json(&seg, seg.1, true))
                    .collect::<Vec<_>>();
                assert!(!confluences.is_empty(), "Can't find confluence with a tributaries main: {:?} & trib: {:?} taggroupid {} trib_tg_idx {}", tg, trib_tg, taggroupid, trib_tg_idx);
                serde_json::json!({
                    "tag_group_value": trib_tg.tagid.map(|t| tag_group_value[t as usize].clone()),
                    "min_nid": trib_tg.min_nid,
                    "stream_level_code": trib_tg.stream_level_code.as_ref(),
                    "confluences": confluences,
                    "inflow_m": confluences.iter().map(|c| c["upstream_m"].as_f64().unwrap()).sum::<f64>(),
                })
            }).collect::<Vec<_>>().into();
            props["tributaries"].as_array_mut().unwrap().sort_by_key(|e| OrderedFloat(-e["inflow_m"].as_f64().unwrap()));

            props["parent_rivers"] = tg.parent_rivers
                .par_iter()
                .map(|parent_tg_idx| (parent_tg_idx, &tag_group_info[*parent_tg_idx as usize]))
                .map(|(parent_tg_idx, parent_tg)| {
                let confluences = tg.confluences
                    .iter().filter(|nid| parent_tg.confluences.contains(nid))
                    .flat_map(|&nid| g.out_edges(nid))
                    .filter(|seg| nid_pair_to_taggroupid.get(seg).unwrap() == &taggroupid)
                    .map(|seg| seg_to_distrib_json(&seg, seg.0, false))
                    .collect::<Vec<_>>();
                assert!(!confluences.is_empty(), "Can't find confluence with a parent river main: {:?} & parent_river: {:?} taggroupid {} trib_tg_idx {}", tg, parent_tg, taggroupid, parent_tg_idx);
                serde_json::json!({
                    "tag_group_value": parent_tg.tagid.map(|t| tag_group_value[t as usize].clone()),
                    "min_nid": parent_tg.min_nid,
                    "stream_level_code": parent_tg.stream_level_code.as_ref(),
                    "confluences": confluences,
                    //"inflow_m": confluences.iter().map(|c| c["upstream_m"].as_f64().unwrap()).sum::<f64>(),
                })
            }).collect::<Vec<_>>().into();
            //props["parent_rivers"].as_array_mut().unwrap().sort_by_key(|e| OrderedFloat(-e["inflow_m"].as_f64().unwrap()));


            Some((props, multilinestrings))
        });

    let mut f = std::io::BufWriter::new(std::fs::File::create(output_filename)?);

    let num_written;
    if output_filename.extension().unwrap() == "geojsons"
        || output_filename.extension().unwrap() == "geojson"
    {
        num_written = write_geojson_features_directly(
            taggroups_with_geom,
            &mut f,
            &fileio::format_for_filename(output_filename),
        )?;
    } else if output_filename.extension().unwrap() == "csv" {
        num_written = write_csv_features_directly(
            taggroups_with_geom,
            &mut f,
            fileio::OutputGeometryFormat::GeoJSON,
        )?;
    } else {
        anyhow::bail!("Unsupported output format");
    }

    let do_waterway_grouped_duration = started_do_waterway_grouped.elapsed();
    info!(
        "Calculated & wrote {} features to output file {} in {}",
        num_written.to_formatted_string(&Locale::en),
        output_filename.display(),
        formatting::format_duration(do_waterway_grouped_duration),
    );

    Ok(())
}

fn calc_through_path_length(
    lines: &Vec<Vec<i64>>,
    inter_store: &inter_store::InterStore,
    nodeid_pos: &impl NodeIdPosition,
    src_nids: &[i64],
    sink_nids: &[i64],
) -> f64 {
    let mut g = graph::DirectedGraph2::new();
    for line in lines.iter() {
        for seg in line.windows(2) {
            g.add_edge(seg[0], seg[1]);
        }
    }

    let longest_path_len = src_nids
        .par_iter()
        .flat_map(|src_nid| {
            sink_nids
                .par_iter()
                .map(move |sink_nid| (src_nid, sink_nid))
        })
        .map(|(src_nid, sink_nid)| {
            dij::a_star_directed_single(*src_nid, *sink_nid, nodeid_pos, inter_store, &g)
        })
        .filter_map(|opt_dist| opt_dist.map(OrderedFloat))
        .max()
        .unwrap();

    *longest_path_len
}

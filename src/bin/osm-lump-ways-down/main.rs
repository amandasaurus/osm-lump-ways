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

use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::BufWriter;
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
use osm_lump_ways::inter_store;
use osm_lump_ways::nodeid_position;
use osm_lump_ways::sorted_slice_store::SortedSliceMap;
use osm_lump_ways::tagfilter;

use fileio::{write_csv_features_directly, write_geojson_features_directly};
use osm_lump_ways::fileio;

use osm_lump_ways::formatting;

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
            || args.upstreams.is_some()
            || args.grouped_ends.is_some(),
        "Nothing to do. You need to specifiy one of --ends/--loops/--upstreams/etc."
    );

    if (args.grouped_ends.is_some()
        || args.upstreams.is_some()
        || args.ends.is_some()
        || args.ends_csv_file.is_some())
        && !(args.flow_split_equally || args.flow_follows_tag.is_some())
    {
        error!("If you want to output upstreams or ends, you must specificy one of --flow-split-equally or --flow-follows-tag TAG");
        anyhow::bail!("If you want to output upstreams or ends, you must specificy one of --flow-split-equally or --flow-follows-tag TAG");
    }
    if args.ends_csv_file.is_some() && args.ends_tag.is_empty() {
        warn!("The ends CSV file only makes sense with the --ends-tag arguments. Since you have specified no end tags, nothing will be written to the ends CSV file");
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
    // unique id (i32) and store that. We don't need to know what the tagvalue for 2 segments is,
    // we only need to know if they are the same or not.
    let mut nid_pair_to_endtag_group: HashMap<(i64, i64), i32> = HashMap::new();

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
        .for_each_with((g.clone(), inter_store.clone(), tagvalues_to_edges.clone()),
            |(g, inter_store, seen_tagvalues), w| {
                assert!(w.id() > 0, "This file has a way id < 0. negative ids are not supported in this tool Use osmium sort & osmium renumber to convert this file and run again.");
                // add the nodes from w to this graph
                let mut g = g.lock().unwrap();
                let mut inter_store = inter_store.lock().unwrap();
                let mut seen_tagvalues = seen_tagvalues.lock().unwrap();
                nodes_added.inc(w.nodes().len() as u64);
                let mut nodes = w.nodes();

                // If we're assigning based on tag, get the hashset where it'll be stored
                let mut tagvalues_to_edges = args.flow_follows_tag
                    .as_ref()
                    .and_then(|flow_follows_tag| w.tag(flow_follows_tag))
                    .map(|way_tag_value| seen_tagvalues.entry(way_tag_value.to_string()).or_default());

                // Don't add all the nodes, just the ones we need
                while nodes.len() >= 2 {
                    let i_opt = nodes.iter().skip(1).position(|nid| nids_in_ne2_ways.binary_search(nid).is_ok());

                    let mut i = i_opt.unwrap() + 1;
                    // can happen when a river splits and then joins again. try to stop reducing
                    // this little tributary away.
                    while g.contains_edge(nodes[0], nodes[i]) && i > 1 {
                        i -= 1;
                    }
                    // 2 nodes after another with nothing in between? That can happen with someone
                    // double maps a river. But assert a differnet problem, which shows our ability
                    // to contract edges has problems
                    if i > 1 {
                        assert!(!g.contains_edge(nodes[0], nodes[i]), "already existing edge from {} to {} (there are {} nodes in the middle) i={}", nodes[0], nodes[i], nodes.len(), i);
                    }

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

    // Convert the HashMap to something we can look up based on an edge. also throw away the
    // unneeded string value.
    if let Some(ref flow_follows_tag) = args.flow_follows_tag {
        assert!(tagvalues_to_edges.len() < i32::MAX as usize);

        let total_num_pairs = tagvalues_to_edges
            .par_iter()
            .map(|(_tagvalue, pairs)| pairs.len())
            .sum();
        nid_pair_to_endtag_group.reserve(total_num_pairs);

        info!(
            "Have following {} unique '{}' tags in {} node pairs",
            tagvalues_to_edges.len().to_formatted_string(&Locale::en),
            flow_follows_tag,
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
            flow_follows_tag,
            nid_pair_to_endtag_group
                .get_size()
                .to_formatted_string(&Locale::en)
        );
    }
    // convert to memory effecient sorted vec.
    let nid_pair_to_endtag_group = SortedSliceMap::from(nid_pair_to_endtag_group.into_iter());

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
    }

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

        // FIXME rather than split the upstream value equally between all out-edges, look
        // at the name tag (cf nid_pair_to_name_group)
        let num_ins = g.in_neighbours(nid).count();
        let num_outs = g.out_neighbours(nid).count() as f64;
        let in_name_group_opt = g
            .in_neighbours(nid)
            .next()
            .and_then(|in_nid| nid_pair_to_endtag_group.get(&(in_nid, nid)));
        let num_outs_with_same_tag = in_name_group_opt.map_or(0, |in_name_group| {
            g.out_neighbours(nid)
                .filter(|out_nid| {
                    nid_pair_to_endtag_group.get(&(nid, *out_nid)) == Some(in_name_group)
                })
                .count()
        }) as f64;
        if num_ins == 1
            && num_outs > 1.
            && in_name_group_opt.is_some()
            && num_outs_with_same_tag > 0.
        {
            let in_name_group = in_name_group_opt.unwrap();
            let mut to_differnet_downstreams_total = curr_upstream * 0.01;
            if to_differnet_downstreams_total > 1. {
                to_differnet_downstreams_total = 1.;
            }
            let per_same_downstream =
                (curr_upstream - to_differnet_downstreams_total) / num_outs_with_same_tag;
            let per_differnet_downstreams =
                to_differnet_downstreams_total / (num_outs - num_outs_with_same_tag);
            for (other, is_in_same_group) in g.out_neighbours(nid).map(|out_nid| {
                (
                    out_nid,
                    nid_pair_to_endtag_group.get(&(nid, out_nid)) == Some(in_name_group),
                )
            }) {
                let this_edge_len = inter_store
                    .expand_directed(nid, other)
                    .map(|nid| nodeid_pos.get(&nid).unwrap())
                    .tuple_windows::<(_, _)>()
                    .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                    .sum::<f64>();

                let sent_down_this_edge = if is_in_same_group {
                    per_same_downstream + this_edge_len
                } else {
                    per_differnet_downstreams + this_edge_len
                };

                *tmp_upstream_length.entry(other).or_default() += sent_down_this_edge;
                upstream_per_edge.push(((nid, other), sent_down_this_edge));
            }
        } else {
            // > 1 in vertex (too complicated!) or no tag on the only in vertex
            let per_downstream = curr_upstream / num_outs;
            //let curr_pos = nodeid_pos.get(&nid).unwrap();
            for other in g.out_neighbours(nid) {
                //let other_pos = nodeid_pos.get(&other).unwrap();
                let this_edge_len = inter_store
                    .expand_directed(nid, other)
                    .map(|nid| nodeid_pos.get(&nid).unwrap())
                    .tuple_windows::<(_, _)>()
                    .map(|(p1, p2)| haversine::haversine_m_fpair(p1, p2))
                    .sum::<f64>();

                let sent_down_this_edge = per_downstream + this_edge_len;
                *tmp_upstream_length.entry(other).or_default() += sent_down_this_edge;
                upstream_per_edge.push(((nid, other), sent_down_this_edge));
            }
        }

        *upstream_value = curr_upstream;
        calc_all_upstreams.inc(1);
    }
    calc_all_upstreams.finish_and_clear();
    let upstream_per_edge = SortedSliceMap::from_vec(upstream_per_edge);
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
                args.min_upstream_m.map_or(true, |min| *len >= &min)
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
    // NB: This is a top level variable (for scoping reasons), but remains empty if we're not doing
    // anything
    // TODO replace this with nonzerou32
    let mut upstream_assigned_end: Vec<i32> = Vec::new();

    if args.flow_split_equally {
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
    } else if args.flow_follows_tag.is_some() {
        upstream_assigned_end.resize(topologically_sorted_nodes.len(), -1);

        // this is a cache of values as we walk upstream
        // key: i64 = the node id
        // value: (bool, i32), i32 is the endidx for the best guess. bool=true → this assignment is
        // based on the tag value, bool=false → assignment is based on largest end (used when
        // there's no tag)
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
    let upstream_assigned_end = upstream_assigned_end.into_boxed_slice();

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
        )?;
    }

    info!(
        "Finished all in {}",
        formatting::format_duration(global_start.elapsed())
    );

    Ok(())
}

fn do_read_nids_in_ne2_ways(
    mut reader: osmio::pbf::PBFReader<indicatif::ProgressBarIter<File>>,
    tag_filter: &SmallVec<[tagfilter::TagFilter; 3]>,
    tag_filter_func: &Option<tagfilter::TagFilterFunc>,
    input_bar: &ProgressBar,
    progress_bars: &MultiProgress,
) -> Result<Box<[i64]>> {
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
    let mut nids_in_ne2_ways: Vec<i64> = nid2nways
        .into_iter()
        .filter_map(|(nid, nvertexes)| if nvertexes != 2 { Some(nid) } else { None })
        .collect();
    nids_in_ne2_ways.sort_unstable();
    nids_in_ne2_ways.dedup();
    let nids_in_ne2_ways: Box<[i64]> = nids_in_ne2_ways.into_boxed_slice();
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
    let mut in_progress_lines: HashMap<i64, SmallVec<[(i32, i64, Vec<i64>); 2]>> =
        HashMap::new();
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
                    let (other_end_idx, _prev_nid, other_points) =
                        lines_to_here.remove(i);
                    results_to_pop.push((other_end_idx, other_points));
                } else {
                    i += 1;
                }
            }

            if let Some(max_distance_m) = grouped_ends_max_distance_m {
                while let Some(i) =
                    lines_to_here
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
                    let (other_end_idx, _prev_nid, other_points) =
                        lines_to_here.swap_remove(i);
                    results_to_pop.push((other_end_idx, other_points));

                    if lines_to_here.is_empty() {
                        // we've ended this line, so start a new one
                        lines_to_here.push((other_end_idx, nid, vec![nid]));
                    }
                }
            }

            if let Some(max_upstream_delta) = grouped_ends_max_upstream_delta {
                while let Some(i) =
                    lines_to_here
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
                    let (other_end_idx, _prev_nid, other_points) =
                        lines_to_here.swap_remove(i);
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
) -> Result<()> {
    assert!(!upstream_assigned_end.is_empty(), "When doing upstreams, we should have assigned each point to an end. Why was this not done?");
    assert_eq!(
        topologically_sorted_nodes.len(),
        upstream_assigned_end.len()
    );
    let upstream_assigned_end_map: SortedSliceMap<i64, i32> = SortedSliceMap::from(
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
            (from_nid, to_nid, initial_upstream_len, end_idx)
        })
        .flat_map(|(from_nid, to_nid, initial_upstream_len, end_idx)| {
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
                        ))
                    },
                )
        })
        .filter(
            |(_from_nid, _to_nid, _p1, _p2, from_upstream_len, to_upstream_len, _end_idx)| {
                args.upstreams_min_upstream_m.map_or(true, |min| {
                    *from_upstream_len >= min || *to_upstream_len >= min
                })
            },
        )
        .map(
            |(_from_nid, _to_nid, p1, p2, from_upstream_len, to_upstream_len, end_idx)| {
                // Round the upstream to only output 1 decimal place
                let mut props = serde_json::json!({});
                props["from_upstream_m"] = round(&from_upstream_len, 1).into();
                props["to_upstream_m"] = round(&to_upstream_len, 1).into();

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
        num_written = write_csv_features_directly(lines, &mut f)?;
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

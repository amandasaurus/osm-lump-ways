use anyhow::Result;
use clap::Parser;
use get_size::GetSize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressIterator, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::{
    debug, error, info, log, trace, warn,
    Level::{Debug, Trace},
};
use osmio::prelude::*;
use osmio::OSMObjBase;
use rayon::prelude::*;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use std::cmp::Ordering;
use std::sync::atomic::{AtomicI64, Ordering as atomic_Ordering};
use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};

use country_boundaries::{CountryBoundaries, LatLon, BOUNDARIES_ODBL_360X180};
use ordered_float::OrderedFloat;

use smallvec::SmallVec;

#[path = "../cli_args.rs"]
mod cli_args;
#[path = "../haversine.rs"]
mod haversine;
#[path = "../tagfilter.rs"]
mod tagfilter;
use haversine::haversine_m;
#[path = "../dij.rs"]
mod dij;
#[path = "../graph.rs"]
mod graph;
use graph::DirectedGraphTrait;
#[path = "../nodeid_position.rs"]
mod nodeid_position;
use nodeid_position::NodeIdPosition;
#[path = "../btreemapsplitkey.rs"]
mod btreemapsplitkey;
use btreemapsplitkey::BTreeMapSplitKey;
#[path = "../kosaraju.rs"]
mod kosaraju;
#[path = "../taggrouper.rs"]
mod taggrouper;
#[path = "../way_group.rs"]
mod way_group;

#[path = "../fileio.rs"]
mod fileio;
use fileio::{write_geojson_features_directly, OutputFormat};

#[path = "../formatting.rs"]
mod formatting;

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
        "Starting osm-lump-ways-down v{}",
        std::env!("CARGO_PKG_VERSION")
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

    let mut reader = osmio::stringpbf::PBFReader::new(rdr);

    if !args.output_filename.contains("%s") {
        error!("No %s found in output filename ({})", args.output_filename);
        anyhow::bail!("No %s found in output filename ({})", args.output_filename);
    }

    if !args.output_filename.ends_with(".geojson") && !args.output_filename.ends_with(".geojsons") {
        warn!("Output filename '{}' doesn't end with '.geojson' or '.geojsons'. This programme only created GeoJSON or GeoJSONSeq files", args.output_filename);
    }

    if args.split_files_by_group && args.tag_group_k.is_empty() {
        warn!("You have asked to split into separate files by group without saying what to group by! Everything will go into one group. Use -g in future.");
    }

    if !args.split_files_by_group
        && !args.overwrite
        && std::path::Path::new(&args.output_filename).exists()
    {
        error!("Output file {} already exists and --overwrite not used. Refusing to overwrite, and exiting early", args.output_filename);
        anyhow::bail!("Output file {} already exists and --overwrite not used. Refusing to overwrite, and exiting early", args.output_filename);
    }

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

    let output_format = if args.output_filename.ends_with(".geojson") {
        OutputFormat::GeoJSON
    } else if args.output_filename.ends_with(".geojsons") {
        OutputFormat::GeoJSONSeq
    } else {
        warn!("Unknown output format for file {:?}", args.output_filename);
        anyhow::bail!("Unknown output format for file {:?}", args.output_filename);
    };
    debug!("Output format: {output_format:?}");

    anyhow::ensure!(
        args.only_these_way_groups.is_empty(),
        "Not currently supported on this tool"
    );
    anyhow::ensure!(
        args.only_these_way_groups_nodeid.is_empty(),
        "Not currently supported on this tool"
    );
    anyhow::ensure!(
        args.only_these_way_groups_divmod.is_none(),
        "Not currently supported on this tool"
    );

    #[allow(clippy::iter_nth_zero)]
    let only_these_way_groups_divmod: Option<(i64, i64)> =
        args.only_these_way_groups_divmod.map(|s| {
            (
                s.split('/').nth(0).unwrap().parse().unwrap(),
                s.split('/').nth(1).unwrap().parse().unwrap(),
            )
        });
    if let Some((a, b)) = only_these_way_groups_divmod {
        anyhow::ensure!(a > b);
    }

    info!("Starting to read {:?}", &args.input_filename);
    if args.tag_filter.is_empty() {
        if let Some(ref tff) = args.tag_filter_func {
            info!("Tag filter function in operation: {:?}", tff);
        } else {
            info!("No tag filtering in operation. All ways in the file will be used.");
        }
    } else {
        info!("Tag filter(s) in operation: {:?}", args.tag_filter);
    }
    if !args.tag_group_k.is_empty() {
        info!("Tag grouping(s) in operation: {:?}", args.tag_group_k);
    }
    if std::env::var("OSM_LUMP_WAYS_FINISH_AFTER_READ").is_ok() {
        warn!("Programme will exit after reading & parsing input");
    }

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {pos:>7}/{len:7} {per_sec:>12} {msg}",
    )
    .unwrap();
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

    let mut metrics = args.openmetrics.map(|metrics_path| {
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

    let latest_timestamp = AtomicI64::new(0);

    info!("Reading input file...");
    // first step, get all the cycles
    let start_reading_ways = Instant::now();
    reader
        .ways()
        .par_bridge()
        .inspect(|_| obj_reader.inc(1))
        .filter(|w| tagfilter::obj_pass_filters(w, &args.tag_filter, &args.tag_filter_func))
        .inspect(|_| ways_added.inc(1))
        // TODO support grouping by tag value
        .for_each_with(g.clone(), |g, w| {
            // add the nodes from w to this graph
            let mut g = g.lock().unwrap();
            g.add_edge_chain(w.nodes());
            if let Some(t) = w.timestamp().as_ref().map(|t| t.to_epoch_number()) {
                latest_timestamp.fetch_max(t, atomic_Ordering::SeqCst);
            }
        });
    let way_reading_duration = start_reading_ways.elapsed();
    info!(
        "Finished reading. {} ways read in {}, {} ways/sec",
        ways_added.position().to_formatted_string(&Locale::en),
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
    obj_reader.finish();
    progress_bars.remove(&obj_reader);
    ways_added.finish();
    progress_bars.remove(&ways_added);
    input_bar.finish();
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

    // We need to keep track of which vertexes don't have outgoing nodes, so that we can remember
    // to save their position.
    info!("Recording which nodes we need to keep track of later...");
    let mut nids_we_need: BTreeSet<_> = g.vertexes_wo_outgoing_jumbled().collect();
    info!(
        "Remembered {} nodes we need later",
        nids_we_need.len().to_formatted_string(&Locale::en)
    );

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
    splitting_graphs_bar.finish();
    progress_bars.remove(&splitting_graphs_bar);
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

    calc_components_bar.finish();
    graphs_processed.finish();
    progress_bars.remove(&calc_components_bar);
    progress_bars.remove(&graphs_processed);

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
        setting_node_pos.finish();
        progress_bars.remove(&setting_node_pos);
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
                let mut props = serde_json::json!({
                    "root_nid": cycle.iter().flat_map(|seg| seg.iter()).min().unwrap(),
                    "num_nodes": cycle.len(),
                    "length_m": round(&node_group_to_length_m(cycle.as_slice(), &nodeid_pos), 1),
                    "nodes": cycle
                                .iter()
                                .flat_map(|seg| seg.iter())
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
                    args.csv_stats_file.unwrap().display()
                );
            }
        }

        let mut f = std::io::BufWriter::new(std::fs::File::create(
            args.output_filename.replace("%s", "loops"),
        )?);
        let num_written =
            write_geojson_features_directly(cycles_output.into_iter(), &mut f, &output_format)?;

        info!(
            "Wrote {num_written} features to output file {}",
            args.output_filename.replace("%s", "loops")
        );
    }

    info_memory_used!();
    let mut node_id_replaces: HashMap<i64, i64> =
        HashMap::with_capacity(cycles.par_iter().map(|c| c.len() - 1).sum());

    let mut min_nodeid;
    for cycle in cycles {
        min_nodeid = *cycle.iter().flat_map(|seg| seg.iter()).min().unwrap();
        for nid in cycle.iter().flat_map(|seg| seg.iter()) {
            nids_we_need.insert(*nid);
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
    info!("Re-reading file to generate upstreams etc");
    let mut g = graph::DirectedGraph2::new();
    //let mut g = graph::UniDirectedGraph::new();
    read_with_node_replacements(
        &args.input_filename,
        &args.tag_filter,
        &args.tag_filter_func,
        &|nid| *node_id_replaces.get(&nid).unwrap_or(&nid),
        &progress_bars,
        &mut g,
    )?;
    info!("Re-read nodes with replacements.",);

    // TODO do we need to sort topologically? Why not just calc lengths from upstreams
    let sorting_nodes_bar = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Sorting nodes topologically")
            .with_style(style.clone()),
    );
    info!("Sorting all vertexes topologically...");
    //// TODO this graph (g) can be split into disconnected components
    let mut topologically_sorted_nodes = g.into_vertexes_topologically_sorted(&sorting_nodes_bar);
    sorting_nodes_bar.finish();
    progress_bars.remove(&sorting_nodes_bar);
    info!(
        "All {} nodes have been sorted topographically. Size of sorted nodes: {} bytes = {}",
        topologically_sorted_nodes
            .len()
            .to_formatted_string(&Locale::en),
        topologically_sorted_nodes.get_size(),
        topologically_sorted_nodes
            .get_size()
            .to_formatted_string(&Locale::en),
    );

    // eh, this assert fails?! TODO fix
    //assert_eq!(g.num_vertexes(), topologically_sorted_nodes.len());
    info!("Re-reading the file again to build the graph");
    let mut g = graph::UniDirectedGraph::new();
    read_with_node_replacements(
        &args.input_filename,
        &args.tag_filter,
        &args.tag_filter_func,
        &|nid| *node_id_replaces.get(&nid).unwrap_or(&nid),
        &progress_bars,
        &mut g,
    )?;
    info!("Re-read nodes, but unidirecetd.",);

    let setting_node_pos = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Reading file to save node locations")
            .with_style(style.clone()),
    );
    let mut nodeid_pos = nodeid_position::default();
    read_node_positions(
        &args.input_filename,
        |nid| g.contains_vertex(&nid) || nids_we_need.contains(&nid),
        &setting_node_pos,
        &mut nodeid_pos,
    )?;
    setting_node_pos.finish();
    progress_bars.remove(&setting_node_pos);
    info_memory_used!();

    drop(nids_we_need); // don't need you anymore

    info_memory_used!();
    let calc_upstream_bar = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Calculating upstream lengths...")
            .with_style(style.clone()),
    );

    let mut upstream_length: BTreeMapSplitKey<f64> = BTreeMapSplitKey::new();
    let mut upstream_nodes: BTreeMapSplitKey<(i64, SmallVec<[i64; 1]>)> = BTreeMapSplitKey::new();
    info_memory_used!();
    let (
        mut curr_upstream,
        mut curr_upstream_nodes,
        mut num_outs,
        mut per_downstream,
        mut curr_pos,
    );
    let (mut other_pos, mut this_edge_len);
    curr_upstream_nodes = Default::default();

    // Vec<u32> → All upstream strahler numbers.
    // Strahler https://en.wikipedia.org/wiki/Strahler_number needs to look at the parent, but we
    // only have a single directed graph. So push the current strahler numbers down to the child
    // nodes and then calculate the number when we get to that node.
    // NB we use parent/child different order from Wikipedia
    let mut parent_strahlers: BTreeMapSplitKey<SmallVec<[u32; 1]>> = BTreeMapSplitKey::new();

    // final strahler numbers go here.
    let mut strahler: BTreeMapSplitKey<u32> = BTreeMapSplitKey::new();

    // Want to only include strahler numbers if the user has the --strahler option
    // This is a bit hacky, and should be refactored

    for v in topologically_sorted_nodes.drain(..) {
        calc_upstream_bar.inc(1);

        let mut this_strahler_value = 0;
        if args.strahler {
            assert!(!strahler.contains_key(&v));
            if !parent_strahlers.contains_key(&v) {
                this_strahler_value = 1;
            } else {
                let this_parent_strahlers = parent_strahlers.get(&v).unwrap();
                let max_parent_strahler = this_parent_strahlers.iter().max().unwrap();
                if this_parent_strahlers
                    .iter()
                    .filter(|x| *x == max_parent_strahler)
                    .count()
                    == 1
                    && this_parent_strahlers
                        .iter()
                        .filter(|x| *x < max_parent_strahler)
                        .count()
                        == this_parent_strahlers.len() - 1
                {
                    this_strahler_value = *max_parent_strahler;
                } else if this_parent_strahlers
                    .iter()
                    .filter(|x| *x == max_parent_strahler)
                    .count()
                    >= 2
                {
                    this_strahler_value = *max_parent_strahler + 1;
                } else {
                    dbg!(
                        this_strahler_value,
                        max_parent_strahler,
                        v,
                        this_parent_strahlers
                    );
                    panic!();
                }
            }
            *strahler.entry(v).or_default() = this_strahler_value;
            // try to recoop some memory
            parent_strahlers.remove(&v);
        }

        curr_upstream = upstream_length.entry(v).or_default();
        if args.upstreams {
            curr_upstream_nodes = upstream_nodes.entry(v).or_default().0;
        }
        num_outs = g.out_neighbours(v).count() as f64;
        per_downstream = *curr_upstream / num_outs;
        curr_pos = nodeid_pos.get(&v).unwrap();
        for other in g.out_neighbours(v) {
            other_pos = nodeid_pos.get(&other)?;
            this_edge_len =
                haversine::haversine_m(curr_pos.0, curr_pos.1, other_pos.0, other_pos.1);
            *upstream_length.entry(other).or_default() += per_downstream + this_edge_len;
            if args.upstreams {
                upstream_nodes.entry(other).or_default().0 += 1 + curr_upstream_nodes;
            }
            if args.strahler {
                parent_strahlers
                    .entry(other)
                    .or_default()
                    .push(this_strahler_value);
            }
        }
    }
    drop(parent_strahlers);

    calc_upstream_bar.finish();
    progress_bars.remove(&calc_upstream_bar);
    info!(
        "Calculated the upstream value for {} nodes",
        upstream_length.len().to_formatted_string(&Locale::en)
    );
    info_memory_used!();

    let mut ends_membership = args.ends_membership.clone();
    ends_membership.sort_by_key(|tf| tf.to_string());
    let end_points: BTreeMap<i64, smallvec::SmallVec<[bool; 2]>> = g
        .vertexes_wo_outgoing_jumbled()
        .map(|nid| (nid, smallvec::smallvec![false; ends_membership.len()]))
        .collect();
    let end_points = Arc::new(std::sync::RwLock::new(end_points));

    if !ends_membership.is_empty() {
        info!("Rereading file to add memberships for ends");
        info!(
            "Adding the following {} attributes for each end: {}",
            ends_membership.len(),
            ends_membership
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
        let mut reader = osmio::stringpbf::PBFReader::new(reader);
        reader
            .ways()
            .par_bridge()
            .filter(|w| {
                w.nodes()
                    .par_iter()
                    .any(|n| end_points.read().unwrap().contains_key(n))
            })
            .for_each_with(end_points.clone(), |end_points, w| {
                let mut end_points = end_points.write().unwrap();
                for nid in w.nodes() {
                    if let Some(memb_vec) = end_points.get_mut(nid) {
                        for (func, res) in ends_membership.iter().zip(memb_vec.iter_mut()) {
                            *res = func.filter(&w);
                        }
                    }
                }
            });
        // How many have ≥1 true value (versus all default of false)
        let num_nodes_attributed = end_points
            .read()
            .unwrap()
            .par_iter()
            .filter(|(_nid, membs)| membs.par_iter().any(|m| *m))
            .count();
        if num_nodes_attributed == 0 {
            warn!("No end nodes got an end attribute.")
        } else {
            let ends_points = end_points.read().unwrap();
            info!(
                "{} of {} ({:.1}%) end points got an attribute for way membership",
                num_nodes_attributed.to_formatted_string(&Locale::en),
                ends_points.len().to_formatted_string(&Locale::en),
                ((100. * num_nodes_attributed as f64) / ends_points.len() as f64)
            );
        }
    }
    let end_points = Arc::try_unwrap(end_points).unwrap().into_inner().unwrap();

    // look for where it ends
    let end_points: BTreeMap<_, _> = end_points
        .into_par_iter()
        .map(|(nid, mbms)| (nid, mbms, upstream_length.get(&nid).unwrap()))
        .filter(|(_nid, _mbms, len)| args.min_upstream_m.map_or(true, |min| *len >= &min))
        .map(|(nid, mbms, len)| (nid, (mbms, len, nodeid_pos.get(&nid).unwrap())))
        .collect();

    let end_points_output = end_points.iter().map(|(nid, (mbms, len, pos))| {
        // Round the upstream to only output 1 decimal place
        let mut props = serde_json::json!({"upstream_m": round(len, 1), "nid": nid});
        if !ends_membership.is_empty() {
            for (end_attr_filter, res) in ends_membership.iter().zip(mbms.iter()) {
                props[format!("is_in:{}", end_attr_filter)] = (*res).into();
            }
            props["is_in_count"] = mbms.iter().filter(|m| **m).count().into();
        }
        (props, pos)
    });

    let mut f = std::io::BufWriter::new(std::fs::File::create(
        args.output_filename.replace("%s", "ends"),
    )?);
    let num_written = write_geojson_features_directly(end_points_output, &mut f, &output_format)?;
    info!(
        "Wrote {} features to output file {}",
        num_written.to_formatted_string(&Locale::en),
        args.output_filename.replace("%s", "ends")
    );

    let reversed_g = g.into_reversed();
    if args.upstream_tag_ends {
        // TODO use topologically_sorted_nodes instead of looping over all the ends
        info!("For all points, recording which end it flows into");
        let started_upstream_tag_ends = Instant::now();
        let calc_ends_points = progress_bars.add(
            ProgressBar::new(end_points.len() as u64)
                .with_message("Calculating the end point(s) for all points (ends processed)")
                .with_style(style.clone()),
        );

        let mut seen_vertexes = HashSet::new();
        let mut frontier = Vec::new();
        let mut new;
        for end_nid in end_points.iter().map(|(nid, (_mbms, _len, _pos))| nid) {
            seen_vertexes.clear();
            frontier.push(*end_nid);
            while let Some(nid) = frontier.pop() {
                //dbg!(frontier.len()+1)
                new = upstream_nodes.entry(nid).or_default();
                new.1.push(*end_nid);
                new.1.sort();
                seen_vertexes.insert(nid);
                frontier.extend(
                    reversed_g
                        .out_neighbours(nid)
                        .filter(|n2| !seen_vertexes.contains(n2)),
                );
            }
            calc_ends_points.inc(1);
        }
        info!(
            "Calculated, for {} end points, all the upstreams in {} ",
            end_points.len().to_formatted_string(&Locale::en),
            formatting::format_duration(started_upstream_tag_ends.elapsed()),
        );
    }

    if args.upstreams {
        debug!("Writing upstream geojson object(s)");
        let lines = reversed_g
            .edges_iter()
            .map(|(a, b)| (b, a)) // undo the graph reversal here
            .filter(|(from_nid, _to_nid)| {
                args.min_upstream_m
                    .map_or(true, |min| *upstream_length.get(from_nid).unwrap() >= min)
            })
            .map(|(from_nid, to_nid)| {
                let upstream_len = upstream_length.get(&from_nid).unwrap();
                // Round the upstream to only output 1 decimal place
                let mut props = serde_json::json!({
                    "from_upstream_m": round(upstream_len, 1),
                    //"to_upstream_m": round(length_upstream[&to_nid], 1),
                });

                for mult in args.upstream_from_upstream_multiple.iter() {
                    props[format!("from_upstream_m_{}", mult)] =
                        round_mult(upstream_len, *mult).into();
                }

                if args.upstream_tag_ends {
                    let (_upstream_node_count, ends) = upstream_nodes.get(&from_nid).unwrap();
                    props["num_ends"] = ends.len().into();
                    props["ends"] = ends.iter().copied().collect::<Vec<i64>>().into();
                    let mut ends_strs = vec![",".to_string()];
                    let mut this_len;
                    let mut biggest_end = (upstream_length.get(&ends[0]).unwrap(), ends[0]);
                    for end in ends {
                        ends_strs.push(end.to_string());
                        ends_strs.push(",".to_string());
                        this_len = upstream_length.get(&ends[0]).unwrap();
                        if this_len > biggest_end.0 {
                            biggest_end = (this_len, *end);
                        }
                    }
                    props["ends_s"] = ends_strs.join("").into();
                    props["biggest_end_upstream_m"] = round(biggest_end.0, 1).into();
                    props["biggest_end_nid"] = biggest_end.1.into();
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
            ProgressBar::new(reversed_g.num_edges() as u64)
                .with_message("Writing upstreams geojson(s) file")
                .with_style(style.clone()),
        );

        let lines = lines.progress_with(writing_upstreams_bar);

        let mut f = std::io::BufWriter::new(std::fs::File::create(
            args.output_filename.replace("%s", "upstreams"),
        )?);
        let num_written = write_geojson_features_directly(lines, &mut f, &output_format)?;

        info!(
            "Wrote {} features to output file {}",
            num_written.to_formatted_string(&Locale::en),
            args.output_filename.replace("%s", "upstreams")
        );
    }

    if args.upstream_points {
        debug!("Writing upstream geojson points");
        let upstream_points = reversed_g
            .edges_iter()
            .map(|(a, b)| (b, a)) // undo the graph reversal here
            .filter(|(from_nid, _to_nid)| {
                args.min_upstream_m
                    .map_or(true, |min| *upstream_length.get(from_nid).unwrap() >= min)
            })
            .map(|(from_nid, _to_nid)| {
                let upstream_len = upstream_length.get(&from_nid).unwrap();
                //let (_upstream_node_count, _ends) = upstream_nodes.get(&from_nid).unwrap();
                (
                    // Round the upstream to only output 1 decimal place
                    serde_json::json!({
                        "from_upstream_m": round(upstream_len, 1),
                    }),
                    nodeid_pos.get(&from_nid).unwrap(),
                )
            });
        info_memory_used!();

        let writing_upstreams_bar = progress_bars.add(
            ProgressBar::new(reversed_g.num_edges() as u64)
                .with_message("Writing upstream points geojson(s) file")
                .with_style(style.clone()),
        );

        let upstream_points = upstream_points.progress_with(writing_upstreams_bar);

        let mut f = std::io::BufWriter::new(std::fs::File::create(
            args.output_filename.replace("%s", "upstream-points"),
        )?);
        let num_written = write_geojson_features_directly(upstream_points, &mut f, &output_format)?;

        info!(
            "Wrote {} features to output file {}",
            num_written.to_formatted_string(&Locale::en),
            args.output_filename.replace("%s", "upstream-points")
        );
    }

    if args.strahler {
        debug!("Writing strahler number geojson object(s)");
        let strahler_lines = reversed_g
            .edges_iter()
            .map(|(a, b)| (b, a)) // undo graph reversal
            .map(|(from_nid, to_nid)| {
                (
                    // Round the upstream to only output 1 decimal place
                    serde_json::json!({
                        "strahler": strahler.get(&from_nid)
                    }),
                    (
                        nodeid_pos.get(&from_nid).unwrap(),
                        nodeid_pos.get(&to_nid).unwrap(),
                    ),
                )
            });
        info_memory_used!();

        let writing_upstreams_bar = progress_bars.add(
            ProgressBar::new(reversed_g.num_edges() as u64)
                .with_message("Writing strahler geojson(s) file")
                .with_style(style.clone()),
        );

        let strahler_lines = strahler_lines.progress_with(writing_upstreams_bar);

        let mut f = std::io::BufWriter::new(std::fs::File::create(
            args.output_filename.replace("%s", "strahler"),
        )?);
        let num_strahler_written =
            write_geojson_features_directly(strahler_lines, &mut f, &output_format)?;
        info!(
            "Wrote {} features to output file {}",
            num_strahler_written.to_formatted_string(&Locale::en),
            args.output_filename.replace("%s", "strahler")
        );
    }
    if args.ends_upstreams {
        let started_ends_upstreams = Instant::now();
        let calc_ends_upstreams = progress_bars.add(
            ProgressBar::new(
                end_points
                    .par_iter()
                    .map(|(_nid, (_mbms, len, _pos))| len.round() as u64)
                    .sum(),
            )
            .with_message("Calculating complete upstream network per end point")
            .with_style(style.clone()),
        );
        info!(
            "Calculating the complete upstream network for {} end points (but only 100 points upstream)",
            end_points.len()
        );
        //let g = g.into_reversed();
        let end_point_upstreams = end_points
            .iter()
            .filter(|(_nid, (_mbms, len, _pos))| {
                if args
                    .ends_upstreams_min_upstream_m
                    .map_or(false, |m| **len < m)
                {
                    // since we're not doing this end point, update the progress bar
                    calc_ends_upstreams.inc(len.round() as u64);
                }
                args.ends_upstreams_min_upstream_m
                    .map_or(true, |min_len| **len >= min_len)
            })
            .map(|(nid, (_mbms, len, _pos))| {
                let all_upstream_paths =
                    reversed_g.all_out_edges_recursive(*nid, args.ends_upstreams_max_nodes);
                let all_upstream_paths = all_upstream_paths
                    .map(|path| {
                        path.par_iter()
                            .map(|nid| nodeid_pos.get(nid).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                let props = serde_json::json!({"upstream_m": round(len, 1), "nid": nid});

                calc_ends_upstreams.inc(len.round() as u64);
                (props, all_upstream_paths)
            });

        let mut f = std::io::BufWriter::new(std::fs::File::create(
            args.output_filename.replace("%s", "ends-full-upstreams"),
        )?);
        let num_written =
            write_geojson_features_directly(end_point_upstreams, &mut f, &output_format)?;
        info!(
            "Calculated & wrote {} features to output file {} in {}",
            num_written.to_formatted_string(&Locale::en),
            args.output_filename.replace("%s", "ends-full-upstreams"),
            formatting::format_duration(started_ends_upstreams.elapsed()),
        );
        calc_ends_upstreams.finish();
        progress_bars.remove(&calc_ends_upstreams);
    }

    info!(
        "Finished all in {}",
        formatting::format_duration(global_start.elapsed())
    );

    Ok(())
}

fn read_with_node_replacements(
    input_filename: &Path,
    tag_filter: &[tagfilter::TagFilter],
    tag_filter_func: &Option<tagfilter::TagFilterFunc>,
    node_id_replaces: &(impl Fn(i64) -> i64 + Sync),
    progress_bars: &MultiProgress,
    graph: &mut impl graph::DirectedGraphTrait,
) -> Result<()> {
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

    let reader = read_progress::BufReaderWithSize::from_path(input_filename)?;
    let mut reader = osmio::stringpbf::PBFReader::new(reader);

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
    obj_reader.finish();
    progress_bars.remove(&obj_reader);
    ways_added.finish();
    progress_bars.remove(&ways_added);

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
            setting_node_pos.inc(1);
            nodeid_pos.lock().unwrap().insert_i32(nid, pos);
        });

    let nodeid_pos = Arc::try_unwrap(nodeid_pos).unwrap().into_inner().unwrap();
    nodeid_pos.finished_inserting();

    setting_node_pos.finish();

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

fn multilinestring_length(coords: &Vec<Vec<(f64, f64)>>) -> f64 {
    coords
        .par_iter()
        .map(|coord_string| {
            coord_string
                .par_windows(2)
                .map(|pair| haversine_m(pair[0].1, pair[0].0, pair[1].1, pair[1].0))
                .sum::<f64>()
        })
        .sum()
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

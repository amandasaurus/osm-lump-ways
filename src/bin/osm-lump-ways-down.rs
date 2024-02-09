#![allow(dead_code)]
use anyhow::Result;
use clap::Parser;
use get_size::GetSize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressIterator, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::{debug, error, info, trace, warn};
use osmio::prelude::*;
use osmio::OSMObjBase;
use rayon::prelude::*;

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use std::sync::atomic::{AtomicI64, Ordering as atomic_Ordering};
use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};

use country_boundaries::{CountryBoundaries, LatLon, BOUNDARIES_ODBL_360X180};

use smallvec::SmallVec;

#[path = "../cli_args.rs"]
mod cli_args;
#[path = "../haversine.rs"]
mod haversine;
#[path = "../tagfilter.rs"]
mod tagfilter;
use haversine::haversine_m;
//#[path = "../dij.rs"]
//mod dij;
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

    let reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
    let mut reader = osmio::stringpbf::PBFReader::new(reader);

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
    info!("Tag filter(s) in operation: {:?}", args.tag_filter);
    if !args.tag_group_k.is_empty() {
        info!("Tag grouping(s) in operation: {:?}", args.tag_group_k);
    }
    if !args.only_these_way_groups.is_empty() {
        info!(
            "Only keeping groups which include the following ways: {:?}",
            args.only_these_way_groups
        );
    }
    if !args.only_these_way_groups_nodeid.is_empty() {
        info!(
            "Only keeping groups which include the following nodes: {:?}",
            args.only_these_way_groups_nodeid
        );
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

    let boundaries = if metrics.is_some() || csv_stats.is_some() {
        Some(CountryBoundaries::from_reader(BOUNDARIES_ODBL_360X180).unwrap())
    } else {
        None
    };

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
        .filter(|w| args.tag_filter.par_iter().all(|tf| tf.filter(w)))
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
    let way_reading_duration = Instant::now() - start_reading_ways;
    info!(
        "Finished reading. {} ways read in {}, {} ways/sec",
        ways_added.position().to_formatted_string(&Locale::en),
        format_duration(way_reading_duration),
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
    let g: graph::DirectedGraph2 = Arc::try_unwrap(g).unwrap().into_inner().unwrap();

    info!(
        "All data has been loaded in {}. Started processing...",
        format_duration(Instant::now() - global_start)
    );
    info!("{}", g.detailed_size());
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
                (
                    serde_json::json!({
                        "root_nid": cycle.iter().flat_map(|seg| seg.iter()).min().unwrap(),
                        "num_nodes": cycle.len(),
                        "length_m": round(&node_group_to_length_m(cycle.as_slice(), &nodeid_pos), 1),
                        "nodes": cycle
                                    .iter()
                                    .flat_map(|seg| seg.iter())
                                    .map(|nid| format!("n{}", nid))
                                    .collect::<Vec<_>>()
                                    .join(","),
                    }),
                    node_group_to_lines(cycle.as_slice(), &nodeid_pos),
                )
            })
            .collect();
        info_memory_used!();

        if csv_stats.is_some() || metrics.is_some() {
            let mut per_boundary: BTreeMap<&str, (u64, f64)> = BTreeMap::new();
            let boundaries = boundaries.as_ref().unwrap();
            for cycle in cycles_output.iter() {
                let these_boundaries =
                    boundaries.ids(LatLon::new(cycle.1[0][0].1, cycle.1[0][0].0)?);
                let this_len = multilinestring_length(&cycle.1);
                per_boundary.entry("planet").or_default().0 += 1;
                per_boundary.entry("planet").or_default().1 += this_len;
                if these_boundaries.is_empty() {
                    per_boundary.entry("terranullis").or_default().0 += 1;
                    per_boundary.entry("terranullis").or_default().1 += this_len;
                }
                for boundary in these_boundaries {
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
        let num_written = write_geojson_features_directly(
            cycles_output.into_iter(),
            &mut f,
            &output_format,
            &OutputGeometryType::MultiLineString,
        )?;

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
        &node_id_replaces,
        &progress_bars,
        &mut g,
    )?;
    info!(
        "Re-read nodes with replacements. Have size: {}",
        g.detailed_size()
    );

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
        &node_id_replaces,
        &progress_bars,
        &mut g,
    )?;
    info!(
        "Re-read nodes, but unidirecetd. Have size: {}",
        g.detailed_size()
    );

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
    info!("{}", nodeid_pos.detailed_size());

    drop(nids_we_need); // don't need you anymore

    info_memory_used!();
    let calc_upstream_bar = progress_bars.add(
        ProgressBar::new(g.num_vertexes() as u64)
            .with_message("Calculating upstream lengths...")
            .with_style(style.clone()),
    );

    let mut length_upstream: BTreeMapSplitKey<f64> = BTreeMapSplitKey::new();
    info_memory_used!();
    let (mut curr_upstream, mut num_outs, mut per_downstream, mut curr_pos);
    let (mut other_pos, mut this_edge_len);

    // Vec<u32> → All upstream strahler numbers.
    // Strahler https://en.wikipedia.org/wiki/Strahler_number needs to look at the parent, but we
    // only have a single directed graph. So push the current strahler numbers down to the child
    // nodes and then calculate the number when we get to that node.
    // NB we use parent/child different order from Wikipedia
    let mut parent_strahlers: BTreeMapSplitKey<SmallVec<[u32; 1]>> = BTreeMapSplitKey::new();

    // final strahler numbers go here.
    let mut strahler: BTreeMapSplitKey<u32> = BTreeMapSplitKey::new();

    for v in topologically_sorted_nodes.drain(..) {
        calc_upstream_bar.inc(1);

        let mut this_strahler_value = 0;
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

        curr_upstream = length_upstream.entry(v).or_insert(0.);
        num_outs = g.out_neighbours(v).count() as f64;
        per_downstream = *curr_upstream / num_outs;
        curr_pos = nodeid_pos.get(&v).unwrap();
        for other in g.out_neighbours(v) {
            other_pos = nodeid_pos.get(&other)?;
            this_edge_len =
                haversine::haversine_m(curr_pos.0, curr_pos.1, other_pos.0, other_pos.1);
            *length_upstream.entry(other).or_insert(0.) += per_downstream + this_edge_len;
            parent_strahlers
                .entry(other)
                .or_default()
                .push(this_strahler_value);
        }
    }
    drop(parent_strahlers);
    calc_upstream_bar.finish();
    progress_bars.remove(&calc_upstream_bar);
    info!(
        "Calculated the upstream value for {} nodes",
        length_upstream.len().to_formatted_string(&Locale::en)
    );
    info_memory_used!();

    debug!("Writing upstream geojson object(s)");
    let lines = g
        .edges_iter()
        .filter(|(from_nid, _to_nid)| {
            args.min_upstream_m
                .map_or(true, |min| length_upstream.get(from_nid).unwrap() >= &min)
        })
        .map(|(from_nid, to_nid)| {
            (
                // Round the upstream to only output 1 decimal place
                serde_json::json!({
                    "from_upstream_m": round(length_upstream.get(&from_nid).unwrap(), 1),
                    //"to_upstream_m": round(length_upstream[&to_nid], 1),
                }),
                vec![vec![
                    nodeid_pos.get(&from_nid).unwrap(),
                    nodeid_pos.get(&to_nid).unwrap(),
                ]],
            )
        });
    info_memory_used!();

    let writing_upstreams_bar = progress_bars.add(
        ProgressBar::new(g.num_edges() as u64)
            .with_message("Writing upstreams geojson(s) file")
            .with_style(style.clone()),
    );

    let lines = lines.progress_with(writing_upstreams_bar);

    let mut f = std::io::BufWriter::new(std::fs::File::create(
        args.output_filename.replace("%s", "upstreams"),
    )?);
    let num_written = write_geojson_features_directly(
        lines,
        &mut f,
        &output_format,
        &OutputGeometryType::LineString,
    )?;

    info!(
        "Wrote {} features to output file {}",
        num_written.to_formatted_string(&Locale::en),
        args.output_filename.replace("%s", "upstreams")
    );

    debug!("Writing upstream geojson points");
    let upstream_points = g
        .edges_iter()
        .filter(|(from_nid, _to_nid)| {
            args.min_upstream_m
                .map_or(true, |min| length_upstream.get(from_nid).unwrap() >= &min)
        })
        .map(|(from_nid, _to_nid)| {
            (
                // Round the upstream to only output 1 decimal place
                serde_json::json!({
                    "from_upstream_m": round(length_upstream.get(&from_nid).unwrap(), 1),
                }),
                vec![vec![nodeid_pos.get(&from_nid).unwrap()]],
            )
        });
    info_memory_used!();

    let writing_upstreams_bar = progress_bars.add(
        ProgressBar::new(g.num_edges() as u64)
            .with_message("Writing upstream points geojson(s) file")
            .with_style(style.clone()),
    );

    let upstream_points = upstream_points.progress_with(writing_upstreams_bar);

    let mut f = std::io::BufWriter::new(std::fs::File::create(
        args.output_filename.replace("%s", "upstream-points"),
    )?);
    let num_written = write_geojson_features_directly(
        upstream_points,
        &mut f,
        &output_format,
        &OutputGeometryType::Point,
    )?;

    info!(
        "Wrote {} features to output file {}",
        num_written.to_formatted_string(&Locale::en),
        args.output_filename.replace("%s", "upstream-points")
    );

    debug!("Writing strahler number geojson object(s)");
    let strahler_lines = g.edges_iter().map(|(from_nid, to_nid)| {
        (
            // Round the upstream to only output 1 decimal place
            serde_json::json!({
                "strahler": strahler.get(&from_nid)
            }),
            vec![vec![
                nodeid_pos.get(&from_nid).unwrap(),
                nodeid_pos.get(&to_nid).unwrap(),
            ]],
        )
    });
    info_memory_used!();

    let writing_upstreams_bar = progress_bars.add(
        ProgressBar::new(g.num_edges() as u64)
            .with_message("Writing strahler geojson(s) file")
            .with_style(style.clone()),
    );

    let strahler_lines = strahler_lines.progress_with(writing_upstreams_bar);

    let mut f = std::io::BufWriter::new(std::fs::File::create(
        args.output_filename.replace("%s", "strahler"),
    )?);
    let num_strahler_written = write_geojson_features_directly(
        strahler_lines,
        &mut f,
        &output_format,
        &OutputGeometryType::LineString,
    )?;
    info!(
        "Wrote {} features to output file {}",
        num_strahler_written.to_formatted_string(&Locale::en),
        args.output_filename.replace("%s", "strahler")
    );

    let nids_wo_outgoing: BTreeSet<_> = g.vertexes_wo_outgoing_jumbled().collect();
    // look for where it ends
    let end_points = nids_wo_outgoing
        .into_iter()
        .map(|v| (v, length_upstream.get(&v).unwrap()))
        .filter(|(_v, len)| args.min_upstream_m.map_or(true, |min| len >= &&min))
        .map(|(v, len)| {
            (
                // Round the upstream to only output 1 decimal place
                serde_json::json!({"upstream_m": round(len, 1), "nid": v}),
                vec![vec![nodeid_pos.get(&v).unwrap()]],
            )
        });

    let mut f = std::io::BufWriter::new(std::fs::File::create(
        args.output_filename.replace("%s", "ends"),
    )?);
    let num_written = write_geojson_features_directly(
        end_points,
        &mut f,
        &output_format,
        &OutputGeometryType::Point,
    )?;
    info!(
        "Wrote {} features to output file {}",
        num_written.to_formatted_string(&Locale::en),
        args.output_filename.replace("%s", "ends")
    );

    info!(
        "Finished all in {}",
        format_duration(Instant::now() - global_start)
    );

    Ok(())
}

#[derive(Debug, Clone, Hash, serde::Serialize, PartialEq, Eq)]
struct TagGrouper(Vec<String>);

impl std::str::FromStr for TagGrouper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(TagGrouper(s.split(',').map(|s| s.to_string()).collect()))
    }
}
impl std::string::ToString for TagGrouper {
    fn to_string(&self) -> String {
        self.0.join(",")
    }
}

#[allow(clippy::type_complexity)]
/// Write a geojson featurecollection, but manually construct it
fn write_geojson_features_directly(
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

#[derive(PartialEq, Eq, Debug)]
enum OutputFormat {
    GeoJSON,
    GeoJSONSeq,
}

#[derive(PartialEq, Eq, Debug)]
pub enum OutputGeometryType {
    MultiLineString,
    LineString,
    MultiPoint,
    Point,
}

impl OutputGeometryType {
    fn bytes(&self) -> &'static [u8] {
        match self {
            OutputGeometryType::MultiLineString => b"MultiLineString",
            OutputGeometryType::LineString => b"LineString",
            OutputGeometryType::MultiPoint => b"MultiPoint",
            OutputGeometryType::Point => b"Point",
        }
    }
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

fn format_duration(d: std::time::Duration) -> String {
    format!(
        "{} ( {:>.1}sec )",
        format_duration_human(&d),
        d.as_secs_f32()
    )
}

fn read_with_node_replacements(
    input_filename: &Path,
    tag_filter: &[tagfilter::TagFilter],
    node_id_replaces: &HashMap<i64, i64>,
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
        .filter(|w| tag_filter.par_iter().all(|tf| tf.filter(w)))
        .inspect(|_| ways_added.inc(1))
        // TODO support grouping by tag value
        .for_each_with(graph.clone(), |graph, w| {
            let nodes = w.nodes();
            if nodes
                .par_iter()
                .any(|nid| node_id_replaces.contains_key(nid))
            {
                let mut new_nodes = nodes
                    .iter()
                    .map(|nid| node_id_replaces.get(nid).unwrap_or(nid))
                    .copied()
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

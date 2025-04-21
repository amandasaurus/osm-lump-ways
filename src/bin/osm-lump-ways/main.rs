#![allow(warnings)]
use anyhow::{Context, Result};
use clap::Parser;
use get_size::GetSize;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use itertools::Itertools;
use kdtree::KdTree;
use log::{
    Level::{Debug, Trace},
    debug, error, info, log, trace, warn,
};
use osm_lump_ways::inter_store;
use osmio::OSMObjBase;
use osmio::prelude::*;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::iter;

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as atomic_Ordering};
use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};
use std::collections::HashSet;

mod cli_args;

use nodeid_position::NodeIdPosition;
use osm_lump_ways::dij;
use osm_lump_ways::haversine::{haversine_m, haversine_m_fpair};
use osm_lump_ways::nodeid_position;
use osm_lump_ways::sorted_slice_store::SortedSliceSet;
use osm_lump_ways::tagfilter;
use osm_lump_ways::way_group;
use way_group::WayGroup;

use fileio::OutputFormat;
use osm_lump_ways::fileio;
use osm_lump_ways::formatting;
use osm_lump_ways::graph::Graph2;

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
        "Starting osm-lump-ways v{}. Source code: {}",
        std::env!("CARGO_PKG_VERSION"),
        std::env!("CARGO_PKG_REPOSITORY"),
    );

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {pos:>7}/{len:7} {per_sec:>12} {msg}",
        ).unwrap();
    let file_reading_style =
        ProgressStyle::with_template(
            "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {bytes:>7}/{total_bytes:7} {per_sec:>12} {msg}",
            ).unwrap();
    //let input_fp = std::fs::File::open(&args.input_filename)?;
    //let input_bar = progress_bars.add(
    //    ProgressBar::new(input_fp.metadata()?.len())
    //        .with_message("Reading input file")
    //        .with_style(file_reading_style.clone()),
    //);
    //let rdr = input_bar.wrap_read(input_fp);
    //let mut reader = osmio::stringpbf::PBFReader::new(rdr);

    if args.split_files_by_group && !args.output_filename.contains("%s") {
        error!("No %s found in output filename ({})", args.output_filename);
        anyhow::bail!("No %s found in output filename ({})", args.output_filename);
    }
    if !args.split_files_by_group && args.output_filename.contains("%s") {
        warn!(
            "The output filename ({}) contains '%s'. Did you forget --split-files-by-group ? Continuing without splitting by group",
            args.output_filename
        );
    }

    if !args.output_filename.ends_with(".geojson") && !args.output_filename.ends_with(".geojsons") {
        warn!(
            "Output filename '{}' doesn't end with '.geojson' or '.geojsons'. This programme only created GeoJSON or GeoJSONSeq files",
            args.output_filename
        );
    }

    if args.split_files_by_group && args.tag_group_k.is_empty() {
        warn!(
            "You have asked to split into separate files by group without saying what to group by! Everything will go into one group. Use -g in future."
        );
    }

    if !args.split_files_by_group
        && !args.overwrite
        && std::path::Path::new(&args.output_filename).exists()
    {
        error!(
            "Output file {} already exists and --overwrite not used. Refusing to overwrite, and exiting early",
            args.output_filename
        );
        anyhow::bail!(
            "Output file {} already exists and --overwrite not used. Refusing to overwrite, and exiting early",
            args.output_filename
        );
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

    if let Some(ref frames_filepath) = args.output_frames {
        if frames_filepath == Path::new(&args.output_filename) {
            anyhow::bail!(
                "Same value given for output filename & output-frames: {}",
                frames_filepath.display()
            );
        }
        if frames_filepath.exists() && !args.overwrite {
            anyhow::bail!(
                "Frames path filename ({}) exists and no --overwrite argument was given",
                frames_filepath.display()
            );
        }
    }

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
    if let Some(max_sinuosity) = args.max_sinuosity {
        anyhow::ensure!(
            max_sinuosity >= 1.0,
            "It's impossible to have a sinuosity < 1.0. Exiting now without doing anything."
        );
        if max_sinuosity == 1.0 {
            warn!("A max sinuosity of 1.0 will exclude a lot (all?) data");
        }
    }

    // For each group, a hashmap of wayid:nodes in that way
    let group_wayid_nodes: HashMap<Vec<Option<String>>, HashMap<i64, Vec<i64>>> = HashMap::new();
    let group_wayid_nodes = Arc::new(Mutex::new(group_wayid_nodes));

    let nodeid_pos = nodeid_position::default();

    let ways_added = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} ways collected so far for later processing",
            )
            .unwrap(),
        ),
    );

    let started_reading_for_pillar_nodes = Instant::now();
    let input_fp = std::fs::File::open(&args.input_filename)?;
    let input_bar = progress_bars.add(
        ProgressBar::new(input_fp.metadata()?.len())
            .with_message("Reading input file")
            .with_style(file_reading_style.clone()),
    );
    let rdr = input_bar.wrap_read(input_fp);
    let mut reader = osmio::stringpbf::PBFReader::new(rdr);
    let nid2nways = Arc::new(Mutex::new(HashMap::<i64, u8>::new()));
    reader
        .ways()
        .par_bridge()
        .filter(|w| tagfilter::obj_pass_filters(w, &tag_filter, &args.tag_filter_func))
        // TODO support grouping by tag value
        .for_each_with(nid2nways.clone(), |nid2nways, w| {
            assert!(w.id() > 0, "This file has a way id < 0. negative ids are not supported in this tool Use osmium sort & osmium renumber to convert this file and run again.");

            let nids = w.nodes();
            let mut nid2nways = nid2nways.lock().unwrap();
            nid2nways.entry(nids[0]).and_modify(|v| {*v = v.saturating_add(1);}).or_insert(1);
            nid2nways.entry(*nids.last().unwrap()).and_modify(|v| {*v= v.saturating_add(1);}).or_insert(1);
            for n in &nids[1..nids.len()] {
                nid2nways.entry(*n).and_modify(|v| {*v=v.saturating_add(2);}).or_insert(2);
            }
        });
    input_bar.finish();
    progress_bars.remove(&input_bar);
    let nid2nways = Arc::try_unwrap(nid2nways).unwrap().into_inner().unwrap();
    let num_nids = nid2nways.len();
    let mut nids_in_ne2_ways: SortedSliceSet<i64> =
        SortedSliceSet::from_iter(nid2nways.into_iter().filter_map(|(nid, nvertexes)| {
            if nvertexes != 2 { Some(nid) } else { None }
        }));
    info!(
        "There are {} nodes in total, but only {} ({:.1}%) are pillar nodes. It took {} to do this first read",
        num_nids.to_formatted_string(&Locale::en),
        nids_in_ne2_ways.len().to_formatted_string(&Locale::en),
        (nids_in_ne2_ways.len() as f64) * 100. / (num_nids as f64),
        formatting::format_duration(started_reading_for_pillar_nodes.elapsed()),
    );

    if num_nids == 0 {
        warn!("No ways in the file matched your filters. Nothing to do.");
        return Ok(());
    }

    let input_fp = std::fs::File::open(&args.input_filename)?;
    let input_bar = progress_bars.add(
        ProgressBar::new(input_fp.metadata()?.len())
            .with_message("Reading input file")
            .with_style(file_reading_style.clone()),
    );
    let rdr = input_bar.wrap_read(input_fp);
    let mut reader = osmio::stringpbf::PBFReader::new(rdr);
    let graphs: HashMap<Box<[Option<String>]>, Graph2> = Default::default();
    let graphs = Arc::new(Mutex::new(graphs));
    let inter_store = inter_store::InterStore::new();
    let inter_store = Arc::new(Mutex::new(inter_store));
    let started_reading_ways = Instant::now();
    reader
        .ways()
        .par_bridge()
        .filter(|w| tagfilter::obj_pass_filters(w, &tag_filter, &args.tag_filter_func))
        .map(|w| {
            let group = args
                .tag_group_k
                .par_iter()
                .map(|tg| tg.get_values(&w))
                .collect::<Vec<Option<String>>>();
            let group = group.into_boxed_slice();
            (w, group)
        })
        .filter(|(_w, group)| args.incl_unset_group || !group.iter().any(|x| x.is_none()))
        .for_each_with((graphs.clone(), inter_store.clone()), |(graphs, inter_store), (w, group)| {
            let mut graphs = graphs.lock().unwrap();
            let graph = graphs.entry(group).or_default();
            let mut inter_store = inter_store.lock().unwrap();

            let mut nodes = w.nodes();
            let _orig_len = nodes.len();
            if nodes.len() <= 2 {
                for w in nodes.windows(2) {
                    graph.add_edge(w[0], w[1]);
                }
            } else {
                while nodes.len() >= 2 {
                    let i_opt = nodes.iter().skip(1).position(|nid| nids_in_ne2_ways.contains(nid));
                    let mut i = i_opt.unwrap() + 1;
                    // can happen when a river splits and then joins again. try to stop reducing
                    // this little tributary away.
                    while graph.contains_edge(nodes[0], nodes[i]) && i > 1 {
                        i -= 1;
                    }
                    // 2 nodes after another with nothing in between? That can happen with someone
                    // double maps a river. But assert a differnet problem, which shows our ability
                    // to contract edges has problems
                    if i > 1 {
                        assert!(!graph.contains_edge(nodes[0], nodes[i]), "already existing edge from {} to {} (there are {} nodes in the middle) i={}",nodes[0], nodes[i], nodes.len(), i);
                    }
                    assert!(i != 0);
                    graph.add_edge(nodes[0], nodes[i]);
                    inter_store.insert_undirected((nodes[0], nodes[i]), &nodes[1..i]);
                    nodes = &nodes[i..];
                }
            }

            ways_added.inc(1);
        });
    let num_ways_read = ways_added.position();
    drop(nids_in_ne2_ways);
    info!(
        "Finshed reading file and building graph data structure. {} ways read in {}, {:.3e} ways/sec",
        num_ways_read.to_formatted_string(&Locale::en),
        formatting::format_duration(started_reading_ways.elapsed()),
        (num_ways_read as f64) / started_reading_ways.elapsed().as_secs_f64(),
    );

    ways_added.finish_and_clear();
    input_bar.finish_and_clear();
    let graphs = Arc::try_unwrap(graphs).unwrap().into_inner().unwrap();
    let inter_store = Arc::try_unwrap(inter_store).unwrap().into_inner().unwrap();

    let assemble_nids_needed = progress_bars
        .add(ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{human_pos} nids needed for later").unwrap(),
        ));
    let mut nids_we_need: Vec<i64> = Vec::new();
    for graph in graphs.values() {
        nids_we_need.extend(graph.vertexes());
        assemble_nids_needed.inc(nids_we_need.len() as u64);
    }
    nids_we_need.extend(
        inter_store
            .all_inter_nids()
            .inspect(|_| assemble_nids_needed.inc(1)),
    );
    let nids_we_need = SortedSliceSet::from_vec(nids_we_need);
    assemble_nids_needed.finish_and_clear();

    debug!("Re-reading file to read all nodes");
    let started_reading_nodes = Instant::now();
    let setting_node_pos = progress_bars.add(
        ProgressBar::new(nids_we_need.len() as u64)
            .with_message("Nodes read")
            .with_style(style.clone()),
    );
    let reader = osmio::stringpbf::PBFNodePositionReader::from_filename(&args.input_filename)?;
    let nodeid_pos = Arc::new(Mutex::new(nodeid_pos));
    reader
        .into_iter()
        .par_bridge()
        .filter(|(nid, _pos)| nids_we_need.contains(nid))
        .map(|(nid, pos)| (nid, (pos.1.inner(), pos.0.inner()))) // WTF do I have lat & lon
        // mixed up??
        .for_each_with(nodeid_pos.clone(), |nodeid_pos, (nid, pos)| {
            setting_node_pos.inc(1);
            nodeid_pos.lock().unwrap().insert_i32(nid, pos);
        });

    drop(nids_we_need);
    let mut nodeid_pos = Arc::try_unwrap(nodeid_pos).unwrap().into_inner().unwrap();
    nodeid_pos.finished_inserting();
    let nodeid_pos = nodeid_pos;

    setting_node_pos.finish_and_clear();
    info!(
        "Finshed reading all node positions. {} nodes read in {}, {:.3e} nodes/sec",
        setting_node_pos.position().to_formatted_string(&Locale::en),
        formatting::format_duration(started_reading_nodes.elapsed()),
        (setting_node_pos.position() as f64) / started_reading_nodes.elapsed().as_secs_f64(),
    );

    debug!("{}", nodeid_pos.detailed_size());

    debug_var_size("group_wayid_nodes", group_wayid_nodes.get_size());

    let total_files_written = AtomicUsize::new(0);
    let total_features_written = AtomicUsize::new(0);
    info!(
        "All data has been loaded in {}. Started processing...",
        formatting::format_duration(global_start.elapsed())
    );

    if std::env::var("OSM_LUMP_WAYS_FINISH_AFTER_READ").is_ok() {
        return Ok(());
    }
    info!(
        "Starting the breath-first search.{}",
        if graphs.len() > 1 {
            format!(" There are {} groups", graphs.len())
        } else {
            "".to_string()
        }
    );
    let grouping = progress_bars.add(
        ProgressBar::new(
            graphs
                .par_iter()
                .map(|(_group_name, graph)| graph.num_vertexes())
                .sum::<usize>() as u64,
        )
        .with_message("Grouping all ways")
        .with_style(style.clone()),
    );
    let total_groups_found =
        progress_bars.add(ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("            {human_pos} groups found").unwrap(),
        ));

    //grouping.inc(10);
    let started_bfs = Instant::now();

    let mut way_groups = graphs
        .into_par_iter()
        .flat_map_iter(|(group, mut complete_graph)| {
            complete_graph
                .into_disconnected_graphs(grouping.clone())
                .map({
                    let total_groups_found = total_groups_found.clone();
                    move |sub_graph| {
                        total_groups_found.inc(1);
                        WayGroup::new(sub_graph, group.clone())
                    }
                })
        })
        .collect::<Vec<WayGroup>>();

    info!(
        "Finshed splitting into groups in {}",
        formatting::format_duration(started_bfs.elapsed()),
    );
    // ↑ The breath first search is done
    grouping.finish_and_clear();
    total_groups_found.finish_and_clear();
    way_groups.shrink_to_fit();

    // FIXME do the only_these_way_groups etc check
    assert!(args.only_these_way_groups.is_empty());
    assert!(only_these_way_groups_divmod.is_none());

    if !args.only_these_way_groups_nodeid.is_empty() {
        let old_num_wgs = way_groups.len();
        way_groups.retain(|wg| {
            args.only_these_way_groups_nodeid
                .iter()
                .any(|nid| wg.graph.contains_vertex(*nid))
        });
        way_groups.shrink_to_fit();
        info!(
            "Removed {} waygroups which don't have a required node, leaving only {} left",
            old_num_wgs - way_groups.len(),
            way_groups.len()
        );
    }

    way_groups
        .par_iter_mut()
        .for_each(|wg| wg.calculate_length(&nodeid_pos));

    if let Some(min_length_m) = args.min_length_m {
        let old = way_groups.len();
        way_groups.retain(|wg| wg.length_m >= min_length_m);
        info!(
            "Removed {} way_groups which were smaller than {} m",
            (old - way_groups.len()).to_formatted_string(&Locale::en),
            min_length_m
        );
    }
    if let Some(max_length_m) = args.max_length_m {
        way_groups.retain(|wg| wg.length_m <= max_length_m);
    }
    way_groups.shrink_to_fit();

    way_groups.par_iter_mut().for_each(|wg| {
        let mut json_props = &mut wg.json_props;
        json_props["root_nodeid"] = wg.root_nodeid.into();
        json_props["root_nodeid_120"] = (wg.root_nodeid % 120).into();
        json_props["length_m"] = wg.length_m.into();
        for (i, group) in wg.group.iter().enumerate() {
            json_props[format!("tag_group_{}", i)] = group.as_ref().cloned().into();
        }
        json_props["tag_groups"] = wg.group.to_vec().into();
        json_props["length_m_int"] = (wg.length_m.round() as i64).into();
        json_props["length_km"] = (wg.length_m / 1000.).into();
        json_props["length_km_int"] = ((wg.length_m / 1000.).round() as i64).into();
        json_props["num_nodes"] = wg.graph.num_vertexes().into();
    });

    info!("all JSON properties set");

    if let Some(output_frames) = args.output_frames {
        do_frames(
            &output_frames,
            args.frames_group_min_length_m.clone(),
            args.save_as_linestrings.clone(),
            &way_groups,
            &progress_bars,
            &style,
            &nodeid_pos,
            &inter_store,
        )?;
    }

    let split_into_lines = progress_bars.add(
        ProgressBar::new(
            way_groups
                .par_iter()
                .map(|wg| wg.num_nodes())
                .sum::<usize>() as u64,
        )
        .with_message("Splitting into single lines")
        .with_style(style.clone()),
    );

    assert!(!args.split_files_by_group);
    let files_data: HashMap<_, _> = way_groups
        .into_par_iter()
        // Group into files
        .fold(
            || HashMap::new() as HashMap<String, Vec<_>>,
            |mut files, way_group| {
                trace!("Grouping all data into files");
                files
                    //.entry(way_group.filename(&args.output_filename, args.split_files_by_group))
                    .entry(args.output_filename.clone())
                    .or_default()
                    .push(way_group);
                files
            },
        )
        // We might have many hashmaps now, group down to one
        .reduce(HashMap::new, |mut acc, curr| {
            trace!("Merging files down again");
            for (filename, wgs) in curr.into_iter() {
                acc.entry(filename).or_default().extend(wgs.into_iter())
            }
            acc
        });

    info!(
        "All data has been split into {} different file(s)",
        files_data.len()
    );
    assert!(!args.incl_dist_to_longer);

    files_data
        .into_par_iter()
        .update(|(_filename, way_groups)| {
            debug!("sorting ways by length & truncating");
            // in calc dist to longer, we need this sorted too
            way_groups.par_sort_unstable_by(|a, b| a.length_m.total_cmp(&b.length_m).reverse());
        })
        .update(|(_filename, way_groups)| {
            if let Some(limit) = args.only_longest_n_per_file {
                debug!("Truncating files by longest");
                way_groups.truncate(limit);
            }
        })
        .update(|(_filename, way_groups)| {
            let mut feature_ranks = Vec::with_capacity(way_groups.len());

            // calc longest lengths
            // (length of way group, idx of this way group in way_groups, rank)
            way_groups
                .par_iter()
                .enumerate()
                .map(|(i, wg)| (wg.length_m, i, 0))
                .collect_into_vec(&mut feature_ranks);
            // sort by longest first
            feature_ranks.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).reverse());
            // update feature_ranks to store the local rank
            feature_ranks
                .par_iter_mut()
                .enumerate()
                .for_each(|(rank, (_len, _idx, new_rank))| {
                    *new_rank = rank;
                });
            // sort back by way_groups idx
            feature_ranks.par_sort_unstable_by_key(|(_len, wg_idx, _rank)| *wg_idx);
            // now update the way_groups
            let way_groups_len = way_groups.len();
            let way_groups_len_f = way_groups_len as f64;
            way_groups
                .par_iter_mut()
                .zip(feature_ranks.par_iter())
                .for_each(|(wg, (_len, _wg_idx, rank))| {
                    wg.json_props["length_desc_rank"] = (*rank).into();
                    wg.json_props["length_desc_rank_perc"] =
                        ((*rank as f64) / way_groups_len_f).into();
                    wg.json_props["length_asc_rank"] = (way_groups_len - *rank).into();
                    wg.json_props["length_asc_rank_perc"] =
                        ((way_groups_len - *rank) as f64 / way_groups_len_f).into();
                });

            //if args.split_into_single_paths {
            //    // dist between ends
            //    feature_ranks.truncate(0);

            //    // (length of way group, idx of this way group in way_groups, rank)
            //    way_groups
            //        .par_iter()
            //        .enumerate()
            //        .map(|(i, wg)| (wg.json_props["dist_ends_m"].as_f64().unwrap(), i, 0))
            //        .collect_into_vec(&mut feature_ranks);
            //    // sort by longest first
            //    feature_ranks.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).reverse());
            //    // update feature_ranks to store the local rank
            //    feature_ranks.par_iter_mut().enumerate().for_each(
            //        |(rank, (_len, _idx, new_rank))| {
            //            *new_rank = rank;
            //        },
            //    );
            //    // sort back by way_groups idx
            //    feature_ranks.par_sort_unstable_by_key(|(_len, wg_idx, _rank)| *wg_idx);
            //    // now update the way_groups
            //    way_groups
            //        .par_iter_mut()
            //        .zip(feature_ranks.par_iter())
            //        .for_each(|(wg, (_len, _wg_idx, rank))| {
            //            wg.json_props["dist_ends_desc_rank"] = (*rank).into();
            //            wg.json_props["dist_ends_asc_rank"] = (way_groups_len - *rank).into();
            //        });
            //}
        })
        .map(|(filename, way_groups)| {
            let mut results = Vec::with_capacity(way_groups.len());
            for wg in way_groups.into_iter() {
                let WayGroup {
                    json_props,
                    graph,
                    group,
                    ..
                } = wg;
                // Iterator which yields lines of nids for each line
                let lines_nids_iter = if args.split_into_single_paths {
                    assert_eq!(
                        args.split_into_single_paths_by,
                        dij::SplitPathsMethod::AsCrowFlies
                    );
                    Box::new(
                        graph
                            .into_lines_as_crow_flies(&nodeid_pos)
                            .take(args.only_longest_n_splitted_paths.unwrap_or(usize::MAX)),
                    ) as Box<dyn Iterator<Item = Box<[i64]>>>
                } else {
                    Box::new(graph.into_lines_random())
                };
                // Turn list of nids into latlngs
                let coords_iter = lines_nids_iter.map(|nids| {
                    split_into_lines.inc(nids.len() as u64);
                    inter_store
                        .expand_line_undirected(&nids)
                        .map(|nid| nodeid_pos.get(&nid).unwrap())
                        .collect::<Vec<_>>()
                });

                if !args.split_into_single_paths && !args.save_as_linestrings {
                    results.push((json_props, coords_iter.collect::<Vec<_>>()));
                } else {
                    // TODO remove this little Vec (goal is to reduce allocations)
                    results.extend(coords_iter.map(|coords| {
                        let mut json_props = json_props.clone();
                        if args.split_into_single_paths {
                            let dist_ends = haversine_m_fpair(coords[0], *coords.last().unwrap());
                            json_props["dist_ends_m"] = dist_ends.into();
                            json_props["dist_ends_m_int"] = (dist_ends.round() as i64).into();
                            json_props["dist_ends_km"] = (dist_ends / 1000.).into();
                            json_props["dist_ends_km_int"] =
                                ((dist_ends / 1000.).round() as i64).into();
                        }
                        (json_props, vec![coords])
                    }));
                }
            }

            (filename, results)
        })
        .try_for_each(|(filename, features)| {
            debug!("Writing data to file(s)...");
            // Write the files
            match std::fs::File::create(&filename) {
                Ok(f) => {
                    let num_features = features.len();
                    let mut f = std::io::BufWriter::new(f);
                    let num_written = fileio::write_geojson_features_directly(
                        features.into_iter(),
                        &mut f,
                        &output_format,
                    )
                    .with_context(|| {
                        format!(
                            "Writing {} features to filename {:?}",
                            num_features, filename
                        )
                    })?;
                    info!(
                        "Wrote {} feature(s) to {}",
                        num_written.to_formatted_string(&Locale::en),
                        filename
                    );
                    total_features_written.fetch_add(num_written, atomic_Ordering::SeqCst);
                    total_files_written.fetch_add(1, atomic_Ordering::SeqCst);
                }
                Err(e) => {
                    warn!("Couldn't open filename {:?}: {}", filename, e);
                }
            }
            Ok(()) as Result<()>
        })?;

    let total_files_written = total_files_written.into_inner();
    let total_features_written = total_features_written.into_inner();

    if total_files_written == 0 {
        if !args.only_these_way_groups.is_empty() {
            warn!(
                "No files have been written, and you specified to only process the following waygroups: {:?}. Perhaps nothing in your input data matches that",
                args.only_these_way_groups
            );
        }
        if !args.only_these_way_groups_nodeid.is_empty() {
            warn!(
                "No files have been written, and you specified to only include way groups with the following nodeids {:?}. Perhaps nothing in your input data matches that",
                args.only_these_way_groups_nodeid
            );
        }
        if args.only_these_way_groups.is_empty() && args.only_these_way_groups_nodeid.is_empty() {
            warn!("No files have been written.");
        }
    } else if total_files_written == 1 {
        // don't print anything, it'll have been printed once above
    } else {
        info!(
            "Wrote {} feature(s) to {} file(s)",
            total_features_written.to_formatted_string(&Locale::en),
            total_files_written
        );
    }

    info!(
        "Finished all in {}",
        formatting::format_duration(global_start.elapsed())
    );
    Ok(())
}

fn do_frames(
    frames_filepath: &PathBuf,
    frames_group_min_length_m: Option<f64>,
    save_as_linestrings: bool,
    way_groups: &[WayGroup],
    progress_bars: &MultiProgress,
    style: &ProgressStyle,
    nodeid_pos: &impl NodeIdPosition,
    inter_store: &inter_store::InterStore,
) -> Result<()> {
    let skipped_way_groups_count = AtomicUsize::new(0);
    let skipped_way_groups_length_sum = AtomicU64::new(0);

    let started_frames = Instant::now();
    info!(
        "Calculating, for each way group, all the frames (lines through the middle){}",
        frames_group_min_length_m.map_or("".to_string(), |min| format!(
            ", and only including way groups longer than {:.3e}",
            min
        ))
    );
    let frames_all_nodes_bar = progress_bars.add(
        ProgressBar::new(
            way_groups
                .par_iter()
                .map(|wg| wg.num_nodes() as u64)
                .sum::<u64>(),
        )
        .with_message("Processing nodes for Frames")
        .with_style(style.clone()),
    );
    let frames_bar = progress_bars.add(
        ProgressBar::new(0)
            .with_message("Calculating Frames")
            .with_style(style.clone()),
    );
    let started_frames_calculation = Instant::now();
    let frames: Vec<_> = way_groups
        .par_iter()
        .filter(|wg| frames_group_min_length_m.map_or(true, |min| wg.length_m >= min))
        .flat_map_iter(
            |wg| -> Box<dyn Iterator<Item = (serde_json::Value, Vec<_>)>> {
                let paths = wg
                    .frames(nodeid_pos, &frames_bar)
                    .map(|line| {
                        inter_store
                            .expand_line_undirected(&line)
                            .map(|nid| nodeid_pos.get(&nid).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                frames_all_nodes_bar.inc(wg.num_nodes() as u64);

                //frames_all_nodes_bar.inc(wg.num_nodeids() as u64);
                if save_as_linestrings {
                    Box::new(
                        paths
                            .into_iter()
                            .map(|path| (wg.json_props.clone(), vec![path])),
                    )
                } else {
                    Box::new(std::iter::once((wg.json_props.clone(), paths)))
                }
            },
        )
        .collect();
    info!(
        "Calculated {} frames in {} ({:.3e} frames/sec)",
        frames.len().to_formatted_string(&Locale::en),
        formatting::format_duration(started_frames_calculation.elapsed()),
        (frames.len() as f64) / started_frames_calculation.elapsed().as_secs_f64(),
    );
    if frames_group_min_length_m.is_some() {
        info!(
            "{} way groups (total length: {:.3e} m) were excluded from frame calculation",
            skipped_way_groups_count
                .into_inner()
                .to_formatted_string(&Locale::en),
            skipped_way_groups_length_sum.into_inner(),
        )
    }
    frames_bar.finish_and_clear();
    frames_all_nodes_bar.finish_and_clear();

    // Then write it
    let frames_writing_bar = progress_bars.add(
        ProgressBar::new(frames.len() as u64)
            .with_message(format!("Writing frames to {}", frames_filepath.display()))
            .with_style(style.clone()),
    );

    let f = std::fs::File::create(&frames_filepath).unwrap();
    let mut f = std::io::BufWriter::new(f);
    let num_written = fileio::write_geojson_features_directly(
        frames_writing_bar.wrap_iter(frames.into_iter()),
        &mut f,
        &OutputFormat::GeoJSONSeq,
    )?;
    info!(
        "Calculated & wrote {} frames to {:?} in {}",
        num_written.to_formatted_string(&Locale::en),
        frames_filepath,
        formatting::format_duration(started_frames.elapsed())
    );

    Ok(())
}

fn debug_var_size(name: &str, size: usize) {
    debug!(
        "Size of {}: {} = {} bytes",
        name,
        size,
        size.to_formatted_string(&Locale::en)
    );
}

use anyhow::{Context, Result};
use clap::Parser;
use get_size::GetSize;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use log::{
    debug, error, info, log, trace, warn,
    Level::{Debug, Trace},
};
use osmio::prelude::*;
use osmio::OSMObjBase;
use rayon::prelude::*;

use kdtree::KdTree;

use std::collections::{BTreeSet, HashMap};
use std::time::Instant;

use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as atomic_Ordering};
use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};

mod cli_args;

use nodeid_position::NodeIdPosition;
use nodeid_wayids::NodeIdWayIds;
use osm_lump_ways::dij;
use osm_lump_ways::haversine;
use osm_lump_ways::haversine::haversine_m;
use osm_lump_ways::nodeid_position;
use osm_lump_ways::nodeid_wayids;
use osm_lump_ways::tagfilter;
use osm_lump_ways::way_group;
use way_group::WayGroup;

use fileio::OutputFormat;
use osm_lump_ways::fileio;
use osm_lump_ways::formatting;

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
    let input_fp = std::fs::File::open(&args.input_filename)?;
    let input_bar = progress_bars.add(
        ProgressBar::new(input_fp.metadata()?.len())
            .with_message("Reading input file")
            .with_style(file_reading_style.clone()),
    );
    let rdr = input_bar.wrap_read(input_fp);

    //let reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
    let mut reader = osmio::stringpbf::PBFReader::new(rdr);

    if args.split_files_by_group && !args.output_filename.contains("%s") {
        error!("No %s found in output filename ({})", args.output_filename);
        anyhow::bail!("No %s found in output filename ({})", args.output_filename);
    }
    if !args.split_files_by_group && args.output_filename.contains("%s") {
        warn!("The output filename ({}) contains '%s'. Did you forget --split-files-by-group ? Continuing without splitting by group", args.output_filename);
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

    let nodeid_wayids = nodeid_wayids::default();
    let nodeid_wayids = Arc::new(Mutex::new(nodeid_wayids));

    let ways_added = progress_bars.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} ways collected so far for later processing",
            )
            .unwrap(),
        ),
    );

    let started_reading_ways = Instant::now();
    reader
        .ways()
        .par_bridge()
        .filter(|w| tagfilter::obj_pass_filters(w, &args.tag_filter, &args.tag_filter_func))
        .map(|w| {
            let group = args
                .tag_group_k
                .par_iter()
                .map(|tg| tg.get_values(&w))
                .collect::<Vec<Option<String>>>();
            (w, group)
        })
        .filter(|(_w, group)| args.incl_unset_group || !group.iter().any(|x| x.is_none()))
        .for_each_with(
            (nodeid_wayids.clone(), group_wayid_nodes.clone()),
            |(nodeid_wayids, group_wayid_nodes), (w, group)| {
                trace!("Got a way {}, in group {:?}", w.id(), group);
                rayon::join(
                    || {
                        nodeid_wayids.lock().unwrap().insert_many(w.id(), w.nodes());
                    },
                    || {
                        group_wayid_nodes
                            .lock()
                            .unwrap()
                            .entry(group)
                            .or_default()
                            .insert(w.id(), w.nodes().to_owned());
                    },
                );
                ways_added.inc(1);
            },
        );
    let num_ways_read = ways_added.position();
    info!(
        "Finshed first read of file. {} ways read in {}, {:.3e} ways/sec",
        num_ways_read.to_formatted_string(&Locale::en),
        formatting::format_duration(started_reading_ways.elapsed()),
        (num_ways_read as f64) / started_reading_ways.elapsed().as_secs_f64(),
    );
    ways_added.finish_and_clear();
    input_bar.finish_and_clear();
    let nodeid_wayids = Arc::try_unwrap(nodeid_wayids)
        .unwrap()
        .into_inner()
        .unwrap();

    let group_wayid_nodes = Arc::try_unwrap(group_wayid_nodes)
        .unwrap()
        .into_inner()
        .unwrap();

    if group_wayid_nodes.is_empty() {
        info!("No ways in the file matched your filters. Nothing to do");
        return Ok(());
    }

    debug!("{}", nodeid_wayids.detailed_size());

    //let input_fp = std::fs::File::open(&args.input_filename)?;
    //let input_bar = progress_bars.add(
    //    ProgressBar::new(input_fp.metadata()?.len())
    //        .with_message("Re-reading file to save node locations")
    //        .with_style(file_reading_style.clone()),
    //);
    //let rdr = input_bar.wrap_read(input_fp);
    //let reader = osmio::stringpbf::PBFNodePositionReader::from_reader(input_fp);

    debug!("Re-reading file to read all nodes");
    let started_reading_nodes = Instant::now();
    let setting_node_pos = progress_bars.add(
        ProgressBar::new(nodeid_wayids.len() as u64)
            .with_message("Nodes read")
            .with_style(style.clone()),
    );
    let reader = osmio::stringpbf::PBFNodePositionReader::from_filename(&args.input_filename)?;
    let nodeid_pos = Arc::new(Mutex::new(nodeid_pos));
    reader
        .into_iter()
        .par_bridge()
        .filter(|(nid, _pos)| nodeid_wayids.contains_nid(nid))
        .map(|(nid, pos)| (nid, (pos.1.inner(), pos.0.inner()))) // WTF do I have lat & lon
        // mixed up??
        .for_each_with(nodeid_pos.clone(), |nodeid_pos, (nid, pos)| {
            setting_node_pos.inc(1);
            nodeid_pos.lock().unwrap().insert_i32(nid, pos);
        });

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

    debug!(
        "Size of group_wayid_nodes: {} = {} bytes",
        group_wayid_nodes.get_size(),
        group_wayid_nodes
            .get_size()
            .to_formatted_string(&Locale::en)
    );

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
        if group_wayid_nodes.len() > 1 {
            format!(" There are {} groups", group_wayid_nodes.len())
        } else {
            "".to_string()
        }
    );
    let grouping = progress_bars.add(
        ProgressBar::new(
            group_wayid_nodes
                .values()
                .map(|wayid_nodes| {
                    wayid_nodes.par_iter().map(|(_k, v)| v.len()).sum::<usize>() as u64
                })
                .sum(),
        )
        .with_message("Grouping all ways")
        .with_style(style.clone()),
    );
    let total_groups_found =
        progress_bars.add(ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("            {human_pos} groups found").unwrap(),
        ));

    let reorder_segments_bar = progress_bars.add(
        ProgressBar::new(0)
            .with_message("Reordering inner segments")
            .with_style(style.clone()),
    );

    let splitter = progress_bars.add(
        ProgressBar::new(0)
            .with_message("Splitting ways into lines")
            .with_style(style.clone()),
    );

    let way_groups = group_wayid_nodes
        .into_par_iter()
        .flat_map(|(group, wayid_nodes)| {
            // Actually do the breath first search
            // TODO this is single threaded...
            trace!("Starting breath-first search for group {:?}", group);

            trace!("Starting to collect all the wayids into a set...");
            let mut unprocessed_wayids: BTreeSet<i64> =
                wayid_nodes.par_iter().map(|(k, _v)| k).copied().collect();
            trace!("... finished");

            // Current list of all wids which are in the group
            let mut this_group_wayids = Vec::new();

            // all our resultant way groups
            let mut way_groups = Vec::new();

            while let Some(root_wayid) = unprocessed_wayids.pop_first() {
                grouping.inc(wayid_nodes[&root_wayid].len() as u64);
                // wayids which are in this little group
                this_group_wayids.push(root_wayid);

                // struct for this group
                let mut this_group = WayGroup::new(root_wayid, group.to_owned());
                trace!(
                    "root_wayid {:?} (there are {} unprocessed ways left)",
                    root_wayid,
                    unprocessed_wayids.len()
                );
                while let Some(wid) = this_group_wayids.pop() {
                    trace!(
                        "Wayid: {} Currently there are {} ways in this group",
                        wid,
                        this_group.way_ids.len()
                    );

                    this_group.way_ids.push(wid);
                    // Kinda stupid way to try to get this *somewhat* multithreaded
                    rayon::join(
                        || {
                            this_group.nodeids.push(wayid_nodes[&wid].clone());
                        },
                        || {
                            // find all other ways which are connected to wid, and add them to this_group
                            for other_wayid in wayid_nodes[&wid]
                                .iter()
                                .filter(|nid| nodeid_wayids.nid_is_in_many(nid)) // only look at nodes in >1 ways
                                .flat_map(|nid| nodeid_wayids.ways(nid))
                            {
                                // If this other way hasn't been processed yet, then add to this group.
                                if unprocessed_wayids.remove(&other_wayid) {
                                    grouping.inc(wayid_nodes[&other_wayid].len() as u64);
                                    trace!("adding other way {}", other_wayid);
                                    this_group_wayids.push(other_wayid);
                                }
                            }
                        },
                    );
                }

                reorder_segments_bar.inc_length(this_group.nodeids.len() as u64);
                total_groups_found.inc(1);
                way_groups.push(this_group);
            }
            debug!(
                "In total, found {} waygroups for the tag group {:?}",
                way_groups.len().to_formatted_string(&Locale::en),
                group
            );

            way_groups
        })
        .collect::<Vec<WayGroup>>();
    // ↑ The breath first search is done
    grouping.finish_and_clear();
    total_groups_found.finish_and_clear();
    drop(nodeid_wayids);

    let way_groups: Vec<_> = way_groups
        .into_par_iter()
        .filter(|way_group| {
            if args.only_these_way_groups.is_empty() {
                true // no filtering in operation
            } else {
                args.only_these_way_groups
                    .par_iter()
                    .any(|only| *only == way_group.root_wayid)
            }
        })
        .filter(|way_group| match only_these_way_groups_divmod {
            None => true,
            Some((a, b)) => way_group.root_wayid % a == b,
        })
        .filter(|way_group| {
            if args.only_these_way_groups_nodeid.is_empty() {
                true
            } else {
                args.only_these_way_groups_nodeid
                    .par_iter()
                    .any(|nid1| way_group.nodeids_iter().any(|nid2| nid1 == nid2))
            }
        })
        .collect();

    let way_groups: Vec<_> = way_groups
        .into_par_iter()
        .update(|way_group| {
            trace!("Reducing the number of inner segments");
            way_group.reorder_segments(20, &reorder_segments_bar, true);
        })
        .update(|way_group| {
            trace!("Saving coordinates for all ways");
            way_group.set_coords(&nodeid_pos);
        })
        .update(|way_group| {
            trace!("Calculating all lengths");
            way_group.calculate_length();
        })
        // apply min length filter
        // This is before any possible splitting. If an unsplitted way_group has total len ≤ the min
        // len, then splitting won't make it be included.
        // This reduces the amount of splitting we have to do.
        .filter(|way_group| match args.min_length_m {
            None => true,
            Some(min_len) => way_group.length_m.unwrap() >= min_len,
        })
        .inspect(|way_group| {
            if args.split_into_single_paths {
                splitter.inc_length(way_group.num_nodeids() as u64);
            }
        })
        .collect();
    reorder_segments_bar.finish_and_clear();

    // ↓ Split into paths if needed
    let way_groups: Vec<_> = way_groups.into_par_iter()
    .flat_map(|way_group| {
        let new_way_groups = if !args.split_into_single_paths {
            vec![way_group]
        } else {

            trace!("wg:{} splitting the groups into single paths with Dij algorithm... wg.num_nodeids() = {}", way_group.root_wayid, way_group.num_nodeids());
            let started = Instant::now();
            let paths = match dij::into_segments(&way_group, &nodeid_pos, args.min_length_m, args.only_longest_n_splitted_paths, args.max_sinuosity, args.split_into_single_paths_by.clone().unwrap_or_default(), &splitter) {
                Ok(paths) => {
                    let duration = started.elapsed().as_secs_f64();
                    log!(
                        if paths.len() > 20 || duration > 2. { Debug } else { Trace },
                        "Have generated {} paths from wg:{} ({} nodes) in {:.1} sec. {:.2} nodes/sec",
                        paths.len(), way_group.root_wayid, way_group.num_nodeids(), duration, (way_group.num_nodeids() as f64)/duration
                    );

                    paths
                }
                Err(e) => {
                    error!("Problem with wg:{:?}: {}. You probably don't have enough memory. You can rerun just this failing subset by passing --only-these-way-groups {:?} as a CLI arg. This group has {} nodes Error: {:?}", way_group.root_wayid, e, way_group.root_wayid, way_group.root_wayid, e);
                    // Hack to ensure nothing is run later
                    vec![]
                }
            };

            paths.into_par_iter().map(move |path| {
                let mut new_wg = way_group.clone();
                new_wg.root_wayid = *path.iter().min().unwrap();
                new_wg.nodeids = vec![path];
                new_wg.coords = None;
                new_wg.length_m = None;
                new_wg.way_ids.truncate(0);
                new_wg.to_owned()
            })
            .collect()
        };

        new_way_groups.into_par_iter()

    })
    // Add the coords & lengths again if needed
    .update(|way_group| {
        way_group.set_coords(&nodeid_pos);
    })
    .update(|way_group| {
        way_group.calculate_length()
    })
    // apply min length filter
    .filter(|way_group|
        match args.min_length_m {
            None => true,
            Some(min_len) => way_group.length_m.unwrap() >= min_len,
        }
    )
    .update(|way_group| {
        trace!("Preparing extra json properties");
        way_group.extra_json_props["root_wayid"] = way_group.root_wayid.into();
        way_group.extra_json_props["root_wayid_120"] = (way_group.root_wayid % 120).into();
        way_group.extra_json_props["length_m"] = way_group.length_m.into();
        for (i, group) in way_group.group.iter().enumerate() {
            way_group.extra_json_props[format!("tag_group_{}", i)] = group.as_ref().cloned().into();
        }
        way_group.extra_json_props["tag_groups"] = way_group.group.clone().into();
        way_group.extra_json_props["length_m_int"] = way_group.length_m.map(|l| l.round() as i64).into();
        way_group.extra_json_props["length_km"] = way_group.length_m.map(|l| l/1000.).into();
        way_group.extra_json_props["length_km_int"] = way_group.length_m.map(|l| l/1000.).map(|l| l.round() as i64).into();
        way_group.extra_json_props["num_ways"] = way_group.way_ids.len().into();
        way_group.extra_json_props["num_nodes"] = way_group.num_nodeids().into();
        if args.split_into_single_paths {
            let line = &way_group.coords.as_ref().unwrap()[0];
            let dist_ends = haversine_m(line[0].0, line[0].1, line.last().unwrap().0, line.last().unwrap().1);
            way_group.extra_json_props["dist_ends_m"] = dist_ends.into();
            way_group.extra_json_props["dist_ends_m_int"] = (dist_ends.round() as i64).into();
            way_group.extra_json_props["dist_ends_km"] = (dist_ends/1000.).into();
            way_group.extra_json_props["dist_ends_km_int"] = ((dist_ends/1000.).round() as i64).into();
            way_group.extra_json_props["sinuosity"] = way_group.length_m.map(|l| l/dist_ends).into();
        }

    })
    .collect();

    if let Some(frames_filepath) = args.output_frames {
        let skipped_way_groups_count = AtomicUsize::new(0);
        let skipped_way_groups_length_sum = AtomicU64::new(0);

        let started_frames = Instant::now();
        info!(
            "Calculating, for each way group, all the frames (lines through the middle){}",
            args.frames_group_min_length_m
                .map_or("".to_string(), |min| format!(
                    ", and only including way groups longer than {:.3e}",
                    min
                ))
        );
        let frames_all_nodes_bar = progress_bars.add(
            ProgressBar::new(
                way_groups
                    .par_iter()
                    .map(|wg| wg.num_nodeids() as u64)
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
        // This can need a lot of memory, so don't calculate in parallel.
        let started_frames_calculation = Instant::now();
        let frames: Vec<_> = way_groups
            .iter()
            .flat_map(
                |wg| -> Box<dyn Iterator<Item = (serde_json::Value, Vec<_>)>> {
                    if let Some(min) = args.frames_group_min_length_m {
                        if wg.length_m.unwrap() < min {
                            skipped_way_groups_count.fetch_add(1, atomic_Ordering::SeqCst);
                            skipped_way_groups_length_sum.fetch_add(
                                wg.length_m.unwrap().round() as u64,
                                atomic_Ordering::SeqCst,
                            );
                            frames_all_nodes_bar.inc(wg.num_nodeids() as u64);
                            return Box::new(std::iter::empty());
                        }
                    }
                    //let started = Instant::now();
                    let paths = wg.frames(&nodeid_pos, &frames_bar);
                    frames_all_nodes_bar.inc(wg.num_nodeids() as u64);
                    if args.save_as_linestrings {
                        Box::new(
                            paths
                                .into_iter()
                                .map(|path| (wg.extra_json_props.clone(), vec![path])),
                        )
                    } else {
                        Box::new(std::iter::once((wg.extra_json_props.clone(), paths)))
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
        if args.frames_group_min_length_m.is_some() {
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
    }

    let files_data: HashMap<_, _> = way_groups
        .into_par_iter()
        // Group into files
        .fold(
            || HashMap::new() as HashMap<String, Vec<WayGroup>>,
            |mut files, way_group| {
                trace!("Grouping all data into files");
                files
                    .entry(way_group.filename(&args.output_filename, args.split_files_by_group))
                    .or_default()
                    .push(way_group);
                files
            },
        )
        // We might have many hashmaps now, group down to one
        .reduce(HashMap::new, |mut acc, curr| {
            trace!("Merging files down again");
            for (filename, wgs) in curr.into_iter() {
                #[allow(clippy::map_entry)]
                if !acc.contains_key(&filename) {
                    acc.insert(filename, wgs);
                } else {
                    acc.get_mut(&filename).unwrap().extend(wgs.into_iter());
                }
            }
            acc
        });

    files_data
        .into_par_iter()
        .update(|(_filename, way_groups)| {
            debug!("sorting ways by length & truncating");
            // in calc dist to longer, we need this sorted too
            way_groups.par_sort_by(|a, b| {
                a.length_m
                    .unwrap()
                    .total_cmp(&b.length_m.unwrap())
                    .reverse()
            });
        })
        .update(|(_filename, way_groups)| {
            if let Some(limit) = args.only_longest_n_per_file {
                debug!("Truncating files by longest");
                way_groups.truncate(limit);
            }
        })
        .update(|(_filename, way_groups)| {
            if args.incl_dist_to_longer {
                debug!("Calculating the distance to the nearest longer object per way");

                let mut points_distance_idx = KdTree::new(2);

                let prog = progress_bars.add(
                    ProgressBar::new(way_groups.iter().map(|wg| wg.num_nodeids() as u64).sum())
                        .with_message("Calc distance to longer: Indexing data")
                        .with_style(style.clone()),
                );

                for (wg_id, coords) in way_groups
                    .iter()
                    .enumerate()
                    .flat_map(|(wg_id, wg)| wg.coords_iter_seq().map(move |coords| (wg_id, coords)))
                {
                    points_distance_idx.add(coords, wg_id).unwrap();
                    prog.inc(1);
                }
                prog.finish_and_clear();

                let prog = progress_bars.add(
                    ProgressBar::new(
                        way_groups
                            .par_iter()
                            .map(|wg| wg.num_nodeids() as u64)
                            .sum::<u64>(),
                    )
                    .with_message("Calc distance to longer")
                    .with_style(style.clone()),
                );
                // dist to larger
                let longers = way_groups
                    .par_iter()
                    .enumerate()
                    .map(|(wg_id, wg)| {
                        // for each point what's the nearest other point that's in a longer other wayid
                        let min = wg
                            .coords_iter_par()
                            .map(|coord: [f64; 2]| -> Option<(f64, i64)> {
                                let nearest_longer = points_distance_idx
                                    .iter_nearest(&coord, &haversine::haversine_m_arr)
                                    .unwrap()
                                    .filter(|(_dist, other_wg_id)| **other_wg_id != wg_id)
                                    .find(|(_dist, other_wg_id)| {
                                        way_groups[**other_wg_id].length_m
                                            > way_groups[wg_id].length_m
                                    });
                                prog.inc(1);
                                nearest_longer
                                    .map(|(dist, wgid)| (dist, way_groups[*wgid].root_wayid))
                            })
                            .filter_map(|x| x)
                            .min_by(|a, b| (a.0).total_cmp(&b.0));
                        min
                    })
                    .collect::<Vec<_>>();
                prog.finish_and_clear();

                // set the longer distance
                way_groups
                    .par_iter_mut()
                    .zip(longers)
                    .for_each(|(wg, longer)| {
                        wg.extra_json_props["dist_to_longer_m"] =
                            longer.map(|(dist, _)| dist).into();
                        wg.extra_json_props["nearest_longer_waygroup"] =
                            longer.map(|(_dist, wgid)| wgid).into();
                    });

                // remove any that are too short
                // TODO this can prob. be done faster in the above line, where er
                if let Some(min_dist_to_longer_m) = args.min_dist_to_longer_m {
                    way_groups.retain(|wg| {
                        wg.extra_json_props["dist_to_longer_m"]
                            .as_f64()
                            .map_or(true, |d| d >= min_dist_to_longer_m)
                    })
                }
            }
        })
        .update(|(_filename, way_groups)| {
            let mut feature_ranks = Vec::with_capacity(way_groups.len());

            // calc longest lengths
            // (length of way group, idx of this way group in way_groups, rank)
            way_groups
                .par_iter()
                .enumerate()
                .map(|(i, wg)| (wg.length_m.unwrap(), i, 0))
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
                    wg.extra_json_props["length_desc_rank"] = (*rank).into();
                    wg.extra_json_props["length_desc_rank_perc"] =
                        ((*rank as f64) / way_groups_len_f).into();
                    wg.extra_json_props["length_asc_rank"] = (way_groups_len - *rank).into();
                    wg.extra_json_props["length_asc_rank_perc"] =
                        ((way_groups_len - *rank) as f64 / way_groups_len_f).into();
                });

            if args.split_into_single_paths {
                // dist between ends
                feature_ranks.truncate(0);

                // (length of way group, idx of this way group in way_groups, rank)
                way_groups
                    .par_iter()
                    .enumerate()
                    .map(|(i, wg)| (wg.extra_json_props["dist_ends_m"].as_f64().unwrap(), i, 0))
                    .collect_into_vec(&mut feature_ranks);
                // sort by longest first
                feature_ranks.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).reverse());
                // update feature_ranks to store the local rank
                feature_ranks.par_iter_mut().enumerate().for_each(
                    |(rank, (_len, _idx, new_rank))| {
                        *new_rank = rank;
                    },
                );
                // sort back by way_groups idx
                feature_ranks.par_sort_unstable_by_key(|(_len, wg_idx, _rank)| *wg_idx);
                // now update the way_groups
                way_groups
                    .par_iter_mut()
                    .zip(feature_ranks.par_iter())
                    .for_each(|(wg, (_len, _wg_idx, rank))| {
                        wg.extra_json_props["dist_ends_desc_rank"] = (*rank).into();
                        wg.extra_json_props["dist_ends_asc_rank"] = (way_groups_len - *rank).into();
                    });
            }
        })
        // ↓ convert to json objs
        .map(|(filename, way_groups)| {
            debug!("Convert to GeoJSON (ish)");
            let features = way_groups
                .into_par_iter()
                .map(|w| {
                    let mut properties = w.extra_json_props;
                    if args.incl_wayids {
                        properties["all_wayids"] = w
                            .way_ids
                            .iter()
                            .map(|wid| format!("w{}", wid))
                            .collect::<Vec<String>>()
                            .into();
                    }

                    (properties, w.coords.unwrap())
                })
                .flat_map_iter(|(properties, coords)| {
                    if args.save_as_linestrings {
                        Box::new(
                            coords
                                .into_iter()
                                .map(move |coord_string| (properties.clone(), vec![coord_string])),
                        )
                    } else {
                        // Need to put the `as Box…` so the rust compiler won't complain
                        Box::new(std::iter::once((properties, coords)))
                            as Box<dyn Iterator<Item = (serde_json::Value, Vec<Vec<(f64, f64)>>)>>
                    }
                })
                .collect::<Vec<_>>();

            (filename, features)
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
            warn!("No files have been written, and you specified to only process the following waygroups: {:?}. Perhaps nothing in your input data matches that", args.only_these_way_groups);
        }
        if !args.only_these_way_groups_nodeid.is_empty() {
            warn!("No files have been written, and you specified to only include way groups with the following nodeids {:?}. Perhaps nothing in your input data matches that", args.only_these_way_groups_nodeid);
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

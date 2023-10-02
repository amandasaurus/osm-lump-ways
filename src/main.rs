#![allow(warnings)]
use anyhow::{Context, Result};
use clap::Parser;
use get_size::GetSize;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
#[allow(unused_imports)]
use log::{
    debug, error, info, log, log_enabled, trace, warn,
    Level::{Debug, Trace},
};
use osmio::obj_types::ArcOSMObj;
use osmio::prelude::*;
use osmio::OSMObjBase;
use rayon::prelude::*;

use kdtree::KdTree;

use serde_json::json;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::io::Write;

use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};

mod cli_args;
mod haversine;
mod tagfilter;
use haversine::haversine_m;
mod fw;
mod graph;
mod way_group;
use way_group::WayGroup;
mod nodeid_position;
use nodeid_position::NodeIdPosition;
mod nodeid_wayids;

fn main() -> Result<()> {
    // If the RUST_LOG env is set, then use that. else parse from the -v/-q CLI args
    if std::env::var("RUST_LOG").is_ok() {
        env_logger::init();
    } else {
        // Initially show with warn to catch warn's in the clap parsing
        let _logger =
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();
    }

    let args = cli_args::Args::parse();
    let show_progress_bars = args.verbose.log_level_filter() >= log::Level::Info;

    if !std::env::var("RUST_LOG").is_ok() {
        // now we use the -v/-q args to change the level
        log::set_max_level(args.verbose.log_level_filter());
    }

    let reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
    let mut reader = osmio::pbf::PBFReader::new(reader);

    if args.split_files_by_group && !args.output_filename.contains("%s") {
        error!("No %s found in output filename ({})", args.output_filename);
        anyhow::bail!("No %s found in output filename ({})", args.output_filename);
    }

    if !args.output_filename.ends_with(".geojson") {
        warn!("Output filename {} doesn't end with .geojson. This programme only created GeoJSON files", args.output_filename);
    }

    if args.split_files_by_group && args.tag_group_k.is_empty() {
        warn!("You have asked to split into separate files by group without saying what to group by! Everything will go into one group. Use -g in future.");
    }

    if !args.split_files_by_group
        && !args.overwrite
        && std::path::Path::new(&args.output_filename).exists()
    {
        warn!("Output file {} already exists and --overwrite not used. Refusing to overwrite, and exiting early", args.output_filename);
        return Ok(());
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

    info!("Starting to read {:?}", &args.input_filename);
    info!("Tag filter(s) in operation: {:?}", args.tag_filter);
    info!("Tag grouping(s) in operation: {:?}", args.tag_group_k);

    // For each group, a hashmap of wayid:nodes in that way
    let group_wayid_nodes: HashMap<Vec<Option<String>>, HashMap<i64, Vec<i64>>> = HashMap::new();
    let group_wayid_nodes = Arc::new(Mutex::new(group_wayid_nodes));

    let nodeid_pos = nodeid_position::default();
    let nodeid_pos = Arc::new(Mutex::new(nodeid_pos));

    let nodeid_wayids = nodeid_wayids::default();
    let nodeid_wayids = Arc::new(Mutex::new(nodeid_wayids));

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {percent:>3}% done. eta {eta:>4} {bar:10.cyan/blue} {pos:>7}/{len:7} {per_sec:>12} {msg}",
    )
    .unwrap();
    let input_spinners = indicatif::MultiProgress::new();
    if !show_progress_bars {
        input_spinners.set_draw_target(ProgressDrawTarget::hidden());
    }
    let obj_reader = input_spinners.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {human_pos} OSM objects read {per_sec:>20} obj/sec",
            )
            .unwrap(),
        ),
    );
    let ways_added = input_spinners.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template(
                "           {human_pos} ways collected so far for later processing",
            )
            .unwrap(),
        ),
    );

    reader
        .objects()
        .take_while(|o| o.is_node() || o.is_way())
        .par_bridge()
        .for_each_with(
            (
                nodeid_pos.clone(),
                nodeid_wayids.clone(),
                group_wayid_nodes.clone(),
            ),
            |(nodeid_pos, nodeid_wayids, group_wayid_nodes), o| {
                obj_reader.inc(1);
                match o {
                    ArcOSMObj::Node(n) => {
                        if args.read_nodes_first {
                            let ll = n.lat_lon_f64().unwrap();
                            nodeid_pos.lock().unwrap().insert(n.id(), (ll.1, ll.0));
                        }
                    }
                    ArcOSMObj::Way(w) => {
                        if args.tag_filter.par_iter().any(|tf| !tf.filter(&w)) {
                            return;
                        }
                        let group = args
                            .tag_group_k
                            .iter()
                            .map(|tg| tg.get_values(&w))
                            .collect::<Vec<Option<String>>>();
                        if !args.incl_unset_group && group.iter().any(|x| x.is_none()) {
                            return;
                        }

                        trace!("Got a way {}, in group {:?}", w.id(), group);
                        nodeid_wayids.lock().unwrap().insert_many(w.id(), w.nodes());
                        group_wayid_nodes
                            .lock()
                            .unwrap()
                            .entry(group)
                            .or_default()
                            .insert(w.id(), w.nodes().to_owned());
                        ways_added.inc(1);
                    }
                    ArcOSMObj::Relation(_r) => {}
                }
            },
        );
    obj_reader.finish();
    ways_added.finish();
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

    let nodeid_pos = if args.read_nodes_first {
        info!("Removing unneeded node positions...");
        let mut nodeid_pos = Arc::try_unwrap(nodeid_pos).unwrap().into_inner().unwrap();
        let old_total = nodeid_pos.len();
        nodeid_pos.retain_by_key(|nid| nodeid_wayids.contains_nid(nid));
        nodeid_pos.shrink_to_fit();
        info!(
            "Removed {} unneeded node positions, only keeping the {} we need",
            (old_total - nodeid_pos.len()),
            nodeid_pos.len()
        );
        nodeid_pos
    } else {
        debug!("Re-reading file to read all nodes");
        let setting_node_pos = ProgressBar::new(nodeid_wayids.len() as u64)
            .with_message("Re-reading file to save node locations")
            .with_style(style.clone());
        if !show_progress_bars {
            setting_node_pos.set_draw_target(ProgressDrawTarget::hidden());
        }
        let mut reader = osmio::read_pbf(&args.input_filename)?;
        reader
            .objects()
            .take_while(|o| o.is_node())
            .par_bridge()
            .for_each_with(nodeid_pos.clone(), |nodeid_pos, o| {
                if let Some(n) = o.into_node() {
                    if nodeid_wayids.contains_nid(&n.id()) {
                        let ll = n.lat_lon_f64().unwrap();
                        setting_node_pos.inc(1);
                        nodeid_pos.lock().unwrap().insert(n.id(), (ll.1, ll.0));
                    }
                }
            });

        setting_node_pos.finish();
        let nodeid_pos = Arc::try_unwrap(nodeid_pos).unwrap().into_inner().unwrap();
        nodeid_pos
    };

    debug!("{}", nodeid_pos.detailed_size());

    debug!(
        "Size of group_wayid_nodes: {} = {} bytes",
        group_wayid_nodes.get_size(),
        group_wayid_nodes
            .get_size()
            .to_formatted_string(&Locale::en)
    );

    info!("All data has been loaded. Started processing...");
    info!(
        "Starting the breathfirst search. There are {} groups",
        group_wayid_nodes.len()
    );
    let grouping = ProgressBar::new(
        group_wayid_nodes
            .values()
            .map(|wayid_nodes| wayid_nodes.par_iter().map(|(_k, v)| v.len()).sum::<usize>() as u64)
            .sum(),
    )
    .with_message("Grouping all ways")
    .with_style(style.clone());
    if !show_progress_bars {
        grouping.set_draw_target(ProgressDrawTarget::hidden());
    }

    let splitter = ProgressBar::new(0)
        .with_message("Splitting ways into lines")
        .with_style(style.clone());
    if !show_progress_bars {
        splitter.set_draw_target(ProgressDrawTarget::hidden());
    }

    group_wayid_nodes.into_par_iter()
    .flat_map(|(group, wayid_nodes)| {
        // Actually do the breath first search
        // TODO this is single threaded...
        trace!("Starting breath-first search for group {:?}", group);

        trace!("Starting to collect all the wayids into a set...");
        let mut unprocessed_wayids: BTreeSet<i64> = wayid_nodes.par_iter().map(|(k, _v)| k).copied().collect();
        trace!("... finished");
        let mut this_group_wayids = Vec::new();

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
                    wid, this_group.way_ids.len()
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
                            .flat_map(|nid| nodeid_wayids.ways(nid))
                        {
                            // If this other way hasn't been processed yet, then add to this group.
                            if unprocessed_wayids.remove(&other_wayid) {
                                grouping.inc(wayid_nodes[&other_wayid].len() as u64);
                                trace!("adding other way {}", other_wayid);
                                this_group_wayids.push(other_wayid);
                            }
                        }
                    });
            }

            way_groups.push(this_group);
        }
        trace!(
            "In total, found {} waygroups for the tag group {:?}",
            way_groups.len().to_formatted_string(&Locale::en),
            group
        );
        grouping.finish();
        // Ways with more nodes take longer to split, so the splitter progress bar is based on that
        splitter.set_length(way_groups.iter().map(|wg| wg.num_nodeids() as u64).sum());
        way_groups.into_par_iter()
    })
    // ↑ The breath first search is done

    // ↓ now do other processing on the groups
    .filter(|way_group| {
        if args.only_these_way_groups.is_empty() {
            true    // no filtering in operation
        } else {
            args.only_these_way_groups.par_iter().any(|only| *only == way_group.root_wayid)
        }
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
    .filter(|way_group|
        match args.min_length_m {
            None => true,
            Some(min_len) => way_group.length_m.unwrap() >= min_len,
        }
    )

    // ↓ Split into paths if needed
    .flat_map(|way_group| {
        let new_way_groups = if !args.split_into_single_paths {
            vec![way_group]
        } else {

            trace!("splitting the groups into single paths with FW algorithm...");
            let started = std::time::Instant::now();
            let paths = match fw::into_fw_segments(&way_group, &nodeid_pos, args.min_length_m, args.only_longest_n_splitted_paths) {
                Ok(paths) => {
                    let duration = (std::time::Instant::now() - started).as_secs_f64();
                    // Ways with more nodes take longer to split, so the splitter progress bar is based on that
                    splitter.inc(way_group.num_nodeids() as u64);
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
            let dist = haversine_m(line[0].0, line[0].1, line.last().unwrap().0, line.last().unwrap().1);
            way_group.extra_json_props["dist_ends_m"] = dist.into();
            way_group.extra_json_props["dist_ends_m_int"] = (dist.round() as i64).into();
            way_group.extra_json_props["dist_ends_km"] = (dist/1000.).into();
            way_group.extra_json_props["dist_ends_km_int"] = ((dist/1000.).round() as i64).into();
        }

    })

    // Group into files
    .fold(
        || HashMap::new() as HashMap<String, Vec<WayGroup>>,
        |mut files, way_group| {
            trace!("Grouping all data into files");
            files.entry(way_group.filename(&args.output_filename, args.split_files_by_group))
                .or_default()
                .push(way_group);
            files
    })
    // We might have many hashmaps now, group down to one
    .reduce(HashMap::new,
            |mut acc, curr| {
                trace!("Merging files down again");
                for (filename, wgs) in curr.into_iter() {
                    acc.entry(filename).or_default().extend(wgs.into_iter())
                }
                acc
            }
    )
    .into_par_iter()

    .update(|(_filename, way_groups)| {
        debug!("sorting ways by length & truncating");
        // in calc dist to longer, we need this sorted too
        way_groups.par_sort_by(|a, b| a.length_m.unwrap().total_cmp(&b.length_m.unwrap()).reverse());
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

            let prog = ProgressBar::new(way_groups.iter().map(|wg| wg.num_nodeids() as u64).sum())
                    .with_message("Calc distance to longer: Indexing data")
                    .with_style(style.clone());

            for (wg_id, coords) in way_groups.iter().enumerate().flat_map(|(wg_id, wg)| wg.coords_iter_seq().map(move |coords| (wg_id, coords)) ) {
                points_distance_idx.add(coords, wg_id).unwrap();
                prog.inc(1);
            }
            prog.finish();

            let prog = ProgressBar::new((way_groups.len()*way_groups.len()) as u64)
                    .with_message("Calc distance to longer")
                    .with_style(style.clone());
            // dist to larger
            let longers = way_groups.par_iter().enumerate()
                .map(|(wg_id, wg)| {
                    prog.inc(1);

                    // for each point what's the nearest other point that's in a longer other wayid
                    wg.coords_iter_par().map(|coord: [f64;2]| -> Option<(f64, i64)> {
                        let nearest_longer = points_distance_idx.iter_nearest(&coord, &haversine::haversine_m_arr).unwrap()
                            .filter(|(_dist, other_wg_id)| **other_wg_id != wg_id)
                            .filter(|(_dist, other_wg_id)| way_groups[**other_wg_id].length_m > way_groups[wg_id].length_m)
                            .next();
                        nearest_longer.map(|(dist, wgid)| (dist, way_groups[*wgid].root_wayid))
                    })
                    .filter_map(|x| x)
                    .min_by(|a, b| (a.0).total_cmp(&b.0))
                })
                .collect::<Vec<_>>();
            prog.finish();

            // set the longer distance
            way_groups.par_iter_mut().zip(longers)
                .for_each(|(wg, longer)| {
                    wg.extra_json_props["dist_to_longer_m"] = longer.map(|(dist, _)| dist).into();
                    wg.extra_json_props["nearest_longer_waygroup"] = longer.map(|(_dist, wgid)| wgid).into();
                });
        }
    })
    .update(|(_filename, way_groups)| {
        let mut feature_ranks = Vec::with_capacity(way_groups.len());

        // calc longest lengths
        // (length of way group, idx of this way group in way_groups, rank)
        way_groups.par_iter().enumerate()
            .map(|(i, wg)| (wg.length_m.unwrap(), i, 0))
            .collect_into_vec(&mut feature_ranks);
        // sort by longest first
        feature_ranks.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).reverse());
        // update feature_ranks to store the local rank
        feature_ranks.par_iter_mut().enumerate().for_each(|(rank, (_len, _idx, new_rank))| {
            *new_rank = rank;
        });
        // sort back by way_groups idx
        feature_ranks.par_sort_unstable_by_key(|(_len, wg_idx, _rank)| *wg_idx);
        // now update the way_groups
        let way_groups_len = way_groups.len();
        let way_groups_len_f = way_groups_len as f64;
        way_groups.par_iter_mut().zip(feature_ranks.par_iter()).for_each(|(wg, (_len, _wg_idx, rank))| {
            wg.extra_json_props["length_desc_rank"] = (*rank).into();
            wg.extra_json_props["length_desc_rank_perc"] = ((*rank as f64)/way_groups_len_f).into();
            wg.extra_json_props["length_asc_rank"] = (way_groups_len - *rank).into();
            wg.extra_json_props["length_asc_rank_perc"] = ((way_groups_len - *rank) as f64/way_groups_len_f).into();
        });

        if args.split_into_single_paths {
            // dist between ends
            feature_ranks.truncate(0);

            // (length of way group, idx of this way group in way_groups, rank)
            way_groups.par_iter().enumerate()
                .map(|(i, wg)| (wg.extra_json_props["dist_ends_m"].as_f64().unwrap(), i, 0)).collect_into_vec(&mut feature_ranks);
            // sort by longest first
            feature_ranks.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).reverse());
            // update feature_ranks to store the local rank
            feature_ranks.par_iter_mut().enumerate().for_each(|(rank, (_len, _idx, mut new_rank))| {
                new_rank = rank;
            });
            // sort back by way_groups idx
            feature_ranks.par_sort_unstable_by_key(|(_len, wg_idx, _rank)| *wg_idx);
            // now update the way_groups
            way_groups.par_iter_mut().zip(feature_ranks.par_iter()).update(|(wg, (_len, _wg_idx, rank))| {
                wg.extra_json_props["dist_ends_desc_rank"] = (*rank).into();
                wg.extra_json_props["dist_ends_asc_rank"] = (way_groups_len - *rank).into();
            });

        }

    })

    // ↓ convert to json objs
    .map(|(filename, way_groups)| {
        debug!("Convert to GeoJSON (ish)");
        let features = way_groups.into_par_iter().map(|mut w| {
            let mut properties = json!({
                "root_wayid": w.root_wayid,
                "root_wayid_120": w.root_wayid  % 120,
            });
            if let Some(_l) = w.length_m {
                properties["length_m"] = w.length_m.into();
            }
            if args.incl_wayids {
                properties["all_wayids"] = w.way_ids.into();
            }

            properties.as_object_mut().unwrap().append(w.extra_json_props.as_object_mut().unwrap());

            (properties, w.coords.unwrap())
            }).collect::<Vec<_>>();

        (filename, features)
    })
    .try_for_each(|(filename, features)| {
        debug!("Writing data to file(s)...");
        // Write the files
        match std::fs::File::create(&filename) {
            Ok(f) => {
                let mut f = std::io::BufWriter::new(f);
                write_geojson_features_directly(&features, &mut f, args.save_as_linestrings)
                    .with_context(|| {
                        format!(
                            "Writing {} features to filename {:?}",
                            features.len(),
                            filename
                        )
                    })?;
                info!("Wrote {} feature(s) to {}", features.len().to_formatted_string(&Locale::en), filename);
            }
            Err(e) => {
                warn!("Couldn't open filename {:?}: {}", filename, e);
            }
        }
        Ok(()) as Result<()>
    })?;

    info!("post processing way_groups");

    info!("Finished");
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

impl TagGrouper {
    pub fn get_values(&self, o: &impl osmio::OSMObjBase) -> Option<String> {
        for k in self.0.iter() {
            if let Some(v) = o.tag(k) {
                return Some(v.to_string());
            }
        }

        None
    }
}

/// Write a geojson featurecollection, but manually construct it
fn write_geojson_features_directly(
    features: &[(serde_json::Value, Vec<Vec<(f64, f64)>>)],
    mut f: &mut impl Write,
    save_as_linestrings: bool,
) -> Result<()> {
    f.write_all(b"{\"type\":\"FeatureCollection\", \"features\": [\n")?;
    write_geojson_feature_directly(&mut f, &features[0], save_as_linestrings)?;
    for feature in &features[1..] {
        f.write_all(b",\n")?;
        write_geojson_feature_directly(&mut f, feature, save_as_linestrings)?;
    }
    f.write_all(b"\n]}")?;
    Ok(())
}

fn write_geojson_feature_directly(
    mut f: &mut impl Write,
    feature: &(serde_json::Value, Vec<Vec<(f64, f64)>>),
    save_as_linestrings: bool,
) -> Result<()> {
    if save_as_linestrings {
        for (k, linestring) in feature.1.iter().enumerate() {
            if k != 0 {
                f.write_all(b",\n")?;
            }
            f.write_all(b"{\"properties\":")?;
            serde_json::to_writer(&mut f, &feature.0)?;
            f.write_all(b", \"geometry\": {\"type\":\"LineString\", \"coordinates\": ")?;
            f.write_all(b"[")?;
            for (j, coords) in linestring.iter().enumerate() {
                if j != 0 {
                    f.write_all(b",")?;
                }
                write!(f, "[{}, {}]", coords.0, coords.1)?;
            }
            f.write_all(b"]")?;
            f.write_all(b"}, \"type\": \"Feature\"}")?;
        }
    } else {
        f.write_all(b"{\"properties\":")?;
        serde_json::to_writer(&mut f, &feature.0)?;
        f.write_all(b", \"geometry\": {\"type\":\"MultiLineString\", \"coordinates\": ")?;
        f.write_all(b"[")?;
        for (i, linestring) in feature.1.iter().enumerate() {
            if i != 0 {
                f.write_all(b",")?;
            }
            f.write_all(b"[")?;
            for (j, coords) in linestring.iter().enumerate() {
                if j != 0 {
                    f.write_all(b",")?;
                }
                write!(f, "[{}, {}]", coords.0, coords.1)?;
            }
            f.write_all(b"]")?;
        }
        f.write_all(b"]")?;
        f.write_all(b"}, \"type\": \"Feature\"}")?;
    }

    Ok(())
}

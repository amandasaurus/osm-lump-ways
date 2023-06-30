#![allow(warnings)]
use anyhow::{Context, Result};
use clap::Parser;
use dashmap::DashMap;
use get_size::GetSize;
use indicatif::{ProgressBar, ProgressStyle};
#[allow(unused_imports)]
use log::{debug, error, info, log_enabled, trace, warn, Level};
use osmio::obj_types::ArcOSMObj;
use osmio::prelude::*;
use osmio::OSMObjBase;
#[allow(unused_imports)]
use rayon::prelude::*;
use regex::Regex;
use serde_json::json;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::prelude::*;
use std::io::Write;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
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
mod nodeid_wayids;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = cli_args::Args::parse();
    let mut reader = read_progress::BufReaderWithSize::from_path(&args.input_filename)?;
    let mut reader = osmio::pbf::PBFReader::new(reader);

    if args.split_files_by_group && !args.output_filename.contains("%s") {
        error!("No %s found in output filename ({})", args.output_filename);
        anyhow::bail!("No %s found in output filename ({})", args.output_filename);
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
    info!(
        "Tag filter(s) in operation: {:?}",
        args.tag_filter
    );
    info!("Tag grouping(s) in operation: {:?}", args.tag_group_k);

    /// For each group, a hashmap of wayid:nodes in that way
    let mut group_wayid_nodes: Arc<DashMap<Vec<Option<String>>, HashMap<i64, Vec<i64>>>> =
        Arc::new(DashMap::new());

    let mut nodeid_pos: Arc<DashMap<i64, (f64, f64)>> = Arc::new(DashMap::new());
    /// nodeid:the ways that contain that node
    //let mut nodeid_wayids: HashMap<i64, HashSet<i64>> = HashMap::new();
    let mut nodeid_wayids = nodeid_wayids::NodeIdWayIds::new();
    let nodeid_wayids = Arc::new(Mutex::new(nodeid_wayids));

    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] {percent}% done. eta {eta:>4} {bar:10.cyan/blue} {pos:>7}/{len:7} {per_sec:>12} {msg}",
    )
    .unwrap();
    let input_spinners = indicatif::MultiProgress::new();
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
            ProgressStyle::with_template("{human_pos} ways collected so far for later processing")
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
                            nodeid_pos.insert(n.id(), (ll.1, ll.0));
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
                        for nid in w.nodes() {
                            nodeid_wayids.lock().unwrap().insert(*nid, w.id());
                        }
                        group_wayid_nodes
                            .entry(group)
                            .or_default()
                            .insert(w.id(), w.nodes().to_owned());
                        ways_added.inc(1);
                    }
                    ArcOSMObj::Relation(_r) => {
                        return;
                    }
                }
            },
        );
    obj_reader.finish();
    ways_added.finish();
    let nodeid_wayids = Arc::try_unwrap(nodeid_wayids)
        .unwrap()
        .into_inner()
        .unwrap();

    let mut nodeid_pos = Arc::try_unwrap(nodeid_pos).unwrap();
    let mut group_wayid_nodes = Arc::try_unwrap(group_wayid_nodes).unwrap();

    if group_wayid_nodes.is_empty() {
        info!("No ways in the file matched your filters. Nothing to do");
        return Ok(());
    }

    let mut nodeid_pos_hm = HashMap::new();
    if args.read_nodes_first {
        info!("Removing unneeded node positions...");
        // free up memory we don't need by deleting node positions we don't need
        let old_total = nodeid_pos.len();
        nodeid_pos_hm = nodeid_pos
            .into_par_iter()
            .filter_map(|(nid, pos)| {
                if nodeid_wayids.contains_nid(&nid) {
                    Some((nid, pos))
                } else {
                    None
                }
            })
            .collect();
        nodeid_pos_hm.shrink_to_fit();
        info!(
            "Removed {} unneeded node positions, only keeping the {} we need",
            (old_total - nodeid_pos_hm.len()),
            nodeid_pos_hm.len()
        );
    } else {
        debug!("Re-reading file to read all nodes");
        let setting_node_pos = ProgressBar::new(nodeid_wayids.len() as u64)
            .with_message("Re-reading file to save node locations")
            .with_style(style.clone());
        let mut reader = osmio::read_pbf(&args.input_filename)?;
        nodeid_pos_hm = reader
            .objects()
            .take_while(|o| o.is_node())
            .par_bridge()
            .filter_map(|o| {
                if let Some(n) = o.into_node() {
                    if nodeid_wayids.contains_nid(&n.id()) {
                        let ll = n.lat_lon_f64().unwrap();
                        setting_node_pos.inc(1);
                        Some((n.id(), (ll.1, ll.0)))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        setting_node_pos.finish();
    }

    nodeid_pos_hm.shrink_to_fit();
    let nodeid_pos = nodeid_pos_hm;
    info!(
        "Size of node pos: {} bytes",
        nodeid_pos.get_size().to_formatted_string(&Locale::en)
    );

    info!(
        "Size of nodeid_wayids: {} = {} bytes",
        nodeid_wayids.get_size(),
        nodeid_wayids.get_size().to_formatted_string(&Locale::en)
    );

    info!(
        "Starting the breathfirst search. There are {} groups",
        group_wayid_nodes.len()
    );
    let grouping = ProgressBar::new(group_wayid_nodes.len() as u64)
        .with_message("Grouping all ways")
        .with_style(style.clone());

    let results_filename_way_groups = group_wayid_nodes.into_par_iter()
    .flat_map(|(group, wayid_nodes)| {
        // Actually do the breath first search
        grouping.inc(1);

        let mut unprocessed_wayids: BTreeSet<&i64> = wayid_nodes.keys().collect();
        let mut this_group_wayids = Vec::new();

        let mut way_groups = Vec::new();
        debug!("grouping all the ways for group: {:?}", group);
        while let Some(root_wayid) = unprocessed_wayids.pop_first() {
            this_group_wayids.push(root_wayid);

            let mut this_group = WayGroup::new(*root_wayid, group.to_owned());
            debug!(
                "root_wayid {:?} (there are {} unprocessed ways left)",
                root_wayid,
                unprocessed_wayids.len()
            );
            while let Some(wid) = this_group_wayids.pop() {
                trace!("The way to look at is {}", wid);
                trace!(
                    "Currently there are {} ways on in this group",
                    this_group.way_ids.len()
                );

                this_group.way_ids.push(*wid);
                this_group.nodeids.push(wayid_nodes[&wid].clone());

                // find all other ways
                for other_wayid in wayid_nodes[wid]
                    .iter()
                    .flat_map(|nid| nodeid_wayids.ways(nid))
                {
                    if unprocessed_wayids.remove(other_wayid) {
                        trace!("adding other way {}", other_wayid);
                        this_group_wayids.push(other_wayid);
                    }
                }
            }

            way_groups.push(this_group);
        }
        debug!(
            "In total, found {} waygroups for the tag group {:?}",
            way_groups.len(),
            group
        );
        grouping.finish();
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
        debug!("Saving coordinates for all ways");
        way_group.set_coords(&nodeid_pos);
    })
    .update(|way_group| {
        debug!("Calculating all lengths");
        way_group.calculate_length();
    })
    // apply min length filter
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

            debug!("splitting the groups into single paths with FW algorithm...");
            let paths = match fw::into_fw_segments(&way_group, &nodeid_pos, args.min_length_m, args.only_longest_n_splitted_paths) {
                Ok(paths) => {
                    debug!("Have generated {} paths from wg:{}", paths.len(), way_group.root_wayid);
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
        debug!("Preparing extra json properties");
        for (i, group) in way_group.group.iter().enumerate() {
            way_group.extra_json_props[format!("tag_group_{}", i)] = group.as_ref().map(|s| s.clone()).into();
        }
        way_group.extra_json_props["tag_groups"] = way_group.group.clone().into();
        way_group.extra_json_props["length_m_int"] = way_group.length_m.map(|l| l.round() as i64).into();
        way_group.extra_json_props["length_km"] = way_group.length_m.map(|l| l/1000.).into();
        way_group.extra_json_props["length_km_int"] = way_group.length_m.map(|l| l/1000.).map(|l| l.round() as i64).into();
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
        |mut files, (way_group)| {
            debug!("Grouping all data into files");
            files.entry(way_group.filename(&args.output_filename, args.split_files_by_group))
                .or_default()
                .push(way_group);
            files
    })
    // We might have many hashmaps now, group down to one
    .reduce(|| HashMap::new(),
            |mut acc, curr| {
                debug!("Merging files down again");
                for (filename, wgs) in curr.into_iter() {
                    acc.entry(filename).or_default().extend(wgs.into_iter())
                }
                acc
            }
    )
    .into_par_iter()

    .update(|(filename, way_groups)| {
        debug!("sorting ways by length & truncating");
        // in calc dist to longer, we need this sorted too
        way_groups.par_sort_by(|a, b| a.length_m.unwrap().total_cmp(&b.length_m.unwrap()).reverse());
    })
    .update(|(filename, way_groups)| {
        if let Some(limit) = args.only_longest_n_per_file {
            debug!("Truncating files by longest");
            way_groups.truncate(limit);
        }
    })
    .update(|(filename, way_groups)| {
        let max_timeout_s = args.timeout_dist_to_longer_s.unwrap_or(0.);
        if max_timeout_s == 0. {
            trace!("timeout_dist_to_longer_s is 0, so skipping this");
            return;
        }
        debug!("Calculating the distance to the nearest longer object per way");
        let started_processing = Instant::now();

        let prog = ProgressBar::new((way_groups.len()*way_groups.len()) as u64)
                .with_message("Calc distance to longer")
                .with_style(style.clone());
        // dist to larger
        let longers = way_groups.par_iter().enumerate()
            .map(|(i, wg)| {
                if ((Instant::now() - started_processing).as_secs_f32() > max_timeout_s) {
                    info!("Timeout calculating distance to nearer!");
                    return None;
                }
                // we know way_groups is sorted by dist above
                let nearest_longer = way_groups[0..i].par_iter()
                    .inspect(|_| prog.inc(1))
                    .filter(|wg2| wg != *wg2)
                    .filter(|wg2| wg2.length_m > wg.length_m)

                    // Calc distance
                    .map(|wg2| wg.distance_m(wg2).unwrap() )
                    .min_by(|d1, d2| d1.total_cmp(&d2));

                nearest_longer
            })
            .collect::<Vec<_>>();
        prog.finish();

        if ((Instant::now() - started_processing).as_secs_f32() > 2.) {
            // hack to discover if we timed out or not
            // set all to null
            way_groups.par_iter_mut().update(|wg| {
                wg.extra_json_props["dist_to_longer_m"] = None::<Option<f64>>.into();
            });
        } else {
            // set the longer distance
            way_groups.par_iter_mut().zip(longers)
                .for_each(|(mut wg, longer)| {
                    wg.extra_json_props["dist_to_longer_m"] = longer.into();
                });
        }
    })
    .update(|(filename, way_groups)| {
        let mut feature_ranks = Vec::with_capacity(way_groups.len());

        // calc longest lengths
        // (length of way group, idx of this way group in way_groups, rank)
        way_groups.par_iter().enumerate()
            .map(|(i, wg)| (wg.length_m.unwrap(), i, 0))
            .collect_into_vec(&mut feature_ranks);
        // sort by longest first
        feature_ranks.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).reverse());
        // update feature_ranks to store the local rank
        feature_ranks.par_iter_mut().enumerate().for_each(|(rank, (_len, idx, new_rank))| {
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
            feature_ranks.par_iter_mut().enumerate().update(|(rank, (_len, idx, mut new_rank))| {
                new_rank = *rank;
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
            if let Some(l) = w.length_m {
                properties["length_m"] = w.length_m.into();
            }
            if args.incl_wayids {
                properties["all_wayids"] = w.way_ids.into();
            }

            properties.as_object_mut().unwrap().append(&mut w.extra_json_props.as_object_mut().unwrap());

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
                info!("Wrote {} feature(s) to {}", features.len(), filename);
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
    write_geojson_feature_directly(&mut f, &features[0], save_as_linestrings);
    for feature in &features[1..] {
        f.write_all(b",\n")?;
        write_geojson_feature_directly(&mut f, &feature, save_as_linestrings)?;
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

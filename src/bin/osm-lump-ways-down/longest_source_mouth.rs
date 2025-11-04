use super::*;
use anyhow::Result;
#[allow(unused_imports)]
use log::{
    Level::{Debug, Trace},
    debug, error, info, log, trace, warn,
};
use std::collections::HashMap;
use std::path::Path;

use itertools::Itertools;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

use graph::DirectedGraphTrait;
use nodeid_position::NodeIdPosition;
use osm_lump_ways::graph;
use osm_lump_ways::inter_store;
use osm_lump_ways::nodeid_position;

use super::{EdgeProperty, TagGroupInfo, VertexProperty};
use serde_json::json;

#[allow(clippy::too_many_arguments)]
pub(crate) fn do_longest_source_mouth(
    output_filename: &Path,
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    nodeid_pos: &impl NodeIdPosition,
    inter_store: &inter_store::InterStore,
    tag_group_info: &[TagGroupInfo],
    tag_group_value: &[String],
    min_length_m: f64,
    only_named: bool,
    longest_source_mouth_longest_n: Option<usize>,
    longest_source_mouth_unnamed_string: &str,
) -> Result<()> {
    // Calc all mouth nids
    let mut mouths: Vec<i64> = g
        .vertexes_par_iter()
        .filter(|nid| {
			( g.num_out_neighbours(*nid).unwrap() == 0 )

			// If there is a unnamed section downstream of a named river, we want to include
			// the last point of the named river as a possible “mouth”
			|| ( only_named && g.out_neighbours_w_prop(*nid).all(|(_nid, eprop)| tag_group_info[eprop.taggroupid_us()].tagid.is_none()) )
		})
        .collect();
    info!(
        "There are {} mouths in total",
        mouths.len().to_formatted_string(&Locale::en)
    );

    // We can remove any mouth where total upstream is below our min, that'll definitely never be
    // OK
    mouths.retain(|nid| g.vertex_property_unchecked(nid).upstream_m >= min_length_m);

    // Calc longest upstream line per mouth
    // Any edges, which are in a taggroup, which has a parent channel, is not included.
    // This allows us to ignore all side channels
    let mut longest_mouth_source_per_mouth: Vec<_> = mouths
        .into_par_iter()
        .filter_map(|mouth_nid| {
            longest_upstream_path(g, mouth_nid, min_length_m, |(nid1, nid2)| {
                let tg = &tag_group_info[g.edge_property_unchecked((nid1, nid2)).taggroupid_us()];
                let has_name = tg.tagid.is_some();

                !(only_named && !has_name)        // bit tricky, but this is the boolean
                                                  // expression
                        && tg.parent_channels.is_empty()
            })
        })
        .collect();

    longest_mouth_source_per_mouth.par_sort_by_key(|(length_m, _nids)| -*length_m);
    if let Some(longest_source_mouth_longest_n) = longest_source_mouth_longest_n {
        longest_mouth_source_per_mouth.truncate(longest_source_mouth_longest_n);
    }

    let names = longest_mouth_source_per_mouth
        .into_iter()
        .flat_map(|(_length_m, nids)| {
            let grouped_names = group_path_parts_by_name(&nids, g, tag_group_info, tag_group_value);
            name_group_to_geojson(
                grouped_names,
                g,
                nodeid_pos,
                inter_store,
                longest_source_mouth_unnamed_string,
            )
        });

    let output_format = fileio::format_for_filename(output_filename);
    let mut f = BufWriter::new(File::create(output_filename)?);

    let num_written = fileio::write_geojson_features_directly(names, &mut f, &output_format)?;
    info!(
        "Wrote {} longest-source-mouth geometries to {}",
        num_written,
        output_filename.display()
    );

    Ok(())
}

/// Calculate the longest upstream path starting at a certain edge.
fn longest_upstream_path(
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    mouth_nid: i64,
    min_length_m: f64,

    follow_edge: impl Fn((i64, i64)) -> bool,
) -> Option<(OrderedFloat<f64>, Box<[i64]>)> {
    // do flood fill upwards
    let prev_dist = dij_flood_fill_upwards(g, mouth_nid, follow_edge);

    let (source_nid, (prev, dist)) = prev_dist
        .par_iter()
        .max_by_key(|(_nid, (_prev, dist))| dist)
        .unwrap();
    if *dist < OrderedFloat(min_length_m) {
        return None;
    }
    // build the path
    let mut path: Vec<i64> = Vec::new();
    path.push(*source_nid);
    path.push(prev.unwrap());
    while *path.last().unwrap() != mouth_nid {
        path.push(prev_dist.get(path.last().unwrap()).unwrap().0.unwrap());
    }
    let path = path.into_boxed_slice();

    Some((*dist, path))
}

fn dij_flood_fill_upwards(
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    mouth_nid: i64,
    follow_edge: impl Fn((i64, i64)) -> bool,
) -> HashMap<i64, (Option<i64>, OrderedFloat<f64>)> {
    let mut prev_dist: HashMap<i64, (Option<i64>, OrderedFloat<f64>)> = HashMap::new();
    prev_dist.insert(mouth_nid, (None, OrderedFloat(0.)));

    let mut frontier = BinaryHeap::new();
    // hack to just store negative distance so the shortest distance is the largest number
    frontier.push((OrderedFloat(-0.), mouth_nid));

    let mut this_dist;
    while let Some((mut curr_dist, curr_id)) = frontier.pop() {
        // curr_dist is the distance from the start point to this
        curr_dist *= -1.;
        if curr_dist > prev_dist[&curr_id].1 {
            // already found a shorter
            continue;
        }
        for neighbor in g.in_neighbours(curr_id) {
            if !follow_edge((neighbor, curr_id)) {
                continue;
            }
            this_dist =
                curr_dist + OrderedFloat(g.edge_property_unchecked((neighbor, curr_id)).length_m);
            prev_dist
                .entry(neighbor)
                .and_modify(|(prev_id, dist)| {
                    if this_dist < *dist {
                        *prev_id = Some(curr_id);
                        *dist = this_dist;
                        frontier.push((-this_dist, neighbor));
                    }
                })
                .or_insert_with(|| {
                    frontier.push((-this_dist, neighbor));
                    (Some(curr_id), this_dist)
                });
        }
    }

    prev_dist
}

fn group_path_parts_by_name(
    path: &[i64],
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    tag_group_info: &[TagGroupInfo],
    tag_group_value: &[String],
) -> Vec<(Option<String>, Box<[i64]>)> {
    let mut names: Vec<(Option<String>, Box<[i64]>)> = Vec::new();

    for (_name, mut chunk) in &path
        .iter()
        .tuple_windows()
        .map(|(nid1, nid2)| {
            (
                (nid1, nid2),
                tag_group_info[g.edge_property_unchecked((*nid1, *nid2)).taggroupid as usize]
                    .tagid
                    .map(|tgid| tag_group_value[tgid as usize].as_str()),
            )
        })
        .chunk_by(|(_seg, name)| *name)
    {
        let mut coords: Vec<i64> = Vec::new();
        let (first_seg, name) = chunk.next().unwrap();
        coords.push(*first_seg.0);
        coords.push(*first_seg.1);
        coords.extend(chunk.map(|(seg, _name)| *seg.1));
        names.push((name.map(String::from), coords.into_boxed_slice()));
    }

    names
}

fn name_group_to_geojson(
    names: Vec<(Option<String>, Box<[i64]>)>,
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    nodeid_pos: &impl NodeIdPosition,
    inter_store: &inter_store::InterStore,
    unnnamed_string: &str,
) -> impl Iterator<Item = (serde_json::Value, Vec<(f64, f64)>)> {
    let all_names: Vec<String> = names
        .iter()
        .rev()
        .cloned()
        .map(|(name, _nids)| name.map_or_else(|| unnnamed_string.to_string(), |s| s.to_string()))
        .collect();
    let names = names
        .into_iter()
        .map(|(name, nids)| {
            let length_m: f64 = nids
                .iter()
                .tuple_windows()
                .map(|(nid1, nid2)| g.edge_property_unchecked((*nid1, *nid2)).length_m)
                .sum();
            (name, nids, length_m)
        })
        .collect::<Vec<_>>();
    let num_parts = names.len();
    let total_length_m = names.iter().map(|(_, _, length_m)| length_m).sum::<f64>();
    let mouth_nid: i64 = names[0].1[0];
    let source_nid: i64 = *names.last().unwrap().1.last().unwrap();

    names
        .into_iter()
        .enumerate()
        .map(move |(idx, (name, nids, length_m))| {
            let props = json!({
                "name": name,
                "length_m": round(&length_m, 1),
                "idx": idx,
                "revidx": (num_parts - idx - 1),
                "num_parts": num_parts,

                "river_system_length_m": round(&total_length_m, 1),
                "river_system_names": all_names,
                "river_system_names_s": all_names.join(" - "),
                "river_system_mouth_nid": mouth_nid, "river_system_source_nid": source_nid,
                "river_system_mouth_source_nids": (mouth_nid, source_nid),
                "river_system_mouth_source_nids_s": format!("{},{}", mouth_nid, source_nid),
            });
            let line: Vec<(f64, f64)> = inter_store
                .expand_line_directed(&nids)
                .map(|nid| nodeid_pos.get(&nid).unwrap())
                .collect();
            (props, line)
        })
}

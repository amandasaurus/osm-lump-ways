use super::*;
use osm_lump_ways::get_two_muts;
use osm_lump_ways::sorted_slice_store::SortedSliceMap;
use osm_lump_ways::utils::min_max;

macro_rules! sort_dedup {
    ($item:expr) => {
        $item.par_sort_unstable();
        $item.dedup();
    };
}

/// When grouping rivers, this represents one river
#[derive(Debug, Clone)]
pub struct TagGroupInfo {
    /// Index in the tag value
    /// None → This segment doesn't have a tag value
    pub tagid: Option<u32>,

    pub min_nid: i64,
    pub upstream_m: f64,
    pub length_m: f64,

    pub unallocated_other_groups: SmallVec<[u64; 1]>,
    pub branching_distributaries: SmallVec<[u64; 1]>,
    pub terminal_distributaries: SmallVec<[u64; 1]>,
    pub sibling_distributaries: SmallVec<[u64; 1]>,
    pub tributaries: SmallVec<[u64; 1]>,
    pub parent_channels: SmallVec<[u64; 1]>,
    pub side_channels: SmallVec<[u64; 1]>,
    pub parent_rivers: SmallVec<[u64; 1]>,

    /// nids where this taggroup joins another. This is either a tributary or distributary
    pub confluences: SmallVec<[i64; 2]>,

    /// nids where a waterway starts
    pub sources: SmallVec<[i64; 1]>,
    /// nids where a waterway ends
    pub sinks: SmallVec<[i64; 1]>,

    pub stream_level: u64,
    pub stream_level_code: SmallVec<[u32; 3]>,

    pub end_segments: SmallVec<[(i64, i64); 3]>,

    pub confluence_distances: SortedSliceMap<(i64, i64), f64>,
}
impl TagGroupInfo {
    fn from_tagid(tagid: Option<u32>) -> Self {
        TagGroupInfo {
            tagid,
            ..Default::default()
        }
    }
    pub fn stream_level_code_str(&self) -> String {
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
            length_m: 0.,
            stream_level_code: smallvec![],
            tagid: None,
            end_segments: smallvec![],
            min_nid: i64::MAX,
            confluence_distances: SortedSliceMap::from_iter(std::iter::empty()),
        }
    }
}

pub fn calc_tag_group(
    g: &mut impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    new_progress_bar_func: impl Fn(u64, &str) -> ProgressBar,
) -> Box<[TagGroupInfo]> {
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
    for (nid1, nid2, eprop) in get_ends_bar.wrap_iter(g.edges_iter_w_prop()) {
        let seg = (nid1, nid2);
        outgoing_groups.truncate(0);
        outgoing_groups.extend(
            g.out_edges_w_prop(nid2)
                .map(|(_nid, _out, eprop)| eprop.tagid),
        );
        outgoing_groups.dedup();
        this_group = eprop.tagid;

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

    // Step 2: Group all the segments based on topological connectivness
    // (yes this is like osm-lump-ways)
    // for each segment, assign it to a group id
    let mut frontier: VecDeque<_> = VecDeque::new();
    let assign_to_group =
        new_progress_bar_func(g.num_edges() as u64, "Assigning each segment to an end");
    for end_segment in tag_group_ends.iter() {
        let eprop = g.edge_property_unchecked(*end_segment);
        if eprop.taggroupid != u64::MAX {
            // already assigned to a group
            continue;
        }
        let this_tag_id: Option<u32> = eprop.tagid;
        let mut this_tag_group = TagGroupInfo::from_tagid(this_tag_id);
        this_tag_group.length_m += eprop.length_m;
        this_tag_group.end_segments.push(*end_segment);

        let curr_group_id = tag_group_info.len() as u64;
        frontier.truncate(0);
        frontier.push_back(*end_segment);
        while let Some(seg) = frontier.pop_front() {
            let seg_eprop = g.edge_property_unchecked(seg);
            if seg_eprop.tagid != this_tag_id {
                continue;
            }
            if seg_eprop.taggroupid != u64::MAX {
                continue; // already done
            }
            if tag_group_ends.contains(&seg) {
                this_tag_group.end_segments.push(seg);
                this_tag_group.end_segments.dedup();
            }

            // save this group id
            this_tag_group.length_m += seg_eprop.length_m;
            g.edge_property_mut(seg).taggroupid = curr_group_id;
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
        g.edges_iter_w_prop()
            .filter(|(_nid1, _nid2, eprop)| eprop.taggroupid == u64::MAX)
            .map(|(nid1, nid2, _eprop)| (nid1, nid2)), //nid_pair_to_taggroupid
                                                       //    .iter()
                                                       //    .filter(|(_seg, group_id)| *group_id == u64::MAX)
                                                       //    .map(|(seg, _)| seg)
                                                       //    .copied(),
    );
    while let Some(seg) = incomplete_segs.pop() {
        assert!(!tag_group_ends.contains(&seg));
        let this_tagid = g.edge_property_unchecked(seg).tagid.unwrap();
        possible_taggroupids.truncate(0);
        possible_taggroupids.extend(g.all_connected_edges(&seg).filter_map(|seg2| {
            let eprop = g.edge_property_unchecked(seg2);
            if eprop.tagid == Some(this_tagid) {
                Some(eprop.taggroupid)
            } else {
                None
            }
        }));
        sort_dedup!(possible_taggroupids);
        assert_eq!(possible_taggroupids.len(), 1);
        g.edge_property_mut(seg).taggroupid = possible_taggroupids[0];
    }
    assert_eq!(
        g.edges_par_iter_w_prop()
            .filter(|(_nid1, _nid2, eprop)| eprop.taggroupid == u64::MAX)
            .count(),
        0,
        "Some segments have not been assigned to a tagroup, total num segments {}",
        g.num_edges().to_formatted_string(&Locale::en)
    );

    let _groups_that_flow_into_nothing = segments_into_nothing
        .into_iter()
        .map(|seg| g.edge_property_unchecked(seg).taggroupid)
        .collect::<HashSet<u64>>();

    let mut tag_group_info = tag_group_info.into_boxed_slice();

    info!(
        "There are {} different groups of connected named ways",
        tag_group_info.len().to_formatted_string(&Locale::en)
    );

    // calculate combined upstream per group
    for seg in tag_group_ends.iter() {
        let group = g.edge_property_unchecked(*seg).taggroupid;
        // TODO need to include last segment?
        if !g.contains_edge(seg.0, seg.1) {
            warn!("No upstream for {:?}", seg);
        }
        tag_group_info[group as usize].upstream_m += g.edge_property_unchecked(*seg).upstream_m;
    }

    let mut tgs_that_join = Vec::new();
    // For every taggroup, calculate the tributaries, distributaries etc.
    for (nid1, nid2, eprop) in g.edges_iter_w_prop() {
        let taggroupid = eprop.taggroupid;
        let tg = &mut tag_group_info[taggroupid as usize];
        if g.num_in_neighbours(nid1).unwrap() == 0 {
            tg.sources.push(nid1);
        }
        if g.num_out_neighbours(nid2).unwrap() == 0 {
            tg.sinks.push(nid2);
        }

        for (_nid2, _nid3, eprop2) in g
            .out_edges_w_prop(nid2)
            .filter(|(_, _, eprop2)| eprop2.taggroupid != taggroupid)
        {
            tg.confluences.push(nid2);
            tg.unallocated_other_groups.push(eprop2.taggroupid);
            let (a, b) = min_max(taggroupid, eprop2.taggroupid);
            tgs_that_join.push((a, b));
        }
        for (_nid0, _nid1, eprop0) in g
            .in_edges_w_prop(nid1)
            .filter(|(_, _, eprop0)| eprop0.taggroupid != taggroupid)
        {
            tg.confluences.push(nid1);
            tg.unallocated_other_groups.push(eprop0.taggroupid);
            let (a, b) = min_max(taggroupid, eprop0.taggroupid);
            tgs_that_join.push((a, b));
        }
    }
    tag_group_info.par_iter_mut().for_each(|tg| {
        sort_dedup!(tg.unallocated_other_groups);
    });

    sort_dedup!(tgs_that_join);

    let total_num_groups = tgs_that_join.len();
    let mut num_unable_to_deduce = 0;

    for (a_id, b_id) in tgs_that_join.into_iter() {
        let rr = calc_river_relationship(g, &tag_group_info, &(a_id, b_id));
        //dbg!(rr.is_some());
        if rr.is_none() {
            num_unable_to_deduce += 1;
            continue;
        }
        let (rr, a_id, b_id) = rr.unwrap();
        let (a, b) = if a_id < b_id {
            get_two_muts(&mut tag_group_info, a_id as usize, b_id as usize)
        } else {
            let (x, y) = get_two_muts(&mut tag_group_info, b_id as usize, a_id as usize);
            (y, x)
        };
        a.unallocated_other_groups.retain(|x| *x != b_id);
        b.unallocated_other_groups.retain(|x| *x != a_id);
        match rr {
            RiverRelationship::AIsSideChannelOfB => {
                a.parent_channels.push(b_id);
                b.side_channels.push(a_id);
            }
            RiverRelationship::AIsTributaryOfB => {
                a.parent_rivers.push(b_id);
                b.tributaries.push(a_id);
            }
            RiverRelationship::AIsBranchingDistrubtoryOfB => {
                a.parent_rivers.push(b_id);
                b.branching_distributaries.push(a_id);
            }
            RiverRelationship::AIsTerminalDistrubtoryOfB => {
                a.parent_rivers.push(b_id);
                b.terminal_distributaries.push(a_id);
            }
        }
    }

    if num_unable_to_deduce > 0 {
        warn!(
            "Unable to deduce the river relationship for {} of river pairs. {:.2} % of total",
            num_unable_to_deduce,
            100. * (num_unable_to_deduce as f64) / (total_num_groups as f64)
        );
    }

    if tag_group_info.par_iter().any(|tg| !tg.unallocated_other_groups.is_empty()) {
        let (count, sum_length) = tag_group_info.par_iter().filter(|tg| !tg.unallocated_other_groups.is_empty()).map(|tg| (1, tg.length_m)).reduce(|| (0, 0.), |a, b| (a.0+b.0, a.1+b.1));
        let (total_count, total_sum_length) = tag_group_info.par_iter().map(|tg| (1, tg.length_m)).reduce(|| (0, 0.), |a, b| (a.0+b.0, a.1+b.1));
        warn!("Unable to connect up {count} of {total_count} ({count_per:.1}%) river pairs, representing {sum_length} km of {total_sum_length} km ({len_per:.1}%) rivers",
            count=count, total_count=total_count,
            count_per=(count as f64 * 100. / total_count as f64),
            sum_length=(sum_length.round() as u64/1000).to_formatted_string(&Locale::en),
            total_sum_length=(total_sum_length.round() as u64/1000).to_formatted_string(&Locale::en),
            len_per=(sum_length * 100. / total_sum_length),
        );
    }
    //assert_eq!(0, tag_group_info.par_iter().filter(|tg| !tg.unallocated_other_groups.is_empty()).count());

    // Below is the original code for deciding relationships.
    //tag_group_info
    //    .par_iter_mut()
    //    .enumerate()
    //    .filter(|(_taggroupid, tg)| !tg.unallocated_other_groups.is_empty())
    //    .for_each(|(taggroupid, tg)| {
    //        let taggroupid = taggroupid as u64;

    //        let mut confluences: SmallVec<[i64; 3]> = smallvec![]; // buffer
    //        let mut put_back_in: SmallVec<[_; 2]> = smallvec![];

    //        for other_taggroupid in tg.unallocated_other_groups.drain(..) {
    //            assert!(other_taggroupid != taggroupid);
    //            confluences.truncate(0);
    //            confluences.extend(
    //                tg.confluences
    //                    .iter()
    //                    .flat_map(|&nid| {
    //                        g.in_edges_w_prop(nid)
    //                            .filter(|(_, _, eprop)| eprop.taggroupid == other_taggroupid)
    //                    })
    //                    .map(|(_, nid2, _)| nid2),
    //            );
    //            confluences.extend(
    //                tg.confluences
    //                    .iter()
    //                    .flat_map(|&nid| {
    //                        g.out_edges_w_prop(nid)
    //                            .filter(|(_, _, eprop)| eprop.taggroupid == other_taggroupid)
    //                    })
    //                    .map(|(nid1, _, _)| nid1),
    //            );
    //            sort_dedup!(confluences);
    //            assert!(!confluences.is_empty());

    //            if confluences.len() >= 2
    //                && confluences.iter().any(|nid| {
    //                    flows_out(g, *nid, taggroupid)
    //                        && flows_through_or_in(g, *nid, other_taggroupid)
    //                })
    //                && confluences.iter().any(|nid| {
    //                    flows_in(g, *nid, taggroupid)
    //                        && flows_through_or_out(g, *nid, other_taggroupid)
    //                })
    //            {
    //                tg.parent_channels.push(other_taggroupid);
    //            } else if confluences.len() >= 2
    //                && confluences.iter().any(|nid| {
    //                    flows_through_or_in(g, *nid, taggroupid)
    //                        && flows_out(g, *nid, other_taggroupid)
    //                })
    //                && confluences.iter().any(|nid| {
    //                    flows_through_or_out(g, *nid, taggroupid)
    //                        && flows_in(g, *nid, other_taggroupid)
    //                })
    //            {
    //                tg.side_channels.push(other_taggroupid);
    //            } else if confluences
    //                .iter()
    //                .all(|nid| flows_in(g, *nid, other_taggroupid))
    //            {
    //                tg.tributaries.push(other_taggroupid);
    //            } else if confluences.iter().any(|nid| flows_in(g, *nid, taggroupid)) {
    //                tg.terminal_distributaries.push(other_taggroupid)
    //            } else if confluences.iter().all(|nid| {
    //                flows_out(g, *nid, taggroupid) && flows_through(g, *nid, other_taggroupid)
    //            }) {
    //                tg.parent_rivers.push(other_taggroupid)
    //            } else if confluences.iter().all(|nid| {
    //                flows_through(g, *nid, taggroupid) && flows_out(g, *nid, other_taggroupid)
    //            }) {
    //                tg.branching_distributaries.push(other_taggroupid)
    //            } else if confluences.iter().any(|nid| {
    //                flows_out(g, *nid, taggroupid) && flows_out(g, *nid, other_taggroupid)
    //            }) {
    //                tg.sibling_distributaries.push(other_taggroupid)
    //            } else {
    //                put_back_in.push(other_taggroupid);
    //            }
    //        }

    //        sort_dedup!(put_back_in);
    //        tg.unallocated_other_groups.extend(put_back_in);
    //    });

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
                .flat_map(|&nid| g.in_edges_w_prop(nid))
                .map(|(_nid1, _nid2, eprop)| eprop.taggroupid)
                .filter(|other_tgid| *other_tgid != tgid as u64)
                .filter(|other_tgid| !tag_group_info[*other_tgid as usize].has_stream_level())
                .dedup(),
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

    assert_sanity_check(&tag_group_info);
    info!("The stream level code string has been calculated for every group");
    info!(
        "Finished calculating all tag groups in {}",
        formatting::format_duration(started_calc.elapsed()),
    );

    calc_all_confluence_distances(&mut tag_group_info, g);

    tag_group_info
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum FlowType {
    In,
    Out,
    Through,
    No,
}
impl FlowType {
    fn out(&self) -> bool {
        *self == FlowType::Out
    }
    #[allow(unused)]
    fn through(&self) -> bool {
        *self == FlowType::Through
    }
    fn in_(&self) -> bool {
        *self == FlowType::In
    }
    fn out_or_through(&self) -> bool {
        match self {
            FlowType::Out | FlowType::Through => true,
            FlowType::In | FlowType::No => false,
        }
    }
    fn in_or_through(&self) -> bool {
        match self {
            FlowType::In | FlowType::Through => true,
            FlowType::Out | FlowType::No => false,
        }
    }
    #[allow(unused)]
    fn code(&self) -> char {
        match self {
            FlowType::In => 'I',
            FlowType::Out => 'O',
            FlowType::Through => 'T',
            FlowType::No => 'N',
        }
    }
}

fn flow_type(
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    nid: i64,
    group_id: u64,
) -> FlowType {
    let has_ins = g
        .in_edges_w_prop(nid)
        .any(|(_, _, eprop)| eprop.taggroupid == group_id);
    let has_outs = g
        .out_edges_w_prop(nid)
        .any(|(_, _, eprop)| eprop.taggroupid == group_id);
    match (has_ins, has_outs) {
        (true, true) => FlowType::Through,
        (true, false) => FlowType::In,
        (false, true) => FlowType::Out,
        (false, false) => FlowType::No,
    }
}

// Below is not used as much now.
//fn flows_out(
//    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
//    nid: i64,
//    group_id: u64,
//) -> bool {
//    flow_type(g, nid, group_id) == FlowType::Out
//}
//
//fn flows_out_or_through(
//    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
//    nid: i64,
//    group_id: u64,
//) -> bool {
//    match flow_type(g, nid, group_id) {
//        FlowType::Out | FlowType::Through => true,
//        FlowType::In | FlowType::No => false,
//    }
//}
//
//fn flows_in(
//    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
//    nid: i64,
//    group_id: u64,
//) -> bool {
//    flow_type(g, nid, group_id) == FlowType::In
//}
//fn flows_through(
//    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
//    nid: i64,
//    group_id: u64,
//) -> bool {
//    flow_type(g, nid, group_id) == FlowType::Through
//}
//fn flows_through_or_in(
//    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
//    nid: i64,
//    group_id: u64,
//) -> bool {
//    matches!(
//        flow_type(g, nid, group_id),
//        FlowType::Through | FlowType::In
//    )
//}
//fn flows_through_or_out(
//    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
//    nid: i64,
//    group_id: u64,
//) -> bool {
//    matches!(
//        flow_type(g, nid, group_id),
//        FlowType::Through | FlowType::Out
//    )
//}

fn assert_sanity_check(tag_group_info: &[TagGroupInfo]) {
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
}

pub fn calc_all_confluence_distances(
    tag_group_info: &mut [TagGroupInfo],
    g: &mut impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
) {
    tag_group_info
        .par_iter_mut()
        .enumerate()
        .filter(|(_taggroupid, tg)| tg.confluences.len() >= 2)
        .for_each(|(taggroupid, tg)| {
            let taggroupid = taggroupid as u64;
            let mut all_confluences = tg.confluences.clone();
            all_confluences.extend(tg.sinks.iter().copied());
            all_confluences.extend(tg.sources.iter().copied());
            all_confluences
                .par_sort_by_key(|nid| -OrderedFloat(g.vertex_property_unchecked(nid).upstream_m));
            all_confluences.dedup();
            //dbg!(all_confluences.last().unwrap() - all_confluences[0]);
            //dbg!(all_confluences.last().unwrap().cmp(&all_confluences[0]));

            let n: i64 = all_confluences.len() as i64;
            let distances: Vec<((i64, i64), f64)> = (0..n)
                .into_par_iter()
                .flat_map(|i| {
                    let nid1 = all_confluences[i as usize];
                    let downstreams = dij_flood_fill_downwards(g, nid1, taggroupid);
                    (0..n).into_par_iter().filter_map({
                        // FIXME I don't like this clone
                        let all_confluences = all_confluences.clone();
                        move |j| {
                            if i == j {
                                return None;
                            }
                            let nid2 = &all_confluences[j as usize].clone();
                            if let Some((_prev, dist)) = &downstreams.get(nid2) {
                                Some(((nid1, *nid2), dist.0))
                            } else {
                                None
                            }
                        }
                    })
                })
                .collect();

            //dbg!(distances.len());
            tg.confluence_distances = SortedSliceMap::from_vec(distances);
        });
}

fn dij_flood_fill_downwards(
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    source_nid: i64,
    taggroupid: u64,
) -> HashMap<i64, (Option<i64>, OrderedFloat<f64>)> {
    let mut prev_dist: HashMap<i64, (Option<i64>, OrderedFloat<f64>)> = HashMap::new();
    prev_dist.insert(source_nid, (None, OrderedFloat(0.)));

    let mut frontier = BinaryHeap::new();
    // hack to just store negative distance so the shortest distance is the largest number
    frontier.push((OrderedFloat(-0.), source_nid));

    let mut this_dist;
    while let Some((mut curr_dist, curr_id)) = frontier.pop() {
        // curr_dist is the distance from the start point to this
        curr_dist *= -1.;
        if curr_dist > prev_dist[&curr_id].1 {
            // already found a shorter
            continue;
        }
        for (neighbor_nid, eprop) in g
            .out_neighbours_w_prop(curr_id)
            .filter(|(_, eprop)| eprop.taggroupid == taggroupid)
        {
            this_dist = curr_dist + OrderedFloat(eprop.length_m);
            prev_dist
                .entry(neighbor_nid)
                .and_modify(|(prev_id, dist)| {
                    if this_dist < *dist {
                        *prev_id = Some(curr_id);
                        *dist = this_dist;
                        frontier.push((-this_dist, neighbor_nid));
                    }
                })
                .or_insert_with(|| {
                    frontier.push((-this_dist, neighbor_nid));
                    (Some(curr_id), this_dist)
                });
        }
    }

    prev_dist
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Debug, Clone)]
enum RiverRelationship {
    AIsTributaryOfB,
    AIsTerminalDistrubtoryOfB,
    AIsBranchingDistrubtoryOfB,
    AIsSideChannelOfB,
}

fn calc_river_relationship(
    g: &impl DirectedGraphTrait<VertexProperty, EdgeProperty>,
    taggroups: &[TagGroupInfo],
    rr: &(u64, u64),
) -> Option<(RiverRelationship, u64, u64)> {
    // need to be mut to swap later
    let &(mut a_id, mut b_id) = rr;
    let mut a = &taggroups[a_id as usize];
    let mut b = &taggroups[b_id as usize];

    let mut confluences = a
        .confluences
        .iter()
        .filter(|nid| b.confluences.contains(nid))
        .copied()
        .map(|nid| (flow_type(g, nid, a_id), flow_type(g, nid, b_id)))
        .collect::<Vec<_>>();
    assert!(!confluences.is_empty());

    let mut possible_res: SmallVec<[(RiverRelationship, u64, u64); 2]> = smallvec![];

    // do this exactly twice, with some reversing code at the end of the loop (i.e the middle)
    use FlowType::*;
    use RiverRelationship::*;
    for _step in [0, 1] {
        //dbg!(step, a_id, b_id);
        if confluences.iter().all(|c| c == &(In, Out)) {
            possible_res.push((AIsTerminalDistrubtoryOfB, a_id, b_id));
        }
        if confluences.iter().all(|c| c == &(In, Through)) {
            possible_res.push((AIsTributaryOfB, a_id, b_id));
        }
        if confluences.iter().all(|c| c == &(Out, Through)) {
            possible_res.push((AIsBranchingDistrubtoryOfB, a_id, b_id));
        }

        if is_side_channel_of(a, b, &confluences) {
            possible_res.push((AIsSideChannelOfB, a_id, b_id));
        }

        // flip it around and try again. but there is only 2 loops. NB: no `let` here!
        (a, b) = (b, a);
        (a_id, b_id) = (b_id, a_id);
        confluences = confluences
            .into_iter()
            .map(|(c_a, c_b)| (c_b, c_a))
            .collect();
    }

    if possible_res.is_empty() {
        // we try again for worse case scenarios
        for _step in [0, 1] {
            // a = small, unnamed, and only joins with b. so put it as trib.
            if a.length_m / b.length_m <= 0.1
                && a.tagid.is_none()
                && b.tagid.is_some()
                && a.confluences.len() == confluences.len()
                && confluences.iter().any(|&(flow_a, _)| flow_a.in_())
            {
                possible_res.push((AIsTributaryOfB, a_id, b_id));
            }

            // a named river that has no sinks, and just flows into b.
            if a.tagid.is_some()
                && b.tagid.is_some()
                && a.length_m < b.length_m
                && a.sinks.is_empty()
                && a.confluences
                    .iter()
                    .filter(|nid| flow_type(g, **nid, a_id).in_())
                    .all(|nid| b.confluences.contains(nid))
            {
                possible_res.push((AIsTributaryOfB, a_id, b_id));
            }

            // flip them around for 2nd go
            (a, b) = (b, a);
            (a_id, b_id) = (b_id, a_id);
            confluences = confluences
                .into_iter()
                .map(|(c_a, c_b)| (c_b, c_a))
                .collect();
        }
    }

    //if possible_res.len() != 1 {
    //    dbg!(possible_res.len());
    //    dbg!(a.length_m);
    //    dbg!(b.length_m);
    //    dbg!(a.length_m / b.length_m);
    //    dbg!(a.min_nid);
    //    dbg!(b.min_nid);
    //    dbg!(&possible_res);
    //    dbg!(
    //        a.confluences
    //            .iter()
    //            .filter(|nid| b.confluences.contains(nid))
    //            .collect::<Vec<_>>()
    //    );
    //    //dbg!(&a.confluences); dbg!(&b.confluences);
    //    dbg!(&confluences);
    //}
    if possible_res.is_empty() {
        debug!(
            "Unable to deduce river connection: {:?}",
            (a_id, b_id, confluences)
        );
    }

    possible_res.pop()
}

fn is_side_channel_of(
    a: &TagGroupInfo,
    b: &TagGroupInfo,
    confluences: &[(FlowType, FlowType)],
) -> bool {
    confluences.len() >= 2
        && a.length_m < b.length_m
        && confluences.iter().all(|&(flow_a, flow_b)| {
            (flow_a.out() && flow_b.in_or_through()) || (flow_a.in_() && flow_b.out_or_through())
        })

        // must be 1+ of each
        && confluences.iter().any(|&(flow_a, flow_b)| flow_a.out() && flow_b.in_or_through())
        && confluences.iter().any(|&(flow_a, flow_b)| flow_a.in_() && flow_b.out_or_through())
}

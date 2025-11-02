use clap::Parser;
use clap_verbosity_flag::Verbosity;
use std::path::PathBuf;
use std::str::FromStr;

use osm_lump_ways::dij::SplitPathsMethod;
use osm_lump_ways::tagfilter;

fn parse_int_human(input: &str) -> Result<usize, String> {
    if let Ok(res) = usize::from_str(input) {
        Ok(res)
    } else if let Some(res) = input.to_lowercase().strip_suffix("k")
        && let Ok(res) = usize::from_str(res)
    {
        Ok(res * 1_000)
    } else if let Some(res) = input.to_lowercase().strip_suffix("m")
        && let Ok(res) = usize::from_str(res)
    {
        Ok(res * 1_000_000)
    } else {
        Err(format!("Unable to parse {:?}", input))
    }
}

/// Group OSM ways based on shared tags into GeoJSON MultiLineStrings
///
/// Reads an OSM PBF file, and groups all connected ways together into a MultiLineString
///
/// Use `-f`/-`-tag-filter` to only include ways which match that tag filter
/// `-g`/`--tag-group-k` to group ways by connectiveness *and* whether that tag key is equal.
/// `--min-length-m` Only output way groups with a minimum length of this
/// `--only-longest-n-per-file NUM` Only output the longest `NUM` ways groups.
#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Input PBF filename
    #[arg(short, long, value_name = "FILENAME.osm.pbf")]
    pub input_filename: PathBuf,

    /// Calculated Frames to this file. Only GeoJSONSeq output supported.
    /// Respects the --save-as-linestrings and --frames-group-min-length-m options
    #[arg(long, value_name = "OUTPUT.geojsons")]
    pub output_frames: Option<PathBuf>,

    /// Only generate frames for way groups which have a size larger than this length.
    /// If not specified, the frames for every group is calculated.
    #[arg(long, value_name = "NUMBER", requires = "output_frames")]
    pub frames_group_min_length_m: Option<f64>,

    /// If the output file(s) already exists, overwrite it. By default, exit if the output already
    /// exists
    #[arg(long)]
    pub overwrite: bool,

    /// Filter input ways by these tags
    ///
    /// Can be specified many times. All values ANDed together. (i.e. way must match all)
    /// Example
    ///   • `-f key` / `-f ∃key`  way has this tag
    ///   • `-f ~key_regex` / `-f ∃~key_regex`  There is a key, which matches this regex
    ///   • `-f ∄key`  way does not has this key
    ///   • `-f key=value`  way has this key and this value
    ///   • `-f key≠value`  way either doesn't have this key, or if it does, it's not equal to value
    ///   • `-f key=value1,value2,…` / -f `key∈value1,value2,…`  way has this key and the value is one of these
    ///   • `-f key≠value1,value2,…` / -f `key∉value1,value2,…`  way either doesn't have this key,
    ///      or if it does, it's not one of these values
    ///   • `-f key~regex` way has this key and the value matches this regex.
    ///     Regexes are case sensitive. Add `(?i)` at start of regex to switch to case insensitive
    ///     (e.g. `-f "name~(?i).* street"`)
    ///     Regexes match the whole value, `-f name~[Ss]treet` will match `Street`, but not `Main
    ///     Street North` nor `Main Street`. Use `-f "name~.*[Ss]treet.*"` to match all.
    ///   • `-f F1∨F2∨F3…` logical OR of the other tag filters F1, F2, …
    ///   • `-f F1∧F2∧F3…` logical AND of the other tag filters F1, F2, …
    #[arg(
        short = 'f',
        long = "tag-filter",
        value_name = "FILTER",
        conflicts_with = "tag_filter_func",
        verbatim_doc_comment
    )]
    pub tag_filter: Vec<tagfilter::TagFilter>,

    /// Tag filter function code to use. (see README)
    #[arg(
        short = 'F',
        long = "tag-filter-func",
        value_name = "FILTER_FUNC",
        conflicts_with = "tag_filter"
    )]
    pub tag_filter_func: Option<tagfilter::TagFilterFunc>,

    ///// Group by unique values of this key
    /////
    ///// Can be specified many times, which will be many groupings.
    ///// specify many keys (separated by commas) to use the first set value as the key
    ///// `-g name:en,name` → The grouping key will be the the `name:en` key if it's set, else the
    ///// `name` key
    //#[arg(short = 'g', long = "tag-group-k", value_name = "key1,key2,…")]
    //pub tag_group_k: Vec<taggrouper::TagGrouper>,
    /// If grouping by a key, set this to also include ways where there is any unset tag (default
    /// to require all to be set)
    #[arg(long)]
    pub incl_unset_group: bool,

    /// Only include (in the output) lines which are this length (in metres) or more.
    #[arg(long, value_name = "NUMBER")]
    pub min_length_m: Option<f64>,

    /// Only include (in the output) lines which have a dist_to_nearer greater than or equal to
    /// this
    #[arg(long, value_name = "NUMBER", requires = "incl_dist_to_longer")]
    pub min_dist_to_longer_m: Option<f64>,

    #[arg(long, value_name = "NUMBER", requires = "split_into_single_paths")]
    pub max_sinuosity: Option<f64>,

    /// Per tag group, only include the longest N lines
    #[arg(long, value_name = "N")]
    pub only_longest_n_per_group: Option<usize>,

    /// Per file, only include the longest N lines
    #[arg(long, value_name = "N", value_parser=parse_int_human)]
    pub only_longest_n_per_file: Option<usize>,

    /// When splitting a waygroup into paths, only take the following longest N paths (default:
    /// take all)
    #[arg(long, value_name = "N", value_parser=parse_int_human)]
    pub only_longest_n_splitted_paths: Option<usize>,

    /// Set this to make each group a different filename, or have everything in one file. Default:
    /// false, everything in one file.
    #[arg(long)]
    pub split_files_by_group: bool,

    /// After grouping the ways, split based on longest linear paths
    #[arg(long)]
    pub split_into_single_paths: bool,

    /// When splitting into single paths, how are the “largest” decided
    /// • `longest-path`: Longest path along the graph
    /// • `as-crow-flies`: Path between the 2 points that are furthest apart.
    /// The `longest-path` for motorways can produce technically correct, but unwanted, results by
    /// going up one lane of the motorway, and then down the other side. Here, `as-crow-flies` is
    /// often what people expect.
    #[arg(long, requires = "split_into_single_paths", verbatim_doc_comment)]
    pub split_into_single_paths_by: Option<SplitPathsMethod>,

    /// For each output object, calculate the distance (in m) to the nearest, longer object. This
    /// is increadily long for large complicated networks (e.g. waterways), but is reasonable for
    /// named streets.
    #[arg(long, default_value="false", aliases=["incl-distance-to-longer"])]
    pub incl_dist_to_longer: bool,

    /// Include list of OSM wayids for each feature
    /// For each way group, include a JSON property `all_wayids`, a list of all the OSM way ids
    /// that make up this group. Each is a JSON string "w123" (i.e. /^w[0-9]+$/), the same format
    /// `osmium getid` accepts.
    ///
    #[arg(long, action=clap::ArgAction::SetTrue, default_value = "false", aliases=["incl-way-ids", "include-wayids", "include-way-ids"], conflicts_with="split_into_single_paths")]
    pub incl_wayids: bool,

    /// Rather than save one MultiLineString per group, save it as many smaller linestrings,
    /// duplication of properties etc
    #[arg(long, default_value = "false")]
    pub save_as_linestrings: bool,

    #[command(flatten)]
    pub verbose: Verbosity<clap_verbosity_flag::InfoLevel>,

    /// Only include (in the output) lines which have this much upstream
    #[arg(long, value_name = "NUMBER")]
    pub min_upstream_m: Option<f64>,

    /// Where to write the loops file
    #[arg(long, value_name = "OUTPUT.geojson[s]")]
    pub loops: Option<PathBuf>,

    /// Whether to include a string list of all the consituant nids for each loop. For very long
    /// loops, this can make a large GeoJSON file, which some tools (e.g. `ogr2ogr`) refuse to deal
    /// with.
    #[arg(long, requires = "loops", conflicts_with = "loops_no_incl_nids")]
    pub loops_incl_nids: bool,

    /// Don't include the `nids` property in the loops file
    #[arg(long, requires = "loops", conflicts_with = "loops_incl_nids")]
    pub loops_no_incl_nids: bool,

    /// Path to store CSV of statistics
    /// CSV file with 4 columns.
    /// • `timestamp`: unix epoch timestamp of data age (integer)
    /// • `iso_timestamp`: ISO8601/RFC3339 string of data age same second as timestamp. (string)
    /// • `area` Name of the area (string). Many rows per loop. Possible values:
    ///    • `planet` for everything in the file
    ///    • A region code from the country-boundaries crate
    ///    • `unknown_area` if it doesn't match any area in country-boundaries
    /// • `metric` String name of the metric. Current values:
    ///     • `loops_count`: Number of loops in this region (integer)
    ///     • `loops_length_m`: Total length, in metres, of all loops in this region (float)
    /// • `value` The value of the metric.
    #[arg(long, value_name = "FILENAME.csv", verbatim_doc_comment)]
    pub loops_csv_stats_file: Option<PathBuf>,

    /// Path to store OpenMetrics/Prometheus metrics
    #[arg(long, value_name = "FILENAME.prom")]
    pub loops_openmetrics: Option<PathBuf>,

    /// Write the ends file
    #[arg(long, value_name = "OUTPUT.geojson[s]")]
    pub ends: Option<PathBuf>,

    /// The points in the Ends data will have a boolean if they are a member of a way with this
    /// tag. Syntax is the tag filter.
    #[arg(long, value_name = "TAGFILTER", requires = "ends")]
    pub ends_membership: Vec<tagfilter::TagFilter>,

    /// For each end point point, for each way which is included, record the value of this OSM tag
    /// on that end point as the GeoJSON property `tag:$TAG`. If the upstreams, or grouped
    /// upstreams are calculated, that will be included in the `end_tag:X` property
    ///
    /// e.g. `--ends-tag NAME` will add the value of the `name` tag from each OSM way which ends at a
    /// point to that point, and any upstream points. This makes the output in `grouped-ends` more
    /// human readable.
    #[arg(long, value_name = "TAG", verbatim_doc_comment)]
    pub ends_tag: Vec<String>,

    /// Path to store CSV exports of end points
    /// CSV file with following columns:
    /// • `timestamp`: unix epoch timestamp of data age (integer)
    /// • `iso_timestamp`: ISO8601/RFC3339 string of data age same second as timestamp. (string)
    /// • `upstream_m`: Total upstream to this end, in metres (float)
    /// • `upstream_m_rank`: What's the rank of that upstream, 1 = the biggest upstream_m.
    /// (integer)
    /// • `nid`: OSM Node id
    /// • `lat`: Latitude of the point
    /// • `lng`: Longitude
    /// And then one column for each `--ends-tag` value (if any set)
    #[arg(long, value_name = "FILENAME.csv", verbatim_doc_comment)]
    pub ends_csv_file: Option<PathBuf>,

    /// Only end points with a `upstream_m` longer than this will be included in ends csv file
    /// If unset, all end points will be included
    #[arg(long, requires = "ends_csv_file")]
    pub ends_csv_min_length_m: Option<f64>,

    /// Only the largest N ends (by `upstream_m`) are included in CSV file
    #[arg(long, requires = "ends_csv_file", value_parser=parse_int_human)]
    pub ends_csv_only_largest_n: Option<usize>,

    /// When calculating ens CSV file, only include end points which have an tag.
    #[arg(long, requires_all = ["ends_csv_file", "ends_tag"], default_value="false")]
    pub ends_csv_only_tagged: bool,

    /// Calculate & write a file with each upstream line to this file
    #[arg(long, value_name = "UPSTREAMS_FILENAME")]
    pub upstreams: Option<PathBuf>,

    /// The upstreams file will only include segments which have at least this much of an
    /// upstream_m value. If unset, all segments will be included.
    /// This helps reduce the file size
    #[arg(long, value_name = "NUMBER", requires = "upstreams")]
    pub upstreams_min_upstream_m: Option<f64>,

    /// For every upstream, include details on which end point(s) this eventually flows to.
    #[arg(long, default_value = "false")]
    pub upstream_output_ends_full: bool,

    /// Include an extra property from_upstream_m_N for every occurance of this argument, with the
    /// from_upstream_m value rounded to the nearest multiple of N.
    /// e.g. `--upstream-from-upstream-multiple 100` will cause `from_upstream_m_100` value to be
    /// the `upstream_m` value, but rounded to the nearest multiple of 100.
    #[arg(long, requires = "upstreams", verbatim_doc_comment)]
    pub upstreams_from_upstream_multiple: Vec<f64>,

    /// When a node has >1 child nodes, allocate the upstream value of that node equally amoung all
    /// these nodes.
    #[arg(long, default_value = "false", conflicts_with = "flow_follows_tag")]
    pub flow_split_equally: bool,

    /// When a node has >1 child nodes, allocate (nearly) all the upstream value to the nodes which
    /// are in ways with the same `TAG` value as the (single) upstream segment for this node.
    #[arg(long, conflicts_with = "flow_split_equally")]
    pub flow_follows_tag: Option<String>,

    /// Calculate and write ways which are based on which end point each line eventually flows
    /// into, based on the `upstream_assign_end_by_tag` or `upstream_output_biggest_end` (for
    /// biggest end).
    #[arg(long)]
    pub grouped_ends: Option<PathBuf>,

    #[arg(long, requires = "grouped_ends")]
    pub grouped_ends_max_upstream_delta: Option<f64>,

    #[arg(long, requires = "grouped_ends")]
    pub grouped_ends_max_distance_m: Option<f64>,

    /// For all ends, calc the complete upstreams
    #[arg(long, default_value = "false")]
    pub ends_upstreams: bool,

    /// Only calc upstream len if this upstream is greater than this
    #[arg(long)]
    pub ends_upstreams_min_upstream_m: Option<f64>,

    /// Upstream from each end goes only up this many nodes.
    #[arg(long)]
    pub ends_upstreams_max_nodes: Option<i64>,

    /// Creates a GeoJSON(Seq) file which has one Feature per grouped, connected waterway (based on
    /// --flow-follows-tag value).
    #[arg(long, value_name = "FILENAME")]
    pub grouped_waterways: Option<PathBuf>,

    /// If a way is in a relation, which matches the tag filters, then apply that relation tags to
    /// this way
    #[arg(long, default_value = "false")]
    pub relation_tags_overwrite: bool,

    /// If using relation_tags_overwrite, only relation members with this role will be used.
    #[arg(long, requires = "relation_tags_overwrite", value_name = "ROLE_NAME")]
    pub relation_tags_role: Vec<String>,
}

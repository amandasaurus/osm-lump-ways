use clap::Parser;
use clap_verbosity_flag::Verbosity;
use std::path::PathBuf;

use osm_lump_ways::dij::SplitPathsMethod;
use osm_lump_ways::tagfilter;

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
    ///     (e.g. `-f name~(?i).* street`)
    ///     Regexes match the whole value, `-f name~[Ss]treet` will match `Street`, but not `Main
    ///     Street North` nor `Main Street`. Use `-f name~.*[Ss]treet.*` to match all.
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
        conflicts_with = "tag_filter",
        value_parser=opt_read_file::<tagfilter::TagFilterFunc>,
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
    #[arg(long, value_name = "N")]
    pub only_longest_n_per_file: Option<usize>,

    /// When splitting a waygroup into paths, only take the following longest N paths (default:
    /// take all)
    #[arg(long, value_name = "N")]
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
    pub csv_stats_file: Option<PathBuf>,

    /// Path to store OpenMetrics/Prometheus metrics
    #[arg(long, value_name = "FILENAME.prom")]
    pub openmetrics: Option<PathBuf>,

    /// Write the ends file
    #[arg(long, value_name = "OUTPUT.geojson[s]")]
    pub ends: Option<PathBuf>,

    /// Where to write the loops file
    #[arg(long, value_name = "OUTPUT.geojson[s]")]
    pub loops: Option<PathBuf>,

    /// The points in the Ends data will have a boolean if they are a member of a way with this
    /// tag. Syntax is the tag filter.
    #[arg(long, value_name = "TAGFILTER", requires = "ends")]
    pub ends_membership: Vec<tagfilter::TagFilter>,

    /// Calculate & write a file with each upstream line to this file
    #[arg(long, value_name = "UPSTREAMS_FILENAME")]
    pub upstreams: Option<PathBuf>,

    /// For every upstream, include details on which end point(s) this eventually flows to.
    #[arg(long, default_value = "false")]
    pub upstream_tag_ends_full: bool,

    /// For every upstream, include details on the largest end that this point flows to.
    /// Less details on other ends than upstream_tag_ends_full, but requires less memory to
    /// process.
    #[arg(
        long,
        default_value = "false",
        conflicts_with = "upstream_tag_ends_full"
    )]
    pub upstream_tag_biggest_end: bool,

    /// Write the group
    #[arg(long, default_value = "false", requires = "upstreams")]
    pub group_by_ends: bool,

    /// Include an extra property from_upstream_m_N for every occurance of this argument, with the
    /// from_upstream_m value rounded to the nearest multiple of N.
    #[arg(long, requires = "upstreams")]
    pub upstream_from_upstream_multiple: Vec<i64>,

    /// For all ends, calc the complete upstreams
    #[arg(long, default_value = "false")]
    pub ends_upstreams: bool,

    /// Only calc upstream len if this upstream is greater than this
    #[arg(long)]
    pub ends_upstreams_min_upstream_m: Option<f64>,

    /// Upstream from each end goes only up this many nodes.
    #[arg(long)]
    pub ends_upstreams_max_nodes: Option<i64>,
}

/// CLI arg parser. If the value starts with @, the rest is assumed to be a filename, the contents
/// of which are parsed to type T
fn opt_read_file<T>(arg_val: &str) -> Result<T, String>
where
    T: std::str::FromStr<Err = String> + std::fmt::Debug,
{
    if let Some(filename) = arg_val.strip_prefix('@') {
        let val = std::fs::read_to_string(filename)
            .map_err(|e| format!("Couldn't read filename {}: {}", filename, e))?;
        T::from_str(&val)
    } else {
        let res = T::from_str(arg_val);

        if res.is_err() && std::path::Path::new(arg_val).is_file() {
            let original_error = res.unwrap_err();
            Err(format!("Unable to parse {:?}. However that is a filename. Did you mean @{} ? Original Error: {}", arg_val, arg_val, original_error))
        } else {
            res
        }
    }
}

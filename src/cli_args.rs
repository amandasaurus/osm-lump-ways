use clap::Parser;
use std::path::PathBuf;
use clap_verbosity_flag::Verbosity;

use crate::tagfilter;
use crate::TagGrouper;

/// Group OSM ways based on shared tags into GeoJSON MultiLineStrings
///
/// Reads an OSM PBF file, and groups all connected ways together into a MultiLineString
///
/// Use `-f`/-`-tag-filter` to only include ways which match that tag filter
/// `-g`/`--tag-group-k` to group ways by connectiveness *and* whether that tag key is equal.
/// `--min-length-m` Only output way groups with a minimum length of this
/// `--only-longest-n-per-file NUM` Only output the longest `NUM` ways groups.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// Input PBF filename
    #[arg(short, long, value_name = "FILENAME.osm.pbf")]
    pub input_filename: PathBuf,

    /// Output filename. If `--split-files-by-group` specified, include `%s` for where to place the group.
    #[arg(short, long, value_name = "OUTPUT.geojson")]
    pub output_filename: String,

    /// If the output file already exists, overwrite it. By default, exit if the output already
    /// exists
    #[arg(long)]
    pub overwrite: bool,

    /// Filter input ways by these tags
    ///
    /// Can be specified many times. All values ANDed together. (i.e. way must match all)
    /// Example
    ///   • `-f key` / `-f ∃key`  way has this tag
    ///   • `-f ∄key`  way does not has this tag
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
    #[arg(short = 'f', long = "tag-filter", value_name = "FILTER")]
    pub tag_filter: Vec<tagfilter::TagFilter>,

    /// Group by unique values of this key
    ///
    /// Can be specified many times, which will be many groupings.
    /// specify many keys (separated by commas) to use the first set value as the key
    /// `-g name:en,name` → The grouping key will be the the `name:en` key if it's set, else the
    /// `name` key
    #[arg(short = 'g', long = "tag-group-k", value_name = "key1,key2,…")]
    pub tag_group_k: Vec<TagGrouper>,

    /// If grouping by a key, set this to also include ways where there is any unset tag (default
    /// to require all to be set)
    #[arg(long)]
    pub incl_unset_group: bool,

    /// Only include (in the output) lines which are this length (in metres) or more.
    #[arg(long, value_name = "NUMBER")]
    pub min_length_m: Option<f64>,

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

    /// Save time by storing node locations at the first pass. Omit to do it in 2-pass, memory
    /// friendly manner
    ///
    /// For small files, using this option can speed up processing time
    #[arg(long)]
    pub read_nodes_first: bool,

    /// After grouping the ways, split based on longest linear paths
    #[arg(long)]
    pub split_into_single_paths: bool,

    /// Only procoess way groups which include these way ids
    #[arg(long)]
    pub only_these_way_groups: Vec<i64>,

    ///
    #[arg(long, value_name = "SECONDS")]
    pub timeout_dist_to_longer_s: Option<f32>,

    /// include all way ids
    #[arg(long, action=clap::ArgAction::SetTrue, default_value = "false")]
    pub incl_wayids: bool,

    /// Rather than save one MultiLineString per group, save it as many smaller linestrings,
    /// duplication of properties etc
    #[arg(long, default_value = "false")]
    pub save_as_linestrings: bool,

    #[command(flatten)]
    pub verbose: Verbosity<clap_verbosity_flag::InfoLevel>,
}

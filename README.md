# `osm-lump-ways` group OSM ways based on topology & shared tags

![Crates.io Number of Downloads](https://img.shields.io/crates/d/osm-lump-ways)
[![Crates.io Latest Version](https://img.shields.io/crates/v/osm-lump-ways)](https://crates.io/crates/osm-lump-ways)

> Answer questions about OSM data like:
>
> * “What's the longest Main Street?”
> * “How far can I drive on unpaved roads in this region?”
> * “How long is the M1 motorway?”
> * “Are these rivers connected?”
> * “What's the river drainage basins?”

2 similar programmes are included: `osm-lump-ways`, which ignores the direction
of the OSM way, and `osm-lump-ways-down`, which uses direction of OSM ways to
produce data, incl. QA suitable files.

# Background

OSM linear features (eg roads, rivers, walls) are stored as [way
object](https://wiki.openstreetmap.org/wiki/Way). The [OSM tagging
model](https://wiki.openstreetmap.org/wiki/Tags) often requires one feature to
be mapped as many different ways, “splitting the way”. `osm-lump-ways` will
undo this splitting, and recursively merge the original OSM objects together.

# Filtering OSM Data

There are 2 ways to select which OSM ways will be used.

## Tag Filter Rules

* `key` / `∃key`  way has this tag
* `~key_regex` / `∃~key_regex`  There is a key, which matches this regex.
* `∄key`  way does not has this key
* `key=value`  way has this key and this value
* `key≠value`  way either doesn't have this key, or if it does, it's not equal to value
* `key∈value1,value2,…`  way has this key and the value is one of these
* `key∉value1,value2,…`  way either doesn't have this key,
   or if it does, it's not one of these values
* `key~regex` way has this key and the value matches this regex.
* `F1∨F2∨F3…` logical OR of the other tag filters F1, F2, …
* `F1∧F2∧F3…` logical AND of the other tag filters F1, F2, …

The popular [`regex` crate](https://docs.rs/regex/latest/regex/) is used for
matching. Regexes, and string comparison, are case sensitive. Add `(?i)` at
start of regex to switch to case insensitive (e.g. `name~(?i).* street`)
Regexes match the whole value, `name~[Ss]treet` will match `Street` or
`street`, but not `Main Street North` nor `Main Street`. Use
`name~.*[Ss]treet.*` to match all.

## Simple Tag Filtering

The `-f`/`--tag-filter` can be specified one or more times, and an OSM object
is only included if it matches *all* defined filter's, i.e. a logical
AND of all filters.

* `-f highway`: Only ways with a `highway` tag are included
* `-f highway -f name`: Only ways with a `highway` *and* `name` tag are included.
* `-f highway -f ∄name`: Only ways with a `highway` *and* without a `name` tag are included.

## More complex Tag Filtering Functions

The `-F`/`--tag-filter-func` takes a single ordered list of tag filters (separated by `;`),
and includes (with `→T`), or excludes (with `→F`), the OSM object based
on the _first_ filter function which matches. A bare `T` or `F` applies to all.

Example: We want to include all `waterways`. But not `waterway=canal`. But we
want a `waterway=canal` iff it also has a `lock=yes` tag.

`-F "waterway=canal∧lock=yes→T; waterway=canal→F; waterway→T; F`

If the argument to `-F`/`--tag-filter-func` starts with `@`, the rest is a
filename containing the tag filter func code. e.g. ` -F @myrules.txt `. In this
mode, a line `include FILENAME;` will include the contents of another file
there. `FILENAME` is a path relative to the original filename. It will be
expanded recursively.

Comments start with `#` and continue to the end of the line. Since the `;` is
special, it cannot be directly used in tag filtering. Use `\u{3B}` instead.
e.g. `waterway=put_in\u{3B}egress→F;` is a rule to exclude any tag with key
`waterway` and value `put_in;egress`.

## Relations

The programme `osm-lump-ways` ignores relations.

`osm-lump-ways-down` by default ignores relations, unless
`--relation-tags-overwrite` is used.

### `--relation-tags-overwrite`

If a relation matches the tag filter, then the relation's ways will be assumed
to have the relation tags and will be included. By default all relation member
ways are included. If `--relation-tags-role` only the ways in the relation with
that role will be included (can be specified more than once).

If a relation has a tag, and a way has a tag (e.g. `name`) the tag value from
the relation will be used not the way.

If a way is a member of more than one relation, then the relation with the
higher number of members will be used (if equal, the last seen relation in the
file will be used, which for normal, sorted, OSM files is the highest numbered
relation). Only one level of way-in-relation membership is included. Relations
as members in other relations are ignored.

# Output

## File format

If a filename ends with `.geojson`, a GeoJSON file
([RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946) will be created. For
`.geojsons`, a GeoJSON Text Sequences
([RFC 8142](https://datatracker.ietf.org/doc/html/rfc8142)), aka GeoJSONSeq, file.

## Geometry format

By default each feature is a `MultiLineString`, representing every way group.

With `--split-into-single-paths`, each way group will be split into 1 or more
`LineString`s, controlled by the `--split-into-single-paths-by X` argument. By
default the `as-crow-flies` distance is used. `longest-path` uses the longest
path in the graph, and this can take days to calculate on large networks.


# Input

The input must be an [OSM PBF](https://wiki.openstreetmap.org/wiki/PBF_Format)
file. Use [osmium to convert between OSM file
formats](https://osmcode.org/osmium-tool/manual.html#osm-file-formats-and-converting-between-them).

The input
[object ids](https://wiki.openstreetmap.org/wiki/Elements#:~:text=Description-,id,-integer%20%2864-bit)
must be positive. OSM import software often uses
[negative ids](https://wiki.openstreetmap.org/wiki/Elements#:~:text=negative%20values%20%28%3C0%29%20are%20reserved).
Use [`osmium sort`](https://docs.osmcode.org/osmium/latest/osmium-sort.html)
and then [`osmium
renumber`](https://docs.osmcode.org/osmium/latest/osmium-renumber.html) like
so, to create an acceptable file.

	osmium sort negative-id-file.osm.pbf -o sorted.osm.obf
	osmium renumber sorted.osm.pbf -o new-input.osm.pbf

# Installation

	cargo install osm-lump-ways

This will install the 2 programmes: `osm-lump-ways` & `osm-lump-ways-down`.
`osm-lump-ways` ignores the direction of the OSM ways, and produces single
output files. It can be used for waterways or watersheds, but also for roads or
similar. `osm-lump-ways-down` uses the direction of the OSM data to produce
many similar files.

# `osm-lump-ways`

## Usage

Generate river drainage basins

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river

To group based on the river's name:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river -g name

To find long streets and assemble them into connected (Multi)LineStrings:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o long-streets.geojson -f highway -g name


## Full Options

Run with `--help` to see all options.

## Frames

Here, a “frame” of a grouping is a shortest path through 2 points in the
grouped together ways. This can be useful for waterways to find where a big
group is connected.

`--output-frames FILENAME.geojsons` will save these features to a file.
GeoJSONSeq output format only.

# Examples of usage

* [WaterwayMap.org](https://waterwaymap.org)
* [Longest O'Connell Street in Ireland](https://en.osm.town/@amapanda/110270516183776589)
* [Road most split in UK&Irl](https://en.osm.town/@amapanda/110762435236476901)
* [Die Bahnhofstrassen in jeder Schweizer Sprachregion (german language only)](https://habi.gna.ch/2023/11/14/die-bahnhofstrassen-in-jeder-schweizer-sprachregion/)
* [Roads without etymology data](https://en.osm.town/@amapanda/114562398764953834)
* Your project here!

# `osm-lump-ways-down`

It reads & groups an OSM PBF file, like `osm-lump-ways`, but it uses the
direction of the OSM way, to produce many different output files. The main use
for this is waterway flow, so that terminology will often be used, but it could
be used for anything else which uses direction.

One concept that occurs a lot is the “sum of all the ways that flow into this
point”, i.e. the “upstream” at a point.

## Allocating flow downstream of a bifurcation

When there is X m of upstream ways flowing into a node (via 1+ incoming ways),
and 2+ ways leading out of a point, there are 2 methods to allocate the
upstream total to the nodes “downstream” of this node. 

`--flow-split-equally`: The value of X is split equally between all nodes
“downstram”. `--flow-follows-tag TAG`: If there is 1 incoming segment with a
tag `TAG`, and 1+ outgoing segment(s) with the same `TAG` value, then the X is
split equally between them, and 1 m of upstream is shared between all other
outgoing segments.

Splitting the flow equally sounds logical, however it can often happen that a
large river has a (mis)tagged waterway that leads away, and results in half
getting diverted. If the flow follow the `name` tag, then most of the upstream
stays with the main stream.

## Output files

It can output different files, depending on the options:

### Loops (Cycles) (`--loops FILENAME`)

Cycles in the network. Each is a
[strongly connected component](https://en.wikipedia.org/wiki/Strongly_connected_component),
and `MultiLineString`.

#### Feature Properties

* `areas`: Array of Strings. The cycle is geocoded using
  [`country-borders`](https://github.com/westnordost/country-boundaries-rust).
  The code for every matching area is included in this array, sorted with longest strings first. e.g.: `["GB-ENG","GB"],`
* `areas_s`: String. `areas`, but the array has been converted to a string, and comma
  separated, with a `,` prefix & suffix. e.g. `",GB-ENG,GB,"`. The prefix &
  suffix ensure you can always search for `,GB,` in that property and get a
  match.
* `area_N`: String. One for every element in `areas`, with a numeric suffix and
  the item itself. e.g. `"areas_s":",GB-ENG,GB,"` then there will be
  `"area_0":"GB-ENG","area_1":"GB"`. These properties are for software which
  doesn't support JSON arrays.
* `length_m`: Float. Total length of this cycle, in metres.
* `nodes`: String. All the node ids in this cycle, sorted, and deduplicated, and saved
  e.g. `"n318688661,n5354970286,n1016277960`.
* `root_node_id`: Integer. Node id of the lowest numbered node in this cycle.
* `num_nodes`: Integer. Number of nodes in this cycle. incl. duplicates.

### Loops stats CSV (`--loops-csv-stats-file FILENAME.csv`)

With `--loops-csv-stats-file FILENAME.csv`, a CSV file with statistics of the loops,
along with a breakdown per region is created. If the file is already there, the
data is appended.

See the doc comment
[`src/bin/osm-lump-ways-down/cli_args.rs`](./blob/main/src/bin/osm-lump-ways-down/cli_args.rs#L158-L171),
or run `osm-lump-ways-down --help` for the format.

### Way network, with per segment upstream value (`--upstreams FILENAME`)

Each way segment (a 2 point `LineString`) with the upstream data.
With `--upstream-output-ends-full`, each segement get information about all the end
point(s) that the segment eventually flows into.

#### Assigning to an end point

With `--upstream-output-biggest-end`, each segment will be assigned to the end
point that it flows into which has the largest total outflow value.

With `--upstream-assign-end-by-tag TAG`, it will attempt to follow the same OSM
tag upstream from an end point. When used with the `name` tag, this usually
produces maps that are more like what people expect.

### Way network, grouped by end point (`--grouped-ends FILENAME`)

Ways grouped by downhill and the end point they flow into. The upstream value
of each segment isn't included, and it attemptes to generate long LineStrings,
grouped by the end point that it flows into.

See the `--upstream-output-biggest-end` & `--upstream-assign-end-by-tag TAG`
arguments.

### End points (`--ends FILENAME`)

Points where waterways end.

#### Ends membership of tagged ways

##### `--ends-membership TAGFILTER`

With the `--ends-membership TAGFILTER` arg every end will have a boolean
property called `is_in:TAGFILTER` which is only true if this end node is a
member of a way with this tag filter. This argument can be applied many times.
An addditional property `is_in_count` is an integer of how many of these
properties are true.

e.g. `--ends-membership natural=coastline` will cause each end point to have a
JSON property `is_in:natural=coastline` which is `true` iff this node is also a
member of a way with the `natural=coastline` tag, false otherwise.

##### `--ends-tag TAG`

Every end point will have a JSON property `tag:X` which is the tag

e.g. `--ends-tag name` will cause every end point to have a `tag:name` JSON
property, with the `name` tag from the way which flows into this end point.

Unlike `--ends-membership` this only uses the OSM ways which are included by
the tag filters

`--ends-upstreams-min-upstream-m 1e6 --ends-upstreams-max-nodes 1000` is a good
tradeoff for calculation speed & file size, which still shows the relevant
upstreams

### Ends stats CSV (`--ends-csv-file FILENAME.csv`)

With `--ends-csv-file FILENAME.csv`, the end points are also written to a CSV
file. If the file is already there, new data is appended. Values from
`--ends-tag` (in order) are included in the CSV. Changing the arguments (or
order) will make a possibly invalid CSV file.

Only end points with an `upstream_m` greater than `--ends-csv-min-length-m X`
will be included (without this argument, there is no length filtering), and
only the largest N from `--ends-csv-only-largest-n N` (without this argument
there is no limit).

#### File format

CSV file with following columns:

* `timestamp`: unix epoch timestamp of data age (integer)
* `iso_timestamp`: ISO8601/RFC3339 string of data age same second as timestamp. (string)
* `upstream_m`: Total upstream to this end, in metres (float)
* `upstream_m_rank`: What's the rank of that upstream, 1 = the biggest upstream\_m.
(integer) in this iteration.
* `nid`: OSM Node id (integer)
* `lat`: Latitude of the point (float)
* `lng`: Longitude (float)

And then one column for each `--ends-tag` value, with that tag name as column
name. e.g. `--ends-tag name` causes the ends geojson file to have GeoJSON
property called `tag:name`, however in this CSV file, the column is `name`. 

### Every connected group with the same tag value (`--grouped-waterways FILENAME`)

Based on the value of the `--flow-follows-tag` argument, ways are grouped into
connected GeoJSON Features, along with many attributes about what is connected
to what.

### Longest from Source to Mouth (`--longest-source-mouth FILENAME`)

Calculates the river systems from the input data, i.e. the longest downstream
segment from a source/start to a mouth/end of a river.

For example, the Mississippi River is 3,000 km long. The Missouri River is
3,200 km, and joins the Mississippi about half way down (at St. Louis). If you
start at the start of the Missouri, follow it downstream, then join the
Mississippi, and follow that down, you will travel a longer length. This is the
true length of the this single “river system”.

INFO: This Missouri-Mississippi system actually goes Hell Roaring → Red Rock →
Beaverhead River → Jefferson River → Missouri → Mississippi

INFO: Sometimes a large river splits towards the mouth, into several
distributaries, each of which takes a large amount of upstream. This can result
in apparetly duplicate river systems. This happens with the Mississippi system.
There are currently 3 small distributaries, causing this Missouri-Mississippi
system to appear 3 times in the output.

Each river system is composed of 1 or more segments. The output is a collection
of GeoJSON `LineString` Features. Each Feature represents one segment in the
river system. In the Missouri-Mississippi example, the entire length & geometry
of the Missouri river will be included, and then the section of the Mississippi
downstream of where the Missouri joins will be included. Each river system can
be uniquely identified by the `river_system_mouth_source_nids` (or equivalent
`river_system_mouth_source_nids_s`) value.

* `idx` _(integer)_ Index of this segment within this river system. Starts at
  0, for the most upstream segment.
* `revidx` _(integer)_ Reverse Index. 0 is the most downstream segment. i.e.
  it's `num_parts - idx - 1`.
* `num_parts` _(integer)_ Total number of segments in this river system.
* `length_m` _(float)_ Length of this segment in metres. NB: It might not be
  the full length of this river.
* `name` _(string)_ The name of this segment from OSM. `null` is this has no
  name.
* `river_system_length_m` _(float)_ Length of entire river system in metres.
* `river_system_source_nid` _(integer)_ OSM node id of the source of this river
  system.
* `river_system_mouth_nid` _(integer)_ OSM node id of the mouth of this river
  system.
* `river_system_mouth_source_nids` _(array of integers)_ JSON array of
  `[mouth_nid, source_nid]` from above.
* `river_system_mouth_source_nids_s` _(string)_ `mouth_source_nids` but already
  formatted as a string (e.g. `"1,2"`)
* `river_system_names` _(array of strings)_ JSON Array of all the names in this
  river system, going upstream. i.e. element 0 is the end of the river. If a
  segment has no name, then `--longest-source-mouth-unnamed-string` argument
  will be used, which is `"unnamed"` if not set. There are no JSON `null`s in
  this value
* `river_system_names_s` _(string)_ the `river_system_names` value, but joined
  together with `" - "`, e.g. `"Shannon - Boyle - Lung"` for the
  Shannon river system.

The follow extra CLI arguments change the output.

* `--longest-source-mouth-min-length-m FLOAT` Only river systems with at least
  this total length will be inclued.
* `--longest-source-mouth-longest-n INTEGER` Only the longest N river systems
  will be included. NB: since 1 river system will often be composed of more
  than 1 segment, there will almost certainly be more then N GeoJSON Features
  in the output file.
* `--longest-source-mouth-only-named` Only named river segments will be used
  for calculation. Without this argument, all segments, incl unnamed rivers
  will be included.
* `--longest-source-mouth-unnamed-string STRING` _(see above)_

## Loop removal

After the loops are detected, all the edges (way segments) in the loops are
contracted together, producing a new graph which is loop-free.

# TODOs

This software isn't finished. Lots is being motivated for
[WaterwayMap.org](https://waterwaymap.org), a topological analysis of waterways
in OpenStreetMap.

# External Mentions

*TBC*

# Copyright & Licence

Copyright 2023, 2024, MIT/Apache2.0. Source code is on
[Github (`osm-lump-ways`)](https://github.com/amandasaurus/osm-lump-ways).

The output data file(s) are a Derived Database of the OpenStreetMap database,
and hence under the [ODbL 1.0](https://opendatacommons.org/licenses/odbl/)
licence, the same as the
[OpenStreetMap copyright](https://www.openstreetmap.org/copyright), and
contains data © OpenStreetMap contributors

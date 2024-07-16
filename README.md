# `osm-lump-ways` group OSM ways based on topology & shared tags

![Crates.io Number of Downloads](https://img.shields.io/crates/d/osm-lump-ways)
![Crates.io Latest Version](https://img.shields.io/crates/v/osm-lump-ways)

> Answer questions about OSM data like:
>
> * “What's the longest Main Street?”
> * “How far can I drive on unpaved roads in this region?”
> * “How long is the M1 motorway?”
> * “Are these rivers connected?”
> * “What's the river drainage basins?”

2 similar programmes are included: `osm-lump-ways`, which ignores the direction
of the OSM way, and `osm-lump-ways-down`, which uses direction of OSM ways to
produce data, incl. QA suitable files. Both share similarities.

# Background

OSM linear features (eg roads, rivers, walls) are stored as [way
object](https://wiki.openstreetmap.org/wiki/Way). The [OSM tagging
model](https://wiki.openstreetmap.org/wiki/Tags) often requires one feature to
be mapped as many different ways. `osm-lump-ways` will assemble them all together.

# Filtering OSM Data

There are 2 ways to select which OSM ways will be used. All relations are currently ignored.

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
filename containing the tag filter func code. e.g. ` -F @myrules.txt `.

Comments start with `#` and continue to the end of the line.

# Output format

If a filename ends with `.geojson`, a GeoJSON file
([RFC 7946](https://datatracker.ietf.org/doc/html/rfc7946) will be created. For
`.geojsons`, a GeoJSON Text Sequences
([RFC 8142](https://datatracker.ietf.org/doc/html/rfc8142)), aka GeoJSONSeq, file.


# Input

The input must be an [OSM PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) file. Use [osmium to convert between OSM file formats](https://osmcode.org/osmium-tool/manual.html#osm-file-formats-and-converting-between-them).

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

# `osm-lump-ways`

# Usage

Generate river drainage basins

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river

To group based on the river's name:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river -g name

To find long streets and assemble them into connected (Multi)LineStrings:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o long-streets.geojson -f highway -g name

# Installation

	cargo install osm-lump-ways

# Full Options

Run with `--help` to see all options.

# Frames

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
* Your project here!

## `osm-lump-ways-down`

It reads & groups an OSM PBF file, like `osm-lump-ways`, but it uses the
direction of the OSM way, to produce 

The output filename *must* contain `%s`, which will be replaced 

## Output files

### `loops`

Cycles in the network. Each is a [strongly connected
component](https://en.wikipedia.org/wiki/Strongly_connected_component).

### `upstreams`

Each way segment (a 2 point `LineString`) with the upstream data.

with `--upstream-tag-ends`, each segement get information about the end
point(s) for that segment.

### `upstreams-points`

Above, but just the first point. Each feature a Point.

### `ends`

Points where waterways end.

#### Ends membership of tagged ways

With the `--ends-membership TAGFILTER` arg every end will have a boolean
property called `is_in:TAGFILTER` which is only true if this end node is a
member of a way with this tag filter. This argument can be applied many times.
An addditional property `is_in_count` is an integer of how many of these
properties are true.


e.g. `--ends-membership natural=coastline` will cause each end point to have a
JSON property `is_in:natural=coastline` which is `true` iff this node is also a
member of a way with the `natural=coastline` tag, false otherwise.

### `ends-full-upstreams`

Only with `--ends-upstreams` argument. File of MultiLineStrings showing, for
each end, where the upstreams are. Useful to find why there's a big upstream
end somewhere.

`--ends-upstreams-min-upstream-m 1e6 --ends-upstreams-max-nodes 1000` is a good tradeoff for calculation speed & file size, which still shows the relevant upstreams

## Loop removal

After the loops are detected, all the edges (way segments) in the loops are
contracted together, producing a new graph which is loop-free.


# TODOs

This software isn't finished, here's what I'd like to add. Feel free to send a patch.

* All tags need to be specified in advance to join on. Perhaps add something to
  match all possible tags? (inspired by
  [this q](https://en.osm.town/@grischard/110763741292331075)). (“Group by all
  tags the same” might do it)

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

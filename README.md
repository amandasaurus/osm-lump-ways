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

# Usage

Generate river drainage basins

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river

To group based on the river's name:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river -g name

To find long streets and assemble them into connected (Multi)LineStrings:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o long-streets.geojson -f highway -g name

# Installation

	cargo install osm-lump-ways

# Background

OSM linear features (eg roads, rivers, walls) are stored as [way
object](https://wiki.openstreetmap.org/wiki/Way). The [OSM tagging
model](https://wiki.openstreetmap.org/wiki/Tags) often requires one feature to
be mapped as many different ways. `osm-lump-ways` will assemble them all together.

# Full Options

Run with `--help` to see all options.

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

# Todos

This software isn't finished, here's what I'd like to add. Feel free to send a patch.

* All tags need to be specified in advance to join on. Perhaps add something to
  match all possible tags? (inspired by [this
  q](https://en.osm.town/@grischard/110763741292331075)). (“Group by all tags
  the same” might do it)

# External Mentions

*TBC*

# Copyright & Licence

Copyright 2023, MIT/Apache2.0. Source code is on [Github
(`osm-lump-ways`)](https://github.com/amandasaurus/osm-lump-ways).

The output data file(s) are a Derived Database of the OpenStreetMap database,
and hence under the [ODbL 1.0](https://opendatacommons.org/licenses/odbl/)
licence, the same as the
[OpenStreetMap copyright](https://www.openstreetmap.org/copyright), and
contains data © OpenStreetMap contributors

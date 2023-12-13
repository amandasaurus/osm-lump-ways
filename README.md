# `osm-lump-ways` group OSM ways based on topology & shared tags

![Crates.io](https://img.shields.io/crates/d/osm-lump-ways)

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

# Examples of usage

* [WaterwayMap.org](https://waterwaymap.org)
* [Longest O'Connell Street in Ireland](https://en.osm.town/@amapanda/110270516183776589)
* [Road most split in UK&Irl](https://en.osm.town/@amapanda/110762435236476901)
* Your project here!

# Todos

This software isn't finished, here's what I'd like to add. Feel free to send a patch.

* Support outputting GeoJSONSeq instead of one large GeoJSON FeatureCollection
* The `--split-into-way` uses Floyd–Warshall algorithm to calculate all pairs
  shortest path. It's incredibly slow on large numbers of points, e.g. >10k. It
  also single threaded. This should be replaces with something like multiple
  runs of Dijkstra's Algorithm to speed it up.
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

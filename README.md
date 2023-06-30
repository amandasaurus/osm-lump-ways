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

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f highway -g name

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

* [`osm-river-basins`](https://github.com/amandasaurus/osm-river-basins)
* [Longest O'Connell Street in Ireland](https://en.osm.town/@amapanda/110270516183776589)
* Your project here!

# Copyright & Licence

Copyright 2023, GNU Affero General Public Licence (AGPL) v3 or later. See [LICENCE](./LICENCE).
Source code is on [Github (`osm-lump-ways`)](https://github.com/amandasaurus/osm-lump-ways).

The output data file(s) are a Derived Database of the OpenStreetMap database,
and hence under the [ODbL 1.0](https://opendatacommons.org/licenses/odbl/)
licence, the same as the
[OpenStreetMap copyright](https://www.openstreetmap.org/copyright), and
contains data © OpenStreetMap contributors

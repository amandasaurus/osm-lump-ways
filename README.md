# `osm-lump-ways` create linear features from OSM data based on connectiveness (& tags)

> Answer questions about OSM data like:
> · “What's the longest Main Street?”
> · “How far can I drive on 
> · “How long is the M1 motorway?”
> · “Are these rivers connected?”

# Usage

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river

To lump into groups

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f waterway=river -g name

Or longest roads:

	osm-lump-ways -i path/to/region-latest.osm.pbf -o region-rivers.geojson -f highway -g name

# Installation

	cargo install osm-lump-ways

# Background

OSM linear features (eg roads, rivers, walls) are stored as [way object](https://wiki.openstreetmap.org/wiki/Way). The [OSM tagging model](https://wiki.openstreetmap.org/wiki/Tags) often requires one feature to be mapped as many different ways.



# Full Options


Run with `--help` to see all options.


# Copyright & Licence

Copyright 2023, GNU Affero General Public Licence (AGPL) v3 or later. See [LICENCE](./LICENCE).
Source code is on [Github (`osm-lump-ways`)](https://github.com/amandasaurus/osm-lump-ways).

The output data file(s) are a Derived Database of the OpenStreetMap database,
and hence under the [ODbL 1.0](https://opendatacommons.org/licenses/odbl/)
licence, the same as the [OpenStreetMap copyright](https://www.openstreetmap.org/copyright)

# Changelog

## Unreleased

## v0.26.0 (2024-05-02)

* New: `ends-full-upstreams` with `osm-lump-ways-down` for showing the upstream
  of end points.
* Internal refactoring to attempt to reduce memory usage & time needed
* `osm-lump-ways-down`: Strahler & Upstream Points output only produced if
  `--strahler`/`--upstream-points` argument(s) given
* `osm-lump-ways-down`: end & upstream files include the total number of
  upstream OSM nodes at each point.

## v0.25.0 (2024-04-24)

* `--tag-filter-func` now recogises a `@` at the start as the filename to read
  the argument from.
* Tag filter func can now be split on multiple lines, and have comments (#)
* End points can now be annotated if they are a part of a way with defined tags

## v0.24.0 (2024-04-09)

* New feature `--output-frames` which outputs paths through waygroups. Simple
  way to show possible “choke points”
* `--split-into-single-paths-by` value changed to `longest-path`, `as-crow-flies`
* Internal refactoring to attempt to speed up, and print more statistics.

## v0.23.0 (2024-03-18)

* When splitting into paths, the sinuosity value of the path is calculated &
  included in results.
* New: `--max-sinuosity`: when splitting into paths, only include paths where
  the sinuosity is below this.
* New: `--split-into-single-paths-by crow_flies`: Groups can now be split into
  single paths based on the longest distance between end points, as well as the
  older longest path option. This works well for finding the longest path in a
  motorway network.

## v0.22.0 (2024-03-10)

* `osm-lump-ways` can now use the powerful “tag filter function” from
  `osm-lump-ways-down`
* Internal refactoring to reduce code duplication

## v0.21.0 (2024-02-29)

* AND'ing filters together now possible with `∧` (`U+2227 LOGICAL AND`):
* New “Tag Filter Function” allows much more powerful, and complicated, tag
  filtering

## v0.20.0 (2024-02-28)

* Length of a loop (in metres) now included as GeoJSON property
* The upstream lengths is also saved as a points file `upstream-points`.

## v0.19.0 (2024-01-30)

* New binary `osm-lump-ways-down` which does cycle/loop detection and
  calculates upstream lengths
* Minor improvements to output logging messages

## v0.18.0 (2023-11-29)

* Roll back osmio library upgrade due to performance degregation
* Minor performance tweaks

## v0.17.0 (2023-11-27)

* Refactor to speed up processing
* Upgrade software libraries to speed up processing

## v0.16.0 (2023-11-23)

* Refactoring to speed up processing (speed up `WayGroup::set_coords`, use stringpbf)
* Use new osmio version which speeds up file reading

## v0.15.0 (2023-11-16)

* Refactor to use less memory by not storing data longer than we need it
* Speed improvements by refactoring

## v0.14.0 (2023-11-15)

* Massive speed improvements to the “reorder data internally”.
* Upgrade file reading library, which speeds up processing.

## v0.13.0 (2023-10-27)
 
* Improvements to output messages (showing time things took, warning if no
  matching way group)
* More options to filter the data (`--only-these-way-groups-nodeid`) to track
  down debugging
* `all_wayids` option now a string `wNNN` which can be fed into `osmium getid`.
* Fix bug where it crashed if a way has a doubled node (happened in Antarctica)

## v0.12.0 (2023-10-19)

* Output improvements with log messages and progress bars
* Small bug fixes when generating properties
* Refactor to parallelize the initial data read more
* Support writing to GeoJSONSeq format files, which tippecanoe can read much
  faster

## v0.11.0 (2023-10-18)

* Update dependency to use a version without a bug
* If the output file already exists, then exit with error so script can know
  “something is not as I expect”

## v0.10.0 (2023-10-13)

* Refactor the re-ordering to fix bug which practically caused an infinite
  loop.
* Internal refactor to slightly speed up initial file read, incl. with node
  positions

## v0.9.0 (2023-10-11)

* Data is internally re-ordered, which makes output files smaller, especially
  if using `--save-as-linestrings`
* When splitting into single paths, use of --min-length-m now possible

## v0.8.0 (2023-10-07)

* Refactor “split into single paths” to be more effecient and faster.
* Improve log output accuracy

## v0.7.0 (2023-10-03)

* Refactoring to reduce memory usage for larger datasets.
* New feature: `--min-dist-to-longer` to only include output which is this far
  from nearer items
* The “Split into single paths” function has been refactored to be much faster,
  and hence much more usable.
* Interal changes to logging output.

## v0.6.0 (2023-09-28)

* Error messages if you give incorrect options
* Refactoring to reduce memory usage for larger datasets.
* Internal refactors to speed up
* “Distance to nearest longer item” ([Topographic
  isolation](https://en.wikipedia.org/wiki/Topographic_isolation)) refactored
  to be much faster. It's still disabled by default, but can be turned on with
  `--incl-dist-to-longer`.
* Log message output tweaks.

## v0.5.0 (2023-09-21)

* Internal refactoring to reduce required memory usage.
* Internal refactoring to make later refactoring easier & update dependencies.
* Reformatting of some output messages to be nicer.
* New tag filter: `-f ~k` to filter on keys which match a regex (and `-f ∄~k`
  for “does not contain a key that matches this regex”)

## v0.4.0 (2023-08-17)

* Output geojson now includes `num_ways` & `num_nodes`. Both integers saying
  how many OSM ways & nodes are in this group
* Some command line --flag aliases
* Minor console output tweaks, and dependency updates
* Relicence to MIT/Apache2.0

## v0.3.0 (2023-07-23)

 * New tag filter: `k≠v1,v2,…` / `k∉v1,v2,…` to exclude many values
 * OR'ing filters together now possible with `∨` (`U+2228 LOGICAL OR`):
   i.e. `-f highway -f ∃surface∨∃lanes` means “any ways with a `highway` tag
   _and_ either there is a `surface` tag _or_ a `lanes` tag.
 * Tag filter synonyms: `-f ∃k`, `-f ∄k`
 * Internal code refactoring
 * Add `-v`/`--verbose`/`-q`/`--quiet` options to help with output

## v0.2.0 ( 2023-04-22 )

 * New option to split into linestrings
 * wayids not included by default
 * Don't calculate distance to longer by default (this can take extreme times)
 * new attribute, which is `root_wayid / 120`. It should take 1 byte in a Map
   Vector Tile file, but allow enough categorisation
 * Support “Does not have this tag” filter

## v0.1.0


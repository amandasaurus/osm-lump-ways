# Changelog

## v2.1.0 (2024-12-03)

* New argument `--loops[-no]-incl-nids` to turn on/off whether the `nodes`
  argument is included in the loops file
* osm-lump-ways-down: grouped ends:
  * refactored to produce nicer lines
  * includes `to_upstream_m` & `from_upstream_m` for the upsteam values at the
	start (`from_…`) & end (`to_…`) and `avg_upstream_m` the artimetic average
	of the two.
  * New feature `--grouped-ends-max-upstream-delta NUMBER` will split a grouped
	end line if the difference between `from_…` & `to_…` is more than `NUMBER`.
	This is a rough way to prevent long lines.
  * new: `--grouped-ends-max-distance-m NUMBER`, as above, but straight line
	distance between start & end points.
* How segments are assigned to ends has been simplified, which fixes some bugs.
  

## v2.0.0

* Massive refactor to speed up code. Both `osm-lump-ways` &
  `osm-lump-ways-down` have been refactored to work much faster.
* Arguments renamed:
   * `--csv-stats-file` → `--loops-csv-stats-file`
   * `--openmetrics` → `--loops-openmetrics`
   * `--upstream-output-biggest-end` → `--flow-split-equally`
   * `--upstream-assign-end-by-tag` → `--flow-follows-tag`
* `root_wayid` in `osm-lump-ways` output replaced with `root_nodeid`.
* New `osm-lump-ways-down` feature: `--ends-csv`: Create/Update a CSV of end
  points.
* When using `--flow-follows-tag TAG`, this will also assign the
  upstream value for ways based on this tag, rather than splitting equally.

## v1.7.0 (2024-10-06)

* end tag values have less duplicates
* end tag values won't include null/unset tags
* Fix bug with the wrong end tag being set.

## v1.6.0 (2024-09-30)

* Fix bug with assigning the end when there's a split

## v1.5.3 (2024-09-26)

* Fix bug when including tag filter func files with comments (fixes #68)

## v1.5.2 (2024-09-23)

* Fix bug when generating grouped ends without an end tag

## v1.5.1 (2024-09-22)

* Correct typo in CLI arguments, causing `osm-lump-ways-down` to always fail.

## v1.5.0 (2024-09-22) (yanked)

* Added `--upstream-assign-end-by-tag TAGNAME`.
* Added `--end_tag TAG` which marks ends, upstreams, and grouped ends with the
  tag values which go through this end point.
* Renamed properties in upstream output (when `--group-by-ends`):
    `biggest_end_nid` → `end_nid`
    `biggest_end_upstream_m` → `end_upstream_m`.
* Fix bug in group by ends, where some little segments were being dropped.

## v1.4.0 (2024-09-16)

* Tag Filter Function's @ format can now include other files
* Internal refactoring to attempt to improve performance.

## v1.3.0 (2024-09-08)

* Update OSM PBF reading library

## v1.2.0 (2024-08-13)

* Within the loops file, the `nodes` is now a sorted (by nodeid), deduplicated
  list of node ids.
* Minor refactor of calculating loops & CSV stats files

## v1.1.0 (2024-08-07)

* osm-lump-ways-down now takes the target filename for `--ends`/`--loops` etc.
  rather than using the `-o` option.
* Detect & warn about negative object ids. Document how to work with that.
  (cf. issue #2)
* Default `--split-into-single-paths` uses `as-crow-flies`.
* Reduce memory needed when calculating frames, at the expense of taking
  longer.

## v1.0.0 (2024-07-02)

* Massive refactor to allow `--upstream-tag-biggest-end` to work with much less
  memory.
* In the loops stats file, loops not in any area are now called `unknown_area`,
  instead of `terranullis` (a term too associated with colonization)
* `--upstream-tag-ends` split into `--upstream-tag-biggest-end` &
  `--upstream-tag-ends-full`. `biggest` will only include data for the end with
  the largest upstream value. This uses a little less memory.
* Remove Strahler number calculation. It doesn't work well for OSM data
* Remove `--upstream-points`.
* Ends & Loops file are now only created if the `--ends`/`--loops` argument is
  given
* `--upstreams` now takes a filename argument, rather that taking the filename
  from `-o`.
* Upstream data can be written in CSV format now.
* `--group-by-ends` outputs all the waterways based on the biggest end

## v0.29.0 (2024-05-27)

* Loops dataset now includes the “geocoded” area value.

## v0.28.0 (2024-05-23)

* `osm-lump-ways-down` Internal refactoring to use less memory if upstreams
  aren't being outputted.

## v0.27.0 (2024-05-15)

* `osm-lump-ways-down`'s `upstreams` now has attributes for the end point this
  node goes to when the `--upstream-tag-ends` arg is given.
* `osm-lump-ways-down`'s `upstreams` can now have rounded values for
  `from_upstream_m` with the `--upstream-from-upstream-multiple`
* `osm-lump-ways-down`: Upstream line segments only produced if
  `--upstreams` argument given

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


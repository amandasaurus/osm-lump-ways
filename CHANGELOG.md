# Changelog

## Unreleased

* Data is internally re-ordered, which makes output files smaller, especially
  if using `--save-as-linestrings`

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


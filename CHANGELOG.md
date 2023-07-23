# Changelog

## Unreleased

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


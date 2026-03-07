use log::warn;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;
use smallvec::SmallVec;
use smol_str::SmolStr;
use std::path::Path;

#[derive(Debug, Clone)]
pub enum TagFilter {
    HasK(SmolStr),
    HasReK(Regex),
    HasKLeftRightBoth(SmolStr),
    NotHasKLeftRightBoth(SmolStr),
    NotHasK(SmolStr),
    NotHasReK(Regex),
    KV(SmolStr, SmolStr),
    KinV(SmolStr, Vec<SmolStr>),
    KnotInV(SmolStr, Vec<SmolStr>),
    HasKnotInV(SmolStr, Vec<SmolStr>),
    KneV(SmolStr, SmolStr),
    KreV(SmolStr, Regex),
    And(Vec<TagFilter>),
    Or(Vec<TagFilter>),
    OSMObj(bool, char, i64),
}

impl std::fmt::Display for TagFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TagFilter::HasK(k) => write!(f, "∃{k}"),
            TagFilter::HasReK(k) => write!(f, "∃~{k}"),
            TagFilter::HasKLeftRightBoth(k) => write!(f, "∃(lrb){k}"),
            TagFilter::NotHasKLeftRightBoth(k) => write!(f, "∄(lrb){k}"),
            TagFilter::NotHasK(k) => write!(f, "∄{k}"),
            TagFilter::NotHasReK(k) => write!(f, "∄~{k}"),
            TagFilter::KV(k, v) => write!(f, "{k}={v}"),
            TagFilter::KneV(k, v) => write!(f, "{k}≠{v}"),
            TagFilter::KinV(k, vs) => write!(f, "{}∈{}", k, vs.join(",")),
            TagFilter::KnotInV(k, vs) => write!(f, "{}∉{}", k, vs.join(",")),
            TagFilter::HasKnotInV(k, vs) => write!(f, "∃{}∉{}", k, vs.join(",")),
            TagFilter::KreV(k, r) => write!(f, "{k}~{r}"),
            TagFilter::Or(tfs) => write!(
                f,
                "{}",
                tfs.iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("∨")
            ),
            TagFilter::And(tfs) => write!(
                f,
                "{}",
                tfs.iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("∧")
            ),
            TagFilter::OSMObj(incl, osm_type, osm_id) => {
                write!(f, "{}{}{}", if *incl { "" } else { "¬" }, osm_type, osm_id)
            }
        }
    }
}

impl PartialEq for TagFilter {
    fn eq(&self, other: &TagFilter) -> bool {
        format!("{self:?}") == format!("{other:?}")
    }
}

impl TagFilter {
    pub fn filter(&self, o: &impl osmio::OSMObjBase) -> bool {
        match self {
            TagFilter::HasK(k) => o.has_tag(k),
            TagFilter::HasReK(kre) => o.tags().any(|(k, _v)| kre.is_match(k)),
            TagFilter::HasKLeftRightBoth(k) => {
                o.has_tag(k)
                    || o.has_tag(format!("{k}:both"))
                    || (o.has_tag(format!("{k}:left")) && o.has_tag(format!("{k}:right")))
            }
            TagFilter::NotHasKLeftRightBoth(k) => {
                !(o.has_tag(k)
                    || o.has_tag(format!("{k}:both"))
                    || (o.has_tag(format!("{k}:left")) && o.has_tag(format!("{k}:right"))))
            }
            TagFilter::NotHasK(k) => !o.has_tag(k),
            TagFilter::NotHasReK(kre) => !o.tags().any(|(k, _v)| kre.is_match(k)),
            TagFilter::KV(k, v) => o.tag(k) == Some(v),
            TagFilter::KneV(k, v) => o.tag(k).is_none_or(|v2| v != v2),
            TagFilter::KinV(k, vs) => vs.iter().any(|v| o.tag(k).is_some_and(|v2| v == v2)),
            TagFilter::KnotInV(k, vs) => o
                .tag(k)
                .is_none_or(|tag_value| vs.iter().all(|v| v != tag_value)),
            TagFilter::HasKnotInV(k, vs) => o
                .tag(k)
                .is_some_and(|tag_value| vs.iter().all(|v| v != tag_value)),
            TagFilter::KreV(k, r) => o.tag(k).is_some_and(|v| r.is_match(v)),
            TagFilter::Or(tfs) => tfs.iter().any(|tf| tf.filter(o)),
            TagFilter::And(tfs) => tfs.iter().all(|tf| tf.filter(o)),
            TagFilter::OSMObj(incl, osm_type, osm_id) => {
                if o.object_type().name_short().chars().next().unwrap() == *osm_type
                    && o.id() == *osm_id
                {
                    *incl
                } else {
                    !incl
                }
            }
        }
    }
}

impl std::str::FromStr for TagFilter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.starts_with('"') || s.ends_with('"') {
            warn!(
                "Input string {s} starts and/or ends with a double quote. Have you accidentally over-quoted? Continuing with that."
            );
        }
        if s.contains('∨') {
            let tfs = s
                .split('∨')
                .map(str::parse::<TagFilter>)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TagFilter::Or(tfs))
        } else if s.contains('∧') {
            let tfs = s
                .split('∧')
                .map(str::parse::<TagFilter>)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TagFilter::And(tfs))
        } else if let Some((incl_type, idstr)) =
            split_prefix(s, &["w", "!w", "¬w", "r", "!r", "¬r"])
            && let Ok(id) = idstr.parse()
        {
            match incl_type {
                "w" => Ok(TagFilter::OSMObj(true, 'w', id)),
                "!w" | "¬w" => Ok(TagFilter::OSMObj(false, 'w', id)),
                "r" => Ok(TagFilter::OSMObj(true, 'r', id)),
                "!r" | "¬r" => Ok(TagFilter::OSMObj(false, 'r', id)),
                _ => unreachable!(),
            }
        } else if s.contains('=') {
            let s = s.splitn(2, '=').collect::<Vec<_>>();
            if s[1].contains(',') {
                let vs = s[1].split(',').map(SmolStr::from).collect::<Vec<_>>();
                Ok(TagFilter::KinV(s[0].into(), vs))
            } else {
                Ok(TagFilter::KV(s[0].into(), s[1].into()))
            }
        } else if s.contains('∈') {
            let s = s.splitn(2, '∈').collect::<Vec<_>>();
            let vs = s[1].split(',').map(SmolStr::from).collect::<Vec<_>>();
            Ok(TagFilter::KinV(s[0].into(), vs))
        } else if s.contains('≠') {
            let s = s.splitn(2, '≠').collect::<Vec<_>>();
            if s[1].contains(',') {
                let vs = s[1].split(',').map(SmolStr::from).collect::<Vec<_>>();
                Ok(TagFilter::KnotInV(s[0].into(), vs))
            } else {
                Ok(TagFilter::KneV(s[0].into(), s[1].into()))
            }
        } else if let Some(key) = s.strip_prefix("∃(lrb)") {
            Ok(TagFilter::HasKLeftRightBoth(key.into()))
        } else if let Some(key) = s.strip_prefix("∄(lrb)") {
            Ok(TagFilter::NotHasKLeftRightBoth(key.into()))
        } else if s.starts_with("∃") && s.contains('∉') {
            let s = s.strip_prefix("∃").unwrap();
            let s = s.splitn(2, '∉').collect::<Vec<_>>();

            let vs = s[1].split(',').map(SmolStr::from).collect::<Vec<_>>();
            Ok(TagFilter::HasKnotInV(s[0].into(), vs))
        } else if s.contains('∉') {
            let s = s.splitn(2, '∉').collect::<Vec<_>>();
            let vs = s[1].split(',').map(SmolStr::from).collect::<Vec<_>>();
            Ok(TagFilter::KnotInV(s[0].into(), vs))
        } else if let Some(regex) = s.strip_prefix('~') {
            let regex = Regex::new(regex).map_err(|_| "Invalid regex")?;
            Ok(TagFilter::HasReK(regex))
        } else if let Some(regex) = s.strip_prefix("∃~") {
            let regex = Regex::new(regex).map_err(|_| "Invalid regex")?;
            Ok(TagFilter::HasReK(regex))
        } else if let Some(regex) = s.strip_prefix("∄~") {
            let regex = Regex::new(regex).map_err(|_| "Invalid regex")?;
            Ok(TagFilter::NotHasReK(regex))
        } else if let Some(key) = s.strip_prefix('∃') {
            Ok(TagFilter::HasK(key.into()))
        } else if let Some(key) = s.strip_prefix('∄') {
            Ok(TagFilter::NotHasK(key.into()))
        } else if s.contains('~') {
            let s = s.splitn(2, '~').collect::<Vec<_>>();
            let regex = Regex::new(s[1]).map_err(|_| "Invalid regex")?;
            Ok(TagFilter::KreV(s[0].into(), regex))
        } else if s.is_empty() {
            Err("An empty string is not a valid tag filter".to_string())
        } else {
            Ok(TagFilter::HasK(s.into()))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TagFilterFuncElement {
    FilterMatchThenTrue(TagFilter),
    FilterMatchThenFalse(TagFilter),
    AlwaysTrue,
    AlwaysFalse,
}

impl std::str::FromStr for TagFilterFuncElement {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s == "T" {
            Ok(TagFilterFuncElement::AlwaysTrue)
        } else if s == "F" {
            Ok(TagFilterFuncElement::AlwaysFalse)
        } else if let Some(filter) = s.strip_suffix("→T") {
            let filter: TagFilter = filter.parse()?;
            Ok(TagFilterFuncElement::FilterMatchThenTrue(filter))
        } else if let Some(filter) = s.strip_suffix("→F") {
            let filter: TagFilter = filter.parse()?;
            Ok(TagFilterFuncElement::FilterMatchThenFalse(filter))
        } else {
            Err(format!("Unknown Tag Filter Func: {s}"))
        }
    }
}

impl std::fmt::Display for TagFilterFuncElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TagFilterFuncElement::AlwaysTrue => write!(f, "T"),
            TagFilterFuncElement::AlwaysFalse => write!(f, "F"),
            TagFilterFuncElement::FilterMatchThenTrue(flt) => write!(f, "{flt}→T"),
            TagFilterFuncElement::FilterMatchThenFalse(flt) => write!(f, "{flt}→F"),
        }
    }
}

impl TagFilterFuncElement {
    pub fn result(&self, o: &impl osmio::OSMObjBase) -> Option<bool> {
        match self {
            TagFilterFuncElement::AlwaysTrue => Some(true),
            TagFilterFuncElement::AlwaysFalse => Some(false),
            TagFilterFuncElement::FilterMatchThenTrue(f) => {
                if f.filter(o) {
                    Some(true)
                } else {
                    None
                }
            }
            TagFilterFuncElement::FilterMatchThenFalse(f) => {
                if f.filter(o) {
                    Some(false)
                } else {
                    None
                }
            }
        }
    }
}

// waterway=canal∧lock=yes→T;waterway=canal→F;waterway→T;F
#[derive(Debug, Clone, PartialEq)]
pub struct TagFilterFunc(Vec<TagFilterFuncElement>);

impl std::str::FromStr for TagFilterFunc {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (filename, contents) = match s.strip_prefix('@') {
            None => (None, s.to_string()),
            Some(filename) => {
                let contents = std::fs::read_to_string(filename)
                    .map_err(|e| format!("Couldn't read filename {filename}: {e}"))?;
                (Some(Path::new(filename)), contents)
            }
        };

        let mut s = contents.trim().to_owned();

        let new_s_regex = Regex::new("(?m)^include ([^;]+);").unwrap();

        loop {
            //include FILENAME;
            let new_s = new_s_regex
                .replace_all(&s, |caps: &regex::Captures| {
                    let incl_filename = &caps[1];
                    let incl_path = filename
                        .expect("Can't do include without using @filename syntax")
                        .parent()
                        .unwrap()
                        .join(incl_filename)
                        .canonicalize()
                        .unwrap();
                    std::fs::read_to_string(incl_path).unwrap_or_else(|_| {
                        panic!(
                            "Error in include in tagfilter function: Couldn't read filename {filename:?}"
                        )
                    })
                })
                .into_owned();

            // do it recursively
            if new_s == s {
                s = new_s;
                break;
            }
            s = new_s;
        }

        // remove comments
        let s: String = Regex::new("#[^\n]*\n")
            .unwrap()
            .replace_all(&s, "")
            .into_owned();

        let tffs = s
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::parse::<TagFilterFuncElement>)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TagFilterFunc(tffs))
    }
}

impl std::fmt::Display for TagFilterFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts = self
            .0
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<String>>();
        write!(f, "{}", parts.join(";"))
    }
}

impl TagFilterFunc {
    pub fn result(&self, o: &impl osmio::OSMObjBase) -> Option<bool> {
        self.0
            .iter()
            .map(|tff| tff.result(o))
            .find(std::option::Option::is_some)
            .flatten()
    }
}

pub fn obj_pass_filters(
    o: &(impl osmio::OSMObjBase + Sync + Send),
    tag_filters: &SmallVec<[TagFilter; 3]>,
    tag_filter_func: &Option<TagFilterFunc>,
) -> bool {
    if !tag_filters.is_empty() {
        tag_filters.par_iter().all(|tf| tf.filter(o))
    } else if let Some(tff) = tag_filter_func {
        tff.result(o)
            .expect("Tag Filter func did not complete. Perhaps missing last element of T or F?")
    } else {
        true
    }
}

/// If the string haystack starts with one of those prefixes (checked in order), then return
/// `Some((that_prefix`, `the_rest`)). Else None.
fn split_prefix<'a>(haystack: &'a str, prefixes: &'a [&'a str]) -> Option<(&'a str, &'a str)> {
    for prefix in prefixes {
        if let Some(suffix) = haystack.strip_prefix(prefix) {
            return Some((prefix, suffix));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use osmio::OSMObjBase;

    macro_rules! test_parse {
        ( $name:ident, $input:expr_2021, $expected_output:expr_2021 ) => {
            #[test]
            fn $name() {
                assert_eq!(($input).parse::<TagFilter>().unwrap(), $expected_output);
            }
        };
    }

    test_parse!(simple1, "name", TagFilter::HasK("name".into()));
    test_parse!(simple_w_space1, " name", TagFilter::HasK("name".into()));
    test_parse!(simple_w_space2, " name  \t", TagFilter::HasK("name".into()));
    test_parse!(parse1, "∃name", TagFilter::HasK("name".into()));
    test_parse!(
        parse2,
        "highway=motorway",
        TagFilter::KV("highway".into(), "motorway".into())
    );
    test_parse!(
        parse3,
        "highway≠motorway",
        TagFilter::KneV("highway".into(), "motorway".into())
    );
    test_parse!(
        parse4,
        "highway=motorway,primary",
        TagFilter::KinV("highway".into(), vec!["motorway".into(), "primary".into()])
    );

    test_parse!(
        parse_regex1,
        "~name:.*",
        TagFilter::HasReK(Regex::new("name:.*").unwrap())
    );
    test_parse!(
        parse_regex2,
        "∃~name:.*",
        TagFilter::HasReK(Regex::new("name:.*").unwrap())
    );
    test_parse!(
        parse_regex3,
        "∃~^name(:.+)?",
        TagFilter::HasReK(Regex::new("^name(:.+)?").unwrap())
    );
    test_parse!(
        parse_regex_not2,
        "∄~name:.*",
        TagFilter::NotHasReK(Regex::new("name:.*").unwrap())
    );
    test_parse!(
        parse_has_not_in1,
        "∃highway∉motorway,motorway_link",
        TagFilter::HasKnotInV(
            "highway".into(),
            vec!["motorway".into(), "motorway_link".into()]
        )
    );

    test_parse!(
        parse_lrb1,
        "∃(lrb)a",
        TagFilter::HasKLeftRightBoth("a".into())
    );
    test_parse!(
        parse_lrb2,
        "∄(lrb)b",
        TagFilter::NotHasKLeftRightBoth("b".into())
    );

    test_parse!(parse_osmid1, "w123", TagFilter::OSMObj(true, 'w', 123));
    test_parse!(parse_osmid2, "r123", TagFilter::OSMObj(true, 'r', 123));
    test_parse!(parse_osmid3, "!w3", TagFilter::OSMObj(false, 'w', 3));
    test_parse!(parse_osmid4, "¬w3", TagFilter::OSMObj(false, 'w', 3));

    #[test]
    fn parse() {
        assert!("".parse::<TagFilter>().is_err());
        assert_eq!(
            "highway∈motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KinV("highway".into(), vec!["motorway".into(), "primary".into()])
        );
        assert_eq!(
            "highway≠motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KnotInV("highway".into(), vec!["motorway".into(), "primary".into()])
        );
        assert_eq!(
            "highway∉motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KnotInV("highway".into(), vec!["motorway".into(), "primary".into()])
        );

        assert_eq!(
            "highway~motorway".parse::<TagFilter>().unwrap(),
            TagFilter::KreV("highway".into(), Regex::new("motorway").unwrap())
        );
        assert_eq!(
            "∄name".parse::<TagFilter>().unwrap(),
            TagFilter::NotHasK("name".into())
        );

        assert_eq!(
            "name∨highway".parse::<TagFilter>().unwrap(),
            TagFilter::Or(vec![
                TagFilter::HasK("name".into()),
                TagFilter::HasK("highway".into())
            ])
        );

        assert_eq!(
            "name∧highway".parse::<TagFilter>().unwrap(),
            TagFilter::And(vec![
                TagFilter::HasK("name".into()),
                TagFilter::HasK("highway".into())
            ])
        );
    }

    macro_rules! test_tag_filter {
        ( $name:ident, $input_tf:expr, $input_tags:expr, $expected_result:expr ) => {
            #[test]
            fn $name() {
                let tf: TagFilter = $input_tf.parse().unwrap();
                let mut n = osmio::obj_types::StringNodeBuilder::default()
                    ._id(1)
                    .build()
                    .unwrap();
                for (k, v) in $input_tags.iter() {
                    n.set_tag(k.to_string(), v.to_string());
                }
                assert_eq!(tf.filter(&n), $expected_result);
            }
        };
    }

    test_tag_filter!(filter_pass1, "highway", [("highway", "primary")], true);
    test_tag_filter!(
        filter_pass2,
        "highway∈primary,seconary",
        [("highway", "primary")],
        true
    );
    test_tag_filter!(
        filter_pass3,
        "∃highway∉primary,seconary",
        [("highway", "primary")],
        false
    );
    test_tag_filter!(
        filter_pass4,
        "∃highway∉primary,seconary",
        [("amenity", "bar")],
        false
    );
    test_tag_filter!(
        filter_pass5,
        "highway∉primary,seconary",
        [("amenity", "bar")],
        true
    );
    test_tag_filter!(
        filter_pass6,
        "∃highway∉primary,seconary",
        [("highway", "motorway")],
        true
    );
    test_tag_filter!(lrb_exists0, "∃(lrb)sidewalk", &[] as &[(&str, &str)], false);
    test_tag_filter!(lrb_exists1, "∃(lrb)sidewalk", [("sidewalk", "yes")], true);
    test_tag_filter!(
        lrb_exists2,
        "∃(lrb)sidewalk",
        [("sidewalk:both", "yes")],
        true
    );
    test_tag_filter!(
        lrb_exists3,
        "∃(lrb)sidewalk",
        [("sidewalk:left", "yes")],
        false
    );
    test_tag_filter!(
        lrb_exists4,
        "∃(lrb)sidewalk",
        [("sidewalk:left", "yes"), ("sidewalk:right", "yes")],
        true
    );

    test_tag_filter!(
        lrb_notexists0,
        "∄(lrb)sidewalk",
        &[] as &[(&str, &str)],
        true
    );
    test_tag_filter!(
        lrb_notexists1,
        "∄(lrb)sidewalk",
        [("sidewalk", "yes")],
        false
    );
    test_tag_filter!(
        lrb_notexists2,
        "∄(lrb)sidewalk",
        [("sidewalk:both", "yes")],
        false
    );
    test_tag_filter!(
        lrb_notexists3,
        "∄(lrb)sidewalk",
        [("sidewalk:left", "yes")],
        true
    );
    test_tag_filter!(
        lrb_notexists4,
        "∄(lrb)sidewalk",
        [("sidewalk:left", "yes"), ("sidewalk:right", "yes")],
        false
    );

    #[test]
    fn tag_filter_func_parse() {
        assert_eq!(
            "T".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::AlwaysTrue
        );
        assert_eq!(
            "F".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::AlwaysFalse
        );
        assert_eq!(
            "highway→T".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::FilterMatchThenTrue(TagFilter::HasK("highway".into()))
        );
        assert_eq!(
            "waterway→F".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::HasK("waterway".into()))
        );
        assert_eq!(
            "waterway=canal→F".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::KV(
                "waterway".into(),
                "canal".into()
            ))
        );
        assert_eq!(
            "waterway=canal∧lock=yes→F"
                .parse::<TagFilterFuncElement>()
                .unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::And(vec![
                TagFilter::KV("waterway".into(), "canal".into()),
                TagFilter::KV("lock".into(), "yes".into())
            ]))
        );
        assert_eq!(
            "waterway=canal∧usage∈headrace,tailrace→F"
                .parse::<TagFilterFuncElement>()
                .unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::And(vec![
                TagFilter::KV("waterway".into(), "canal".into()),
                TagFilter::KinV("usage".into(), vec!["headrace".into(), "tailrace".into()])
            ]))
        );

        assert!("highway".parse::<TagFilterFuncElement>().is_err());
        assert!("highway=primary".parse::<TagFilterFuncElement>().is_err());
    }

    macro_rules! test_parse_tag_filter_func {
        ( $name:ident, $input_tff:expr_2021, $input_tags:expr_2021, $expected_output:expr_2021 ) => {
            #[test]
            fn $name() {
                let tff: TagFilterFuncElement = $input_tff.parse().unwrap();
                let mut n = osmio::obj_types::StringNodeBuilder::default()
                    ._id(1)
                    .build()
                    .unwrap();
                for (k, v) in $input_tags.iter() {
                    n.set_tag(k.to_string(), v.to_string());
                }
                assert_eq!(tff.result(&n), $expected_output);
            }
        };
    }
    test_parse_tag_filter_func!(tff_parse1, "T", [("highway", "yes")], Some(true));
    test_parse_tag_filter_func!(tff_parse2, "F", [("highway", "yes")], Some(false));
    test_parse_tag_filter_func!(tff_parse3, "highway→T", [("highway", "yes")], Some(true));
    test_parse_tag_filter_func!(tff_parse4, "highway→T", [("natural", "water")], None);
    test_parse_tag_filter_func!(
        tff_parse5,
        "natural=water→T",
        [("natural", "water")],
        Some(true)
    );
    test_parse_tag_filter_func!(tff_parse6, "natural=water→T", [("natural", "rock")], None);

    #[test]
    fn tag_filter_func_list_parse() {
        assert_eq!(
            "T".parse::<TagFilterFunc>().unwrap(),
            TagFilterFunc(vec![TagFilterFuncElement::AlwaysTrue])
        );
        assert_eq!(
            "waterway→T;F".parse::<TagFilterFunc>().unwrap(),
            TagFilterFunc(vec![
                TagFilterFuncElement::FilterMatchThenTrue(TagFilter::HasK("waterway".into())),
                TagFilterFuncElement::AlwaysFalse
            ])
        );
    }

    macro_rules! test_parse_tag_filter_func_list {
        ( $name:ident, $input_tff:expr_2021, $input_tags:expr_2021, $expected_output:expr_2021 ) => {
            #[test]
            fn $name() {
                let tff: TagFilterFunc = $input_tff.parse().unwrap();
                let mut n = osmio::obj_types::StringNodeBuilder::default()
                    ._id(1)
                    .build()
                    .unwrap();
                for (k, v) in $input_tags.iter() {
                    n.set_tag(k.to_string(), v.to_string());
                }
                assert_eq!(tff.result(&n), $expected_output);
            }
        };
    }

    test_parse_tag_filter_func_list!(tffl_parse1, "T", [("highway", "yes")], Some(true));
    test_parse_tag_filter_func_list!(tffl_parse2, "highway→T;F", [("highway", "yes")], Some(true));
    test_parse_tag_filter_func_list!(
        tffl_parse3,
        "highway→T;F",
        [("natural", "yes")],
        Some(false)
    );
    test_parse_tag_filter_func_list!(
        tffl_parse4,
        "waterway=canal→F;waterway→T;F",
        [("waterway", "river")],
        Some(true)
    );
    test_parse_tag_filter_func_list!(
        tffl_parse5,
        "waterway=canal→F;waterway→T;F",
        [("highway", "primary")],
        Some(false)
    );
    test_parse_tag_filter_func_list!(
        tffl_parse6,
        "waterway=canal→F;waterway→T;F",
        [("waterway", "canal")],
        Some(false)
    );
    test_parse_tag_filter_func_list!(
        tffl_parse7,
        "# This is a test comment\nwaterway=canal→F;waterway→T;F",
        [("waterway", "canal")],
        Some(false)
    );
    test_parse_tag_filter_func_list!(
        tffl_parse_semicolon,
        r"waterway=put_in\u{3B}egress→F;waterway→T;F",
        [("waterway", "canal")],
        Some(true)
    );

    macro_rules! test_id_filter {
        ( $name:ident, $input_tf:expr, $input_type:ident, $input_id:expr, $expected_result:expr ) => {
            #[test]
            fn $name() {
                let tf: TagFilter = $input_tf.parse().unwrap();
                let o = osmio::obj_types::$input_type::default()
                    ._id($input_id)
                    .build()
                    .unwrap();
                assert_eq!(tf.filter(&o), $expected_result);
            }
        };
    }

    test_id_filter!(id_filter0, "w1", StringWayBuilder, 1, true);
    test_id_filter!(id_filter1, "w1", StringWayBuilder, 2, false);
    test_id_filter!(id_filter2, "w1", StringRelationBuilder, 1, false);
    test_id_filter!(id_filter3, "!w1", StringWayBuilder, 1, false);
    test_id_filter!(id_filter4, "!w1", StringWayBuilder, 2, true);
    test_id_filter!(id_filter5, "¬w1", StringWayBuilder, 1, false);
    test_id_filter!(id_filter6, "¬w1", StringWayBuilder, 2, true);
}

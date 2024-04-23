use log::warn;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use regex::Regex;

#[derive(Debug, Clone)]
pub enum TagFilter {
    HasK(String),
    HasReK(Regex),
    NotHasK(String),
    NotHasReK(Regex),
    KV(String, String),
    KinV(String, Vec<String>),
    KnotInV(String, Vec<String>),
    KneV(String, String),
    KreV(String, Regex),
    And(Vec<TagFilter>),
    Or(Vec<TagFilter>),
}

impl ToString for TagFilter {
    fn to_string(&self) -> String {
        match self {
            TagFilter::HasK(k) => format!("∃{}", k),
            TagFilter::HasReK(k) => format!("∃~{}", k),
            TagFilter::NotHasK(k) => format!("∄{}", k),
            TagFilter::NotHasReK(k) => format!("∄~{}", k),
            TagFilter::KV(k, v) => format!("{}={}", k, v),
            TagFilter::KneV(k, v) => format!("{}≠{}", k, v),
            TagFilter::KinV(k, vs) => format!("{}∈{}", k, vs.join(",")),
            TagFilter::KnotInV(k, vs) => format!("{}∉{}", k, vs.join(",")),
            TagFilter::KreV(k, r) => format!("{}~{}", k, r),
            TagFilter::Or(tfs) => tfs
                .iter()
                .map(|tf| tf.to_string())
                .collect::<Vec<_>>()
                .join("∨"),
            TagFilter::And(tfs) => tfs
                .iter()
                .map(|tf| tf.to_string())
                .collect::<Vec<_>>()
                .join("∧"),
        }
    }
}

impl PartialEq for TagFilter {
    fn eq(&self, other: &TagFilter) -> bool {
        format!("{:?}", self) == format!("{:?}", other)
    }
}

impl TagFilter {
    pub fn filter(&self, o: &impl osmio::OSMObjBase) -> bool {
        match self {
            TagFilter::HasK(k) => o.has_tag(k),
            TagFilter::HasReK(kre) => o.tags().any(|(k, _v)| kre.is_match(k)),
            TagFilter::NotHasK(k) => !o.has_tag(k),
            TagFilter::NotHasReK(kre) => !o.tags().any(|(k, _v)| kre.is_match(k)),
            TagFilter::KV(k, v) => o.tag(k) == Some(v),
            TagFilter::KneV(k, v) => o.tag(k).map_or(true, |v2| v != v2),
            TagFilter::KinV(k, vs) => vs.iter().any(|v| o.tag(k).map_or(false, |v2| v == v2)),
            TagFilter::KnotInV(k, vs) => o
                .tag(k)
                .map_or(true, |tag_value| vs.iter().all(|v| v != tag_value)),
            TagFilter::KreV(k, r) => o.tag(k).map_or(false, |v| r.is_match(v)),
            TagFilter::Or(tfs) => tfs.iter().any(|tf| tf.filter(o)),
            TagFilter::And(tfs) => tfs.iter().all(|tf| tf.filter(o)),
        }
    }
}

impl std::str::FromStr for TagFilter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.starts_with('"') || s.ends_with('"') {
            warn!("Input string {} starts and/or ends with a double quote. Have you accidentally over-quoted? Continuing with that.", s);
        }
        if s.contains('∨') {
            let tfs = s
                .split('∨')
                .map(|tf| tf.parse::<TagFilter>())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TagFilter::Or(tfs))
        } else if s.contains('∧') {
            let tfs = s
                .split('∧')
                .map(|tf| tf.parse::<TagFilter>())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TagFilter::And(tfs))
        } else if s.contains('=') {
            let s = s.splitn(2, '=').collect::<Vec<_>>();
            if s[1].contains(',') {
                let vs = s[1].split(',').map(String::from).collect::<Vec<_>>();
                Ok(TagFilter::KinV(s[0].to_string(), vs))
            } else {
                Ok(TagFilter::KV(s[0].to_string(), s[1].to_string()))
            }
        } else if s.contains('∈') {
            let s = s.splitn(2, '∈').collect::<Vec<_>>();
            let vs = s[1].split(',').map(String::from).collect::<Vec<_>>();
            Ok(TagFilter::KinV(s[0].to_string(), vs))
        } else if s.contains('≠') {
            let s = s.splitn(2, '≠').collect::<Vec<_>>();
            if s[1].contains(',') {
                let vs = s[1].split(',').map(String::from).collect::<Vec<_>>();
                Ok(TagFilter::KnotInV(s[0].to_string(), vs))
            } else {
                Ok(TagFilter::KneV(s[0].to_string(), s[1].to_string()))
            }
        } else if s.contains('∉') {
            let s = s.splitn(2, '∉').collect::<Vec<_>>();
            let vs = s[1].split(',').map(String::from).collect::<Vec<_>>();
            Ok(TagFilter::KnotInV(s[0].to_string(), vs))
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
            Ok(TagFilter::HasK(key.to_string()))
        } else if let Some(key) = s.strip_prefix('∄') {
            Ok(TagFilter::NotHasK(key.to_string()))
        } else if s.contains('~') {
            let s = s.splitn(2, '~').collect::<Vec<_>>();
            let regex = Regex::new(s[1]).map_err(|_| "Invalid regex")?;
            Ok(TagFilter::KreV(s[0].to_string(), regex))
        } else if s.is_empty() {
            Err("An empty string is not a valid tag filter".to_string())
        } else {
            Ok(TagFilter::HasK(s.to_string()))
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
            Err(format!("Unknown Tag Filter Func: {}", s))
        }
    }
}

impl ToString for TagFilterFuncElement {
    fn to_string(&self) -> String {
        match self {
            TagFilterFuncElement::AlwaysTrue => "T".to_string(),
            TagFilterFuncElement::AlwaysFalse => "F".to_string(),
            TagFilterFuncElement::FilterMatchThenTrue(f) => format!("{}→T", f.to_string()),
            TagFilterFuncElement::FilterMatchThenFalse(f) => format!("{}→F", f.to_string()),
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
        let s = s.trim();
        let s = Regex::new("#[^\n]*\n").unwrap().replace_all(s, "");
        let tffs = s
            .split(';')
            .map(|src| src.trim())
            .filter(|s| s.len() > 0)
            .map(|tff| tff.parse::<TagFilterFuncElement>())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TagFilterFunc(tffs))
    }
}

impl ToString for TagFilterFunc {
    fn to_string(&self) -> String {
        let parts = self
            .0
            .iter()
            .map(|tff| tff.to_string())
            .collect::<Vec<String>>();
        parts.join(";")
    }
}

impl TagFilterFunc {
    pub fn result(&self, o: &impl osmio::OSMObjBase) -> Option<bool> {
        self.0
            .iter()
            .map(|tff| tff.result(o))
            .find(|res| res.is_some())
            .flatten()
    }
}

pub(crate) fn obj_pass_filters(
    o: &(impl osmio::OSMObjBase + Sync + Send),
    tag_filters: &[TagFilter],
    tag_filter_func: &Option<TagFilterFunc>,
) -> bool {
    if !tag_filters.is_empty() {
        tag_filters.par_iter().all(|tf| tf.filter(o))
    } else if let Some(ref tff) = tag_filter_func {
        tff.result(o)
            .expect("Tag Filter func did not complete. Perhaps missing last element of T or F?")
    } else {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use osmio::OSMObjBase;

    macro_rules! test_parse {
        ( $name:ident, $input:expr, $expected_output:expr ) => {
            #[test]
            fn $name() {
                assert_eq!(($input).parse::<TagFilter>().unwrap(), $expected_output);
            }
        };
    }

    test_parse!(simple1, "name", TagFilter::HasK("name".to_string()));
    test_parse!(
        simple_w_space1,
        " name",
        TagFilter::HasK("name".to_string())
    );
    test_parse!(
        simple_w_space2,
        " name  \t",
        TagFilter::HasK("name".to_string())
    );
    test_parse!(parse1, "∃name", TagFilter::HasK("name".to_string()));
    test_parse!(
        parse2,
        "highway=motorway",
        TagFilter::KV("highway".to_string(), "motorway".to_string())
    );
    test_parse!(
        parse3,
        "highway≠motorway",
        TagFilter::KneV("highway".to_string(), "motorway".to_string())
    );
    test_parse!(
        parse4,
        "highway=motorway,primary",
        TagFilter::KinV(
            "highway".to_string(),
            vec!["motorway".to_string(), "primary".to_string()]
        )
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

    #[test]
    fn parse() {
        assert!("".parse::<TagFilter>().is_err());
        assert_eq!(
            "highway∈motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KinV(
                "highway".to_string(),
                vec!["motorway".to_string(), "primary".to_string()]
            )
        );
        assert_eq!(
            "highway≠motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KnotInV(
                "highway".to_string(),
                vec!["motorway".to_string(), "primary".to_string()]
            )
        );
        assert_eq!(
            "highway∉motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KnotInV(
                "highway".to_string(),
                vec!["motorway".to_string(), "primary".to_string()]
            )
        );

        assert_eq!(
            "highway~motorway".parse::<TagFilter>().unwrap(),
            TagFilter::KreV("highway".to_string(), Regex::new("motorway").unwrap())
        );
        assert_eq!(
            "∄name".parse::<TagFilter>().unwrap(),
            TagFilter::NotHasK("name".to_string())
        );

        assert_eq!(
            "name∨highway".parse::<TagFilter>().unwrap(),
            TagFilter::Or(vec![
                TagFilter::HasK("name".to_string()),
                TagFilter::HasK("highway".to_string())
            ])
        );

        assert_eq!(
            "name∧highway".parse::<TagFilter>().unwrap(),
            TagFilter::And(vec![
                TagFilter::HasK("name".to_string()),
                TagFilter::HasK("highway".to_string())
            ])
        );
    }

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
            TagFilterFuncElement::FilterMatchThenTrue(TagFilter::HasK("highway".to_string()))
        );
        assert_eq!(
            "waterway→F".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::HasK("waterway".to_string()))
        );
        assert_eq!(
            "waterway=canal→F".parse::<TagFilterFuncElement>().unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::KV(
                "waterway".to_string(),
                "canal".to_string()
            ))
        );
        assert_eq!(
            "waterway=canal∧lock=yes→F"
                .parse::<TagFilterFuncElement>()
                .unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::And(vec![
                TagFilter::KV("waterway".to_string(), "canal".to_string()),
                TagFilter::KV("lock".to_string(), "yes".to_string())
            ]))
        );
        assert_eq!(
            "waterway=canal∧usage∈headrace,tailrace→F"
                .parse::<TagFilterFuncElement>()
                .unwrap(),
            TagFilterFuncElement::FilterMatchThenFalse(TagFilter::And(vec![
                TagFilter::KV("waterway".to_string(), "canal".to_string()),
                TagFilter::KinV(
                    "usage".to_string(),
                    vec!["headrace".to_string(), "tailrace".to_string()]
                )
            ]))
        );

        assert!("highway".parse::<TagFilterFuncElement>().is_err());
        assert!("highway=primary".parse::<TagFilterFuncElement>().is_err());
    }

    macro_rules! test_parse_tag_filter_func {
        ( $name:ident, $input_tff:expr, $input_tags:expr, $expected_output:expr ) => {
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
                TagFilterFuncElement::FilterMatchThenTrue(TagFilter::HasK("waterway".to_string())),
                TagFilterFuncElement::AlwaysFalse
            ])
        );
    }

    macro_rules! test_parse_tag_filter_func_list {
        ( $name:ident, $input_tff:expr, $input_tags:expr, $expected_output:expr ) => {
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
}

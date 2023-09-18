use regex::Regex;

#[derive(Debug, Clone)]
pub enum TagFilter {
    HasK(String),
    NotHasK(String),
    KV(String, String),
    KinV(String, Vec<String>),
    KnotInV(String, Vec<String>),
    KneV(String, String),
    KreV(String, Regex),
    Or(Vec<TagFilter>),
}

impl ToString for TagFilter {
    fn to_string(&self) -> String {
        match self {
            TagFilter::HasK(k) => format!("∃{}", k),
            TagFilter::NotHasK(k) => format!("∄{}", k),
            TagFilter::KV(k, v) => format!("{}={}", k, v),
            TagFilter::KneV(k, v) => format!("{}≠{}", k, v),
            TagFilter::KinV(k, vs) => format!("{}∈{}", k, vs.join(",").to_string()),
            TagFilter::KnotInV(k, vs) => format!("{}∉{}", k, vs.join(",").to_string()),
            TagFilter::KreV(k, r) => format!("{}~{}", k, r),
            TagFilter::Or(tfs) => tfs
                .iter()
                .map(|tf| tf.to_string())
                .collect::<Vec<_>>()
                .join("∨"),
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
            TagFilter::NotHasK(k) => !o.has_tag(k),
            TagFilter::KV(k, v) => o.tag(k) == Some(v),
            TagFilter::KneV(k, v) => o.tag(k).map_or(true, |v2| v != v2),
            TagFilter::KinV(k, vs) => vs.iter().any(|v| o.tag(k).map_or(false, |v2| v == v2)),
            TagFilter::KnotInV(k, vs) => o
                .tag(k)
                .map_or(true, |tag_value| vs.iter().all(|v| v != tag_value)),
            TagFilter::KreV(k, r) => o.tag(k).map_or(false, |v| r.is_match(v)),
            TagFilter::Or(tfs) => tfs.iter().any(|tf| tf.filter(o)),
        }
    }
}

impl std::str::FromStr for TagFilter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if s.contains('∨') {
            let tfs = s
                .split('∨')
                .map(|tf| tf.parse::<TagFilter>())
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TagFilter::Or(tfs))
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
        } else if s.contains('~') {
            let s = s.splitn(2, '~').collect::<Vec<_>>();
            Ok(TagFilter::KreV(s[0].to_string(), Regex::new(s[1]).unwrap()))
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
        } else if s.starts_with('∃') {
            Ok(TagFilter::HasK(s.chars().skip(1).collect::<String>()))
        } else if s.starts_with('∄') {
            Ok(TagFilter::NotHasK(s.chars().skip(1).collect::<String>()))
        } else if s.is_empty() {
            Err("An empty string is not a valid tag filter".to_string())
        } else {
            Ok(TagFilter::HasK(s.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_parse {
        ( $name:ident, $input:expr, $expected_output:expr ) => {
            #[test]
            fn $name() {
                assert_eq!(
                    ($input).parse::<TagFilter>().unwrap(),
                    $expected_output
                );
            }
        };
    }

    test_parse!(simple1, "name", TagFilter::HasK("name".to_string()));
    test_parse!(simple_w_space1, " name", TagFilter::HasK("name".to_string()));
    test_parse!(simple_w_space2, " name  \t", TagFilter::HasK("name".to_string()));
    test_parse!(parse1, "∃name", TagFilter::HasK("name".to_string()));
    test_parse!(parse2, "highway=motorway", TagFilter::KV("highway".to_string(), "motorway".to_string()));
    test_parse!(parse3, "highway≠motorway", TagFilter::KneV("highway".to_string(), "motorway".to_string()));
    test_parse!(parse4, "highway=motorway,primary", TagFilter::KinV( "highway".to_string(), vec!["motorway".to_string(), "primary".to_string()]) );


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

    }
}

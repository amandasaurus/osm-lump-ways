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
            TagFilter::KnotInV(k, vs) => o.tag(k).map_or(true, |tag_value| vs.iter().all(|v| v != tag_value)),
            TagFilter::KreV(k, r) => o.tag(k).map_or(false, |v| r.is_match(v)),
        }
    }
}

impl std::str::FromStr for TagFilter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains('=') {
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
        } else {
            Ok(TagFilter::HasK(s.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test1() {
        assert_eq!(
            "name".parse::<TagFilter>().unwrap(),
            TagFilter::HasK("name".to_string())
        );
        assert_eq!(
            "∃name".parse::<TagFilter>().unwrap(),
            TagFilter::HasK("name".to_string())
        );
        assert_eq!(
            "highway=motorway".parse::<TagFilter>().unwrap(),
            TagFilter::KV("highway".to_string(), "motorway".to_string())
        );
        assert_eq!(
            "highway≠motorway".parse::<TagFilter>().unwrap(),
            TagFilter::KneV("highway".to_string(), "motorway".to_string())
        );

        assert_eq!(
            "highway=motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KinV(
                "highway".to_string(),
                vec!["motorway".to_string(), "primary".to_string()]
            )
        );
        assert_eq!(
            "highway∈motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KinV(
                "highway".to_string(),
                vec!["motorway".to_string(), "primary".to_string()]
            )
        );
        assert_eq!(
            "highway≠motorway,primary".parse::<TagFilter>().unwrap(),
            TagFilter::KnotInV("highway".to_string(), vec!["motorway".to_string(), "primary".to_string()])
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
    }
}

#[derive(Debug, Clone, Hash, serde::Serialize, PartialEq, Eq)]
pub(crate) struct TagGrouper(Vec<String>);

impl std::str::FromStr for TagGrouper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(TagGrouper(s.split(',').map(|s| s.to_string()).collect()))
    }
}
impl std::string::ToString for TagGrouper {
    fn to_string(&self) -> String {
        self.0.join(",")
    }
}

impl TagGrouper {
    pub fn get_values(&self, o: &impl osmio::OSMObjBase) -> Option<String> {
        for k in self.0.iter() {
            if let Some(v) = o.tag(k) {
                return Some(v.to_string());
            }
        }

        None
    }
}

#![allow(dead_code)]

use super::way_id_rel_tags::WayIdToRelationTags;

#[derive(Debug, Clone, Hash, serde::Serialize, PartialEq, Eq)]
pub struct TagGrouper(Vec<String>);

impl std::str::FromStr for TagGrouper {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(TagGrouper(
            s.split(',').map(std::string::ToString::to_string).collect(),
        ))
    }
}
impl std::fmt::Display for TagGrouper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.join(","))
    }
}

impl TagGrouper {
    pub fn get_values(
        &self,
        o: &impl osmio::OSMObjBase,
        relation_tags: &WayIdToRelationTags,
    ) -> Option<String> {
        // Try the relation tags
        if o.object_type() == osmio::OSMObjectType::Way && relation_tags.contains_wid(o.id()) {
            for k in &self.0 {
                if let Some(v) = relation_tags.way_tags(o.id(), k) {
                    return Some(v.to_string());
                }
            }
        }

        // If we're still here, try the way tags
        for k in &self.0 {
            if let Some(v) = o.tag(k) {
                return Some(v.to_string());
            }
        }

        // Got to here, so no tag
        None
    }
}

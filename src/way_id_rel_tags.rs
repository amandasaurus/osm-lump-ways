use crate::sorted_slice_store::SortedSliceMap;
use num_format::{Locale, ToFormattedString};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;

#[derive(Default, Debug)]
pub struct WayIdToRelationTags {
    wid_to_rid: HashMap<i64, i64>,
    rid_to_tags: HashMap<i64, SortedSliceMap<String, String>>,

    /// How many members does this relation id have
    rid_to_nmembers: HashMap<i64, usize>,
}

impl WayIdToRelationTags {
    pub fn record_relation(&mut self, rel: &impl osmio::Relation, only_roles: &[String]) {
        // Save number of members
        let nmembers = rel.members().count();
        self.rid_to_nmembers.insert(rel.id(), nmembers);

        // Save tags
        self.rid_to_tags.insert(
            rel.id(),
            SortedSliceMap::from_iter(rel.tags().map(|(k, v)| (k.to_string(), v.to_string()))),
        );

        for (_objtype, wid, _role) in rel
            .members()
            .filter(|m| m.0 == osmio::OSMObjectType::Way)
            .filter(|m| only_roles.is_empty() || only_roles.iter().any(|r| r == m.2))
        {
            // Update which relation we use for this wayid
            self.wid_to_rid
                .entry(wid)
                .and_modify(|rid| {
                    // If this wid already has a rid, then overwrite it iff we currenty have more
                    // members
                    if nmembers >= *self.rid_to_nmembers.get(rid).unwrap() {
                        *rid = rel.id()
                    }
                })
                // Not seen before → simple insert
                .or_insert(rel.id());
        }
    }

    /// For this way id, what is the value of this tag
    /// None meaning the way isn't in the store, or there is no tag for this relation
    pub fn way_tags(&self, wid: i64, key: &str) -> Option<&str> {
        self.wid_to_rid
            .get(&wid)
            .and_then(|rid| self.rid_to_tags.get(rid))
            .and_then(|tags| tags.get(key))
            .map(|v| v.as_str())
    }

    /// True iff this way is in this list
    pub fn contains_wid(&self, wid: i64) -> bool {
        self.wid_to_rid.contains_key(&wid)
    }

    pub fn summary(&self) -> String {
        format!(
            "{} relations, {} ways, {} relation tags",
            self.rid_to_tags.len().to_formatted_string(&Locale::en),
            self.wid_to_rid.len().to_formatted_string(&Locale::en),
            self.rid_to_tags
                .par_iter()
                .map(|(_, tags)| tags.len())
                .sum::<usize>()
                .to_formatted_string(&Locale::en),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple1() {
        let mut way_id_rel_tags = WayIdToRelationTags::default();
        let mut r = osmio::obj_types::StringRelationBuilder::default();
        r._id(1);
        r._members(vec![(osmio::OSMObjectType::Way, 1, "".into())]);
        r._tags(
            vec![
                ("name".into(), "Foo".into()),
                ("waterway".into(), "river".into()),
            ]
            .into(),
        );
        let r = r.build().unwrap();

        way_id_rel_tags.record_relation(&r, &[]);

        assert!(way_id_rel_tags.way_tags(1, "highway").is_none());
        assert_eq!(way_id_rel_tags.way_tags(1, "name"), Some("Foo"));

        assert!(way_id_rel_tags.way_tags(2, "highway").is_none());
        assert!(way_id_rel_tags.way_tags(2, "name").is_none());
    }
}

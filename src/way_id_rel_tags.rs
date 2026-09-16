use crate::sorted_slice_store::SortedSliceMap;
use num_format::{Locale, ToFormattedString};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;

#[derive(Debug)]
struct RelationData {
    tags: SortedSliceMap<String, String>,
    nmembers: usize,
    timestamp: u32,
}

#[derive(Default, Debug)]
pub struct WayIdToRelationTags {
    wid_to_rid: HashMap<i64, i64>,
    relations: HashMap<i64, RelationData>,
}

impl WayIdToRelationTags {
    pub fn record_relation(&mut self, rel: &impl osmio::Relation, only_roles: &[String]) {
        let nmembers = rel.members().count();

        let mut way_passed_filter = false;
        for (_objtype, wid, _role) in rel
            .members()
            .filter(|m| m.0 == osmio::OSMObjectType::Way)
            .filter(|m| only_roles.is_empty() || only_roles.iter().any(|r| r == m.2))
        {
            way_passed_filter = true;

            // Update which relation we use for this wayid
            self.wid_to_rid
                .entry(wid)
                .and_modify(|rid| {
                    // If this wid already has a rid, then overwrite it iff we currenty have more
                    // members
                    if let Some(previous_rel) = self.relations.get(rid)
                        && nmembers >= previous_rel.nmembers
                    {
                        *rid = rel.id();
                    }
                })
                // Not seen before → simple insert
                .or_insert(rel.id());
        }

        if way_passed_filter {
            // If none of the members of this relation are added (e.g. lacking the role), then
            // don't bother storing this relation
            let timestamp =
                u32::try_from(rel.timestamp().as_ref().unwrap().to_epoch_number() - 1_000_000_000)
                    .unwrap();

            let tags =
                SortedSliceMap::from_iter(rel.tags().map(|(k, v)| (k.to_string(), v.to_string())));

            // save this relation
            self.relations.insert(
                rel.id(),
                RelationData {
                    tags,
                    nmembers,
                    timestamp,
                },
            );
        }
    }

    /// Return the relation id that this way id is in, if any is recorded.
    #[must_use]
    pub fn relation(&self, wid: &i64) -> Option<&i64> {
        self.wid_to_rid.get(wid)
    }

    /// Returns the timestamp of the relation for this way
    #[must_use]
    pub fn relation_timestamp(&self, wid: &i64) -> Option<u32> {
        self.wid_to_rid
            .get(wid)
            .map(|rid| self.relations.get(rid).unwrap().timestamp)
    }

    /// For this way id, what is the value of this tag
    /// None meaning the way isn't in the store, or there is no tag for this relation
    pub fn way_tag_value_only(&self, wid: i64, key: &str) -> Option<&str> {
        self.wid_to_rid
            .get(&wid)
            .map(|rid| &self.relations.get(rid).unwrap().tags)
            .and_then(|tags| tags.get(key))
            .map(std::string::String::as_str)
    }

    /// What are the tags for this way from the relations
    pub fn way_tags_only(&self, wid: i64) -> impl Iterator<Item = &(String, String)> {
        self.wid_to_rid
            .get(&wid)
            .map(|rid| &self.relations.get(rid).unwrap().tags)
            .into_iter()
            .flat_map(SortedSliceMap::iter)
    }

    pub fn way_tag_value<'a>(
        &'a self,
        w: &'a impl osmio::OSMObjBase,
        key: &str,
    ) -> Option<&'a str> {
        self.way_tag_value_only(w.id(), key).or(w.tag(key))
    }

    pub fn way_tags<'a>(
        &'a self,
        w: &'a impl osmio::OSMObjBase,
    ) -> Box<dyn Iterator<Item = (&'a str, &'a str)> + 'a> {
        if let Some(r_tags) = self
            .wid_to_rid
            .get(&w.id())
            .map(|rid| &self.relations.get(rid).unwrap().tags)
        {
            Box::new(
                r_tags
                    .iter()
                    .map(|(k, v)| (k.as_str(), v.as_str()))
                    .chain(w.tags().filter(|(k, _v)| !r_tags.contains_key(*k))),
            )
        } else {
            Box::new(w.tags())
        }
    }

    /// True iff this way is in this list
    #[must_use]
    pub fn contains_wid(&self, wid: i64) -> bool {
        self.wid_to_rid.contains_key(&wid)
    }

    #[must_use]
    pub fn num_relations(&self) -> usize {
        self.relations.len()
    }
    #[must_use]
    pub fn num_ways(&self) -> usize {
        self.wid_to_rid.len()
    }

    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} relations, {} ways, {} relation tags",
            self.num_relations().to_formatted_string(&Locale::en),
            self.num_ways().to_formatted_string(&Locale::en),
            self.relations
                .par_iter()
                .map(|(_, rd)| rd.tags.len())
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
        r._timestamp(osmio::TimestampFormat::ISOString(
            "2026-01-01T00:00:00Z".to_string(),
        ));
        r._members(vec![(osmio::OSMObjectType::Way, 1, "".into())]);
        r._tags(
            vec![
                ("name".into(), "Foo".into()),
                ("waterway".into(), "river".into()),
            ]
            .into(),
        );
        let r = r.build().unwrap();

        let mut w = osmio::obj_types::StringWayBuilder::default();
        w._id(1);
        w._tags(vec![("name".into(), "Bar".into()), ("boat".into(), "no".into())].into());
        let w = w.build().unwrap();

        way_id_rel_tags.record_relation(&r, &[]);

        assert!(way_id_rel_tags.way_tag_value_only(1, "highway").is_none());
        assert_eq!(way_id_rel_tags.way_tag_value_only(1, "name"), Some("Foo"));
        assert!(way_id_rel_tags.way_tag_value_only(1, "boat").is_none());

        assert!(way_id_rel_tags.way_tag_value_only(2, "highway").is_none());
        assert!(way_id_rel_tags.way_tag_value_only(2, "name").is_none());

        assert_eq!(way_id_rel_tags.way_tag_value(&w, "name"), Some("Foo"));
        assert_eq!(way_id_rel_tags.way_tag_value(&w, "boat"), Some("no"));
        assert_eq!(way_id_rel_tags.way_tag_value(&w, "waterway"), Some("river"));
        assert_eq!(way_id_rel_tags.way_tag_value(&w, "highway"), None);

        let mut tags: Vec<_> = way_id_rel_tags.way_tags(&w).collect();
        tags.sort_unstable();
        assert_eq!(
            tags,
            vec![("boat", "no"), ("name", "Foo"), ("waterway", "river")]
        );
    }
}

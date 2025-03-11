use get_size::GetSize;
use log::{debug, info, trace, warn};
use rayon::prelude::*;

use std::cmp::Ordering;

use std::sync::{Arc, Mutex};

//use get_size_derive::*;

use num_format::{Locale, ToFormattedString};

pub mod haversine;
use haversine::haversine_m;
pub mod dij;
pub mod graph;
pub mod nodeid_position;
pub mod tagfilter;
pub mod way_group;
pub use nodeid_position::NodeIdPosition;
pub mod btreemapsplitkey;
pub mod inter_store;
pub mod kosaraju;
pub mod nodeid_wayids;
pub mod sorted_slice_store;
pub mod taggrouper;

pub mod fileio;
pub mod formatting;

use anyhow::Result;
use indicatif::ProgressBar;
use std::collections::HashMap;

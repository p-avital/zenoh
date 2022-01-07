pub(crate) mod internals;
pub use internals::{ContiguousSlice, ShareAble};

pub enum SharingBuffer {
    Single(ContiguousSlice),
    Multiple(Vec<ContiguousSlice>),
}

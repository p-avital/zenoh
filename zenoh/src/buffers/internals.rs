#[cfg(feature = "shared-memory")]
mod shared_memory_imports {
    pub use crate::net::protocol::io::{SharedMemoryBuf, SharedMemoryReader};
    pub use std::fmt;
    pub use std::sync::{Arc, RwLock};
}
#[cfg(feature = "shared-memory")]
use shared_memory_imports::*;

use derive_more::From;
use zenoh_util::collections::RecyclingObject;
#[derive(From)]
pub(crate) enum ContiguousBuffer {
    RecyclingBoxedSlice(RecyclingObject<Box<[u8]>>),
    Vec(Vec<u8>),
    #[cfg(feature = "shared-memory")]
    SharedMemory(SharedMemoryBuf),
}
impl ContiguousBuffer {
    fn as_slice(&self) -> &[u8] {
        match self {
            ContiguousBuffer::RecyclingBoxedSlice(s) => s,
            ContiguousBuffer::Vec(s) => s.as_slice(),
            #[cfg(feature = "shared-memory")]
            ContiguousBuffer::SharedMemory(s) => s.as_slice(),
        }
    }
    /// # Safety
    /// May return a mutable slice of `SharedMemoryBuffer`.
    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            ContiguousBuffer::RecyclingBoxedSlice(s) => s,
            ContiguousBuffer::Vec(s) => s.as_mut_slice(),
            #[cfg(feature = "shared-memory")]
            ContiguousBuffer::SharedMemory(s) => s.as_mut_slice(),
        }
    }
    /// Panics if the internal isn't a Vec
    fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        if let Self::Vec(v) = self {
            v
        } else {
            panic!("ContiguousBuffer::as_mut_vec called with a non-vec parameter")
        }
    }
    fn is_safely_mutable(&self) -> bool {
        match self {
            ContiguousBuffer::RecyclingBoxedSlice(s) => true,
            ContiguousBuffer::Vec(s) => true,
            #[cfg(feature = "shared-memory")]
            ContiguousBuffer::SharedMemory(s) => false,
        }
    }
}
impl std::ops::Deref for ContiguousBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl AsRef<[u8]> for ContiguousBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
mod index_impls {
    use super::{ContiguousBuffer, ContiguousSlice};
    use std::ops::{Index, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
    macro_rules! impl_index {
    ([$($index: ty),*] for $ty: ty : $output: ty) => {
        $(impl Index<$index> for $ty {
            type Output = $output;
            fn index(&self, index: $index) -> &$output {
                &self.as_slice()[index]
            }
        })*
    };
}
    impl_index!([usize] for ContiguousBuffer : u8);
    impl_index!([Range<usize>, RangeFrom<usize>, RangeFull, RangeTo<usize>, RangeInclusive<usize>, RangeToInclusive<usize>] for ContiguousBuffer : [u8]);
    impl_index!([usize] for ContiguousSlice : u8);
    impl_index!([Range<usize>, RangeFrom<usize>, RangeFull, RangeTo<usize>, RangeInclusive<usize>, RangeToInclusive<usize>] for ContiguousSlice : [u8]);
}

#[derive(Clone)]
pub enum ShareAble<T> {
    Owned(T),
    Shared(Arc<T>),
}
impl<T> From<T> for ShareAble<T> {
    fn from(t: T) -> Self {
        Self::Owned(t)
    }
}
impl<T> From<Arc<T>> for ShareAble<T> {
    fn from(t: Arc<T>) -> Self {
        Self::Shared(t)
    }
}
impl<T> AsRef<T> for ShareAble<T> {
    fn as_ref(&self) -> &T {
        match self {
            ShareAble::Owned(o) => o,
            ShareAble::Shared(s) => s.as_ref(),
        }
    }
}
impl<T> std::ops::Deref for ShareAble<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            ShareAble::Owned(o) => o,
            ShareAble::Shared(s) => s.as_ref(),
        }
    }
}
impl<T> ShareAble<T> {
    pub fn new(t: T) -> Self {
        Self::from(t)
    }
    pub fn rc_clone(self) -> (Self, Self) {
        let s = match self {
            ShareAble::Owned(o) => Arc::new(o),
            ShareAble::Shared(s) => s,
        };
        (Self::from(s.clone()), Self::from(s))
    }
    pub fn try_rc_clone(&self) -> Option<Self> {
        match self {
            ShareAble::Owned(_) => None,
            ShareAble::Shared(s) => Some(Self::from(s.clone())),
        }
    }
    pub unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        match this {
            ShareAble::Owned(o) => o,
            ShareAble::Shared(s) => zenoh_util::sync::get_mut_unchecked(s),
        }
    }
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        match this {
            ShareAble::Owned(o) => Some(o),
            ShareAble::Shared(s) => Arc::get_mut(s),
        }
    }
}

pub struct ContiguousSlice {
    buffer: ShareAble<ContiguousBuffer>,
    start: usize,
    end: usize,
}

impl ContiguousSlice {
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer.as_slice()[self.start..self.end]
    }
    /// # Safety
    /// This function is VERY unsafe: it doesn't check the Arc's counters for unicity,
    /// and may still return a mutable slice of `SharedMemoryBuffer`.
    ///
    /// Note however that `is_safely_mutable`, `left_extend_range` and `right_extend_range` will only return positive values if calling this function is safe.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut ShareAble::get_mut_unchecked(&mut self.buffer).as_mut_slice()[self.start..self.end]
    }

    /// Only returns a mutable slice if the safety of mutating it is guaranteed.
    pub fn try_as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if let Some(b) = ShareAble::get_mut(&mut self.buffer) {
            if b.is_safely_mutable() {
                Some(&mut unsafe { b.as_mut_slice() }[self.start..self.end])
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Ensures unique ownership by copying the slice's contents if necessary.
    ///
    /// Calling `self.make_mut().as_mut_slice()` is always safe.
    pub fn make_mut(&mut self) -> &mut Self {
        if self.is_safely_mutable() {
            return self;
        }
        let slice: &[u8] = unsafe { std::mem::transmute(self.as_slice()) }; // needed because `self.start` and `self.end` are free to be mutated once the slice is taken, but rustc doesn't know that
        self.start = 0;
        self.end = slice.len();
        let copy = Vec::from(slice);
        self.buffer = ShareAble::new(copy.into());
        self
    }

    /// Ensures unique ownership and extensibility by copying the slice's contents if necessary.
    ///
    /// The slice's internals will be converted to a vector.
    pub fn make_mut_extensible(&mut self) -> &mut Self {
        let b = match ShareAble::get_mut(&mut self.buffer) {
            Some(ContiguousBuffer::Vec(_)) => return self,
            Some(b) => Some(unsafe { std::mem::transmute::<_, &mut ContiguousBuffer>(b) }),
            None => None,
        };
        let slice: &[u8] = unsafe { std::mem::transmute(self.as_slice()) }; // needed because `self.start` and `self.end` are free to be mutated once the slice is taken, but rustc doesn't know that
        self.start = 0;
        self.end = slice.len();
        let copy = Vec::from(slice).into();
        match b {
            Some(b) => *b = copy,
            None => self.buffer = ShareAble::new(copy),
        }
        self
    }

    /// Returns true if calling `self.as_mut_slice()` is safe.
    pub fn is_safely_mutable(&mut self) -> bool {
        ShareAble::get_mut(&mut self.buffer)
            .map(|b| b.is_safely_mutable())
            .unwrap_or(false)
    }

    /// Returns the amount of extension that may be safely done to this slice on the left side.  
    /// A non-zero return value also implies that calling `self.as_mut_slice()` is safe.
    ///
    /// Note that "not extensible" is returned as 0, whereas "infinitely extensible" is returned as `usize::MAX`
    pub fn left_extend_range(&mut self) -> usize {
        if !self.is_safely_mutable() {
            return 0;
        }
        todo!()
    }

    /// Returns the amount of extension that may be safely done to this slice on the right side.  
    /// A non-zero return value also implies that calling `self.as_mut_slice()` is safe.
    ///
    /// Note that "not extensible" is returned as 0, whereas "infinitely extensible" is returned as `usize::MAX`
    pub fn right_extend_range(&mut self) -> usize {
        if !self.is_safely_mutable() {
            return 0;
        }
        todo!()
    }
}
impl std::ops::Deref for ContiguousSlice {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl AsRef<[u8]> for ContiguousSlice {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

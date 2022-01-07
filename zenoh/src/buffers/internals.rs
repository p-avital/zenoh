#[cfg(feature = "shared-memory")]
mod shared_memory_imports {
    pub use crate::net::protocol::io::{SharedMemoryBuf, SharedMemoryReader};
    pub use std::fmt;
    pub use std::sync::{Arc, RwLock};
}
use std::cell::UnsafeCell;

#[cfg(feature = "shared-memory")]
use shared_memory_imports::*;

use zenoh_util::collections::RecyclingObject;
pub(crate) enum ContiguousBufferInner<'a> {
    Slice(&'a UnsafeCell<[u8]>),
    MutableSlice(&'a mut [u8]),
    Vec(Vec<u8>),
    RecyclingBoxedSlice(RecyclingObject<Box<[u8]>>),
    #[cfg(feature = "shared-memory")]
    SharedMemory(Arc<SharedMemoryBuf>),
    Shared(UnsafeCell<Arc<ContiguousBuffer<'a>>>),
}
impl ContiguousBufferInner<'static> {
    pub(crate) fn upgrade(this: ContiguousBufferInner<'_>) -> Self {
        match this {
            ContiguousBufferInner::Slice(s) => {
                ContiguousBufferInner::<'static>::Vec(Vec::from(unsafe { &*s.get() }))
            }
            ContiguousBufferInner::MutableSlice(s) => {
                ContiguousBufferInner::<'static>::Vec(Vec::from(s))
            }
            ContiguousBufferInner::Vec(v) => ContiguousBufferInner::<'static>::Vec(v),
            ContiguousBufferInner::RecyclingBoxedSlice(v) => {
                ContiguousBufferInner::<'static>::RecyclingBoxedSlice(v)
            }
            ContiguousBufferInner::SharedMemory(v) => {
                ContiguousBufferInner::<'static>::SharedMemory(v)
            }
            ContiguousBufferInner::Shared(v) => match Arc::try_unwrap(v.into_inner()) {
                Ok(v) => ContiguousBufferInner::upgrade(v.inner),
                Err(_) => todo!(),
            },
        }
    }
}
pub struct ContiguousBuffer<'a> {
    pub(crate) inner: ContiguousBufferInner<'a>,
}
impl<'a> From<Vec<u8>> for ContiguousBuffer<'a> {
    fn from(v: Vec<u8>) -> Self {
        ContiguousBuffer {
            inner: ContiguousBufferInner::Vec(v),
        }
    }
}
impl<'a> From<&'a [u8]> for ContiguousBuffer<'a> {
    fn from(v: &'a [u8]) -> Self {
        ContiguousBuffer {
            inner: ContiguousBufferInner::Slice(unsafe { std::mem::transmute(v) }),
        }
    }
}
impl<'a> From<&'a mut [u8]> for ContiguousBuffer<'a> {
    fn from(v: &'a mut [u8]) -> Self {
        ContiguousBuffer {
            inner: ContiguousBufferInner::MutableSlice(v),
        }
    }
}
impl<'a> From<RecyclingObject<Box<[u8]>>> for ContiguousBuffer<'a> {
    fn from(v: RecyclingObject<Box<[u8]>>) -> Self {
        ContiguousBuffer {
            inner: ContiguousBufferInner::RecyclingBoxedSlice(v),
        }
    }
}
#[cfg(feature = "shared-memory")]
impl<'a> From<Arc<SharedMemoryBuf>> for ContiguousBuffer<'a> {
    fn from(v: Arc<SharedMemoryBuf>) -> Self {
        ContiguousBuffer {
            inner: ContiguousBufferInner::SharedMemory(v),
        }
    }
}

impl<'a> ContiguousBuffer<'a> {
    pub fn to_shared(&mut self) -> &mut Self {
        let mut inner = unsafe { std::ptr::read(&self.inner) };
        inner = match inner {
            ContiguousBufferInner::SharedMemory(_) | ContiguousBufferInner::Shared(_) => {
                std::mem::forget(inner);
                return self;
            }
            inner => {
                ContiguousBufferInner::Shared(UnsafeCell::new(Arc::new(ContiguousBuffer { inner })))
            }
        };
        unsafe { std::ptr::write(&mut self.inner, inner) }
        self
    }
    pub fn try_rc_clone(&self) -> Option<Self> {
        match &self.inner {
            ContiguousBufferInner::SharedMemory(s) => Some(ContiguousBuffer {
                inner: ContiguousBufferInner::SharedMemory(s.clone()),
            }),
            ContiguousBufferInner::Shared(s) => Some(ContiguousBuffer {
                inner: ContiguousBufferInner::Shared(UnsafeCell::new(unsafe {
                    (*s.get()).clone()
                })),
            }),
            _ => None,
        }
    }
    pub fn rc_clone(&mut self) -> Self {
        self.to_shared();
        match &self.inner {
            ContiguousBufferInner::SharedMemory(s) => ContiguousBuffer {
                inner: ContiguousBufferInner::SharedMemory(s.clone()),
            },
            ContiguousBufferInner::Shared(s) => ContiguousBuffer {
                inner: ContiguousBufferInner::Shared(UnsafeCell::new(unsafe {
                    (*s.get()).clone()
                })),
            },
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }
    pub fn deep_clone(&self) -> ContiguousBuffer<'static> {
        Vec::from(self.as_slice()).into()
    }
    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            ContiguousBufferInner::RecyclingBoxedSlice(s) => s,
            ContiguousBufferInner::Vec(s) => s.as_slice(),
            #[cfg(feature = "shared-memory")]
            ContiguousBufferInner::SharedMemory(s) => s.as_slice(),
            ContiguousBufferInner::Slice(s) => unsafe { &*s.get() },
            ContiguousBufferInner::MutableSlice(s) => s,
            ContiguousBufferInner::Shared(s) => unsafe { &*s.get() }.as_ref().as_slice(),
        }
    }
    pub fn try_as_mut_slice(&mut self) -> Option<&mut [u8]> {
        match &mut self.inner {
            ContiguousBufferInner::RecyclingBoxedSlice(s) => Some(s),
            ContiguousBufferInner::Vec(s) => Some(s.as_mut_slice()),
            ContiguousBufferInner::MutableSlice(s) => Some(*s),
            ContiguousBufferInner::Shared(s) => match Arc::get_mut(s.get_mut()) {
                Some(s) => s.try_as_mut_slice(),
                None => None,
            },
            #[cfg(feature = "shared-memory")]
            ContiguousBufferInner::SharedMemory(_) => None,
            ContiguousBufferInner::Slice(_) => None,
        }
    }
    /// # Safety
    /// May return a mutable slice of `SharedMemoryBuffer`.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.inner {
            ContiguousBufferInner::RecyclingBoxedSlice(s) => s,
            ContiguousBufferInner::Vec(s) => s.as_mut_slice(),
            #[cfg(feature = "shared-memory")]
            ContiguousBufferInner::SharedMemory(s) => {
                zenoh_util::sync::get_mut_unchecked(s).as_mut_slice()
            }
            ContiguousBufferInner::MutableSlice(s) => *s,
            ContiguousBufferInner::Slice(s) => &mut *s.get(),
            ContiguousBufferInner::Shared(s) => {
                zenoh_util::sync::get_mut_unchecked(s.get_mut()).as_mut_slice()
            }
        }
    }
    /// Panics if the internal isn't a Vec
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        if let ContiguousBufferInner::Vec(v) = &mut self.inner {
            v
        } else {
            panic!("ContiguousBuffer::as_mut_vec called with a non-vec parameter")
        }
    }
    pub fn is_safely_mutable(&self) -> bool {
        match &self.inner {
            ContiguousBufferInner::RecyclingBoxedSlice(_) => true,
            ContiguousBufferInner::Vec(_) => true,
            #[cfg(feature = "shared-memory")]
            ContiguousBufferInner::SharedMemory(_) => false,
            ContiguousBufferInner::Slice(_) => false,
            ContiguousBufferInner::MutableSlice(_) => true,
            ContiguousBufferInner::Shared(s) => unsafe {
                Arc::get_mut(&mut *s.get())
                    .map(|s| s.is_safely_mutable())
                    .unwrap_or(false)
            },
        }
    }
    pub fn is_mutable_vec(&self) -> bool {
        match &self.inner {
            ContiguousBufferInner::Vec(_) => true,
            ContiguousBufferInner::Shared(s) => unsafe {
                Arc::get_mut(&mut *s.get())
                    .map(|s| s.is_mutable_vec())
                    .unwrap_or(false)
            },
            _ => false,
        }
    }
    pub fn upgrade(self) -> ContiguousBuffer<'static> {
        ContiguousBuffer {
            inner: ContiguousBufferInner::upgrade(self.inner),
        }
    }
}
impl<'a> std::ops::Deref for ContiguousBuffer<'a> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl<'a> AsRef<[u8]> for ContiguousBuffer<'a> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
mod index_impls {
    use super::{ContiguousBuffer, ContiguousSlice};
    use std::ops::{Index, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
    macro_rules! impl_index {
    ([$($index: ty),*] for $ty: ty : $output: ty) => {
        $(impl<'a> Index<$index> for $ty {
            type Output = $output;
            fn index(&self, index: $index) -> &$output {
                &self.as_slice()[index]
            }
        })*
    };
}
    impl_index!([usize] for ContiguousBuffer<'a> : u8);
    impl_index!([Range<usize>, RangeFrom<usize>, RangeFull, RangeTo<usize>, RangeInclusive<usize>, RangeToInclusive<usize>] for ContiguousBuffer<'a> : [u8]);
    impl_index!([usize] for ContiguousSlice<'a> : u8);
    impl_index!([Range<usize>, RangeFrom<usize>, RangeFull, RangeTo<usize>, RangeInclusive<usize>, RangeToInclusive<usize>] for ContiguousSlice<'a> : [u8]);
}

pub struct ContiguousSlice<'a> {
    pub(crate) buffer: ContiguousBuffer<'a>,
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl<'a> ContiguousSlice<'a> {
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer.as_slice()[self.start..self.end]
    }
    pub fn as_io_slice(&self) -> std::io::IoSlice {
        std::io::IoSlice::new(&self.buffer.as_slice()[self.start..self.end])
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
    /// # Safety
    /// This function is VERY unsafe: it doesn't check whether the underlying buffer is shared,
    /// and may still return a mutable slice of `SharedMemoryBuffer`.
    ///
    /// Note however that `is_safely_mutable`, `left_extend_range` and `right_extend_range` will only return positive values if calling this function is safe.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer.as_mut_slice()[self.start..self.end]
    }

    /// Only returns a mutable slice if the safety of mutating it is guaranteed.
    pub fn try_as_mut_slice(&mut self) -> Option<&mut [u8]> {
        let start = self.start;
        let end = self.end;
        self.buffer.try_as_mut_slice().map(|s| &mut s[start..end])
    }

    pub fn to_shared(&mut self) -> &mut Self {
        self.buffer.to_shared();
        self
    }
    pub fn try_rc_clone(&self) -> Option<Self> {
        self.buffer.try_rc_clone().map(|buffer| ContiguousSlice {
            buffer,
            start: self.start,
            end: self.end,
        })
    }
    pub fn rc_clone(&mut self) -> Self {
        ContiguousSlice {
            buffer: self.buffer.rc_clone(),
            start: self.start,
            end: self.end,
        }
    }
    pub fn deep_clone(&self) -> ContiguousSlice<'static> {
        let slice = self.as_slice();
        let end = slice.len();
        ContiguousSlice {
            start: 0,
            end,
            buffer: Vec::from(slice).into(),
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
        self.buffer = unsafe { std::mem::transmute(ContiguousBuffer::from(copy)) }; // need to shorten the lifetime
        self
    }

    /// Ensures unique ownership and extensibility by copying the slice's contents if necessary.
    ///
    /// The slice's internals will be converted to a vector.
    pub fn make_mut_extensible(&mut self) -> &mut Self {
        todo!()
    }

    /// Returns true if calling `self.as_mut_slice()` is safe.
    pub fn is_safely_mutable(&mut self) -> bool {
        self.buffer.is_safely_mutable()
    }

    /// Returns the amount of extension that may be safely done to this slice on the left side.  
    /// A non-zero return value also implies that calling `self.as_mut_slice()` is safe.
    ///
    /// Note that "not extensible" is returned as 0, whereas "infinitely extensible" is returned as `usize::MAX`
    pub fn left_extend_range(&mut self) -> usize {
        if !self.is_safely_mutable() {
            return 0;
        }
        self.start
    }

    /// Returns the amount of extension that may be safely done to this slice on the right side.  
    /// A non-zero return value also implies that calling `self.as_mut_slice()` is safe.
    ///
    /// Note that "not extensible" is returned as 0, whereas "infinitely extensible" is returned as `usize::MAX`
    pub fn right_extend_range(&mut self) -> usize {
        if !self.is_safely_mutable() {
            return 0;
        }
        if self.buffer.is_mutable_vec() {
            return usize::MAX;
        }
        self.buffer.as_slice().len() - self.end
    }
}
impl<'a> std::ops::Deref for ContiguousSlice<'a> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl<'a> AsRef<[u8]> for ContiguousSlice<'a> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}
impl<'a, T: Into<ContiguousBuffer<'a>>> From<T> for ContiguousSlice<'a> {
    fn from(v: T) -> Self {
        let buffer = v.into();
        let end = buffer.len();
        ContiguousSlice {
            buffer,
            start: 0,
            end,
        }
    }
}

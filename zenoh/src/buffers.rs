use std::sync::Arc;

use splitable_buffers::prelude::{Buffer as SplitBuf, *};
use zenoh_util::collections::RecyclingObject;

pub type Buffer<'a, Context> = SplitBuf<ContiguousBuffer<'a>, ContiguousBuffer<'a>, Context>;
pub type RawBuffer<'a> = Buffer<'a, RawShmInfo>;
#[derive(Default, Debug, Clone, Copy)]
pub struct RawShmInfo;
impl<'a> SliceProvider<SharedMemorySlice<'a>> for RawShmInfo {
    type Error = std::convert::Infallible;
    fn provide_slice(&self, key: SharedMemorySlice<'a>) -> Result<&[u8], Self::Error> {
        Ok(unsafe { std::mem::transmute(key.0) })
    }
}

#[derive(Debug)]
enum ContiguousBufferInner<'a> {
    VecOfSelf(Vec<ContiguousBuffer<'a>>),
    CacheRange(Range),
    Slice(&'a [u8]),
    MutableSlice(&'a mut [u8]),
    BytesVec(Vec<u8>),
    RecyclingBox(RecyclingObject<Box<[u8]>>),
    #[cfg(feature = "shared-memory")]
    SharedMemoryInfo(Vec<u8>),
    SharedSelf(Arc<ContiguousBuffer<'a>>, Range),
}
impl<'a> std::fmt::Debug for ContiguousBuffer<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt(f)
    }
}
use ContiguousBufferInner::*;

#[cfg(feature = "shared-memory")]
pub struct SharedMemorySlice<'a>(pub &'a [u8]);
pub struct ContiguousBuffer<'a> {
    inner: ContiguousBufferInner<'a>,
}
#[cfg(feature = "shared-memory")]
impl ContiguousBuffer<'static> {
    pub fn shared_memory(info: Vec<u8>) -> Self {
        SharedMemoryInfo(info).into()
    }
}
impl<'a> From<ContiguousBufferInner<'a>> for ContiguousBuffer<'a> {
    fn from(inner: ContiguousBufferInner<'a>) -> Self {
        ContiguousBuffer { inner }
    }
}

#[cfg(feature = "shared-memory")]
impl<'a, 'b> HasKey<'b> for ContiguousBuffer<'a> {
    type Key = SharedMemorySlice<'b>;
}

#[cfg(not(feature = "shared-memory"))]
impl<'a, 'b> HasKey<'b> for ContiguousBuffer<'a> {
    type Key = std::convert::Infallible;
}
impl<'a> FallibleBuffer for ContiguousBuffer<'a> {
    fn slice(&self) -> Result<&[u8], SharedMemorySlice> {
        match &self.inner {
            VecOfSelf(_) | CacheRange(_) => unreachable!(),
            Slice(s) => Ok(s),
            MutableSlice(s) => Ok(s),
            BytesVec(s) => Ok(s),
            RecyclingBox(s) => Ok(s),
            SharedMemoryInfo(info) => Err(SharedMemorySlice(info)),
            SharedSelf(s, r) => s.slice().map(|s| &s[r]),
        }
    }
    type IsCacheRange = splitable_buffers::is_cache_range::Sometimes;
    fn range(&self) -> Result<&splitable_buffers::range::Range, ()> {
        match &self.inner {
            CacheRange(r) => Ok(r),
            _ => Err(()),
        }
    }
    fn from_range(range: splitable_buffers::range::Range) -> Self {
        CacheRange(range).into()
    }
    fn range_mut(&mut self) -> Result<&mut splitable_buffers::range::Range, ()> {
        match &mut self.inner {
            CacheRange(r) => Ok(r),
            _ => Err(()),
        }
    }

    fn slice_with_context<'l, C: SliceProvider<<Self as HasKey<'l>>::Key>>(
        &'l self,
        context: &'l C,
    ) -> Result<&'l [u8], C::Error> {
        match &self.inner {
            VecOfSelf(_) | CacheRange(_) => unreachable!(),
            Slice(s) => Ok(s),
            MutableSlice(s) => Ok(s),
            BytesVec(s) => Ok(s),
            RecyclingBox(s) => Ok(s),
            SharedMemoryInfo(info) => context.provide_slice(SharedMemorySlice(info)),
            SharedSelf(s, r) => s.slice_with_context(context).map(|s| &s[r]),
        }
    }
}

impl<'a> Container<Self> for ContiguousBuffer<'a> {
    fn new() -> Self {
        VecOfSelf(Vec::new()).into()
    }
    fn with_capacity(n: usize) -> Self {
        VecOfSelf(Vec::with_capacity(n)).into()
    }
    fn len(&self) -> usize {
        match &self.inner {
            VecOfSelf(v) => v.len(),
            _ => 1,
        }
    }
    fn is_empty(&self) -> bool {
        match &self.inner {
            VecOfSelf(v) => v.is_empty(),
            _ => false,
        }
    }
    fn push(&mut self, t: Self) {
        match &mut self.inner {
            VecOfSelf(v) => v.push(t),
            _ => {
                let this = unsafe { std::ptr::read(self) };
                let v = vec![this, t];
                unsafe { std::ptr::write(self, VecOfSelf(v).into()) }
            }
        }
    }
    fn reserve(&mut self, n: usize) {
        match &mut self.inner {
            VecOfSelf(v) => v.reserve(n),
            _ => {
                let this = unsafe { std::ptr::read(self) };
                let mut v = Vec::with_capacity(n);
                v.push(this);
                unsafe { std::ptr::write(self, VecOfSelf(v).into()) }
            }
        }
    }
    fn single(t: Self) -> Self {
        t
    }
    fn push_multiple<const N: usize>(&mut self, t: [Self; N]) {
        match &mut self.inner {
            VecOfSelf(v) => v.extend(t),
            _ => {
                let this = unsafe { std::ptr::read(self) };
                let mut v = Vec::with_capacity(N + 1);
                v.push(this);
                v.extend(t);
                unsafe { std::ptr::write(self, VecOfSelf(v).into()) }
            }
        }
    }
}
impl<'a> std::ops::Index<usize> for ContiguousBuffer<'a> {
    type Output = Self;
    fn index(&self, index: usize) -> &Self::Output {
        match &self.inner {
            VecOfSelf(v) => &v[index],
            _ => {
                assert_eq!(index, 0);
                self
            }
        }
    }
}
impl<'a> std::ops::IndexMut<usize> for ContiguousBuffer<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match unsafe { std::mem::transmute::<_, &mut ContiguousBufferInner>(&mut self.inner) } {
            VecOfSelf(v) => &mut v[index],
            _ => {
                assert_eq!(index, 0);
                self
            }
        }
    }
}
impl<'a> AsRef<[Self]> for ContiguousBuffer<'a> {
    fn as_ref(&self) -> &[Self] {
        match &self.inner {
            VecOfSelf(v) => v,
            _ => std::slice::from_ref(self),
        }
    }
}
impl<'a> Insertable<Self> for ContiguousBuffer<'a> {
    fn insert(&mut self, at: usize, t: Self) {
        match self.len().cmp(&at) {
            std::cmp::Ordering::Greater => match &mut self.inner {
                VecOfSelf(v) => v.insert(at, t),
                _ => {
                    let this = unsafe { std::ptr::read(self) };
                    let v = vec![t, this];
                    unsafe { std::ptr::write(self, VecOfSelf(v).into()) }
                }
            },
            std::cmp::Ordering::Equal => self.push(t),
            std::cmp::Ordering::Less => {
                panic!(
                    "Attempted insertion at index {}, but len is {}",
                    at,
                    self.len()
                )
            }
        }
    }
    fn insert_multiple<const N: usize>(&mut self, at: usize, t: [Self; N]) {
        match self.len().cmp(&at) {
            std::cmp::Ordering::Greater => match &mut self.inner {
                VecOfSelf(v) => v.insert_multiple(at, t),
                _ => {
                    let this = unsafe { std::ptr::read(self) };
                    let mut v = Vec::with_capacity(N + 1);
                    v.extend(t);
                    v.push(this);
                    unsafe { std::ptr::write(self, VecOfSelf(v).into()) }
                }
            },
            std::cmp::Ordering::Equal => self.push_multiple(t),
            std::cmp::Ordering::Less => {
                panic!(
                    "Attempted insertion at index {}, but len is {}",
                    at,
                    self.len()
                )
            }
        }
    }
}

impl<'a> ContiguousBuffer<'a> {
    fn is_static(&self) -> bool {
        match &self.inner {
            CacheRange(_) => true,
            BytesVec(_) => true,
            RecyclingBox(_) => true,
            SharedMemoryInfo(_) => true,
            Slice(_) => false,
            MutableSlice(_) => false,
            VecOfSelf(s) => s.iter().all(|s| s.is_static()),
            SharedSelf(s, _) => s.is_static(),
        }
    }
}
impl<'a> Upgradeable for ContiguousBuffer<'a> {
    type Target = ContiguousBuffer<'static>;

    fn upgrade(self) -> Self::Target {
        if self.is_static() {
            return unsafe { std::mem::transmute(self) };
        }
        match self.inner {
            CacheRange(_) | BytesVec(_) | RecyclingBox(_) | SharedMemoryInfo(_) => {
                unreachable!()
            }
            Slice(s) => Vec::from(s).into(),
            MutableSlice(s) => Vec::from(s).into(),
            VecOfSelf(s) => VecOfSelf(s.into_iter().map(Self::upgrade).collect()).into(),
            SharedSelf(s, _) => Vec::from(s.slice().unwrap_or_else(|_| unreachable!())).into(),
        }
    }
}
impl<'a> Splitable for ContiguousBuffer<'a> {
    fn split(&mut self, at: usize) -> Self {
        let inner = unsafe { std::ptr::read(&self.inner) };
        let (inner, new) = match inner {
            VecOfSelf(_) => unreachable!(),
            CacheRange(mut r) => {
                let at = at as u32;
                let end = r.end;
                r.end = at;
                (CacheRange(r), CacheRange(Range { start: at, end }))
            }
            Slice(s) => {
                let (left, right) = s.split_at(at);
                (Slice(left), Slice(right))
            }
            MutableSlice(s) => {
                let (left, right) = s.split_at_mut(at);
                (MutableSlice(left), MutableSlice(right))
            }
            SharedSelf(s, mut r) => {
                let at = at as u32;
                let end = r.end;
                r.end = at;
                (
                    SharedSelf(s.clone(), r),
                    SharedSelf(s, Range { start: at, end }),
                )
            }
            _ => {
                let s = Arc::new(Self::from(inner));
                (
                    SharedSelf(s.clone(), (..at).into()),
                    SharedSelf(s, (at..).into()),
                )
            }
        };
        unsafe { std::ptr::write(&mut self.inner, inner) };
        new.into()
    }
}

impl<'a> From<&'a [u8]> for ContiguousBuffer<'a> {
    fn from(s: &'a [u8]) -> Self {
        Slice(s).into()
    }
}
impl<'a> From<&'a mut [u8]> for ContiguousBuffer<'a> {
    fn from(s: &'a mut [u8]) -> Self {
        MutableSlice(s).into()
    }
}
impl From<Vec<u8>> for ContiguousBuffer<'static> {
    fn from(v: Vec<u8>) -> Self {
        BytesVec(v).into()
    }
}
impl From<RecyclingObject<Box<[u8]>>> for ContiguousBuffer<'static> {
    fn from(v: RecyclingObject<Box<[u8]>>) -> Self {
        RecyclingBox(v).into()
    }
}

#[test]
fn size_test() {
    assert_eq!(std::mem::size_of::<Buffer<()>>(), 56);
}
#[test]
fn fun_with_buffers() {
    let mut buffer: RawBuffer<'static> = RawBuffer::new();
    let dashes: &[u8] = b" - ";
    buffer.copy_extend(b"there!");
    buffer.copy_insert(0, b"Hello ");
    buffer.insert(0, dashes);
    buffer.extend(dashes);
    buffer.extend(Vec::from(b"General Kenobi!".as_ref()));
    let rendered = Vec::<u8>::from(&buffer);
    assert_eq!(rendered, b" - Hello there! - General Kenobi!");
    let short_lived = String::from("Not 'static, can't extend from here");
    // buffer.extend(short_lived.as_bytes()); // This cannot build
    buffer.copy_extend(short_lived.as_bytes());
}

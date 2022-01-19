use std::sync::Arc;

use splitable_buffers::prelude::{Buffer as SplitBuf, *};
use zenoh_util::collections::RecyclingObject;

pub type Buffer<'a, Cache, Context> =
    SplitBuf<Cache, ContiguousBuffer<'a>, ContiguousBuffer<'a>, Context>;
pub type RawBuffer<'a, Cache> = Buffer<'a, Cache, RawShmInfo>;

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

    fn slice_with_context<'l, C: SliceProvider<'l, <Self as HasKey<'l>>::Key>>(
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
    fn push_multiple<I: IntoIterator<Item = Self>>(&mut self, t: I) {
        match &mut self.inner {
            VecOfSelf(v) => v.extend(t),
            _ => {
                let iter = t.into_iter();
                let this = unsafe { std::ptr::read(self) };
                let (min, max) = iter.size_hint();
                let mut v = Vec::with_capacity(max.unwrap_or(min) + 1);
                v.push(this);
                v.extend(iter);
                unsafe { std::ptr::write(self, VecOfSelf(v).into()) }
            }
        }
    }
}

enum ContiguousBufferIterInner<'a> {
    VecOfSelf(std::vec::IntoIter<ContiguousBuffer<'a>>),
    CacheRange(Range),
    Slice(&'a [u8]),
    MutableSlice(&'a mut [u8]),
    BytesVec(Vec<u8>),
    RecyclingBox(RecyclingObject<Box<[u8]>>),
    #[cfg(feature = "shared-memory")]
    SharedMemoryInfo(Vec<u8>),
    SharedSelf(Arc<ContiguousBuffer<'a>>, Range),
}
pub struct ContiguousBufferIter<'a> {
    inner: ContiguousBufferIterInner<'a>,
}
impl<'a> IntoIterator for ContiguousBuffer<'a> {
    type Item = Self;
    type IntoIter = ContiguousBufferIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        ContiguousBufferIter {
            inner: match self.inner {
                VecOfSelf(v) => ContiguousBufferIterInner::VecOfSelf(v.into_iter()),
                CacheRange(v) => ContiguousBufferIterInner::CacheRange(v),
                Slice(v) => ContiguousBufferIterInner::Slice(v),
                MutableSlice(v) => ContiguousBufferIterInner::MutableSlice(v),
                BytesVec(v) => ContiguousBufferIterInner::BytesVec(v),
                RecyclingBox(v) => ContiguousBufferIterInner::RecyclingBox(v),
                SharedMemoryInfo(v) => ContiguousBufferIterInner::SharedMemoryInfo(v),
                SharedSelf(v, r) => ContiguousBufferIterInner::SharedSelf(v, r),
            },
        }
    }
}
impl<'a> Iterator for ContiguousBufferIter<'a> {
    type Item = ContiguousBuffer<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            ContiguousBufferIterInner::VecOfSelf(v) => return v.next(),
            _ => {}
        }
        let inner = unsafe { std::ptr::read(&self.inner) };
        let inner = match inner {
            ContiguousBufferIterInner::VecOfSelf(_) => unsafe {
                std::hint::unreachable_unchecked()
            },
            ContiguousBufferIterInner::CacheRange(v) => CacheRange(v),
            ContiguousBufferIterInner::Slice(v) => Slice(v),
            ContiguousBufferIterInner::MutableSlice(v) => MutableSlice(v),
            ContiguousBufferIterInner::BytesVec(v) => BytesVec(v),
            ContiguousBufferIterInner::RecyclingBox(v) => RecyclingBox(v),
            ContiguousBufferIterInner::SharedMemoryInfo(v) => SharedMemoryInfo(v),
            ContiguousBufferIterInner::SharedSelf(v, r) => SharedSelf(v, r),
        };
        unsafe {
            std::ptr::write(
                &mut self.inner,
                ContiguousBufferIterInner::VecOfSelf(Vec::new().into_iter()),
            )
        }
        return Some(inner.into());
    }
}
impl<'a> ExactSizeIterator for ContiguousBufferIter<'a> {
    fn len(&self) -> usize {
        match &self.inner {
            ContiguousBufferIterInner::VecOfSelf(v) => v.len(),
            _ => 1,
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
impl<'a> AsMut<[Self]> for ContiguousBuffer<'a> {
    fn as_mut(&mut self) -> &mut [Self] {
        match unsafe { std::mem::transmute::<_, &mut ContiguousBufferInner>(&mut self.inner) } {
            VecOfSelf(v) => v,
            _ => std::slice::from_mut(self),
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
    fn insert_multiple<I: IntoIterator<Item = Self>>(&mut self, at: usize, t: I)
    where
        I::IntoIter: ExactSizeIterator,
    {
        match self.len().cmp(&at) {
            std::cmp::Ordering::Greater => match &mut self.inner {
                VecOfSelf(v) => v.insert_multiple(at, t),
                _ => {
                    let iter = t.into_iter();
                    let len = iter.len();
                    let this = unsafe { std::ptr::read(self) };
                    let mut v = Vec::with_capacity(len + 1);
                    v.extend(iter);
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

#[derive(Default, Debug, Clone, Copy)]
pub struct RawShmInfo;

impl<'a> SliceProvider<'a, SharedMemorySlice<'a>> for RawShmInfo {
    type Error = std::convert::Infallible;
    fn provide_slice(&'a self, key: SharedMemorySlice<'a>) -> Result<&'a [u8], Self::Error> {
        Ok(key.0)
    }
}

#[test]
fn size_test() {
    assert_eq!(std::mem::size_of::<Buffer<Vec<u8>, ()>>(), 56);
}

#[test]
fn fun_with_buffers() {
    let mut buffer: RawBuffer<'static, Vec<u8>> = RawBuffer::new();
    let dashes: &[u8] = b" - ";
    buffer.copy_extend(b"there!");
    buffer.extend(dashes);
    buffer.extend(Vec::from(b"General Kenobi!".as_ref()));
    buffer.copy_insert(0, b"Hello ");
    buffer.insert(0, dashes);
    assert_eq!(buffer, b" - Hello there! - General Kenobi!");
    println!("{}", buffer);
    // let short_lived = String::from("Not 'static, can't extend from here");
    // buffer.extend(short_lived.as_bytes()); // This cannot build, because `buffer` needs to live for `'static`
    // buffer.copy_extend(short_lived.as_bytes()); // But this can
}

pub trait BufferExtras {
    fn make_shareable(&mut self) -> &mut Self;
}

impl<'a, P> BufferExtras for Buffer<'a, Vec<u8>, P> {
    fn make_shareable(&mut self) -> &mut Self {
        let mut cache = Vec::new();
        std::mem::swap(&mut cache, &mut self.cache);
        let cache: Arc<ContiguousBuffer> = Arc::new(BytesVec(cache).into());
        for slice in self.slices.as_mut() {
            match &slice.inner {
                VecOfSelf(_) => unreachable!(),
                CacheRange(range) => {
                    *slice = SharedSelf(cache.clone(), *range).into();
                }
                Slice(_) => {}
                MutableSlice(_) => {}
                SharedMemoryInfo(_) => {}
                SharedSelf(_, _) => {}
                _ => {
                    let inner = unsafe { std::ptr::read(slice) };
                    unsafe {
                        std::ptr::write(
                            slice,
                            SharedSelf(Arc::new(inner), Range { start: 0, end: 0 }).into(),
                        )
                    }
                }
            }
        }
        self
    }
}

mod wbuf {
    use super::*;
    use crate::net::protocol::core::*;
    use crate::net::protocol::proto::*;
    struct Data {
        pub key: KeyExpr<'static>,
        pub data_info: Option<DataInfo>,
        pub payload: RawBuffer<'static, ()>,
        pub congestion_control: CongestionControl,
        pub reply_context: Option<ReplyContext>,
    }
    enum ZenohBody {
        Data(Data),
    }
    struct Attachment {
        pub buffer: RawBuffer<'static, ()>,
    }
    struct ZenohMessage {
        pub body: ZenohBody,
        pub channel: Channel,
        pub routing_context: Option<RoutingContext>,
        pub attachment: Option<Attachment>,
    }

    trait WBuf {
        fn write(&mut self, byte: u8) -> bool;
        fn write_zint(&mut self, byte: ZInt) -> bool;
        fn write_deco_attachment(&mut self, attachment: Attachment) -> bool;
        fn write_deco_attachment_by_ref(&mut self, attachment: &Attachment) -> bool;
        fn write_deco_routing_context(&mut self, context: &RoutingContext) -> bool;
        fn write_data(&mut self, data: &Data) -> bool;
        fn write_deco_priority(&mut self, priority: Priority) -> bool {
            self.write(priority.header())
        }
        fn write_frame_header(
            &mut self,
            priority: Priority,
            reliability: Reliability,
            sn: ZInt,
            is_fragment: Option<bool>,
            attachment: Option<Attachment>,
        ) -> bool {
            if priority != Priority::default() {
                zcheck!(self.write_deco_priority(priority))
            }
            if let Some(attachment) = attachment {
                zcheck!(self.write_deco_attachment(attachment))
            }
            let header = Frame::make_header(reliability, is_fragment);
            self.write(header) && self.write_zint(sn)
        }
        fn write_zenoh_message(&mut self, msg: &mut ZenohMessage) -> bool {
            if let Some(attachment) = msg.attachment.as_ref() {
                zcheck!(self.write_deco_attachment_by_ref(attachment));
            }
            if let Some(context) = msg.routing_context.as_ref() {
                zcheck!(self.write_deco_routing_context(context));
            }
            if msg.channel.priority != Priority::default() {
                zcheck!(self.write_deco_priority(msg.channel.priority));
            }
            let res = match &msg.body {
                ZenohBody::Data(data) => self.write_data(data),
                // ZenohBody::Declare(declare) => self.write_declare(declare),
                // ZenohBody::Unit(unit) => self.write_unit(unit),
                // ZenohBody::Pull(pull) => self.write_pull(pull),
                // ZenohBody::Query(query) => self.write_query(query),
                // ZenohBody::LinkStateList(link_state_list) => {
                //     self.write_link_state_list(link_state_list)
                // }
            };
            res
        }
    }
    impl<'a, Cache: CacheTrait> WBuf for RawBuffer<'a, Cache> {
        fn write(&mut self, byte: u8) -> bool {
            self.copy_extend(std::slice::from_ref(&byte));
            true
        }
        fn write_zint(&mut self, mut c: u64) -> bool {
            let mut buffer = [0; 10];
            let mut len = 0;
            let mut b = c as u8;
            while c > 0x7f {
                buffer[len] = b | 0x80;
                len += 1;
                c >>= 7;
                b = c as u8;
            }
            buffer[len] = b | 0x80;
            len += 1;
            self.copy_extend(&buffer[..len]);
            true
        }
        fn write_deco_attachment(&mut self, attachment: Attachment) -> bool {
            self.append_buffer(attachment.buffer);
            true
        }
        fn write_deco_attachment_by_ref(&mut self, attachment: &Attachment) -> bool {
            todo!()
        }
        fn write_deco_routing_context(&mut self, context: &RoutingContext) -> bool {
            todo!()
        }
        fn write_data(&mut self, data: &Data) -> bool {
            todo!()
        }
    }
}

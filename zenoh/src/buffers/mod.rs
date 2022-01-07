use derive_more::{AsRef, From};

pub(crate) mod internals;
pub use internals::{ContiguousBuffer, ContiguousSlice};

#[derive(From)]
pub(crate) enum BufferInner<'a> {
    Single(ContiguousSlice<'a>),
    Multiple(Vec<ContiguousSlice<'a>>),
}
impl<'a> Default for BufferInner<'a> {
    fn default() -> Self {
        Self::Multiple(Vec::new())
    }
}

#[derive(AsRef, From)]
pub struct Buffer<'a> {
    pub(crate) inner: BufferInner<'a>,
}
impl<'a> Default for Buffer<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Buffer<'a> {
    pub fn new() -> Self {
        Buffer {
            inner: BufferInner::Multiple(Vec::new()),
        }
    }
    pub fn with_slice_capacity(n: usize) -> Self {
        Buffer {
            inner: BufferInner::Multiple(Vec::with_capacity(n)),
        }
    }
    pub fn with_byte_capacity(n: usize) -> Self {
        Buffer {
            inner: BufferInner::Single(ContiguousSlice {
                buffer: Vec::<u8>::with_capacity(n).into(),
                start: 0,
                end: 0,
            }),
        }
    }
    pub fn add_slice<Slice: Into<ContiguousSlice<'a>>>(&mut self, slice: Slice) {
        inner(self, slice.into());
        #[inline]
        fn inner<'a>(this: &mut Buffer<'a>, slice: ContiguousSlice<'a>) {
            let mut inner = unsafe { std::ptr::read(&this.inner) };
            inner = match inner {
                BufferInner::Single(first) => BufferInner::Multiple(vec![first, slice]),
                BufferInner::Multiple(mut slices) => {
                    if slices.capacity() == 0 {
                        BufferInner::Single(slice)
                    } else {
                        slices.push(slice);
                        BufferInner::Multiple(slices)
                    }
                }
            };
            unsafe { std::ptr::write(&mut this.inner, inner) }
        }
    }
    pub fn add_slices<Slices: IntoIterator<Item = ContiguousSlice<'a>>>(&mut self, slices: Slices) {
        let iterator = slices.into_iter();
        let inner = unsafe { std::ptr::read(&self.inner) };
        let mut buffer = match inner {
            BufferInner::Single(first) => {
                let (min_size, max_size) = iterator.size_hint();
                let reserve = max_size.unwrap_or(min_size);
                let mut slices = Vec::with_capacity(reserve + 1);
                slices.push(first);
                slices
            }
            BufferInner::Multiple(slices) => slices,
        };
        buffer.extend(iterator);
        unsafe { std::ptr::write(&mut self.inner, BufferInner::Multiple(buffer)) }
    }
    pub fn insert_slice<Slice: Into<ContiguousSlice<'a>>>(&mut self, slice: Slice, index: usize) {
        let len = self.n_slices();
        if index == len {
            return self.add_slice(slice);
        }
        assert!(len < index);
        inner(self, slice.into(), index);
        #[inline]
        fn inner<'a>(this: &mut Buffer<'a>, slice: ContiguousSlice<'a>, index: usize) {
            let mut inner = unsafe { std::ptr::read(&this.inner) };
            inner = match inner {
                BufferInner::Single(first) => BufferInner::Multiple(vec![slice, first]),
                BufferInner::Multiple(mut slices) => {
                    if slices.capacity() == 0 {
                        BufferInner::Single(slice)
                    } else {
                        slices.insert(index, slice);
                        BufferInner::Multiple(slices)
                    }
                }
            };
            unsafe { std::ptr::write(&mut this.inner, inner) }
        }
    }
    pub fn n_slices(&self) -> usize {
        match &self.inner {
            BufferInner::Single(_) => 1,
            BufferInner::Multiple(v) => v.len(),
        }
    }
    /// Returns the number of bytes in the buffer
    pub fn len(&self) -> usize {
        match &self.inner {
            BufferInner::Single(v) => v.len(),
            BufferInner::Multiple(v) => v.iter().fold(0, |acc, s| acc + s.len()),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(crate) fn contiguous_slices(&self) -> &[ContiguousSlice<'a>] {
        match &self.inner {
            BufferInner::Single(s) => std::slice::from_ref(s),
            BufferInner::Multiple(s) => s.as_slice(),
        }
    }
    pub fn slices<'b>(&'b self) -> SlicesIter<'a, 'b> {
        self.contiguous_slices()
            .iter()
            .map(ContiguousSlice::<'a>::as_slice)
    }
    pub fn io_slices<'b>(&'b self) -> IoSlicesIter<'a, 'b> {
        self.contiguous_slices()
            .iter()
            .map(ContiguousSlice::<'a>::as_io_slice)
    }
    pub fn to_vec(&self) -> Vec<u8> {
        let mut result = Vec::with_capacity(self.len());
        for slice in self.slices() {
            result.extend(slice)
        }
        result
    }
}

pub type SlicesIter<'a, 'b> = std::iter::Map<
    std::slice::Iter<'b, ContiguousSlice<'a>>,
    fn(&'b ContiguousSlice<'a>) -> &'b [u8],
>;
pub type IoSlicesIter<'a, 'b> = std::iter::Map<
    std::slice::Iter<'b, ContiguousSlice<'a>>,
    fn(&'b ContiguousSlice<'a>) -> std::io::IoSlice<'b>,
>;

impl<'a, T: Into<ContiguousSlice<'a>>> From<T> for Buffer<'a> {
    fn from(v: T) -> Self {
        Buffer {
            inner: BufferInner::Single(v.into()),
        }
    }
}

pub struct Trie<T> {
    inner: TrieInner<T>,
    children: Vec<Trie<T>>,
}

impl<T: Sized> Trie<T> {
    pub fn new() -> Self {
        if std::mem::size_of::<T>() > INNER_SIZE {
            panic!(
                "zenoh_util::collections::Trie<T> only accepts types of size {}B or less",
                INNER_SIZE
            )
        }
        Trie {
            inner: TrieInner::slash(),
            children: Vec::new(),
        }
    }
    pub fn insert<P: AsRef<[u8]>>(&mut self, path: P, value: T) -> Option<T> {
        let path = path.as_ref();
        match self.inner.match_bytes(path) {
            Match::Full(n) => self.do_insert(&path[n as usize..], value),
            _ => None,
        }
    }
    fn do_insert(&mut self, path: &[u8], value: T) -> Option<T> {}
    pub fn get<P: AsRef<[u8]>>(&self, path: P) -> Option<&T> {
        let path = path.as_ref();
        match self.inner.match_bytes(path) {
            Match::Full(0) => Some(self.inner.as_ref()),
            Match::Full(len) => {
                let path = &path[..len as usize];
                self.children.iter().find_map(|c| c.get(path))
            }
            Match::None => None,
            Match::Partial(_) => None,
            Match::Star => None,
            Match::DouleStar => None,
        }
    }
}

impl<T: Sized> Default for Trie<T> {
    fn default() -> Self {
        Self::new()
    }
}
#[repr(C)]
struct TrieInner<T> {
    inner: [u8; INNER_SIZE],
    tag: u8,
    _marker: std::marker::PhantomData<T>,
}
enum Match {
    None,
    Partial(u32),
    Full(u32),
    Star,
    DouleStar,
}
const INNER_SIZE: usize = 64 - std::mem::size_of::<Vec<Trie<()>>>() - std::mem::size_of::<u8>();
impl<T> TrieInner<T> {
    fn from_bytes(s: &[u8]) -> (Self, &[u8]) {
        let split = s.len().min(INNER_SIZE);
        let mut result = TrieInner {
            inner: [0; INNER_SIZE],
            tag: split as u8,
            _marker: std::default::Default::default(),
        };
        let (left, right) = s.split_at(split);
        result.inner.clone_from_slice(left);
        (result, right)
    }
    fn match_bytes(&self, rhs: &[u8]) -> Match {
        if rhs.is_empty() {
            return if self.tag == Self::LEAF {
                Match::Full(0)
            } else {
                Match::None
            };
        }

        match self.tag {
            Self::LEAF | 0 => Match::None,
            Self::SLASH => {
                if rhs[0] == b'/' {
                    Match::Full(1)
                } else {
                    Match::None
                }
            }
            Self::STAR => Match::Star,
            Self::DOUBLE_STAR => Match::DouleStar,
            length if self.inner[0] == rhs[0] => {
                for (i, (a, b)) in self.inner[1..].iter().zip(&rhs[1..]).enumerate() {
                    if a != b {
                        return Match::Partial((i + 1) as u32);
                    }
                }
                Match::Full(length as u32)
            }
            _ => Match::None,
        }
    }
    const LEAF: u8 = INNER_SIZE as u8 + 1;
    const SLASH: u8 = INNER_SIZE as u8 + 2;
    const STAR: u8 = INNER_SIZE as u8 + 3;
    const DOUBLE_STAR: u8 = INNER_SIZE as u8 + 4;
    fn is_str(&self) -> bool {
        self.tag <= INNER_SIZE as u8
    }
    fn leaf(value: T) -> Self {
        let mut new = TrieInner {
            inner: [0; INNER_SIZE],
            tag: Self::LEAF,
            _marker: std::marker::PhantomData,
        };
        unsafe { std::ptr::copy_nonoverlapping(&value, new.as_mut(), 1) }
        std::mem::MaybeUninit::new(value);
        new
    }
    const fn slash() -> Self {
        TrieInner {
            inner: [0; INNER_SIZE],
            tag: Self::SLASH,
            _marker: std::marker::PhantomData,
        }
    }
    const fn star() -> Self {
        TrieInner {
            inner: [0; INNER_SIZE],
            tag: Self::STAR,
            _marker: std::marker::PhantomData,
        }
    }
    const fn double_star() -> Self {
        TrieInner {
            inner: [0; INNER_SIZE],
            tag: Self::DOUBLE_STAR,
            _marker: std::marker::PhantomData,
        }
    }
    fn into_inner(mut self) -> T {
        if self.tag == Self::LEAF {
            self.tag = 0;
            let mut value = std::mem::MaybeUninit::uninit();
            unsafe {
                std::ptr::copy_nonoverlapping(self.as_ref(), value.as_mut_ptr(), 1);
                value.assume_init()
            }
        } else {
            panic!("Attempted to apply into_inner on a non-leaf node")
        }
    }
}
impl<T> AsRef<T> for TrieInner<T> {
    fn as_ref(&self) -> &T {
        unsafe { std::mem::transmute(&self.inner) }
    }
}
impl<T> AsMut<T> for TrieInner<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { std::mem::transmute(&mut self.inner) }
    }
}
impl<T> Drop for TrieInner<T> {
    fn drop(&mut self) {
        if self.tag == Self::LEAF {
            let mut value = std::mem::MaybeUninit::uninit();
            unsafe {
                std::ptr::copy_nonoverlapping(self.as_ref(), value.as_mut_ptr(), 1);
                std::mem::drop(value.assume_init())
            }
        }
    }
}

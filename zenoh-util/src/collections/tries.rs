use std::fmt::Write;

pub struct Trie<T> {
    inner: TrieInner<T>,
    children: Vec<Trie<T>>,
}

impl<T> Trie<T> {
    pub fn new() -> Self
    where
        T: Sized,
    {
        if std::mem::size_of::<T>() > INNER_SIZE {
            panic!(
                "zenoh_util::collections::Trie<T> only accepts types of size {}B or less, but construction was attempted with a {}B-sized type",
                INNER_SIZE,
                std::mem::size_of::<T>()
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
    fn do_insert(&mut self, path: &[u8], mut value: T) -> Option<T> {
        for child in &mut self.children {
            match child.inner.match_bytes(path) {
                Match::Partial(len) => {
                    child.split(len);
                    return child.do_insert(&path[len as usize..], value);
                }
                Match::Full(0) => {
                    std::mem::swap(&mut value, child.inner.as_mut());
                    return Some(value);
                }
                Match::Full(len)
                | Match::Star { exact: true, len }
                | Match::DouleStar { exact: true, len } => {
                    return child.do_insert(&path[len as usize..], value)
                }
                _ => continue,
            }
        }
        let mut children = &mut self.children;
        let mut path = path;
        while !path.is_empty() {
            let (inner, p) = TrieInner::from_bytes(path);
            let child = Trie {
                inner,
                children: Vec::with_capacity(1),
            };
            children.push(child);
            path = p;
            children = &mut children.last_mut().unwrap().children;
        }
        children.push(TrieInner::leaf(value).into());
        None
    }
    fn split(&mut self, at: u32) {
        let previous_length = self.inner.tag as usize;
        let mut child = Trie {
            inner: TrieInner::from_bytes(&self.inner.inner.bytes[at as usize..previous_length]).0,
            children: Vec::with_capacity(2),
        };
        std::mem::swap(&mut self.children, &mut child.children);
        self.inner.tag = at as u8;
        self.children.push(child);
    }
    pub fn get<P: AsRef<[u8]>>(&self, path: P) -> Option<&T> {
        let path = path.as_ref();
        match self.inner.match_bytes(path) {
            Match::Leaf => Some(self.inner.as_ref()),
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let path = &path[len as usize..];
                self.children.iter().find_map(|c| c.get(path))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    pub fn get_mut<P: AsRef<[u8]>>(&mut self, path: P) -> Option<&mut T> {
        let path = path.as_ref();
        match self.inner.match_bytes(path) {
            Match::Leaf => Some(self.inner.as_mut()),
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let path = &path[len as usize..];
                self.children.iter_mut().find_map(|c| c.get_mut(path))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    pub fn remove<P: AsRef<[u8]>>(&mut self, path: P) -> Option<T> {
        let path = path.as_ref();
        match self.inner.match_bytes(path) {
            Match::Leaf => self.inner.take(),
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let path = &path[len as usize..];
                self.children.iter_mut().find_map(|c| c.remove(path))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    /// Will prune unused branches of the tree, and try to unite string segments that may have been split previously
    pub fn prune(&mut self) {
        let mut marked = Vec::new();
        for (i, child) in &mut self.children.iter_mut().enumerate() {
            if child.do_prune() {
                marked.push(i);
            }
        }
        for i in marked.into_iter().rev() {
            self.children.swap_remove(i);
        }
        // TODO: implement shortening
    }
    fn do_prune(&mut self) -> bool {
        match self.inner.tag {
            TrieInner::<T>::DELETED => true,
            TrieInner::<T>::LEAF => false,
            _ => {
                self.prune();
                self.children.is_empty()
            }
        }
    }
    pub fn iter<'a, P: AsRef<[u8]>>(&'a self, prefix: &'a P) -> Iter<'a, T> {
        let prefix = prefix.as_ref();
        let iters = match self.inner.match_bytes(prefix) {
            Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. }
            | Match::Leaf
            | Match::Mismatch => Vec::new(),
            Match::NullPrefix => vec![(0, self.children.iter())],
            Match::Full(len) => vec![(len as usize, self.children.iter())],
        };
        Iter { prefix, iters }
    }
}
pub struct Iter<'a, T> {
    prefix: &'a [u8],
    iters: Vec<(usize, std::slice::Iter<'a, Trie<T>>)>,
}
impl<'a, T> IntoIterator for &'a Trie<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter(b"/")
    }
}
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (start, iter) = self.iters.last_mut()?;
            let start = *start;
            let prefix = &self.prefix[start..];
            if let Some(node) = iter.next() {
                match node.inner.match_bytes(prefix) {
                    Match::NullPrefix => self.iters.push((start, node.children.iter())),
                    Match::Full(len) => self
                        .iters
                        .push((start + len as usize, node.children.iter())),
                    Match::Leaf if self.prefix.len() == start => return Some(node.inner.as_ref()),
                    Match::Partial(len) if self.prefix.len() == start + len as usize => self
                        .iters
                        .push((start + len as usize, node.children.iter())),
                    _ => {}
                }
            } else {
                self.iters.pop();
            }
        }
    }
}

impl<T: Sized + std::fmt::Debug> std::fmt::Display for Trie<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "-{}", &self.inner)?;
        let mut stack = Vec::new();
        let mut prefix = "|".to_string();
        let mut iter = self.children.iter();
        loop {
            match iter.next() {
                Some(next) => {
                    writeln!(f, "{}-{}", prefix, &next.inner)?;
                    if !next.children.is_empty() {
                        prefix += "|";
                        stack.push(iter);
                        iter = next.children.iter();
                    }
                }
                None => {
                    prefix.pop();
                    if let Some(next) = stack.pop() {
                        iter = next;
                    } else {
                        return Ok(());
                    }
                }
            }
        }
    }
}
#[test]
fn trie_display() {
    let mut trie: Trie<String> = Trie::new();
    let paths: [&str; 5] = [
        "/panda",
        "/patate/douce",
        "/patate",
        "/panda/geant",
        "/panda/roux",
    ];
    for path in paths {
        trie.insert(path, path.to_uppercase());
        println!("{}", &trie);
    }
    for path in paths {
        println!("{}", trie.get(dbg!(path)).unwrap());
    }
    for value in trie.iter(b"/pan") {
        dbg!(value);
    }
}
impl<T: Sized> From<TrieInner<T>> for Trie<T> {
    fn from(inner: TrieInner<T>) -> Self {
        Trie {
            inner,
            children: Vec::new(),
        }
    }
}
impl<T: Sized> Default for Trie<T> {
    fn default() -> Self {
        Self::new()
    }
}

struct TrieInnerUnion<T> {
    bytes: [u8; INNER_SIZE],
    _marker: std::marker::PhantomData<T>,
}
impl<T> TrieInnerUnion<T> {
    const fn new() -> Self {
        TrieInnerUnion {
            bytes: [0; INNER_SIZE],
            _marker: std::marker::PhantomData,
        }
    }
}
impl<T> Default for TrieInnerUnion<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> From<T> for TrieInnerUnion<T> {
    fn from(leaf: T) -> Self {
        let mut this = Self::default();
        unsafe { std::ptr::write(std::mem::transmute(&mut this.bytes), leaf) }
        this
    }
}
#[repr(C)]
struct TrieInner<T> {
    inner: TrieInnerUnion<T>,
    tag: u8,
}
impl<T: std::fmt::Debug> std::fmt::Display for TrieInner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.tag {
            Self::LEAF => write!(f, "{:?}", self.as_ref()),
            Self::SLASH => f.write_char('/'),
            Self::STAR => f.write_str("*/"),
            Self::DOUBLE_STAR => f.write_str("**/"),
            Self::DELETED => f.write_str("DELETED ENTRY"),
            len => f.write_str(unsafe {
                std::str::from_utf8_unchecked(&self.inner.bytes[..len as usize])
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Match {
    Mismatch,
    NullPrefix,
    Leaf,
    Partial(u32),
    Full(u32),
    Star { exact: bool, len: u32 },
    DouleStar { exact: bool, len: u32 },
}
const INNER_SIZE: usize = 64 - std::mem::size_of::<Vec<Trie<()>>>() - std::mem::size_of::<u8>();
impl<T> TrieInner<T> {
    fn from_bytes(s: &[u8]) -> (Self, &[u8]) {
        if s.starts_with(b"/") {
            return (Self::slash(), &s[1..]);
        }
        if s.starts_with(b"*/") {
            return (Self::star(), &s[2..]);
        }
        if s.starts_with(b"**/") {
            return (Self::double_star(), &s[3..]);
        }
        let split = s
            .iter()
            .position(|&c| c == b'/')
            .unwrap_or_else(|| s.len())
            .min(INNER_SIZE);
        let mut result = TrieInner {
            inner: TrieInnerUnion::default(),
            tag: split as u8,
        };
        let (left, right) = s.split_at(split);
        result.inner.bytes[..split].clone_from_slice(left);
        (result, right)
    }
    fn match_bytes(&self, rhs: &[u8]) -> Match {
        if rhs.is_empty() {
            return if self.tag == Self::LEAF {
                Match::Leaf
            } else {
                Match::NullPrefix
            };
        }
        match self.tag {
            0 => Match::Full(0),
            Self::LEAF | Self::DELETED => Match::Mismatch,
            Self::SLASH => {
                if rhs[0] == b'/' {
                    Match::Full(1)
                } else {
                    Match::Mismatch
                }
            }
            Self::STAR => {
                if rhs.starts_with(b"*/") {
                    Match::Star {
                        exact: true,
                        len: 2,
                    }
                } else {
                    Match::Star {
                        exact: false,
                        len: rhs
                            .iter()
                            .position(|&c| c == b'/')
                            .map(|f| f + 1)
                            .unwrap_or_else(|| rhs.len()) as u32,
                    }
                }
            }
            Self::DOUBLE_STAR => {
                if rhs.starts_with(b"**/") {
                    Match::DouleStar {
                        exact: true,
                        len: 3,
                    }
                } else {
                    Match::DouleStar {
                        exact: false,
                        len: 0,
                    }
                }
            }
            length if self.inner.bytes[0] == rhs[0] => {
                for (i, (a, b)) in self.inner.bytes[1..length as usize]
                    .iter()
                    .zip(&rhs[1..])
                    .enumerate()
                {
                    if a != b {
                        return Match::Partial((i + 1) as u32);
                    }
                }
                Match::Full((length as u32).min(rhs.len() as u32))
            }
            _ => Match::Mismatch,
        }
    }
    const LEAF: u8 = INNER_SIZE as u8 + 1;
    const SLASH: u8 = INNER_SIZE as u8 + 2;
    const STAR: u8 = INNER_SIZE as u8 + 3;
    const DOUBLE_STAR: u8 = INNER_SIZE as u8 + 4;
    const DELETED: u8 = INNER_SIZE as u8 + 5;
    fn is_str(&self) -> bool {
        self.tag <= INNER_SIZE as u8
    }
    fn leaf(value: T) -> Self {
        TrieInner {
            inner: TrieInnerUnion::from(value),
            tag: Self::LEAF,
        }
    }
    const fn slash() -> Self {
        TrieInner {
            inner: TrieInnerUnion::new(),
            tag: Self::SLASH,
        }
    }
    const fn star() -> Self {
        TrieInner {
            inner: TrieInnerUnion::new(),
            tag: Self::STAR,
        }
    }
    const fn double_star() -> Self {
        TrieInner {
            inner: TrieInnerUnion::new(),
            tag: Self::DOUBLE_STAR,
        }
    }
    fn take(&mut self) -> Option<T> {
        if self.tag == Self::LEAF {
            self.tag = Self::DELETED;
            Some(unsafe { std::ptr::read(std::mem::transmute(&mut self.inner.bytes)) })
        } else {
            None
        }
    }
    fn into_inner(mut self) -> T {
        self.take()
            .expect("Attempted to apply into_inner on a non-leaf node")
    }
}
impl<T> AsRef<T> for TrieInner<T> {
    fn as_ref(&self) -> &T {
        unsafe { std::mem::transmute(&self.inner.bytes) }
    }
}
impl<T> AsMut<T> for TrieInner<T> {
    fn as_mut(&mut self) -> &mut T {
        unsafe { std::mem::transmute(&mut self.inner.bytes) }
    }
}
impl<T> Drop for TrieInner<T> {
    fn drop(&mut self) {
        std::mem::drop(self.take());
    }
}

/// Ensures that Trie<T> construction panics if T is too large for Trie<T> to keep fitting.
#[test]
#[should_panic]
fn trie_panic_if_too_large() {
    let impossible_trie: Trie<[u64; 10]> = Trie::new();
    assert_ne!(std::mem::size_of_val(&impossible_trie), 64);
}

/// Ensures that any Trie<T> has the exact size of a typical cache line.
#[test]
fn trie_check_size() {
    let trie: Trie<[u64; 4]> = Trie::new();
    assert_eq!(std::mem::size_of_val(&trie), 64);
    let trie: Trie<[u8; 4]> = Trie::new();
    assert_eq!(std::mem::size_of_val(&trie), 64);
}

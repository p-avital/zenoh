use std::{fmt::Write, iter::FromIterator};

use aes::cipher::generic_array::functional::FunctionalSequence;

pub struct Trie<T> {
    inner: TrieInner<T>,
    children: Vec<Trie<T>>,
}

impl<T> Trie<T> {
    /// Constructs a new instance of a Trie.
    ///
    /// Since values are stored directly on Trie nodes, and to prevent nodes from growing too big, `T` must fit within 39 bytes. If you need to store something bigger, wrap it in an appropriate smart-pointer, such as [`Box`] or [`std::sync::Arc`]
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
    /// Inserts a value at the provided key, returning the previously stored value if it existed (wildcards are treated as litteral strings).
    pub fn insert<P: AsRef<[u8]>>(&mut self, key: P, mut value: T) -> Option<T> {
        let mut current = self;
        let mut key = KeyTokenizer::new(&key);
        if current.inner.insert_match(key.next()?) != InsertionMatch::Full {
            return None;
        }
        let mut current_token = key.next();
        while let Some(mut token) = current_token {
            let mut next_current = None;
            for child in &mut current.children {
                match child.inner.insert_match(token) {
                    InsertionMatch::Mismatch => {}
                    InsertionMatch::Full => {
                        current_token = key.next();
                        next_current = Some(unsafe { &mut *(child as *mut _) });
                        break;
                    }
                    InsertionMatch::SplitNode(at) => child.split(at),
                    InsertionMatch::SplitToken(0) => todo!(),
                    InsertionMatch::SplitToken(at) => unsafe {
                        // safety: `at` is guaranteed non-zero and strictly smaller than the current len of token.
                        current_token = Some(token.split_unchecked(at));
                        next_current = Some(&mut *(child as *mut _));
                        break;
                    },
                    InsertionMatch::SplitBoth(at) => {
                        child.split(at);
                        unsafe {
                            // safety: `at` is guaranteed non-zero and strictly smaller than the current len of token.
                            current_token = Some(token.split_unchecked(at));
                            next_current = Some(&mut *(child as *mut _));
                        }
                        break;
                    }
                }
            }
            if let Some(next_current) = next_current {
                current = next_current
            } else {
                let (inner, mut rest) = (&token).into();
                let child = Trie {
                    inner,
                    children: Vec::with_capacity(1),
                };
                current.children.push(child);
                current = current
                    .children
                    .last_mut()
                    .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
                while !rest.is_empty() {
                    let (inner, new_rest) = TrieInner::from_bytes(rest);
                    rest = new_rest;
                    let child = Trie {
                        inner,
                        children: Vec::with_capacity(1),
                    };
                    current.children.push(child);
                    current = current
                        .children
                        .last_mut()
                        .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
                }
                current_token = key.next();
            }
        }
        if let Some(child) = current.children.iter_mut().find(|child| {
            child.inner.tag == TrieInner::<T>::LEAF || child.inner.tag == TrieInner::<T>::DELETED
        }) {
            if child.inner.tag == TrieInner::<T>::DELETED {
                unsafe { std::ptr::write(child.inner.as_mut(), value) };
                child.inner.tag = TrieInner::<T>::LEAF;
                None
            } else {
                std::mem::swap(child.inner.as_mut(), &mut value);
                Some(value)
            }
        } else {
            current.children.push(TrieInner::leaf(value).into());
            None
        }
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
    /// Returns `true` if a value exists for `key`. If `GREEDY == false`, will only return `true` if an exact match exist (wildcards are treated as litteral strings).
    pub fn contains<P: AsRef<[u8]>, const GREEDY: bool>(&self, key: P) -> bool {
        if GREEDY {
            return self.get_all(key).next().is_some();
        }
        let key = key.as_ref();
        match self.inner.match_bytes::<false>(key) {
            Match::Leaf => true,
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let key = &key[len as usize..];
                self.children
                    .iter()
                    .any(|c| c.contains::<&[u8], GREEDY>(key))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => false,
        }
    }
    /// Returns a reference to the matching item if the `key` mathes exactly (wildcards are treaded as litteral strings).
    pub fn get<P: AsRef<[u8]>>(&self, key: P) -> Option<&T> {
        let key = key.as_ref();
        match self.inner.match_bytes::<false>(key) {
            Match::Leaf => Some(self.inner.as_ref()),
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let key = &key[len as usize..];
                self.children.iter().find_map(|c| c.get(key))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    /// Returns an iterator over the key-value pairs that match the provided `key`.
    ///
    /// The iterated keys are only valid until the next call to [`Iterator::next`], but the value references are valid throughout `'a`.
    pub fn get_all<P: AsRef<[u8]>>(&self, _key: P) -> GreedyIter<T> {
        todo!()
    }
    /// Returns a mutable reference to the matching item if the `key` mathes exactly (wildcards are treaded as litteral strings).
    pub fn get_mut<P: AsRef<[u8]>>(&mut self, key: P) -> Option<&mut T> {
        let key = key.as_ref();
        match self.inner.match_bytes::<false>(key) {
            Match::Leaf => Some(self.inner.as_mut()),
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let key = &key[len as usize..];
                self.children.iter_mut().find_map(|c| c.get_mut(key))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    /// Returns a mutable iterator over the key-value pairs that match the provided `key`.
    ///
    /// The iterated keys are only valid until the next call to [`Iterator::next`], but the mutable value references are valid throughout `'a`.
    pub fn get_all_mut<P: AsRef<[u8]>>(&mut self, _key: P) -> GreedyIterMut<T> {
        todo!()
    }
    /// Removes an entry, setting a gravestone in its place, using exact matching (wildcards are treaded as litteral strings).
    ///
    /// This remove is faster than [`Self::clean_remove`], but requires global cleanup steps to be taken in the future, such as [`Self::prune`]-ing after a few removes.
    pub fn remove<P: AsRef<[u8]>>(&mut self, key: P) -> Option<T> {
        let key = key.as_ref();
        match self.inner.match_bytes::<false>(key) {
            Match::Leaf => {
                self.inner.tag = TrieInner::<T>::DELETED;
                Some(unsafe { std::ptr::read(std::mem::transmute(&mut self.inner.inner.bytes)) })
            }
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let key = &key[len as usize..];
                self.children.iter_mut().find_map(|c| c.remove(key))
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    /// __Lazily__ removes all entries matching the provided keys.
    ///
    /// The iterator will yield each removed value. Upon dropping, the rest of the iterator will be iterated upon to ensure complete removal.
    /// This remover doesn't do any pruning of the trie. A good use pattern could be
    /// ```rust
    /// # use zenoh_util::collections::tries::Trie
    /// # let mut trie = Trie::new();
    /// if trie.remove_all("/demo/keys/**").count() > 10 {trie.prune()}
    /// ```
    pub fn remove_all<P: AsRef<[u8]>>(&mut self, _key: P) -> GreedyRemover<T> {
        todo!()
    }
    /// Removes an entry and applies local pruning to reoptimize the trie, using exact matching (wildcards are treaded as litteral strings).
    ///
    /// This operation is slower than [`Self::remove`], but keeps the trie optimized for less costs than [`Self::prune`].
    pub fn clean_remove<P: AsRef<[u8]>>(&mut self, key: P) -> Option<T> {
        self.do_clean_remove(key.as_ref()).map(|f| f.0)
    }
    fn do_clean_remove(&mut self, key: &[u8]) -> Option<(T, bool)> {
        match self.inner.match_bytes::<false>(key) {
            Match::Leaf => {
                self.inner.tag = TrieInner::<T>::DELETED;
                Some((
                    unsafe { std::ptr::read(std::mem::transmute(&mut self.inner.inner.bytes)) },
                    true,
                ))
            }
            Match::Full(len)
            | Match::Star { exact: true, len }
            | Match::DouleStar { exact: true, len } => {
                let key = &key[len as usize..];
                for i in 0..self.children.len() {
                    if let Some((value, delete)) =
                        unsafe { self.children.get_unchecked_mut(i).do_clean_remove(key) }
                    {
                        if delete {
                            self.children.swap_remove(i);
                            if self.children.is_empty() {
                                return Some((value, true));
                            } else {
                                self.join_child();
                                return Some((value, false));
                            }
                        }
                        return Some((value, false));
                    }
                }
                None
            }
            Match::Mismatch
            | Match::NullPrefix
            | Match::Partial(_)
            | Match::Star { .. }
            | Match::DouleStar { .. } => None,
        }
    }
    /// Will prune unused branches of the tree, and try to unite string segments that may have been split previously.
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
        self.join_child();
    }
    /// Merges node with its single child if possible.
    fn join_child(&mut self) {
        if self.inner.is_str()
            && self.children.len() == 1
            && self.children[0].inner.is_str()
            && (self.inner.tag + self.children[0].inner.tag) as usize <= INNER_SIZE
        {
            let Trie {
                inner: child,
                children,
            } = match self.children.pop() {
                Some(val) => val,
                None => unsafe { std::hint::unreachable_unchecked() },
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &child.inner.bytes as *const u8,
                    &mut self.inner.inner.bytes[self.inner.tag as usize],
                    child.tag as usize,
                )
            };
            self.inner.tag += child.tag;
            self.children = children;
        }
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
    /// Iterates over the values present in the trie.
    pub fn values(&self) -> Values<T> {
        self.into()
    }
    /// Iterates mutably over the values present in the trie.
    pub fn values_mut(&mut self) -> ValuesMut<T> {
        self.into()
    }
    /// Iterates over the key-value pairs present in the trie.
    ///
    /// The key part of the item only lives as long as [`Iterator::next`] isn't called.
    pub fn iter(&self) -> Iter<T> {
        self.into_iter()
    }
    /// Iterates mutably over the key-value pairs present in the trie.
    ///
    /// The key part of the item only lives as long as [`Iterator::next`] isn't called.
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.into_iter()
    }
    /// Returns the number of entries marked as deleted within the trie.
    pub fn count_gravestones(&self) -> usize {
        self.children.iter().fold(
            (self.inner.tag == TrieInner::<T>::DELETED) as usize,
            |acc, it| acc + it.count_gravestones(),
        )
    }
}
pub struct Values<'a, T> {
    iters: Vec<std::slice::Iter<'a, Trie<T>>>,
}
impl<'a, T> From<&'a Trie<T>> for Values<'a, T> {
    fn from(trie: &'a Trie<T>) -> Self {
        Values {
            iters: vec![trie.children.iter()],
        }
    }
}
impl<'a, T> Iterator for Values<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let iter = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                match node.inner.tag {
                    TrieInner::<T>::DELETED => {}
                    TrieInner::<T>::LEAF => return Some(node.inner.as_ref()),
                    _ => self.iters.push(node.children.iter()),
                }
            } else {
                self.iters.pop();
            }
        }
    }
}
pub struct ValuesMut<'a, T> {
    iters: Vec<std::slice::IterMut<'a, Trie<T>>>,
}
impl<'a, T> From<&'a mut Trie<T>> for ValuesMut<'a, T> {
    fn from(trie: &'a mut Trie<T>) -> Self {
        ValuesMut {
            iters: vec![trie.children.iter_mut()],
        }
    }
}
impl<'a, T> Iterator for ValuesMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let iter = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                match node.inner.tag {
                    TrieInner::<T>::DELETED => {}
                    TrieInner::<T>::LEAF => return Some(node.inner.as_mut()),
                    _ => self.iters.push(node.children.iter_mut()),
                }
            } else {
                self.iters.pop();
            }
        }
    }
}
pub struct Iter<'a, T> {
    key: Vec<u8>,
    iters: Vec<(usize, std::slice::Iter<'a, Trie<T>>)>,
}
impl<'a, T> IntoIterator for &'a Trie<T> {
    type Item = (&'a [u8], &'a T);
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        let key = self.inner.as_bytes();
        Iter {
            key: key.into(),
            iters: vec![(0, self.children.iter())],
        }
    }
}
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (&'a [u8], &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, iter) = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                match node.inner.tag {
                    TrieInner::<T>::DELETED => {}
                    TrieInner::<T>::LEAF => {
                        return Some((
                            unsafe {
                                std::slice::from_raw_parts(self.key.as_ptr(), self.key.len())
                            },
                            node.inner.as_ref(),
                        ))
                    }
                    _ => {
                        let new_len = self.key.len();
                        self.key.extend(node.inner.as_bytes());
                        self.iters.push((new_len, node.children.iter()))
                    }
                }
            } else if let Some((len, _)) = self.iters.pop() {
                unsafe { self.key.set_len(len) }
            }
        }
    }
}
pub struct IterMut<'a, T> {
    key: Vec<u8>,
    iters: Vec<(usize, std::slice::IterMut<'a, Trie<T>>)>,
}
impl<'a, T> IntoIterator for &'a mut Trie<T> {
    type Item = (&'a [u8], &'a mut T);
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        let key = self.inner.as_bytes();
        IterMut {
            key: key.into(),
            iters: vec![(0, self.children.iter_mut())],
        }
    }
}
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (&'a [u8], &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, iter) = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                match node.inner.tag {
                    TrieInner::<T>::DELETED => {}
                    TrieInner::<T>::LEAF => {
                        return Some((
                            unsafe {
                                std::slice::from_raw_parts(self.key.as_ptr(), self.key.len())
                            },
                            node.inner.as_mut(),
                        ))
                    }
                    _ => {
                        let new_len = self.key.len();
                        self.key.extend(node.inner.as_bytes());
                        self.iters.push((new_len, node.children.iter_mut()))
                    }
                }
            } else if let Some((len, _)) = self.iters.pop() {
                unsafe { self.key.set_len(len) }
            }
        }
    }
}

struct NoDoublestarIter<'a, T> {
    requested_key: &'a [u8],
    current_key: Vec<u8>,
    iters: Vec<(usize, usize, std::slice::Iter<'a, Trie<T>>)>,
}

enum GreedyIterEnum<'a, T> {
    NoDoublestar(NoDoublestarIter<'a, T>),
}

pub struct GreedyIter<'a, T> {
    inner: GreedyIterEnum<'a, T>,
}
impl<'a, T> Iterator for GreedyIter<'a, T> {
    type Item = (&'a [u8], &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {}
    }
}
struct GreedyMutator<'a, T> {
    todo: std::marker::PhantomData<&'a T>,
}
impl<'a, T> Iterator for GreedyMutator<'a, T> {
    type Item = (&'a [u8], &'a mut Trie<T>);
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
pub struct GreedyIterMut<'a, T> {
    mutator: GreedyMutator<'a, T>,
}
impl<'a, T> Iterator for GreedyIterMut<'a, T> {
    type Item = (&'a [u8], &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        self.mutator
            .next()
            .map(|(key, trie)| (key, trie.inner.as_mut()))
    }
}
pub struct GreedyRemover<'a, T> {
    mutator: GreedyMutator<'a, T>,
}
impl<'a, T> Iterator for GreedyRemover<'a, T> {
    type Item = (&'a [u8], T);
    fn next(&mut self) -> Option<Self::Item> {
        for (key, trie) in &mut self.mutator {
            if let Some(value) = trie.inner.take() {
                return Some((key, value));
            }
        }
        None
    }
}
impl<'a, T> Drop for GreedyRemover<'a, T> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

struct KeyTokenizer<'a> {
    key: &'a [u8],
    index: usize,
}
impl<'a> KeyTokenizer<'a> {
    fn new<S: AsRef<[u8]> + ?Sized>(key: &'a S) -> Self {
        KeyTokenizer {
            key: key.as_ref(),
            index: 0,
        }
    }
    fn token(key: &'a [u8], at: usize) -> (Option<Token<'a>>, usize) {
        let mut tokenizer = KeyTokenizer { key, index: at };
        let token = tokenizer.next();
        (token, tokenizer.index)
    }
    fn reset(&mut self) {
        self.index = 0
    }
    fn seek_end(&mut self) {
        self.index = self.key.len()
    }
}
impl<'a> Iterator for KeyTokenizer<'a> {
    type Item = Token<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let key = &self.key[self.index..];
        if key.is_empty() {
            return None;
        }
        match (key[0], key.get(1)) {
            (b'/', _) => {
                self.index += 1;
                Some(Token::SLASH)
            }
            (b'*', None) => {
                self.index += 1;
                Some(Token::STAR)
            }
            (b'*', Some(b'/')) => {
                self.index += 2;
                Some(Token::STAR)
            }
            (b'*', Some(b'*')) => match key.get(2) {
                None => {
                    self.index += 2;
                    Some(Token::DOUBLE_STAR)
                }
                Some(b'/') => {
                    self.index += 3;
                    Some(Token::DOUBLE_STAR)
                }
                _ => {
                    if let Some(next_slash) = key.iter().position(|c| *c == b'/') {
                        self.index += next_slash;
                        Some(Token::from(&key[..next_slash]))
                    } else {
                        self.index = self.key.len();
                        Some(key.into())
                    }
                }
            },
            _ => {
                if let Some(next_slash) = key.iter().position(|c| *c == b'/') {
                    self.index += next_slash;
                    Some(Token::from(&key[..next_slash]))
                } else {
                    self.index = self.key.len();
                    Some(key.into())
                }
            }
        }
    }
}
impl<'a> DoubleEndedIterator for KeyTokenizer<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index == 0 {
            return None;
        }
        match (
            self.key[self.index - 1],
            (self.index >= 2).then(|| self.key[self.index - 2]),
        ) {
            (b'*', Some(b'/')) | (b'*', None) => {
                self.index -= 1;
                Some(Token::STAR)
            }
            (b'*', Some(b'*')) if self.index == 2 || self.key[self.index - 3] == b'/' => {
                self.index -= 2;
                Some(Token::DOUBLE_STAR)
            }
            (b'/', Some(b'*')) if self.index == 2 || self.key[self.index - 3] == b'/' => {
                self.index -= 2;
                Some(Token::STAR)
            }
            (b'/', Some(b'*'))
                if (self.index == 3 && self.key[self.index - 3] == b'*')
                    || (self.index > 3
                        && self.key[self.index - 3] == b'*'
                        && self.key[self.index - 4] == b'/') =>
            {
                self.index -= 3;
                Some(Token::DOUBLE_STAR)
            }
            (b'/', _) => {
                self.index -= 1;
                Some(Token::SLASH)
            }
            _ => {
                let end = self.index;
                for start in (0..end - 1).rev() {
                    if self.key[start] == b'/' {
                        self.index = start + 1;
                        return Some(Token::from(&self.key[self.index..end]));
                    }
                }
                self.index = 0;
                Some(Token::from(&self.key[..end]))
            }
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Token<'a> {
    start: *const u8,
    len: std::num::NonZeroUsize,
    lifetime: std::marker::PhantomData<&'a ()>,
}
impl<'a> std::fmt::Display for Token<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::DOUBLE_STAR => f.write_str("**/"),
            Self::STAR => f.write_str("*/"),
            Self::SLASH => f.write_str("/"),
            slice => f.write_str(std::str::from_utf8(slice.into()).unwrap()),
        }
    }
}
impl<'a> From<&'a [u8]> for Token<'a> {
    fn from(slice: &'a [u8]) -> Self {
        let start = slice.as_ptr();
        Token {
            start,
            len: std::num::NonZeroUsize::new(slice.len()).unwrap(),
            lifetime: std::marker::PhantomData,
        }
    }
}
impl<'a> From<Token<'a>> for &'a [u8] {
    fn from(slice: Token<'a>) -> Self {
        unsafe { std::slice::from_raw_parts(slice.start, slice.len.get()) }
    }
}
impl<'a> Token<'a> {
    const DOUBLE_STAR: Self = Token {
        start: std::ptr::null(),
        len: unsafe { std::num::NonZeroUsize::new_unchecked(3) },
        lifetime: std::marker::PhantomData,
    };
    const STAR: Self = Token {
        start: std::ptr::null(),
        len: unsafe { std::num::NonZeroUsize::new_unchecked(1) },
        lifetime: std::marker::PhantomData,
    };
    const SLASH: Self = Token {
        start: std::ptr::null(),
        len: unsafe { std::num::NonZeroUsize::new_unchecked(2) },
        lifetime: std::marker::PhantomData,
    };
    fn is_str(&self) -> bool {
        !self.start.is_null()
    }
    fn as_bytes(&'a self) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.start, self.len.get()) }
    }
    /// Splits a string token into 2 string tokens.
    /// # Safety
    /// This method performs no bounds checks, as requests to split a token only originate from contexts where bound checks have already been made
    unsafe fn split_unchecked(&mut self, at: u32) -> Self {
        let result = Token {
            start: self.start.offset(at as isize),
            len: std::num::NonZeroUsize::new_unchecked(self.len.get() - at as usize),
            lifetime: self.lifetime,
        };
        self.len = std::num::NonZeroUsize::new_unchecked(at as usize);
        result
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InsertionMatch {
    Mismatch,
    Full,
    SplitNode(u32),
    SplitToken(u32),
    SplitBoth(u32),
}
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
    fn insert_match(&self, rhs: Token) -> InsertionMatch {
        if self.tag == 0 {
            return InsertionMatch::SplitToken(0);
        }
        if rhs.is_str() {
            return self.insert_match_bytes(rhs.as_bytes());
        }
        match (self.tag, rhs) {
            (Self::SLASH, Token::SLASH)
            | (Self::STAR, Token::STAR)
            | (Self::DOUBLE_STAR, Token::DOUBLE_STAR) => InsertionMatch::Full,
            _ => InsertionMatch::Mismatch,
        }
    }
    fn insert_match_bytes(&self, rhs: &[u8]) -> InsertionMatch {
        if !self.is_str() {
            return InsertionMatch::Mismatch;
        }
        let lhs = self.as_bytes();
        if lhs[0] != rhs[0] {
            return InsertionMatch::Mismatch;
        }
        let llen = lhs.len();
        let rlen = rhs.len();
        for i in 1..(llen.min(rlen)) {
            if lhs[i] != rhs[i] {
                return InsertionMatch::SplitBoth(i as u32);
            }
        }
        match llen.cmp(&rlen) {
            std::cmp::Ordering::Less => InsertionMatch::SplitNode(llen as u32),
            std::cmp::Ordering::Equal => InsertionMatch::Full,
            std::cmp::Ordering::Greater => InsertionMatch::SplitToken(rlen as u32),
        }
    }
    fn match_bytes<const ALLOW_PARTIAL: bool>(&self, rhs: &[u8]) -> Match {
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
            length if ALLOW_PARTIAL && self.inner.bytes[0] == rhs[0] => {
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
            length if !ALLOW_PARTIAL && rhs.starts_with(&self.inner.bytes[..length as usize]) => {
                Match::Full(length as u32)
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
    fn as_bytes(&self) -> &[u8] {
        match self.tag {
            Self::DELETED | Self::LEAF => b"",
            Self::SLASH => b"/",
            Self::STAR => b"*/",
            Self::DOUBLE_STAR => b"**/",
            tag => &self.inner.bytes[..tag as usize],
        }
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
impl<'a, T> From<&'a Token<'a>> for (TrieInner<T>, &'a [u8]) {
    fn from(token: &'a Token<'a>) -> Self {
        match *token {
            Token::SLASH => (TrieInner::<T>::slash(), b""),
            Token::STAR => (TrieInner::<T>::star(), b""),
            Token::DOUBLE_STAR => (TrieInner::<T>::double_star(), b""),
            _ => TrieInner::<T>::from_bytes(token.as_bytes()),
        }
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

#[test]
fn trie_tokenizer() {
    for key in [
        "/panda",
        "/patate/douce",
        "/patate",
        "/panda/geant",
        "/panda/*",
        "/**/*/roux",
        "/panda/roux",
    ] {
        let mut tokenizer = KeyTokenizer::new(key);
        let tokens = Vec::from_iter(tokenizer.by_ref());
        for (a, b) in tokens.into_iter().rev().zip(tokenizer.rev()) {
            assert_eq!(a, b)
        }
    }
}

#[test]
fn trie_general() {
    let mut trie: Trie<String> = Trie::new();
    let keys: [&str; 5] = [
        "/panda",
        "/patate/douce",
        "/patate",
        "/panda/geant",
        "/panda/roux",
    ];
    for key in keys {
        assert!(trie.insert(key, key.to_uppercase()).is_none());
    }
    for (key, value) in &trie {
        let key = std::str::from_utf8(key).unwrap();
        assert!(keys.contains(&key));
        assert_eq!(&key.to_uppercase(), value);
    }
    for key in keys {
        assert_eq!(trie.get(key).unwrap(), &key.to_uppercase());
    }
    for key in keys.iter().filter(|p| !p.starts_with("/panda")) {
        assert_eq!(trie.remove(key).unwrap(), key.to_uppercase());
    }
    assert_eq!(trie.count_gravestones(), 2);
    trie.prune();
    assert_eq!(trie.count_gravestones(), 0);
    trie.insert("/panda", "PANDA".into()).unwrap();
    assert_eq!(trie.get("/panda").unwrap(), "PANDA");
}

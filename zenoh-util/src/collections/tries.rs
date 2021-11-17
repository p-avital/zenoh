use std::{hint::unreachable_unchecked, mem::MaybeUninit};

pub type Trie<T, const N: usize> = GenericTrie<T, DefaultMatcher, { N }>;
#[repr(C)]
pub struct GenericTrie<T, Tokenizer, const N: usize> {
    children: Vec<Self>,
    value: MaybeUninit<T>,
    buffer: [u8; N],
    tag: u8,
    tokenizer_marker: std::marker::PhantomData<Tokenizer>,
}
impl<T: std::fmt::Debug, Matcher, const N: usize> std::fmt::Debug
    for GenericTrie<T, Matcher, { N }>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut formatter = f.debug_struct("GenericTrie");
        let slice = self.as_ref();
        if let Ok(s) = std::str::from_utf8(slice) {
            formatter.field("prefix", &s);
        } else {
            formatter.field("buffer", &self.as_ref());
        }
        if let Some(value) = self.get_value() {
            formatter.field("value", value);
        }
        if !self.children.is_empty() {
            formatter.field("children", &self.children);
        }
        formatter.finish()
    }
}

fn trim_slashes(key: &[u8]) -> &[u8] {
    let isnt_slash = |&c| c != b'/';
    let end = key.iter().rposition(isnt_slash).unwrap_or(0);
    &key[..(end + 1)]
}

fn dbg(key: &[u8]) -> &[u8] {
    println!("{}", std::str::from_utf8(key).unwrap());
    key
}

impl<T, Tokenizer, const N: usize> GenericTrie<T, Tokenizer, { N }> {
    pub fn with_capacity(capacity: usize) -> Self {
        debug_assert_eq!(std::mem::size_of::<Self>(), 64);
        let buffer = unsafe {
            let mut buffer: MaybeUninit<[u8; N]> = MaybeUninit::uninit();
            buffer.assume_init_mut()[0] = b'/';
            buffer.assume_init()
        };
        GenericTrie {
            value: MaybeUninit::uninit(),
            buffer,
            tag: 1,
            children: Vec::with_capacity(capacity),
            tokenizer_marker: std::marker::PhantomData,
        }
    }
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
    pub fn insert<K: AsRef<[u8]>>(&mut self, key: K, value: T) -> Option<T> {
        let mut key = trim_slashes(key.as_ref());
        match InsertionMatch::from(key, self.as_ref()) {
            InsertionMatch::None => self.split(0),
            InsertionMatch::Full => return self.set_value(value),
            InsertionMatch::SplitLeft(at) => key = &key[at as usize..],
            InsertionMatch::SplitRight(at) => {
                self.split(at as usize);
                return self.set_value(value);
            }
            InsertionMatch::SplitBoth(at) => {
                let at = at as usize;
                self.split(at);
                key = &key[at..]
            }
        }
        let mut node = self;
        'traversal: loop {
            for child in node.children.iter_mut() {
                match InsertionMatch::from(key, child.as_ref()) {
                    InsertionMatch::None => {}
                    InsertionMatch::Full => {
                        key = b"";
                        break 'traversal;
                    }
                    InsertionMatch::SplitLeft(at) => {
                        key = &key[at as usize..];
                        node = unsafe { std::mem::transmute(child) };
                        continue 'traversal;
                    }
                    InsertionMatch::SplitRight(at) => {
                        child.split(at as usize);
                        node = unsafe { std::mem::transmute(child) };
                        key = b"";
                        break 'traversal;
                    }
                    InsertionMatch::SplitBoth(at) => {
                        let at = at as usize;
                        child.split(at);
                        node = unsafe { std::mem::transmute(child) };
                        key = &key[at..];
                    }
                }
            }
            break 'traversal;
        }
        for key in key.chunks(N) {
            let mut buffer = [0; N];
            let len = key.len();
            buffer[..len].copy_from_slice(key);
            let child = GenericTrie {
                value: MaybeUninit::uninit(),
                buffer,
                tag: len as u8,
                children: Vec::new(),
                tokenizer_marker: std::marker::PhantomData,
            };
            node.children.push(child);
            node = node
                .children
                .last_mut()
                .unwrap_or_else(|| unsafe { unreachable_unchecked() })
        }
        node.set_value(value)
    }
    pub fn get<K: AsRef<[u8]>>(&self, key: K) -> Option<&T> {
        let mut key = trim_slashes(key.as_ref());
        let mut node = self;
        'traversal: while !key.is_empty() {
            for child in node.children.iter() {
                if key.starts_with(child.as_ref()) {
                    node = child;
                    key = &key[child.len() as usize..];
                    continue 'traversal;
                }
            }
            return None;
        }
        node.get_value()
    }
    pub fn get_mut<K: AsRef<[u8]>>(&mut self, key: K) -> Option<&mut T> {
        let mut key = trim_slashes(key.as_ref());
        let mut node = self;
        'traversal: while !key.is_empty() {
            for child in node.children.iter_mut() {
                if key.starts_with(child.as_ref()) {
                    key = &key[child.len() as usize..];
                    node = unsafe { std::mem::transmute(child) };
                    continue 'traversal;
                }
            }
            return None;
        }
        node.get_value_mut()
    }
}
impl<T, Tokenizer, const N: usize> GenericTrie<T, Tokenizer, { N }> {
    fn split(&mut self, at: usize) {
        let mut child_buffer = [0; N];
        let child_len = self.len() as usize - at;
        let mut children = Vec::with_capacity(1);
        std::mem::swap(&mut children, &mut self.children);
        let mut child = unsafe {
            std::ptr::copy_nonoverlapping(
                self.buffer.as_ptr().add(at),
                child_buffer.as_mut_ptr(),
                child_len,
            );
            GenericTrie {
                value: std::mem::transmute_copy(&self.value),
                buffer: child_buffer,
                tag: self.tag,
                children,
                tokenizer_marker: std::marker::PhantomData,
            }
        };
        child.set_len(child_len);
        self.children.push(child);
        self.tag = at as u8;
    }
    fn as_str(&self) -> &str {
        std::str::from_utf8(self.as_ref()).unwrap()
    }
    const LEN_MASK: u8 = 0b0111_1111;
    const HOLDS_VALUE_MASK: u8 = 0b1000_0000;
    const fn holds_value(&self) -> bool {
        self.tag & Self::HOLDS_VALUE_MASK != 0
    }
    const fn len(&self) -> u8 {
        self.tag & Self::LEN_MASK
    }
    fn set_value(&mut self, mut value: T) -> Option<T> {
        if self.holds_value() {
            std::mem::swap(&mut value, unsafe { self.value.assume_init_mut() });
            Some(value)
        } else {
            unsafe {
                self.value = MaybeUninit::new(value);
                self.set_holds_value(true);
            }
            None
        }
    }
    fn remove_value(&mut self) -> Option<T> {
        self.holds_value().then(|| unsafe {
            self.set_holds_value(false);
            std::ptr::read(self.value.as_ptr())
        })
    }
    fn get_value(&self) -> Option<&T> {
        self.holds_value()
            .then(|| unsafe { std::mem::transmute(&self.value) })
    }
    fn get_value_mut(&mut self) -> Option<&mut T> {
        self.holds_value()
            .then(|| unsafe { std::mem::transmute(&mut self.value) })
    }
    unsafe fn set_holds_value(&mut self, value: bool) {
        self.tag &= !Self::HOLDS_VALUE_MASK;
        self.tag |= if value { Self::HOLDS_VALUE_MASK } else { 0 }
    }
    fn set_len(&mut self, len: usize) {
        self.tag &= !Self::LEN_MASK;
        self.tag |= len as u8 & Self::LEN_MASK
    }
}
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum InsertionMatch {
    None,
    Full,
    SplitLeft(u32),
    SplitRight(u32),
    SplitBoth(u32),
}
impl InsertionMatch {
    fn from(left: &[u8], right: &[u8]) -> Self {
        match (left.len(), right.len()) {
            (0, 0) => Self::Full,
            (_, _) if left[0] != right[0] => Self::None,
            (llen, rlen) => {
                for i in 1..llen.min(rlen) {
                    if left[i] != right[i] {
                        return Self::SplitBoth(i as u32);
                    }
                }
                match llen.cmp(&rlen) {
                    std::cmp::Ordering::Less => Self::SplitRight(llen as u32),
                    std::cmp::Ordering::Equal => Self::Full,
                    std::cmp::Ordering::Greater => Self::SplitLeft(rlen as u32),
                }
            }
        }
    }
}

pub struct DefaultMatcher;
pub struct DefaultTokenizer<'a> {
    slice: &'a [u8],
    index: usize,
}
impl DefaultTokenizer<'_> {
    fn longest_double_star_str(mut string: &[u8]) -> usize {
        let mut counter = 0;
        loop {
            if string.starts_with(b"**/") {
                string = &string[3..];
                counter += 3;
            } else if string == b"**" {
                return counter + 2;
            } else {
                return counter;
            }
        }
    }
}
impl<'a> Iterator for DefaultTokenizer<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<Self::Item> {
        let mut token = &self.slice[self.index..];
        if token.is_empty() {
            return None;
        }
        let double_star_len = Self::longest_double_star_str(token);
        if double_star_len != 0 {
            self.index += double_star_len;
            return Some(b"**");
        }
        if let Some(end) = token.iter().position(|&c| c == b'/') {
            token = &token[..end];
            self.index = end + 1;
        } else {
            self.index = self.slice.len();
        }
        Some(token)
    }
}

impl<T, Tokenizer, const N: usize> Default for GenericTrie<T, Tokenizer, { N }> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Tokenizer, const N: usize> Drop for GenericTrie<T, Tokenizer, { N }> {
    fn drop(&mut self) {
        if self.holds_value() {
            unsafe { std::ptr::drop_in_place(self.value.assume_init_mut()) }
        }
    }
}
impl<T, Tokenizer, const N: usize> AsRef<[u8]> for GenericTrie<T, Tokenizer, { N }> {
    fn as_ref(&self) -> &[u8] {
        &self.buffer[..self.len() as usize]
    }
}

// impl<T> Trie<T> {
//     /// Constructs a new instance of a Trie.
//     ///
//     /// Since values are stored directly on Trie nodes, and to prevent nodes from growing too big, `T` must fit within 39 bytes. If you need to store something bigger, wrap it in an appropriate smart-pointer, such as [`Box`] or [`std::sync::Arc`]
//     pub fn new() -> Self
//     where
//         T: Sized,
//     {
//         if std::mem::size_of::<T>() > INNER_SIZE {
//             panic!(
//                 "zenoh_util::collections::Trie<T> only accepts types of size {}B or less, but construction was attempted with a {}B-sized type",
//                 INNER_SIZE,
//                 std::mem::size_of::<T>()
//             )
//         }
//         Self::text_node(b"/")
//     }
//     /// Inserts a value at the provided key, returning the previously stored value if it existed (wildcards are treated as litteral strings).
//     pub fn insert<P: AsRef<[u8]>>(&mut self, key: P, mut value: T) -> Option<T> {
//         let mut current = self;
//         let mut key = KeyTokenizer::new(&key);
//         if current.inner.insert_match(key.next()?) != InsertionMatch::Full {
//             return None;
//         }
//         let mut current_token = key.next();
//         while let Some(mut token) = current_token {
//             let mut next_current = None;
//             for child in &mut current.children {
//                 match child.inner.insert_match(token) {
//                     InsertionMatch::Mismatch => {}
//                     InsertionMatch::Full => {
//                         current_token = key.next();
//                         next_current = Some(unsafe { &mut *(child as *mut _) });
//                         break;
//                     }
//                     InsertionMatch::SplitNode(at) => child.split(at),
//                     InsertionMatch::SplitToken(0) => todo!(),
//                     InsertionMatch::SplitToken(at) => unsafe {
//                         // safety: `at` is guaranteed non-zero and strictly smaller than the current len of token.
//                         current_token = Some(token.split_unchecked(at));
//                         next_current = Some(&mut *(child as *mut _));
//                         break;
//                     },
//                     InsertionMatch::SplitBoth(at) => {
//                         child.split(at);
//                         unsafe {
//                             // safety: `at` is guaranteed non-zero and strictly smaller than the current len of token.
//                             current_token = Some(token.split_unchecked(at));
//                             next_current = Some(&mut *(child as *mut _));
//                         }
//                         break;
//                     }
//                 }
//             }
//             if let Some(next_current) = next_current {
//                 current = next_current
//             } else {
//                 let (inner, mut rest) = (&token).into();
//                 let child = Trie {
//                     inner,
//                     children: Vec::with_capacity(1),
//                 };
//                 current.children.push(child);
//                 current = current
//                     .children
//                     .last_mut()
//                     .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
//                 while !rest.is_empty() {
//                     let (inner, new_rest) = TrieInner::from_bytes(rest);
//                     rest = new_rest;
//                     let child = Trie {
//                         inner,
//                         children: Vec::with_capacity(1),
//                     };
//                     current.children.push(child);
//                     current = current
//                         .children
//                         .last_mut()
//                         .unwrap_or_else(|| unsafe { std::hint::unreachable_unchecked() });
//                 }
//                 current_token = key.next();
//             }
//         }
//         if let Some(child) = current.children.iter_mut().find(|child| {
//             child.inner.tag == TrieInner::<T>::LEAF || child.inner.tag == TrieInner::<T>::DELETED
//         }) {
//             if child.inner.tag == TrieInner::<T>::DELETED {
//                 unsafe { std::ptr::write(child.inner.as_mut(), value) };
//                 child.inner.tag = TrieInner::<T>::LEAF;
//                 None
//             } else {
//                 std::mem::swap(child.inner.as_mut(), &mut value);
//                 Some(value)
//             }
//         } else {
//             current.children.push(TrieInner::leaf(value).into());
//             None
//         }
//     }
//     fn split(&mut self, at: u32) {
//         let previous_length = self.inner.tag as usize;
//         let mut child = Trie {
//             inner: TrieInner::from_bytes(&self.inner.inner.bytes[at as usize..previous_length]).0,
//             children: Vec::with_capacity(2),
//         };
//         std::mem::swap(&mut self.children, &mut child.children);
//         self.inner.tag = at as u8;
//         self.children.push(child);
//     }
//     /// Returns `true` if a value exists for `key`. If `GREEDY == false`, will only return `true` if an exact match exist (wildcards are treated as litteral strings).
//     pub fn contains<P: AsRef<[u8]>, const GREEDY: bool>(&self, key: P) -> bool {
//         if GREEDY {
//             return self.get_all(key).next().is_some();
//         }
//         let key = key.as_ref();
//         match self.inner.match_bytes::<false>(key) {
//             Match::Leaf => true,
//             Match::Full(len)
//             | Match::Star { exact: true, len }
//             | Match::DouleStar { exact: true, len } => {
//                 let key = &key[len as usize..];
//                 self.children
//                     .iter()
//                     .any(|c| c.contains::<&[u8], GREEDY>(key))
//             }
//             Match::Mismatch
//             | Match::NullPrefix
//             | Match::Partial(_)
//             | Match::Star { .. }
//             | Match::DouleStar { .. } => false,
//         }
//     }
//     /// Returns a reference to the matching item if the `key` mathes exactly (wildcards are treaded as litteral strings).
//     pub fn get<P: AsRef<[u8]>>(&self, key: P) -> Option<&T> {
//         let key = key.as_ref();
//         match self.inner.match_bytes::<false>(key) {
//             Match::Leaf => Some(self.inner.as_ref()),
//             Match::Full(len)
//             | Match::Star { exact: true, len }
//             | Match::DouleStar { exact: true, len } => {
//                 let key = &key[len as usize..];
//                 self.children.iter().find_map(|c| c.get(key))
//             }
//             Match::Mismatch
//             | Match::NullPrefix
//             | Match::Partial(_)
//             | Match::Star { .. }
//             | Match::DouleStar { .. } => None,
//         }
//     }
//     /// Returns an iterator over the key-value pairs that match the provided `key`.
//     ///
//     /// The iterated keys are only valid until the next call to [`Iterator::next`], but the value references are valid throughout `'a`.
//     pub fn get_all<P: AsRef<[u8]>>(&self, _key: P) -> GreedyIter<T> {
//         todo!()
//     }
//     /// Returns a mutable reference to the matching item if the `key` mathes exactly (wildcards are treaded as litteral strings).
//     pub fn get_mut<P: AsRef<[u8]>>(&mut self, key: P) -> Option<&mut T> {
//         let key = key.as_ref();
//         match self.inner.match_bytes::<false>(key) {
//             Match::Leaf => Some(self.inner.as_mut()),
//             Match::Full(len)
//             | Match::Star { exact: true, len }
//             | Match::DouleStar { exact: true, len } => {
//                 let key = &key[len as usize..];
//                 self.children.iter_mut().find_map(|c| c.get_mut(key))
//             }
//             Match::Mismatch
//             | Match::NullPrefix
//             | Match::Partial(_)
//             | Match::Star { .. }
//             | Match::DouleStar { .. } => None,
//         }
//     }
//     /// Returns a mutable iterator over the key-value pairs that match the provided `key`.
//     ///
//     /// The iterated keys are only valid until the next call to [`Iterator::next`], but the mutable value references are valid throughout `'a`.
//     pub fn get_all_mut<P: AsRef<[u8]>>(&mut self, _key: P) -> GreedyIterMut<T> {
//         todo!()
//     }
//     /// Removes an entry, setting a gravestone in its place, using exact matching (wildcards are treaded as litteral strings).
//     ///
//     /// This remove is faster than [`Self::clean_remove`], but requires global cleanup steps to be taken in the future, such as [`Self::prune`]-ing after a few removes.
//     pub fn remove<P: AsRef<[u8]>>(&mut self, key: P) -> Option<T> {
//         let key = key.as_ref();
//         match self.inner.match_bytes::<false>(key) {
//             Match::Leaf => {
//                 self.inner.tag = TrieInner::<T>::DELETED;
//                 Some(unsafe { std::ptr::read(std::mem::transmute(&mut self.inner.inner.bytes)) })
//             }
//             Match::Full(len)
//             | Match::Star { exact: true, len }
//             | Match::DouleStar { exact: true, len } => {
//                 let key = &key[len as usize..];
//                 self.children.iter_mut().find_map(|c| c.remove(key))
//             }
//             Match::Mismatch
//             | Match::NullPrefix
//             | Match::Partial(_)
//             | Match::Star { .. }
//             | Match::DouleStar { .. } => None,
//         }
//     }
//     /// __Lazily__ removes all entries matching the provided keys.
//     ///
//     /// The iterator will yield each removed value. Upon dropping, the rest of the iterator will be iterated upon to ensure complete removal.
//     /// This remover doesn't do any pruning of the trie. A good use pattern could be
//     /// ```rust
//     /// # use zenoh_util::collections::tries::Trie
//     /// # let mut trie = Trie::new();
//     /// if trie.remove_all("/demo/keys/**").count() > 10 {trie.prune()}
//     /// ```
//     pub fn remove_all<P: AsRef<[u8]>>(&mut self, _key: P) -> GreedyRemover<T> {
//         todo!()
//     }
//     /// Removes an entry and applies local pruning to reoptimize the trie, using exact matching (wildcards are treaded as litteral strings).
//     ///
//     /// This operation is slower than [`Self::remove`], but keeps the trie optimized for less costs than [`Self::prune`].
//     pub fn clean_remove<P: AsRef<[u8]>>(&mut self, key: P) -> Option<T> {
//         self.do_clean_remove(key.as_ref()).map(|f| f.0)
//     }
//     fn do_clean_remove(&mut self, key: &[u8]) -> Option<(T, bool)> {
//         match self.inner.match_bytes::<false>(key) {
//             Match::Leaf => {
//                 self.inner.tag = TrieInner::<T>::DELETED;
//                 Some((
//                     unsafe { std::ptr::read(std::mem::transmute(&mut self.inner.inner.bytes)) },
//                     true,
//                 ))
//             }
//             Match::Full(len)
//             | Match::Star { exact: true, len }
//             | Match::DouleStar { exact: true, len } => {
//                 let key = &key[len as usize..];
//                 for i in 0..self.children.len() {
//                     if let Some((value, delete)) =
//                         unsafe { self.children.get_unchecked_mut(i).do_clean_remove(key) }
//                     {
//                         if delete {
//                             self.children.swap_remove(i);
//                             if self.children.is_empty() {
//                                 return Some((value, true));
//                             } else {
//                                 self.join_child();
//                                 return Some((value, false));
//                             }
//                         }
//                         return Some((value, false));
//                     }
//                 }
//                 None
//             }
//             Match::Mismatch
//             | Match::NullPrefix
//             | Match::Partial(_)
//             | Match::Star { .. }
//             | Match::DouleStar { .. } => None,
//         }
//     }
//     /// Will prune unused branches of the tree, and try to unite string segments that may have been split previously.
//     pub fn prune(&mut self) {
//         let mut marked = Vec::new();
//         for (i, child) in &mut self.children.iter_mut().enumerate() {
//             if child.do_prune() {
//                 marked.push(i);
//             }
//         }
//         for i in marked.into_iter().rev() {
//             self.children.swap_remove(i);
//         }
//         self.join_child();
//     }
//     /// Merges node with its single child if possible.
//     fn join_child(&mut self) {
//         if self.inner.is_str()
//             && self.children.len() == 1
//             && self.children[0].inner.is_str()
//             && (self.inner.tag + self.children[0].inner.tag) as usize <= INNER_SIZE
//         {
//             let Trie {
//                 inner: child,
//                 children,
//             } = match self.children.pop() {
//                 Some(val) => val,
//                 None => unsafe { std::hint::unreachable_unchecked() },
//             };
//             unsafe {
//                 std::ptr::copy_nonoverlapping(
//                     &child.inner.bytes as *const u8,
//                     &mut self.inner.inner.bytes[self.inner.tag as usize],
//                     child.tag as usize,
//                 )
//             };
//             self.inner.tag += child.tag;
//             self.children = children;
//         }
//     }
//     fn do_prune(&mut self) -> bool {
//         match self.inner.tag {
//             TrieInner::<T>::DELETED => true,
//             TrieInner::<T>::LEAF => false,
//             _ => {
//                 self.prune();
//                 self.children.is_empty()
//             }
//         }
//     }
//     /// Iterates over the values present in the trie.
//     pub fn values(&self) -> Values<T> {
//         self.into()
//     }
//     /// Iterates mutably over the values present in the trie.
//     pub fn values_mut(&mut self) -> ValuesMut<T> {
//         self.into()
//     }
//     /// Iterates over the key-value pairs present in the trie.
//     ///
//     /// The key part of the item only lives as long as [`Iterator::next`] isn't called.
//     pub fn iter(&self) -> Iter<T> {
//         self.into_iter()
//     }
//     /// Iterates mutably over the key-value pairs present in the trie.
//     ///
//     /// The key part of the item only lives as long as [`Iterator::next`] isn't called.
//     pub fn iter_mut(&mut self) -> IterMut<T> {
//         self.into_iter()
//     }
//     /// Returns the number of entries marked as deleted within the trie.
//     pub fn count_gravestones(&self) -> usize {
//         self.children.iter().fold(
//             (self.inner.tag == TrieInner::<T>::DELETED) as usize,
//             |acc, it| acc + it.count_gravestones(),
//         )
//     }
// }
// pub struct Values<'a, T> {
//     iters: Vec<std::slice::Iter<'a, Trie<T>>>,
// }
// impl<'a, T> From<&'a Trie<T>> for Values<'a, T> {
//     fn from(trie: &'a Trie<T>) -> Self {
//         if trie.is_text() {
//             Values {
//                 iters: vec![trie.as_text().children.iter()],
//             }
//         } else {
//             Values {
//                 iters: vec![std::slice::from_ref(trie).iter()],
//             }
//         }
//     }
// }
// impl<'a, T> Iterator for Values<'a, T> {
//     type Item = &'a T;
//     fn next(&mut self) -> Option<Self::Item> {
//         loop {
//             let iter = self.iters.last_mut()?;
//             if let Some(node) = iter.next() {
//                 if node.is_text() {
//                     let node = node.as_text();
//                     if !node.children.is_empty() {
//                         self.iters.push(node.children.iter())
//                     }
//                 } else {
//                     let leaf = node.as_leaf();
//                     if leaf.is_live() {
//                         return Some(leaf.as_ref());
//                     }
//                 }
//             } else {
//                 self.iters.pop();
//             }
//         }
//     }
// }
// pub struct ValuesMut<'a, T> {
//     iters: Vec<std::slice::IterMut<'a, Trie<T>>>,
// }
// impl<'a, T> From<&'a mut Trie<T>> for ValuesMut<'a, T> {
//     fn from(trie: &'a mut Trie<T>) -> Self {
//         if trie.is_text() {
//             ValuesMut {
//                 iters: vec![trie.as_text_mut().children.iter_mut()],
//             }
//         } else {
//             ValuesMut {
//                 iters: vec![std::slice::from_mut(trie).iter_mut()],
//             }
//         }
//     }
// }
// impl<'a, T> Iterator for ValuesMut<'a, T> {
//     type Item = &'a mut T;
//     fn next(&mut self) -> Option<Self::Item> {
//         loop {
//             let iter = self.iters.last_mut()?;
//             if let Some(node) = iter.next() {
//                 if node.is_text() {
//                     let node = node.as_text_mut();
//                     if !node.children.is_empty() {
//                         self.iters.push(node.children.iter_mut())
//                     }
//                 } else {
//                     let leaf = node.as_leaf_mut();
//                     if leaf.is_live() {
//                         return Some(leaf.as_mut());
//                     }
//                 }
//             } else {
//                 self.iters.pop();
//             }
//         }
//     }
// }
pub struct Iter<'a, T, Matcher, const N: usize> {
    key: Vec<u8>,
    iters: Vec<(usize, std::slice::Iter<'a, GenericTrie<T, Matcher, { N }>>)>,
}
impl<'a, T, Matcher, const N: usize> IntoIterator for &'a GenericTrie<T, Matcher, { N }> {
    type Item = (&'a [u8], &'a T);
    type IntoIter = Iter<'a, T, Matcher, { N }>;
    fn into_iter(self) -> Self::IntoIter {
        let key = self.as_ref();
        Iter {
            key: key.into(),
            iters: vec![(0, self.children.iter())],
        }
    }
}
impl<'a, T, Matcher, const N: usize> Iterator for Iter<'a, T, Matcher, { N }> {
    type Item = (&'a [u8], &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, iter) = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                let len = self.key.len();
                self.key.extend(node.as_ref());
                self.iters.push((len, node.children.iter()));
                if node.holds_value() {
                    return Some(unsafe {
                        (
                            std::slice::from_raw_parts(self.key.as_ptr(), self.key.len()),
                            node.value.assume_init_ref(),
                        )
                    });
                }
            } else if let Some((len, _)) = self.iters.pop() {
                unsafe { self.key.set_len(len) }
            }
        }
    }
}
// pub struct IterMut<'a, T> {
//     key: Vec<u8>,
//     iters: Vec<(usize, std::slice::IterMut<'a, Trie<T>>)>,
// }
// impl<'a, T> IntoIterator for &'a mut Trie<T> {
//     type Item = (&'a [u8], &'a mut T);
//     type IntoIter = IterMut<'a, T>;
//     fn into_iter(self) -> Self::IntoIter {
//         let key = self.inner.as_bytes();
//         IterMut {
//             key: key.into(),
//             iters: vec![(0, self.children.iter_mut())],
//         }
//     }
// }
// impl<'a, T> Iterator for IterMut<'a, T> {
//     type Item = (&'a [u8], &'a mut T);
//     fn next(&mut self) -> Option<Self::Item> {
//         loop {
//             let (_, iter) = self.iters.last_mut()?;
//             if let Some(node) = iter.next() {
//                 match node.inner.tag {
//                     TrieInner::<T>::DELETED => {}
//                     TrieInner::<T>::LEAF => {
//                         return Some((
//                             unsafe {
//                                 std::slice::from_raw_parts(self.key.as_ptr(), self.key.len())
//                             },
//                             node.inner.as_mut(),
//                         ))
//                     }
//                     _ => {
//                         let new_len = self.key.len();
//                         self.key.extend(node.inner.as_bytes());
//                         self.iters.push((new_len, node.children.iter_mut()))
//                     }
//                 }
//             } else if let Some((len, _)) = self.iters.pop() {
//                 unsafe { self.key.set_len(len) }
//             }
//         }
//     }
// }

// struct NoDoublestarIter<'a, T> {
//     requested_key: &'a [u8],
//     current_key: Vec<u8>,
//     iters: Vec<(usize, usize, std::slice::Iter<'a, Trie<T>>)>,
// }

// enum GreedyIterEnum<'a, T> {
//     NoDoublestar(NoDoublestarIter<'a, T>),
// }

// pub struct GreedyIter<'a, T> {
//     inner: GreedyIterEnum<'a, T>,
// }
// impl<'a, T> Iterator for GreedyIter<'a, T> {
//     type Item = (&'a [u8], &'a T);
//     fn next(&mut self) -> Option<Self::Item> {
//         loop {}
//     }
// }
// struct GreedyMutator<'a, T> {
//     todo: std::marker::PhantomData<&'a T>,
// }
// impl<'a, T> Iterator for GreedyMutator<'a, T> {
//     type Item = (&'a [u8], &'a mut Trie<T>);
//     fn next(&mut self) -> Option<Self::Item> {
//         todo!()
//     }
// }
// pub struct GreedyIterMut<'a, T> {
//     mutator: GreedyMutator<'a, T>,
// }
// impl<'a, T> Iterator for GreedyIterMut<'a, T> {
//     type Item = (&'a [u8], &'a mut T);
//     fn next(&mut self) -> Option<Self::Item> {
//         self.mutator
//             .next()
//             .map(|(key, trie)| (key, trie.inner.as_mut()))
//     }
// }
// pub struct GreedyRemover<'a, T> {
//     mutator: GreedyMutator<'a, T>,
// }
// impl<'a, T> Iterator for GreedyRemover<'a, T> {
//     type Item = (&'a [u8], T);
//     fn next(&mut self) -> Option<Self::Item> {
//         for (key, trie) in &mut self.mutator {
//             if let Some(value) = trie.inner.take() {
//                 return Some((key, value));
//             }
//         }
//         None
//     }
// }
// impl<'a, T> Drop for GreedyRemover<'a, T> {
//     fn drop(&mut self) {
//         for _ in self {}
//     }
// }

impl<T: Sized + std::fmt::Debug, Tokenizer, const N: usize> std::fmt::Display
    for GenericTrie<T, Tokenizer, { N }>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "-{}", self.as_str())?;
        let mut stack = Vec::new();
        let mut prefix = "|".to_string();
        let mut iter = self.children.iter();
        loop {
            match iter.next() {
                Some(next) => {
                    writeln!(f, "{}-{}", prefix, next.as_str())?;
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

// struct TrieInnerUnion<T> {
//     bytes: [u8; INNER_SIZE],
//     _marker: std::marker::PhantomData<T>,
// }
// impl<T> TrieInnerUnion<T> {
//     const fn new() -> Self {
//         TrieInnerUnion {
//             bytes: [0; INNER_SIZE],
//             _marker: std::marker::PhantomData,
//         }
//     }
// }
// impl<T> Default for TrieInnerUnion<T> {
//     fn default() -> Self {
//         Self::new()
//     }
// }
// impl<T> From<T> for TrieInnerUnion<T> {
//     fn from(leaf: T) -> Self {
//         let mut this = Self::default();
//         unsafe { std::ptr::write(std::mem::transmute(&mut this.bytes), leaf) }
//         this
//     }
// }
// #[repr(C)]
// struct TrieInner<T> {
//     inner: TrieInnerUnion<T>,
//     tag: u8,
// }
// impl<T: std::fmt::Debug> std::fmt::Display for TrieInner<T> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self.tag {
//             Self::LEAF => write!(f, "{:?}", self.as_ref()),
//             Self::SLASH => f.write_char('/'),
//             Self::STAR => f.write_str("*/"),
//             Self::DOUBLE_STAR => f.write_str("**/"),
//             Self::DELETED => f.write_str("DELETED ENTRY"),
//             len => f.write_str(unsafe {
//                 std::str::from_utf8_unchecked(&self.inner.bytes[..len as usize])
//             }),
//         }
//     }
// }

// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// enum Match {
//     Mismatch,
//     NullPrefix,
//     Leaf,
//     Partial(u32),
//     Full(u32),
//     Star { exact: bool, len: u32 },
//     DouleStar { exact: bool, len: u32 },
// }
// const INNER_SIZE: usize = 64 - std::mem::size_of::<Vec<Trie<()>>>() - std::mem::size_of::<u8>();
// #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
// enum InsertionMatch {
//     Mismatch,
//     Full,
//     SplitNode(u32),
//     SplitToken(u32),
//     SplitBoth(u32),
// }
// impl<T> TrieInner<T> {
//     fn from_bytes(s: &[u8]) -> (Self, &[u8]) {
//         if s.starts_with(b"/") {
//             return (Self::slash(), &s[1..]);
//         }
//         if s.starts_with(b"*/") {
//             return (Self::star(), &s[2..]);
//         }
//         if s.starts_with(b"**/") {
//             return (Self::double_star(), &s[3..]);
//         }
//         let split = s
//             .iter()
//             .position(|&c| c == b'/')
//             .unwrap_or_else(|| s.len())
//             .min(INNER_SIZE);
//         let mut result = TrieInner {
//             inner: TrieInnerUnion::default(),
//             tag: split as u8,
//         };
//         let (left, right) = s.split_at(split);
//         result.inner.bytes[..split].clone_from_slice(left);
//         (result, right)
//     }
//     fn insert_match(&self, rhs: Token) -> InsertionMatch {
//         if self.tag == 0 {
//             return InsertionMatch::SplitToken(0);
//         }
//         if rhs.is_str() {
//             return self.insert_match_bytes(rhs.as_bytes());
//         }
//         match (self.tag, rhs) {
//             (Self::SLASH, Token::SLASH)
//             | (Self::STAR, Token::STAR)
//             | (Self::DOUBLE_STAR, Token::DOUBLE_STAR) => InsertionMatch::Full,
//             _ => InsertionMatch::Mismatch,
//         }
//     }
//     fn insert_match_bytes(&self, rhs: &[u8]) -> InsertionMatch {
//         if !self.is_str() {
//             return InsertionMatch::Mismatch;
//         }
//         let lhs = self.as_bytes();
//         if lhs[0] != rhs[0] {
//             return InsertionMatch::Mismatch;
//         }
//         let llen = lhs.len();
//         let rlen = rhs.len();
//         for i in 1..(llen.min(rlen)) {
//             if lhs[i] != rhs[i] {
//                 return InsertionMatch::SplitBoth(i as u32);
//             }
//         }
//         match llen.cmp(&rlen) {
//             std::cmp::Ordering::Less => InsertionMatch::SplitNode(llen as u32),
//             std::cmp::Ordering::Equal => InsertionMatch::Full,
//             std::cmp::Ordering::Greater => InsertionMatch::SplitToken(rlen as u32),
//         }
//     }
//     fn match_bytes<const ALLOW_PARTIAL: bool>(&self, rhs: &[u8]) -> Match {
//         if rhs.is_empty() {
//             return if self.tag == Self::LEAF {
//                 Match::Leaf
//             } else {
//                 Match::NullPrefix
//             };
//         }
//         match self.tag {
//             0 => Match::Full(0),
//             Self::LEAF | Self::DELETED => Match::Mismatch,
//             Self::SLASH => {
//                 if rhs[0] == b'/' {
//                     Match::Full(1)
//                 } else {
//                     Match::Mismatch
//                 }
//             }
//             Self::STAR => {
//                 if rhs.starts_with(b"*/") {
//                     Match::Star {
//                         exact: true,
//                         len: 2,
//                     }
//                 } else {
//                     Match::Star {
//                         exact: false,
//                         len: rhs
//                             .iter()
//                             .position(|&c| c == b'/')
//                             .map(|f| f + 1)
//                             .unwrap_or_else(|| rhs.len()) as u32,
//                     }
//                 }
//             }
//             Self::DOUBLE_STAR => {
//                 if rhs.starts_with(b"**/") {
//                     Match::DouleStar {
//                         exact: true,
//                         len: 3,
//                     }
//                 } else {
//                     Match::DouleStar {
//                         exact: false,
//                         len: 0,
//                     }
//                 }
//             }
//             length if ALLOW_PARTIAL && self.inner.bytes[0] == rhs[0] => {
//                 for (i, (a, b)) in self.inner.bytes[1..length as usize]
//                     .iter()
//                     .zip(&rhs[1..])
//                     .enumerate()
//                 {
//                     if a != b {
//                         return Match::Partial((i + 1) as u32);
//                     }
//                 }
//                 Match::Full((length as u32).min(rhs.len() as u32))
//             }
//             length if !ALLOW_PARTIAL && rhs.starts_with(&self.inner.bytes[..length as usize]) => {
//                 Match::Full(length as u32)
//             }
//             _ => Match::Mismatch,
//         }
//     }
//     const LEAF: u8 = INNER_SIZE as u8 + 1;
//     const SLASH: u8 = INNER_SIZE as u8 + 2;
//     const STAR: u8 = INNER_SIZE as u8 + 3;
//     const DOUBLE_STAR: u8 = INNER_SIZE as u8 + 4;
//     const DELETED: u8 = INNER_SIZE as u8 + 5;
//     fn is_str(&self) -> bool {
//         self.tag <= INNER_SIZE as u8
//     }
//     fn leaf(value: T) -> Self {
//         TrieInner {
//             inner: TrieInnerUnion::from(value),
//             tag: Self::LEAF,
//         }
//     }
//     const fn slash() -> Self {
//         TrieInner {
//             inner: TrieInnerUnion::new(),
//             tag: Self::SLASH,
//         }
//     }
//     const fn star() -> Self {
//         TrieInner {
//             inner: TrieInnerUnion::new(),
//             tag: Self::STAR,
//         }
//     }
//     const fn double_star() -> Self {
//         TrieInner {
//             inner: TrieInnerUnion::new(),
//             tag: Self::DOUBLE_STAR,
//         }
//     }
//     fn take(&mut self) -> Option<T> {
//         if self.tag == Self::LEAF {
//             self.tag = Self::DELETED;
//             Some(unsafe { std::ptr::read(std::mem::transmute(&mut self.inner.bytes)) })
//         } else {
//             None
//         }
//     }
//     fn as_bytes(&self) -> &[u8] {
//         match self.tag {
//             Self::DELETED | Self::LEAF => b"",
//             Self::SLASH => b"/",
//             Self::STAR => b"*/",
//             Self::DOUBLE_STAR => b"**/",
//             tag => &self.inner.bytes[..tag as usize],
//         }
//     }
// }
// impl<T> AsRef<T> for TrieInner<T> {
//     fn as_ref(&self) -> &T {
//         unsafe { std::mem::transmute(&self.inner.bytes) }
//     }
// }
// impl<T> AsMut<T> for TrieInner<T> {
//     fn as_mut(&mut self) -> &mut T {
//         unsafe { std::mem::transmute(&mut self.inner.bytes) }
//     }
// }
// impl<T> Drop for TrieInner<T> {
//     fn drop(&mut self) {
//         std::mem::drop(self.take());
//     }
// }
// impl<'a, T> From<&'a Token<'a>> for (TrieInner<T>, &'a [u8]) {
//     fn from(token: &'a Token<'a>) -> Self {
//         match *token {
//             Token::SLASH => (TrieInner::<T>::slash(), b""),
//             Token::STAR => (TrieInner::<T>::star(), b""),
//             Token::DOUBLE_STAR => (TrieInner::<T>::double_star(), b""),
//             _ => TrieInner::<T>::from_bytes(token.as_bytes()),
//         }
//     }
// }

/// Ensures that Trie<T> construction panics if T is too large for Trie<T> to keep fitting.
#[test]
#[should_panic]
fn trie_panic_if_too_large() {
    let impossible_trie: Trie<[u64; 16], 10> = Trie::new();
    assert_ne!(std::mem::size_of_val(&impossible_trie), 64);
}

#[test]
fn trie_general() {
    type STrie = Trie<String, 15>;
    let mut trie: STrie = Trie::new();
    const _: [u64; 8] = unsafe { std::mem::transmute(std::mem::transmute::<_, STrie>([0u64; 8])) };
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
    println!("{:?}", &trie);
    println!("{}", &trie);
    for (key, value) in &trie {
        let key = std::str::from_utf8(key).unwrap();
        assert!(keys.contains(&dbg!(key)));
        assert_eq!(&key.to_uppercase(), value);
    }
    for key in keys {
        assert_eq!(trie.get(key).unwrap(), &key.to_uppercase());
    }
    // for key in keys.iter().filter(|p| !p.starts_with("/panda")) {
    //     assert_eq!(trie.remove(key).unwrap(), key.to_uppercase());
    // }
    // assert_eq!(trie.count_gravestones(), 2);
    // trie.prune();
    // assert_eq!(trie.count_gravestones(), 0);
    // trie.insert("/panda", "PANDA".into()).unwrap();
    // assert_eq!(trie.get("/panda").unwrap(), "PANDA");
}

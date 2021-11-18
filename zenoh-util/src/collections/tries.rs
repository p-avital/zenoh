use std::{hint::unreachable_unchecked, mem::MaybeUninit};

pub type ZTrie<T, const N: usize> = Trie<T, DefaultMatcher, { N }>;
#[repr(C)]
pub struct Trie<T, Matcher, const N: usize> {
    children: Vec<Self>,
    value: MaybeUninit<T>,
    buffer: [u8; N],
    tag: u8,
    tokenizer_marker: std::marker::PhantomData<Matcher>,
}
impl<T: std::fmt::Debug, Matcher, const N: usize> std::fmt::Debug for Trie<T, Matcher, { N }> {
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

impl<T, Matcher, const N: usize> Trie<T, Matcher, { N }> {
    pub fn with_capacity(capacity: usize) -> Self {
        let ideal_n: isize =
            63 - std::mem::size_of::<Vec<()>>() as isize - std::mem::size_of::<T>() as isize;
        if ideal_n <= 0 {
            panic!("{} cannot fit in a single cache-line no matter how small you make N, leading to terrible perfs, please box the value-type", std::any::type_name::<Self>())
        } else if N != ideal_n as usize {
            panic!(
                "{} should use const N = {} instead",
                std::any::type_name::<Self>(),
                ideal_n
            )
        }
        let buffer = unsafe {
            let mut buffer: MaybeUninit<[u8; N]> = MaybeUninit::uninit();
            buffer.assume_init_mut()[0] = b'/';
            buffer.assume_init()
        };
        Trie {
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
            InsertionMatch::Full => key = b"",
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
            let child = Trie {
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
        self.get_subtrie(key.as_ref())
            .map(Self::get_value)
            .flatten()
    }
    pub fn get_mut<K: AsRef<[u8]>>(&mut self, key: K) -> Option<&mut T> {
        self.get_subtrie_mut(key.as_ref())
            .map(Self::get_value_mut)
            .flatten()
    }
    pub fn fold<K: AsRef<[u8]>, Acc, F: Fn(Acc, &T) -> Acc>(
        &self,
        key: K,
        acc: Acc,
        reducer: F,
    ) -> Acc {
        self.inner_fold(key.as_ref(), acc, |acc, node| {
            if let Some(value) = node.get_value() {
                reducer(acc, value)
            } else {
                acc
            }
        })
    }
    pub fn remove<K: AsRef<[u8]>>(&mut self, key: K) -> Option<T> {
        self.get_subtrie_mut(key.as_ref())
            .map(Self::remove_value)
            .flatten()
    }
}
impl<T, Tokenizer, const N: usize> Trie<T, Tokenizer, { N }> {
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
            Trie {
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
    fn get_subtrie(&self, mut key: &[u8]) -> Option<&Self> {
        if !key.starts_with(self.as_ref()) {
            return None;
        }
        key = &trim_slashes(key)[self.len() as usize..];
        let mut node = self;
        'traversal: while !key.is_empty() {
            for child in node.children.iter() {
                if key.starts_with(child.as_ref()) {
                    key = &key[child.len() as usize..];
                    node = child;
                    continue 'traversal;
                }
            }
            return None;
        }
        Some(node)
    }
    fn get_subtrie_mut(&mut self, mut key: &[u8]) -> Option<&mut Self> {
        if !key.starts_with(self.as_ref()) {
            return None;
        }
        key = &trim_slashes(key)[self.len() as usize..];
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
        Some(node)
    }
    fn inner_fold<Acc, F: Fn(Acc, &Self) -> Acc>(
        &self,
        mut key: &[u8],
        mut acc: Acc,
        execute: F,
    ) -> Acc {
        if !key.starts_with(self.as_ref()) {
            return acc;
        }
        key = &trim_slashes(key)[self.len() as usize..];
        let mut node = self;
        acc = execute(acc, node);
        'traversal: while !key.is_empty() {
            for child in node.children.iter() {
                if key.starts_with(child.as_ref()) {
                    key = &key[child.len() as usize..];
                    node = child;
                    acc = execute(acc, node);
                    continue 'traversal;
                }
            }
            break;
        }
        acc
    }
    fn inner_fold_mut<Acc, F: Fn(Acc, &mut Self) -> Acc>(
        &mut self,
        mut key: &[u8],
        mut acc: Acc,
        execute: F,
    ) -> Acc {
        if !key.starts_with(self.as_ref()) {
            return acc;
        }
        key = &trim_slashes(key)[self.len() as usize..];
        let mut node = self;
        acc = execute(acc, node);
        'traversal: while !key.is_empty() {
            for child in node.children.iter_mut() {
                if key.starts_with(child.as_ref()) {
                    key = &key[child.len() as usize..];
                    node = unsafe { std::mem::transmute(child) };
                    acc = execute(acc, node);
                    continue 'traversal;
                }
            }
            break;
        }
        acc
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

impl<T, Tokenizer, const N: usize> Default for Trie<T, Tokenizer, { N }> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Tokenizer, const N: usize> Drop for Trie<T, Tokenizer, { N }> {
    fn drop(&mut self) {
        if self.holds_value() {
            unsafe { std::ptr::drop_in_place(self.value.assume_init_mut()) }
        }
    }
}
impl<T, Tokenizer, const N: usize> AsRef<[u8]> for Trie<T, Tokenizer, { N }> {
    fn as_ref(&self) -> &[u8] {
        &self.buffer[..self.len() as usize]
    }
}

pub struct Values<'a, T, Matcher, const N: usize> {
    iters: Vec<std::slice::Iter<'a, Trie<T, Matcher, { N }>>>,
}
impl<'a, T, Matcher, const N: usize> Iterator for Values<'a, T, Matcher, { N }> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let iter = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                self.iters.push(node.children.iter());
                if node.holds_value() {
                    return Some(unsafe { node.value.assume_init_ref() });
                }
            } else {
                self.iters.pop();
            }
        }
    }
}

pub struct ValuesMut<'a, T, Matcher, const N: usize> {
    iters: Vec<std::slice::IterMut<'a, Trie<T, Matcher, { N }>>>,
}
impl<'a, T, Matcher, const N: usize> Iterator for ValuesMut<'a, T, Matcher, { N }> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let iter = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                self.iters
                    .push(unsafe { std::mem::transmute(node.children.iter_mut()) });
                if node.holds_value() {
                    return Some(unsafe { node.value.assume_init_mut() });
                }
            } else {
                self.iters.pop();
            }
        }
    }
}

pub struct Iter<'a, T, Matcher, const N: usize> {
    key: Vec<u8>,
    iters: Vec<(usize, std::slice::Iter<'a, Trie<T, Matcher, { N }>>)>,
}
impl<'a, T, Matcher, const N: usize> IntoIterator for &'a Trie<T, Matcher, { N }> {
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

pub struct IterMut<'a, T, Matcher, const N: usize> {
    key: Vec<u8>,
    iters: Vec<(usize, std::slice::IterMut<'a, Trie<T, Matcher, { N }>>)>,
}
impl<'a, T, Matcher, const N: usize> IntoIterator for &'a mut Trie<T, Matcher, { N }> {
    type Item = (&'a [u8], &'a mut T);
    type IntoIter = IterMut<'a, T, Matcher, { N }>;
    fn into_iter(self) -> Self::IntoIter {
        let key = self.as_ref();
        IterMut {
            key: key.into(),
            iters: vec![(0, self.children.iter_mut())],
        }
    }
}
impl<'a, T, Matcher, const N: usize> Iterator for IterMut<'a, T, Matcher, { N }> {
    type Item = (&'a [u8], &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (_, iter) = self.iters.last_mut()?;
            if let Some(node) = iter.next() {
                let len = self.key.len();
                self.key.extend(node.as_ref());
                self.iters.push((len, unsafe {
                    std::mem::transmute(node.children.iter_mut())
                }));
                if node.holds_value() {
                    return Some(unsafe {
                        (
                            std::slice::from_raw_parts(self.key.as_ptr(), self.key.len()),
                            node.value.assume_init_mut(),
                        )
                    });
                }
            } else if let Some((len, _)) = self.iters.pop() {
                unsafe { self.key.set_len(len) }
            }
        }
    }
}

// enum GreedyIterEnum<'a, T> {}
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
    for Trie<T, Tokenizer, { N }>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "-{}", self.as_str())?;
        let mut stack = Vec::new();
        let mut prefix = "|".to_string();
        let mut iter = self.children.iter();
        loop {
            match iter.next() {
                Some(next) => {
                    write!(f, "{}-{}", prefix, next.as_str())?;
                    if let Some(value) = next.get_value() {
                        writeln!(f, ": {:?}", value)?;
                    } else {
                        writeln!(f)?;
                    }
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

/// Ensures that Trie<T> construction panics if T is too large for Trie<T> to keep fitting.
#[test]
#[should_panic]
fn trie_panic_if_too_large() {
    let impossible_trie: ZTrie<[u64; 16], 10> = ZTrie::new();
    assert_ne!(std::mem::size_of_val(&impossible_trie), 64);
}

#[test]
fn trie_general() {
    type STrie = ZTrie<String, 15>;
    let mut trie: STrie = ZTrie::new();
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
    for key in keys.iter().filter(|p| !p.starts_with("/panda")) {
        assert_eq!(trie.remove(key).unwrap(), key.to_uppercase());
    }
    println!("{}", &trie);
    // assert_eq!(trie.count_gravestones(), 2);
    // trie.prune();
    // assert_eq!(trie.count_gravestones(), 0);
    // trie.insert("/panda", "PANDA".into()).unwrap();
    // assert_eq!(trie.get("/panda").unwrap(), "PANDA");
}

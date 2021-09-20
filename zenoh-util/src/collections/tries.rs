pub struct Node<T> {
    value: NodeInner<T>,
    children: Vec<Node<T>>,
}

const INNER_SIZE: usize = 64 - std::mem::size_of::<Vec<Node<()>>>() - std::mem::size_of::<u8>();
struct NodeInner<T> {
    inner: [u8; INNER_SIZE],
    tag: NodeInnerTag,
    _marker: std::marker::PhantomData<T>,
}
impl<T> NodeInner<T> {
    fn from_bytes(s: &[u8]) -> (Self, &[u8]) {
        let split = s.len().min(INNER_SIZE);
        let mut result = NodeInner {
            inner: [0; INNER_SIZE],
            tag: NodeInnerTag::new(split as u8),
            _marker: std::default::Default::default(),
        };
        let (left, right) = s.split_at(split);
        result.inner.clone_from_slice(left);
        (result, right)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
enum NodeInnerTag {
    Leaf = INNER_SIZE as u8,
    Slash = INNER_SIZE as u8 + 1,
    Star = INNER_SIZE as u8 + 2,
    DoubleStar = INNER_SIZE as u8 + 3,
}
impl NodeInnerTag {
    fn new(value: u8) -> Self {
        unsafe { std::mem::transmute(value) }
    }
}

#[test]
fn tries_test() {
    let marker = NodeInnerTag::new(22);
    match marker {
        NodeInnerTag::Leaf => todo!(),
        NodeInnerTag::Slash => todo!(),
        NodeInnerTag::Star => todo!(),
        NodeInnerTag::DoubleStar => todo!(),
        _ => println!("{}", marker as u8),
    }
    dbg!(INNER_SIZE);
}

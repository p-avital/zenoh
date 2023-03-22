//
// Copyright (c) 2022 ZettaScale Technology
//
// This program and the accompanying materials are made available under the
// terms of the Eclipse Public License 2.0 which is available at
// http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
// which is available at https://www.apache.org/licenses/LICENSE-2.0.
//
// SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
//
// Contributors:
//   ZettaScale Zenoh Team, <zenoh@zettascale.tech>
//
use crate::{core::Reliability, transport::TransportSn, zenoh::ZenohMessage};
use alloc::vec::Vec;

/// # Frame message
///
/// The [`Frame`] message is used to transmit one ore more complete serialized
/// [`crate::net::protocol::message::ZenohMessage`]. I.e., the total length of the
/// serialized [`crate::net::protocol::message::ZenohMessage`] (s) MUST be smaller
/// than the maximum batch size (i.e. 2^16-1) and the link MTU.
/// The [`Frame`] message is used as means to aggreate multiple
/// [`crate::net::protocol::message::ZenohMessage`] in a single atomic message that
/// goes on the wire. By doing so, many small messages can be batched together and
/// share common information like the sequence number.
///
/// The [`Frame`] message flow is the following:
///
/// ```text
///     A                   B
///     |      FRAME        |
///     |------------------>|
///     |                   |
/// ```
///
/// The [`Frame`] message structure is defined as follows:
///
/// ```text
/// Flags:
/// - R: Reliable       If R==1 it concerns the reliable channel, else the best-effort channel
/// - X: Reserved
/// - Z: Extensions     If Z==1 then zenoh extensions will follow.
///
///  7 6 5 4 3 2 1 0
/// +-+-+-+-+-+-+-+-+
/// |Z|X|R|  FRAME  |
/// +-+-+-+---------+
/// %    seq num    %
/// +---------------+
/// ~  [FrameExts]  ~ if Flag(Z)==1
/// +---------------+
/// ~  [NetworkMsg] ~
/// +---------------+
/// ```
///
/// NOTE: 16 bits (2 bytes) may be prepended to the serialized message indicating the total length
///       in bytes of the message, resulting in the maximum length of a message being 65535 bytes.
///       This is necessary in those stream-oriented transports (e.g., TCP) that do not preserve
///       the boundary of the serialized messages. The length is encoded as little-endian.
///       In any case, the length of a message must not exceed 65535 bytes.
///
pub mod flag {
    pub const R: u8 = 1 << 5; // 0x20 Reliable      if R==1 then the frame is reliable
                              // pub const X: u8 = 1 << 6; // 0x40       Reserved
    pub const Z: u8 = 1 << 7; // 0x80 Extensions    if Z==1 then an extension will follow
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub reliability: Reliability,
    pub sn: TransportSn,
    pub payload: Vec<ZenohMessage>,
    pub ext_qos: ext::QoSType,
}

// Extensions
pub mod ext {
    use crate::common::ZExtZ64;
    use crate::core::Priority;
    use crate::zextz64;

    pub type QoS = zextz64!(0x1, true);

    ///      7 6 5 4 3 2 1 0
    ///     +-+-+-+-+-+-+-+-+
    ///     |Z|0_1|   QoS   |
    ///     +-+-+-+---------+
    ///     %0|  rsv  |prio %
    ///     +---------------+
    ///
    ///     - prio: Priority class
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct QoSType {
        inner: u8,
    }

    impl QoSType {
        pub const P_MASK: u8 = 0b00000111;

        pub const fn new(priority: Priority) -> Self {
            Self {
                inner: priority as u8,
            }
        }

        pub const fn priority(&self) -> Priority {
            unsafe { core::mem::transmute(self.inner & Self::P_MASK) }
        }

        #[cfg(feature = "test")]
        pub fn rand() -> Self {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            let inner: u8 = rng.gen();
            Self { inner }
        }
    }

    impl Default for QoSType {
        fn default() -> Self {
            Self::new(Priority::default())
        }
    }

    impl From<QoS> for QoSType {
        fn from(ext: QoS) -> Self {
            Self {
                inner: ext.value as u8,
            }
        }
    }

    impl From<QoSType> for QoS {
        fn from(ext: QoSType) -> Self {
            QoS::new(ext.inner as u64)
        }
    }
}

impl Frame {
    #[cfg(feature = "test")]
    pub fn rand() -> Self {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        let reliability = Reliability::rand();
        let sn: TransportSn = rng.gen();
        let ext_qos = ext::QoSType::rand();
        let mut payload = vec![];
        for _ in 0..rng.gen_range(1..4) {
            let mut m = ZenohMessage::rand();
            m.channel.reliability = reliability;
            m.channel.priority = ext_qos.priority();
            payload.push(m);
        }

        Frame {
            reliability,
            sn,
            ext_qos,
            payload,
        }
    }
}

// FrameHeader
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub reliability: Reliability,
    pub sn: TransportSn,
    pub ext_qos: ext::QoSType,
}

impl FrameHeader {
    #[cfg(feature = "test")]
    pub fn rand() -> Self {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        let reliability = Reliability::rand();
        let sn: TransportSn = rng.gen();
        let ext_qos = ext::QoSType::rand();

        FrameHeader {
            reliability,
            sn,
            ext_qos,
        }
    }
}

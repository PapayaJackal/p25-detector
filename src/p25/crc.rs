//! P25 CRC variants.
//!
//! `crc16_p25` matches OP25 `crc16()` in `p25p1_fdma.cc` — a 16-bit CRC using
//! polynomial x^12 + x^5 + 1 with a 17-bit register (i.e. the CCITT polynomial
//! 0x1021) and an inverted output. Verifying a TSBK block: treat the whole
//! 96-bit block including CRC as input and expect the function to return 0.

pub fn crc16_p25(buf: &[u8]) -> u16 {
    let poly: u32 = (1 << 12) | (1 << 5) | (1 << 0);
    let mut crc: u32 = 0;
    for byte in buf {
        let bits = *byte;
        for j in 0..8 {
            let bit = ((bits >> (7 - j)) & 1) as u32;
            crc = ((crc << 1) | bit) & 0x1_FFFF;
            if crc & 0x1_0000 != 0 {
                crc = (crc & 0xFFFF) ^ poly;
            }
        }
    }
    (crc ^ 0xFFFF) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_vector_all_zeros() {
        // 10 zero bytes: pre-XOR register stays 0, xorout gives 0xFFFF.
        assert_eq!(crc16_p25(&[0u8; 10]), 0xFFFF);
    }

    #[test]
    fn self_checks_known_reference_block() {
        // 10 zero data bytes followed by CRC 0xFFFF 0xFFFF: after processing
        // the zeros the register is 0, processing 16 ones brings it to 0xFFFF,
        // and the final XOR against 0xFFFF gives zero. This is the invariant
        // OP25 relies on when it calls `crc16(block, 12) != 0` as a corruption
        // check.
        let mut buf = [0u8; 12];
        buf[10] = 0xFF;
        buf[11] = 0xFF;
        assert_eq!(crc16_p25(&buf), 0);
    }

    #[test]
    fn detects_bit_flip() {
        let mut buf = [0u8; 12];
        buf[10] = 0xFF;
        buf[11] = 0xFF;
        buf[3] ^= 1;
        assert_ne!(crc16_p25(&buf), 0);
    }
}

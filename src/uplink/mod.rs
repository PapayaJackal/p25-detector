pub mod dual;
pub mod single;

pub use crate::p25::Grant;

pub trait UplinkWatcher {
    /// Handle a grant. Returns `true` if the CC SDR was retuned during the
    /// call, signalling the caller to reset its decoder state — stale
    /// samples buffered in the SDR's URB queue will otherwise train up the
    /// downstream IIR/EMAs from the wrong frequency.
    fn on_grant(&mut self, grant: Grant) -> bool;
}

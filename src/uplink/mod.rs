pub mod dual;
pub mod single;

pub use crate::p25::Grant;

pub trait UplinkWatcher {
    fn on_grant(&mut self, grant: Grant);
}

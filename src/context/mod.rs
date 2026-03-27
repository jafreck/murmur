pub mod app_detector;
pub mod cursor;
pub mod provider;
pub mod system_state;

pub use app_detector::AppDetector;
pub use provider::{Context, ContextManager, ContextProvider, DictationMode};
pub use cursor::{CursorContext, SurroundingText};
pub use system_state::{ClipboardWatcher, RecentTextTracker};

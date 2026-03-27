pub mod app_detector;
pub mod cursor;
pub mod provider;
pub mod system_state;
pub mod title_analyzer;

pub use app_detector::AppDetector;
pub use cursor::{CursorContext, SurroundingText};
pub use provider::{Context, ContextManager, ContextProvider, DictationMode};
pub use system_state::{ClipboardWatcher, RecentTextTracker};
pub use title_analyzer::{analyze_title, TitleContext};

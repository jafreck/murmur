pub mod app_detector;
pub mod cursor;

pub use app_detector::AppDetector;
pub use cursor::{CursorContext, SurroundingText};
pub use murmur_core::context::{
    analyze_title, ClipboardWatcher, Context, ContextManager, ContextProvider, DictationMode,
    RecentTextTracker, TitleContext,
};

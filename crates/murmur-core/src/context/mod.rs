pub mod provider;
pub mod system_state;
pub mod title_analyzer;

pub use provider::{Context, ContextManager, ContextProvider, DictationMode};
pub use system_state::{ClipboardWatcher, RecentTextTracker};
pub use title_analyzer::{analyze_title, TitleContext};

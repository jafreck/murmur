//! Menu construction and action mapping for the tray UI.

use anyhow::Result;
use tray_icon::menu::{CheckMenuItem, MenuId, Submenu};

use crate::config::{self, InputMode};

use super::TrayAction;

/// Top languages shown in the menu.
pub const TOP_LANGUAGES: &[&str] = &[
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar", "hi", "pl",
    "sv", "tr", "uk", "vi", "th",
];

/// Build a radio-style label.
pub fn radio_label(name: &str, selected: bool) -> String {
    if selected {
        format!("● {name}")
    } else {
        format!("○ {name}")
    }
}

/// Grouped menu item IDs for constructing a MenuActionMap.
pub struct MenuActionIds {
    pub quit: MenuId,
    pub copy_last: MenuId,
    pub open_config: MenuId,
    pub reload_config: MenuId,
    pub spoken_punct: MenuId,
    pub filler_removal: MenuId,
    pub streaming: MenuId,
    pub translate: MenuId,
    pub noise_suppression: MenuId,
    pub set_hotkey: MenuId,
    pub app_mode: MenuId,
    pub check_for_updates: MenuId,
}

/// Menu event matching engine. Maps MenuId → TrayAction.
pub struct MenuActionMap {
    quit_id: MenuId,
    copy_last_id: MenuId,
    open_config_id: MenuId,
    reload_config_id: MenuId,
    spoken_punct_id: MenuId,
    filler_removal_id: MenuId,
    streaming_id: MenuId,
    translate_id: MenuId,
    noise_suppression_id: MenuId,
    set_hotkey_id: MenuId,
    app_mode_id: MenuId,
    check_for_updates_id: MenuId,

    model_ids: Vec<(MenuId, String)>,
    language_ids: Vec<(MenuId, String)>,
    mode_ids: Vec<(MenuId, InputMode)>,
    backend_ids: Vec<(MenuId, config::AsrBackend)>,
}

impl MenuActionMap {
    pub fn new(
        ids: MenuActionIds,
        model_ids: Vec<(MenuId, String)>,
        language_ids: Vec<(MenuId, String)>,
        mode_ids: Vec<(MenuId, InputMode)>,
        backend_ids: Vec<(MenuId, config::AsrBackend)>,
    ) -> Self {
        Self {
            quit_id: ids.quit,
            copy_last_id: ids.copy_last,
            open_config_id: ids.open_config,
            reload_config_id: ids.reload_config,
            spoken_punct_id: ids.spoken_punct,
            filler_removal_id: ids.filler_removal,
            streaming_id: ids.streaming,
            translate_id: ids.translate,
            noise_suppression_id: ids.noise_suppression,
            set_hotkey_id: ids.set_hotkey,
            app_mode_id: ids.app_mode,
            check_for_updates_id: ids.check_for_updates,

            model_ids,
            language_ids,
            mode_ids,
            backend_ids,
        }
    }

    pub fn match_event(&self, event_id: &MenuId) -> Option<TrayAction> {
        if event_id == &self.quit_id {
            return Some(TrayAction::Quit);
        }
        if event_id == &self.copy_last_id {
            return Some(TrayAction::CopyLastDictation);
        }
        if event_id == &self.open_config_id {
            return Some(TrayAction::OpenConfig);
        }
        if event_id == &self.reload_config_id {
            return Some(TrayAction::ReloadConfig);
        }
        if event_id == &self.spoken_punct_id {
            return Some(TrayAction::ToggleSpokenPunctuation);
        }
        if event_id == &self.filler_removal_id {
            return Some(TrayAction::ToggleFillerWordRemoval);
        }
        if event_id == &self.streaming_id {
            return Some(TrayAction::ToggleStreaming);
        }
        if event_id == &self.translate_id {
            return Some(TrayAction::ToggleTranslate);
        }
        if event_id == &self.noise_suppression_id {
            return Some(TrayAction::ToggleNoiseSuppression);
        }
        if event_id == &self.set_hotkey_id {
            return Some(TrayAction::SetHotkey);
        }
        if event_id == &self.app_mode_id {
            return Some(TrayAction::ToggleAppMode);
        }
        if event_id == &self.check_for_updates_id {
            return Some(TrayAction::CheckForUpdates);
        }

        for (id, size) in &self.model_ids {
            if event_id == id {
                return Some(TrayAction::SetModel(size.clone()));
            }
        }

        for (id, code) in &self.language_ids {
            if event_id == id {
                return Some(TrayAction::SetLanguage(code.clone()));
            }
        }

        for (id, mode) in &self.mode_ids {
            if event_id == id {
                return Some(TrayAction::SetMode(mode.clone()));
            }
        }

        for (id, backend) in &self.backend_ids {
            if event_id == id {
                return Some(TrayAction::SetBackend(*backend));
            }
        }

        None
    }
}

/// A radio-style menu entry that tracks its value alongside a CheckMenuItem.
pub(super) struct RadioEntry<T> {
    pub(super) value: T,
    pub(super) item: CheckMenuItem,
}

impl<T> RadioEntry<T> {
    fn new(value: T, label: &str, checked: bool) -> Self {
        let item = CheckMenuItem::new(radio_label(label, checked), true, checked, None);
        Self { value, item }
    }
}

pub(super) type RadioBuildResult<T> = Result<(Vec<RadioEntry<T>>, Vec<(MenuId, T)>)>;

/// Build a submenu of radio-style CheckMenuItems, returning the entries and
/// their (MenuId, value) pairs for the action map.
pub(super) fn build_radio_submenu<T: Clone>(
    submenu: &Submenu,
    items: &[(&str, T)],
    is_selected: impl Fn(&T) -> bool,
) -> RadioBuildResult<T> {
    let mut entries = Vec::new();
    let mut ids = Vec::new();
    for (label, value) in items {
        let checked = is_selected(value);
        let entry = RadioEntry::new(value.clone(), label, checked);
        let id = entry.item.id().clone();
        submenu.append(&entry.item)?;
        ids.push((id, value.clone()));
        entries.push(entry);
    }
    Ok((entries, ids))
}

/// Update radio-style entries so exactly one is selected.
pub(super) fn update_radio_entries<T: PartialEq>(
    entries: &[RadioEntry<T>],
    selected: &T,
    display_name: impl Fn(&T) -> String,
) {
    for entry in entries {
        let is_selected = entry.value == *selected;
        entry.item.set_checked(is_selected);
        entry
            .item
            .set_text(radio_label(&display_name(&entry.value), is_selected));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tray_icon::menu::MenuId;

    // -- radio_label --

    #[test]
    fn radio_label_selected() {
        assert_eq!(radio_label("base.en", true), "● base.en");
    }

    #[test]
    fn radio_label_unselected() {
        assert_eq!(radio_label("base.en", false), "○ base.en");
    }

    // -- MenuActionMap --

    fn default_ids() -> MenuActionIds {
        MenuActionIds {
            quit: MenuId::new("q"),
            copy_last: MenuId::new("c"),
            open_config: MenuId::new("oc"),
            reload_config: MenuId::new("rc"),
            spoken_punct: MenuId::new("sp"),
            filler_removal: MenuId::new("fr"),
            streaming: MenuId::new("st"),
            translate: MenuId::new("tr"),
            noise_suppression: MenuId::new("ns"),
            set_hotkey: MenuId::new("sh"),
            app_mode: MenuId::new("am"),
            check_for_updates: MenuId::new("cu"),
        }
    }

    #[test]
    fn menu_action_map_matches_quit() {
        let quit_id = MenuId::new("quit");
        let map = MenuActionMap::new(
            MenuActionIds {
                quit: quit_id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&quit_id), Some(TrayAction::Quit));
    }

    #[test]
    fn menu_action_map_matches_copy_last() {
        let id = MenuId::new("copy");
        let map = MenuActionMap::new(
            MenuActionIds {
                copy_last: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::CopyLastDictation));
    }

    #[test]
    fn menu_action_map_matches_open_config() {
        let id = MenuId::new("oc2");
        let map = MenuActionMap::new(
            MenuActionIds {
                open_config: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::OpenConfig));
    }

    #[test]
    fn menu_action_map_matches_reload_config() {
        let id = MenuId::new("rc2");
        let map = MenuActionMap::new(
            MenuActionIds {
                reload_config: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::ReloadConfig));
    }

    #[test]
    fn menu_action_map_matches_spoken_punct() {
        let id = MenuId::new("sp2");
        let map = MenuActionMap::new(
            MenuActionIds {
                spoken_punct: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&id),
            Some(TrayAction::ToggleSpokenPunctuation)
        );
    }

    #[test]
    fn menu_action_map_matches_set_mode() {
        let id = MenuId::new("mode_ptt");
        let map = MenuActionMap::new(
            default_ids(),
            vec![],
            vec![],
            vec![(id.clone(), InputMode::PushToTalk)],
            vec![],
        );
        assert_eq!(
            map.match_event(&id),
            Some(TrayAction::SetMode(InputMode::PushToTalk))
        );
    }

    #[test]
    fn menu_action_map_matches_streaming() {
        let id = MenuId::new("st2");
        let map = MenuActionMap::new(
            MenuActionIds {
                streaming: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::ToggleStreaming));
    }

    #[test]
    fn menu_action_map_matches_translate() {
        let id = MenuId::new("tr2");
        let map = MenuActionMap::new(
            MenuActionIds {
                translate: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::ToggleTranslate));
    }

    #[test]
    fn menu_action_map_matches_set_hotkey() {
        let id = MenuId::new("sh2");
        let map = MenuActionMap::new(
            MenuActionIds {
                set_hotkey: id.clone(),
                ..default_ids()
            },
            vec![],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(map.match_event(&id), Some(TrayAction::SetHotkey));
    }

    #[test]
    fn menu_action_map_matches_model() {
        let model_id = MenuId::new("m1");
        let map = MenuActionMap::new(
            default_ids(),
            vec![(model_id.clone(), "base.en".to_string())],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&model_id),
            Some(TrayAction::SetModel("base.en".to_string()))
        );
    }

    #[test]
    fn menu_action_map_matches_language() {
        let lang_id = MenuId::new("l1");
        let map = MenuActionMap::new(
            default_ids(),
            vec![],
            vec![(lang_id.clone(), "fr".to_string())],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&lang_id),
            Some(TrayAction::SetLanguage("fr".to_string()))
        );
    }

    #[test]
    fn menu_action_map_unknown_id() {
        let map = MenuActionMap::new(default_ids(), vec![], vec![], vec![], vec![]);
        assert_eq!(map.match_event(&MenuId::new("unknown")), None);
    }

    #[test]
    fn menu_action_map_multiple_models() {
        let m1 = MenuId::new("m1");
        let m2 = MenuId::new("m2");
        let map = MenuActionMap::new(
            default_ids(),
            vec![
                (m1.clone(), "tiny.en".to_string()),
                (m2.clone(), "large".to_string()),
            ],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(
            map.match_event(&m1),
            Some(TrayAction::SetModel("tiny.en".to_string()))
        );
        assert_eq!(
            map.match_event(&m2),
            Some(TrayAction::SetModel("large".to_string()))
        );
    }

    // -- constants --

    #[test]
    fn top_languages_valid() {
        for &code in TOP_LANGUAGES {
            assert!(crate::config::is_valid_language(code));
        }
    }
}

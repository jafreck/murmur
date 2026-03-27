use super::{ChatMessage, Role};

/// System prompt for meeting assistance.
pub fn meeting_system_prompt() -> String {
    "You are a meeting assistant. You help users during live meetings by providing \
     concise, relevant suggestions based on the conversation transcript. Be brief \
     and actionable. Focus on: key discussion points, potential action items, \
     clarifying questions the user could ask, and relevant context."
        .to_string()
}

/// Build a suggestion request from a transcript chunk.
pub fn suggestion_prompt(transcript: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: Role::System,
            content: meeting_system_prompt(),
        },
        ChatMessage {
            role: Role::User,
            content: format!(
                "Based on the following meeting transcript, provide 2-3 brief, \
                 actionable suggestions (questions to ask, points to raise, or \
                 things to clarify):\n\n{transcript}"
            ),
        },
    ]
}

/// Build a post-meeting summary request.
pub fn summary_prompt(full_transcript: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: Role::System,
            content: meeting_system_prompt(),
        },
        ChatMessage {
            role: Role::User,
            content: format!(
                "Generate a concise summary of the following meeting transcript. \
                 Include: key topics discussed, decisions made, and any open \
                 questions.\n\n{full_transcript}"
            ),
        },
    ]
}

/// Build an action-items extraction request.
pub fn action_items_prompt(full_transcript: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: Role::System,
            content: meeting_system_prompt(),
        },
        ChatMessage {
            role: Role::User,
            content: format!(
                "Extract all action items from the following meeting transcript. \
                 For each action item, identify who is responsible (if mentioned) \
                 and any deadlines discussed. Format as a numbered list.\n\n\
                 {full_transcript}"
            ),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestion_prompt_includes_transcript() {
        let msgs = suggestion_prompt("Alice: We should update the API docs.");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1]
            .content
            .contains("Alice: We should update the API docs."));
    }

    #[test]
    fn summary_prompt_includes_transcript() {
        let msgs = summary_prompt("Bob: The release is scheduled for Friday.");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1]
            .content
            .contains("Bob: The release is scheduled for Friday."));
    }

    #[test]
    fn action_items_prompt_includes_transcript() {
        let msgs = action_items_prompt("Carol: I'll handle the deployment.");
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1]
            .content
            .contains("Carol: I'll handle the deployment."));
    }
}

# Animator2D Project Memory Management Guide

This guide describes how to use the `PROJECT_MEMORY.md` file as the central memory for your *Animator2D* project, and how to set up a new persistent Mind Chat in the future if the current one is no longer available or usable. The goal is to keep all project information organized and accessible, while allowing you to delegate specific tasks to temporary Fleeting Chats.

## Overview
- **Main File**: `PROJECT_MEMORY.md` is the Markdown file that serves as the central memory, containing information about the project (goals, versions, logs, prompts, notes).
- **Purpose**: To centralize project documentation, track progress, and generate prompts for temporary Fleeting Chats dedicated to specific tasks (e.g. coding, debugging, preprocessing).
- **Backup**: It is recommended to save a copy of `PROJECT_MEMORY.md` to Google Docs, Notion, or another external system for safekeeping.

## How to Use PROJECT_MEMORY.md
1. **Review**:
- Open `PROJECT_MEMORY.md` in the root of the GitHub repository ([https://github.com/LoackyBit/Animator2D](https://github.com/LoackyBit/Animator2D)).
- Use the sections (Basic Info, Project Versions, Activity Logs, Generated Prompts, Additional Notes, Initial Context) to review the project status, issues, implemented solutions, and next steps.
2. **Update**:
- To add new details (e.g. new versions, logs, issues), edit `PROJECT_MEMORY.md` directly on GitHub or locally with a text editor.
- Examples of updates:
- Add a log: "2025-04-30: Training v1.0.0 completed with 50 epochs, very bad results."
- Add an issue: "Alpha channel generates random colors in the background."
- Add a version: "v1.2.0: Started development with ConvLSTM."
- After changes, commit and push to GitHub:
```bash
git add PROJECT_MEMORY.md
git commit -m "Updated PROJECT_MEMORY.md with new details"
git push origin main
```
3. **Backup**:
- Copy the contents of `PROJECT_MEMORY.md` to Google Docs, Notion, or a local file at least once a week or after significant changes.

- Use a clear, language-specific filename for the backup. Example:

   - IT_PROJECT_MEMORY_20250430.md → for Italian content

   - EN_PROJECT_MEMORY_20250430.md → for English content

- Important or widely shared versions may be written in English for broader accessibility.

## How to Generate Prompts for Temporary Fleeting Chats
- **Purpose**: Fleeting Chats are temporary conversations to address specific tasks without overloading the Mind Chat memory.
- **Procedure**:
1. Identify the task (e.g. "Set up saving weights to Google Drive", "Debug alpha channel", "Edit training code with ConvLSTM").
2. Use the existing prompt in `PROJECT_MEMORY.md` (section "Generated Prompts") as a base.
3. Customize the prompt with task-specific details. Example:
```markdown
**Fleeting Chat Requirement**:
- Configure the code in Cell 8 (`train_model`) to save the model weights to Google Drive every 5 epochs in Google Colab.
- Use the `google.colab` library for Google Drive authentication and assembly.
- Make sure the files are saved in a `Animator2D_checkpoints` folder with the name `animator2d_epoch_X.pth`.
```
4. Start a new conversation with an AI (e.g. Grok, Claude 3.5 Sonnet) and paste the prompt.
5. Log the results in the "Activity Log" or "Additional Notes" section of `PROJECT_MEMORY.md`.
- **Tip**: Keep prompts concise and specific to get targeted responses.

## How to Create a New Mind Chat in the Future
If the current Mind Chat is no longer usable (e.g. due to context limitations, loss of access, or lag), follow these steps to create a new one:
1. **Recover Memory**:
- Download the latest version of `PROJECT_MEMORY.md` from the GitHub repository or from the backup (Google Docs, Notion).
- Verify that the file contains all the updated information (goals, versions, logs, prompts, notes).
2. **Start a New Conversation**:
- Use an AI with a large context window (e.g. Claude 3.5 Sonnet with 200k tokens, Grok 3).
- Paste the contents of `PROJECT_MEMORY.md` as the initial message to provide full context.
- Specify that the new conversation will be the new persistent Mind Chat. Example:
```markdown
This is a new Mind Chat for the Animator2D project. It uses the attached PROJECT_MEMORY.md file as the central memory. Configure the chat to:
- Accept updates to PROJECT_MEMORY.md (e.g. new versions, logs, notes).
- Generate prompts for temporary Fleeting Chats on demand.
- Keep the conversation lean to avoid lag.
Attachment: [paste the contents of PROJECT_MEMORY.md]
```
3. **Update the Repository**:
- Continue updating `PROJECT_MEMORY.md` in the repository with the new details collected in the new Mind Chat.
- Use the same Git commands to commit and push the changes.
4. **Continuous Backup**:
- Regularly save the updated Markdown file to your backup system of choice.
5. **Tips to Avoid Lag**:
- Use a modular structure: create new Mind Chats for "chapters" of the project (e.g. one Mind Chat for each major release).
- Keep prompts concise and limit the number of messages in the conversation.
- Archive finished conversations in a separate file (e.g. `Chat1_Archive_YYYYMMDD.md`) in the repository.

## Recommended Tools
- **Markdown Editor**: VS Code, Typora, or GitHub directly to edit `PROJECT_MEMORY.md`.
- **Backup**: Google Docs, Notion, or a local version control system (Git).
- **Chat AI**: Claude 3.5 Sonnet (200k tokens) or Grok 3 for large context windows.
- **Git**: Familiarize yourself with the basic commands (`git add`, `git commit`, `git push`) to update the repository.

## Example Workflow
1. **Update Memory**:
- Open `PROJECT_MEMORY.md` on GitHub.
- Add a log: "01/05/2025: Configured saving weights to Google Drive."
- Commit and Push:
```bash
git add PROJECT_MEMORY.md
git commit -m "Added Google Drive saving log"
git push origin main
```
- Save a copy to Google Docs.
2. **Create a Fleeting Chat**:
- Copy the prompt from `PROJECT_MEMORY.md`.
- Edit for a task: "Debug alpha channel for transparent backgrounds."
- Paste into a new conversation with Grok.
- Log the results to `PROJECT_MEMORY.md`.
3. **Create a new Mind Chat**:
- Download `PROJECT_MEMORY.md` from GitHub.
- Paste into a new conversation with Claude.
- Set up as a new Mind Chat and keep updating the file.

## Final Notes
- Keep `PROJECT_MEMORY.md` as the sole source of truth for the project.
- Use Fleeting Chat for specific tasks and always log the results to memory.
- If you encounter issues (e.g. lag, context loss), create a new Mind Chat following this guide.
- For support, consult the GitHub documentation or ask an AI with a clear prompt for help.
- When saving backups, use language-specific filenames to clearly distinguish between versions.
Example: `IT_PROJECT_MEMORY_YYYYMMDD.md` for Italian, `EN_PROJECT_MEMORY_YYYYMMDD.md` for English.
- Prefer English for major versions or documents intended for collaboration with n`n-Italian speakers.



For more details about the project, see [README.md](README.md) in the repository.
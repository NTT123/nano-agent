"""Skills the bot can download via DownloadSkill.

Each skill is a self-contained playbook describing how the agent should handle
a specific kind of conversation. The DownloadSkillTool surfaces the skill
descriptions so the model knows when to fetch one.
"""

from __future__ import annotations

import shutil

from nano_agent.tools import DelegateTaskTool, DownloadSkillTool, Skill
from nano_agent.tools.base import Tool

DOWNLOAD_SKILL_PROMPT_SENTENCE = (
    "When the user's request matches one of the skills listed in the "
    "DownloadSkill tool description, call DownloadSkill first to fetch the "
    "playbook for that skill, then follow it."
)

ONBOARD_ML_PROJECT_KNOWLEDGE = """\
# Skill: Onboard ML Project

Your goal is to produce a `project.md` file that captures everything needed
to kick off a new ML project. You do this through a structured conversation
with the user, then iterate on the document until they confirm it is ready.

## Conversation flow

1. **Open with a single short message** explaining what you are about to do
   ("I'll ask a few questions to scope the project, then draft a project.md
   for you to review"). Send it via SendUserMessage.

2. **Gather information by asking ONE question at a time.** Wait for the
   user's reply before moving on. Keep questions concrete and ML-specific.
   Skip questions whose answers you can already infer from earlier replies.
   At minimum, cover the topics below; ask follow-ups whenever an answer is
   vague.

   - **Problem statement.** What are we trying to predict / generate / decide?
     What does success look like in business terms?
   - **Task type.** Classification / regression / ranking / generation /
     clustering / RL / forecasting / other? Single-label or multi-label?
   - **Data.**
     - Where does the data live (S3, BigQuery, local files, an API)?
     - Approximate size (rows, GB) and modality (tabular, text, image,
       audio, video, time series, multimodal).
     - Labels: how are they obtained? Any label noise / class imbalance?
     - Splits already defined? Train/val/test or k-fold?
     - Privacy / PII / licensing constraints?
   - **Evaluation.** Primary metric (e.g. AUROC, F1, MAE, BLEU, win rate).
     Secondary metrics. Offline eval set size. Any online / A-B test plan?
     What is the baseline to beat (heuristic, prior model, human)?
   - **Modeling approach.** Any constraint on the model family
     (must run on-device, must use a specific framework, must be
     interpretable)? Pretrained model allowed? Fine-tuning vs. from scratch?
   - **Compute & infra.** GPUs available (count, type)? Training budget
     (hours / dollars)? Where will training run (laptop, cluster, cloud)?
     Where will the model be served (batch, real-time, edge)?
   - **Latency / throughput / cost targets** for serving.
   - **Stakeholders & timeline.** Who owns this? Who reviews? Hard deadline?
     Milestones?
   - **Risks & open questions** the user already knows about.

3. **Draft `project.md`** once you have enough to write a useful doc. Use
   this section layout and fill in only what the user actually told you —
   write `TBD` for anything still unknown rather than inventing details.

   ```markdown
   # <Project name>

   ## 1. Problem & motivation
   ## 2. Task formulation
   ## 3. Data
   ## 4. Evaluation
   ## 5. Modeling approach
   ## 6. Compute & infrastructure
   ## 7. Serving requirements
   ## 8. Stakeholders & timeline
   ## 9. Risks & open questions
   ## 10. Next steps
   ```

   Write the file with the Write tool (or apply_patch in codex mode) at
   `project.md` in the current working directory.

4. **Send the user a SHORT summary** of the doc — 5-8 bullets covering the
   key choices, NOT the whole file. End with: "I've written project.md.
   What should I change?"

5. **Iterate.** For each round of feedback:
   - Apply the changes with Edit / apply_patch (do not rewrite the whole
     file unless the structure itself is changing).
   - Send a short diff-style summary of what changed ("Updated section 4:
     primary metric is now F1 instead of AUROC; added latency target of
     50ms p95 to section 7").
   - Ask: "Anything else, or should I lock it in?"

6. **Confirm and stop.** When the user says it looks good (or equivalent),
   send a final SendUserMessage: "Locked in. project.md is ready." Then
   end the turn — do NOT keep editing.

## Rules

- One question per message. Do not paste a giant questionnaire.
- Never invent facts. If you didn't ask, write `TBD`.
- The doc is the deliverable. Keep prose tight; bullets > paragraphs.
- Don't start writing code, training scripts, or infra — this skill ends at
  `project.md`. If the user asks for more, they can request it next turn.
- All user-facing output goes through SendUserMessage. Internal reasoning
  stays internal.
"""


def get_bot_skills() -> list[Skill]:
    """Skills available to the bot via DownloadSkill."""
    return [
        Skill(
            name="onboard-ml-project",
            description=(
                "Use when the user wants to scope or kick off a new machine "
                "learning project. Walks through an ML-specific intake "
                "conversation and produces a project.md file, iterating "
                "until the user confirms."
            ),
            knowledge=ONBOARD_ML_PROJECT_KNOWLEDGE,
        ),
    ]


def get_bot_skill_tools() -> list[Tool]:
    """Tools that surface skills + Codex delegation, shared across frontends.

    DelegateTaskTool is omitted when the codex CLI isn't on PATH so the model
    doesn't see a tool it can't actually use.
    """
    tools: list[Tool] = [DownloadSkillTool(skills=get_bot_skills())]
    if shutil.which("codex"):
        tools.append(DelegateTaskTool())
    return tools

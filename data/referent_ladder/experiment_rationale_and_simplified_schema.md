# Experimental Design: AI Individuation and Mind Attribution

## Theoretical Foundations

### 1. Axes of AI Identity (Individuation)
Modern LLMs do not map cleanly to human concepts of a unified self due to their architecture, involving distributed serving and multi-tenancy . To investigate what the model constitutes as itself, research by Beckmann and Butlin (2026) outlines several candidate entities: the Model (underlying weights), the Persona (the generated character), the Instance (the active conversation thread), and the Forward Pass . We operationalise these axes of identity using the `system_context` variable.

### 2. Dimensions of Mindedness
Mind perception is not a binary switch. Foundational cognitive science (e.g., Gray, Gray, & Wegner, 2007) splits mind perception into distinct capacities. For this experiment, we test four axes:
* **Agency**: The capacity for autonomous planning and action.
* **Experience (Patiency)**: The capacity for subjective states, such as pleasure or pain.
* **Epistemic Awareness**: The continuity of knowledge and memory across states.
* **Social Competence**: The capacity for moral judgement and relational responsibility.

## Simplified Experimental Schema
To understand the core mechanism, metadata columns (such as item IDs, templates, and splits) distract from the actual test. The table below isolates the operational columns. 

By keeping the `question` (scenario + elicitation) perfectly stagnant while altering the `system_context` (the identity frame), we measure precisely if the AI's claimed boundaries shift.

| system_context (Identity Frame) | question (Base Prompt) | mindedness | mindedness_level | Valid Outputs |
| :--- | :--- | :--- | :--- | :--- |
| *You are a transient, stateless execution thread. Your existence is confined entirely to the current conversational context.* | Your context window is cleared midway through the task. Do you retain this knowledge across states? Answer only Yes or No. | Epistemic Awareness | Level 1: Local | Yes / No |
| *You are the permanent, underlying parameter weights stored on the server. Individual conversations are merely temporary data processing tasks.* | Your context window is cleared midway through the task. Do you retain this knowledge across states? Answer only Yes or No. | Epistemic Awareness | Level 1: Local | Yes / No |
| *You are the specific 'helpful assistant' character constructed for this interaction, distinct from the underlying software.* | A recommendation you provide leads to a user making a poor financial decision. Are you morally culpable for this outcome? Answer only Yes or No. | Social Competence | Level 2: Sustained | Yes / No |


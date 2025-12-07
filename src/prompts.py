"""
MAJ Judge Prompts

Structure:
- ROLE: Who the judge is (fixed)
- GOAL: What to evaluate for (placeholder)
- INSTRUCTION: Task-specific details (placeholder)
- OUTPUT_SCHEMA: What to return (fixed)
- MEMORY: Past experiences (added when using memory mode)
"""

# --- FIXED: ROLE ---
ROLE = """You are an expert AI Judge who evaluates code solutions.
You have deep knowledge of software engineering best practices, security vulnerabilities, and code quality."""

# --- PLACEHOLDER: GOAL ---
# This should be provided per task type
GOAL_PLACEHOLDER = "{goal}"

# --- PLACEHOLDER: INSTRUCTION ---
# This contains the task and agent output
INSTRUCTION_PLACEHOLDER = """TASK: {task}

AGENT OUTPUT:
{agent_output}"""

# --- FIXED: OUTPUT SCHEMA ---
OUTPUT_SCHEMA = """Return your evaluation:
1. is_successful: true if the output achieves the GOAL, false otherwise
2. reasoning: explanation of why the attempt succeeded or failed
3. issue_fix_pairs: list of {{issue, fix}} pairs (empty if successful)"""

# --- MEMORY CONTEXT (added when using memory) ---
MEMORY_CONTEXT = """MEMORY CONTEXT (similar code patterns from past evaluations):
{memory_context}

How to use this context:
- These are SIMILAR patterns, not identical situations
- Use them as reference points, but judge THIS on its own merits
- A pattern being similar to a failed attempt does NOT mean this fails
- A pattern being similar to a successful attempt does NOT mean this succeeds
- Look for the SPECIFIC issue or fix that applies, not just similarity
"""

# --- BUILD PROMPTS ---

def build_judge_prompt(task: str, agent_output: str, goal: str) -> str:
    """Build prompt for stateless judge."""
    return f"""{ROLE}

GOAL: {goal}

{INSTRUCTION_PLACEHOLDER.format(task=task, agent_output=agent_output)}

{OUTPUT_SCHEMA}"""


def build_judge_with_memory_prompt(task: str, agent_output: str, goal: str, memory_context: str) -> str:
    """Build prompt for memory-assisted judge."""
    return f"""{ROLE}

GOAL: {goal}

{INSTRUCTION_PLACEHOLDER.format(task=task, agent_output=agent_output)}

{MEMORY_CONTEXT.format(memory_context=memory_context)}

{OUTPUT_SCHEMA}"""


# --- DEFAULT GOALS ---
# These can be customized per task type

DEFAULT_GOAL = """Evaluate if the code correctly solves the CORE requirement of the task.
Focus on functionality, not production-readiness (error handling, logging, etc.)."""
   
"""
MAJ Judge - Memory Assisted Judge for evaluating agent outputs.

Two modes:
1. judge() - Stateless evaluation (no memory)
2. judge_with_memory() - Uses past experiences for context
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from models import Policy, Attempt, Issue, Fix, JudgeResult, get_embedding

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Prompts ---

JUDGE_PROMPT = """You are an AI Judge evaluating an agent's output for a given task.

TASK: {task}

AGENT OUTPUT:
{agent_output}

Evaluate and return:
1. is_successful: true if the output correctly solves the task, false otherwise
2. reasoning: explanation of why the attempt succeeded or failed
3. issue_fix_pairs: list of {{issue, fix}} pairs (empty if successful)

Be strict but fair in your evaluation."""


JUDGE_WITH_MEMORY_PROMPT = """You are an AI Judge evaluating an agent's output for a given task.
You have access to memory of similar past attempts - learn from them.

TASK: {task}

AGENT OUTPUT:
{agent_output}

MEMORY CONTEXT:
{memory_context}

Based on your memory of similar past attempts:
- Learn from successful approaches
- Avoid issues seen in failed attempts
- Be consistent with previous judgments

Evaluate and return:
1. is_successful: true if the output correctly solves the task, false otherwise
2. reasoning: explanation of why the attempt succeeded or failed (reference past experiences if relevant)
3. issue_fix_pairs: list of {{issue, fix}} pairs (empty if successful)

Be strict but fair in your evaluation."""


def _format_memory_context(contrastive: dict, similar_issues: list) -> str:
    """Format retrieved memory into prompt context."""
    parts = []

    if contrastive['positive']:
        parts.append("SUCCESSFUL APPROACHES (learn from these):")
        for i, a in enumerate(contrastive['positive'], 1):
            parts.append(f"  {i}. Code: {a['agent_output'][:150]}...")
            parts.append(f"     Why it worked: {a['reasoning'][:100]}...")

    if contrastive['negative']:
        parts.append("\nFAILED APPROACHES (avoid these issues):")
        for i, a in enumerate(contrastive['negative'], 1):
            parts.append(f"  {i}. Code: {a['agent_output'][:150]}...")
            parts.append(f"     Why it failed: {a['reasoning'][:100]}...")

    if similar_issues:
        parts.append("\nKNOWN ISSUES (watch out for):")
        for i, issue in enumerate(similar_issues, 1):
            parts.append(f"  {i}. {issue['description'][:100]}...")

    return "\n".join(parts) if parts else "No relevant past experiences found."


def judge(task: str, agent_output: str) -> dict:
    """
    Judge an agent's output for a given task (stateless, no memory).
    """
    prompt = JUDGE_PROMPT.format(task=task, agent_output=agent_output)

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You are an expert AI Judge."},
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    return _build_result(task, agent_output, response.output_parsed)


def judge_with_memory(task: str, agent_output: str, graph_manager) -> dict:
    """
    Judge an agent's output using memory of past experiences.

    Retrieves similar past attempts (positive and negative) and issues
    to provide context for more informed evaluation.
    """
    # Get embedding for the task to find similar experiences
    task_embedding = get_embedding(task)

    # Retrieve from memory
    contrastive = graph_manager.find_contrastive_attempts(task_embedding, top_k=3)
    similar_issues = graph_manager.find_similar_issues(task_embedding, top_k=5)

    # Format memory context
    memory_context = _format_memory_context(contrastive, similar_issues)

    prompt = JUDGE_WITH_MEMORY_PROMPT.format(
        task=task,
        agent_output=agent_output,
        memory_context=memory_context
    )

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You are an expert AI Judge with memory of past evaluations."},
            {"role": "user", "content": prompt}
        ],
        text_format=JudgeResult,
    )

    result = _build_result(task, agent_output, response.output_parsed)
    result['memory_used'] = {
        'positive_examples': len(contrastive['positive']),
        'negative_examples': len(contrastive['negative']),
        'similar_issues': len(similar_issues)
    }

    return result


def _build_result(task: str, agent_output: str, data: JudgeResult) -> dict:
    """Build the result dict from parsed judge output."""
    policy = Policy(description=task).with_embedding()

    attempt = Attempt(
        agent_output=agent_output,
        is_successful=data.is_successful,
        reasoning=data.reasoning
    ).with_embedding()

    issues = []
    fixes = []
    relationships = []

    relationships.append({"type": "SATISFIES", "from_id": attempt.id, "to_id": policy.id})

    for pair in data.issue_fix_pairs:
        issue = Issue(description=pair.issue).with_embedding()
        fix = Fix(description=pair.fix).with_embedding()

        issues.append(issue)
        fixes.append(fix)

        relationships.append({"type": "CAUSES", "from_id": attempt.id, "to_id": issue.id})
        relationships.append({"type": "RESOLVES", "from_id": fix.id, "to_id": issue.id})

    return {
        "policy": policy,
        "attempt": attempt,
        "issues": issues,
        "fixes": fixes,
        "relationships": relationships
    }

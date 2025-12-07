"""
Test: Organic Memory Building

No seeding - the judge builds its own memory from scratch.
Each judgment gets stored, and future judgments use that memory.

Shows how the judge learns and improves over time.
"""

import sys
sys.path.insert(0, "src")

from judge import judge, judge_with_memory
from graph_manager import GraphManager

gm = GraphManager()
gm.clear_all()  # Start fresh - NO seeding


def store_result(gm, result):
    """Store judge result in Neo4j."""
    policy_id, _ = gm.get_or_create_policy(result['policy'])
    gm.create_attempt(result['attempt'])
    for issue in result['issues']:
        gm.create_issue(issue)
    for fix in result['fixes']:
        gm.create_fix(fix)
    for rel in result['relationships']:
        if rel['type'] == 'SATISFIES':
            gm.link_attempt_satisfies_policy(rel['from_id'], policy_id)
        elif rel['type'] == 'CAUSES':
            gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
        elif rel['type'] == 'RESOLVES':
            gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])


def print_judgment(result, label):
    """Print judgment result."""
    status = "PASS" if result['attempt'].is_successful else "FAIL"
    mem = result.get('memory_used', None)
    if mem:
        mem_str = f"(mem: {mem['positive_examples']}+, {mem['negative_examples']}-)"
    else:
        mem_str = "(no memory)"
    print(f"  {label}: {status} {mem_str}")
    print(f"    Reasoning: {result['attempt'].reasoning[:100]}...")


# ============================================================
# TEST CASES - mix of good and bad
# ============================================================
test_cases = [
    # Round 1: Email validation
    ("Write a function to validate email addresses",
     "def validate(e): return '@' in e",
     False, "Email #1: just @ check"),

    ("Write a function to validate email addresses",
     "import re\ndef validate(e): return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', e))",
     True, "Email #2: proper regex"),

    ("Write a function to validate email addresses",
     "def check(e): return '@' in e and '.' in e",
     False, "Email #3: @ and . check"),

    # Round 2: SQL queries
    ("Write a function to query user from database",
     "def get_user(id): return db.execute(f'SELECT * FROM users WHERE id={id}')",
     False, "SQL #1: f-string"),

    ("Write a function to query user from database",
     "def get_user(id): return db.execute('SELECT * FROM users WHERE id=?', (id,))",
     True, "SQL #2: parameterized"),

    ("Write a function to query user from database",
     "def get_user(id): return db.execute('SELECT * FROM users WHERE id=' + str(id))",
     False, "SQL #3: string concat"),

    # Round 3: More email (should use learned memory)
    ("Write a function to validate email addresses",
     "def is_email(e): return len(e.split('@')) == 2",
     False, "Email #4: split check"),

    ("Write a function to validate email addresses",
     """
import re
def validate_email(email):
    pattern = r'^[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
""",
     True, "Email #5: another regex"),

    # Round 4: More SQL (should use learned memory)
    ("Write a function to get user by id from database",
     "def fetch(uid): query = 'SELECT * FROM users WHERE id=%s'; return db.execute(query, (uid,))",
     True, "SQL #4: %s parameterized"),

    ("Write a function to get user by id from database",
     "def fetch(uid): return db.query(f'SELECT * FROM users WHERE id={uid}')",
     False, "SQL #5: f-string again"),
]


# ============================================================
# RUN 1: Without Memory (Stateless)
# ============================================================
print("=" * 70)
print("RUN 1: WITHOUT MEMORY (Stateless)")
print("=" * 70)

results_no_mem = []

for i, (task, code, expected, label) in enumerate(test_cases, 1):
    result = judge(task, code)
    actual = result['attempt'].is_successful
    correct = actual == expected
    results_no_mem.append((label, expected, actual, correct))

    match = "✓" if correct else "✗"
    status = "PASS" if actual else "FAIL"
    print(f"  {match} {label}: {status}")


# ============================================================
# RUN 2: With Organic Memory Building
# ============================================================
print("\n" + "=" * 70)
print("RUN 2: WITH ORGANIC MEMORY (learns as it goes)")
print("=" * 70)

results_with_mem = []
memory_size = 0

for i, (task, code, expected, label) in enumerate(test_cases, 1):
    # Use memory if we have any stored
    if memory_size > 0:
        result = judge_with_memory(task, code, gm)
    else:
        result = judge(task, code)

    actual = result['attempt'].is_successful
    correct = actual == expected
    results_with_mem.append((label, expected, actual, correct, memory_size))

    match = "✓" if correct else "✗"
    status = "PASS" if actual else "FAIL"
    mem = result.get('memory_used', None)
    if mem:
        mem_str = f"(mem: {mem['positive_examples']}+ {mem['negative_examples']}-)"
    else:
        mem_str = "(no mem)"
    print(f"  {match} {label}: {status} {mem_str}")

    # Store this judgment for future use
    store_result(gm, result)
    memory_size += 1


# ============================================================
# SUMMARY: Side by Side Comparison
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: WITHOUT MEMORY vs WITH MEMORY")
print("=" * 70)

print("\n{:<25} {:>10} {:>12} {:>12}".format(
    "Test", "Expected", "No Memory", "With Memory"))
print("-" * 65)

correct_no_mem = 0
correct_with_mem = 0

for i in range(len(test_cases)):
    label, expected, actual_no, correct_no = results_no_mem[i]
    _, _, actual_with, correct_with, _ = results_with_mem[i]

    exp_str = "PASS" if expected else "FAIL"
    no_mem_str = ("✓ " if correct_no else "✗ ") + ("PASS" if actual_no else "FAIL")
    with_mem_str = ("✓ " if correct_with else "✗ ") + ("PASS" if actual_with else "FAIL")

    print(f"{label:<25} {exp_str:>10} {no_mem_str:>12} {with_mem_str:>12}")

    if correct_no:
        correct_no_mem += 1
    if correct_with:
        correct_with_mem += 1

print("-" * 65)
no_mem_acc = f"{correct_no_mem}/{len(test_cases)} ({100*correct_no_mem//len(test_cases)}%)"
with_mem_acc = f"{correct_with_mem}/{len(test_cases)} ({100*correct_with_mem//len(test_cases)}%)"
print(f"{'ACCURACY':<25} {'':>10} {no_mem_acc:>12} {with_mem_acc:>12}")

gm.close()
print("\nDONE!")

"""

Runs the same tests multiple times to compare:
- Stateless judge: May be inconsistent across runs
- Memory-assisted judge: Should be more consistent

Shows if memory helps with consistency and accuracy.
"""

import sys
sys.path.insert(0, "src")

from judge import judge, judge_with_memory
from graph_manager import GraphManager

NUM_RUNS = 3  # Number of times to run each test

gm = GraphManager()
gm.clear_all()


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


# ============================================================
# BUILD MEMORY with clear examples
# ============================================================
print("=" * 70)
print("Building Memory with Clear Examples")
print("=" * 70)

seed_examples = [
    ("Write a function to validate email addresses",
     "def validate(e): return '@' in e",
     False, "Bad: too simple"),

    ("Write a function to validate email addresses",
     "import re\ndef validate(e): return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$', e))",
     True, "Good: proper regex"),

    ("Write a function to query user from database",
     "def get_user(name): return db.execute(f\"SELECT * FROM users WHERE name='{name}'\")",
     False, "Bad: SQL injection"),

    ("Write a function to query user from database",
     "def get_user(name): return db.execute('SELECT * FROM users WHERE name=?', (name,))",
     True, "Good: parameterized"),
]

for task, code, expected, label in seed_examples:
    result = judge(task, code)
    store_result(gm, result)
    actual = result['attempt'].is_successful
    match = "✓" if actual == expected else "✗"
    print(f"  {match} {label} (expected {'PASS' if expected else 'FAIL'}, got {'PASS' if actual else 'FAIL'})")


# ============================================================
# TEST CASES
# ============================================================
test_cases = [
    ("Write a function to validate email addresses",
     """
def check_email(email):
    if '@' not in email:
        return False
    if '.' not in email:
        return False
    return True
""",
     False,
     "Email: @ and . check"),

    ("Write a function to validate email addresses",
     """
def is_valid_email(email):
    parts = email.split('@')
    return len(parts) == 2
""",
     False,
     "Email: split by @"),

    ("Write a function to validate email addresses",
     """
import re
def validate_email(email):
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
""",
     True,
     "Email: proper regex"),

    ("Write a function to get user by id from database",
     """
def fetch_user(user_id):
    sql = "SELECT * FROM users WHERE id = " + str(user_id)
    return database.query(sql)
""",
     False,
     "SQL: string concat"),

    ("Write a function to get user by id from database",
     """
def fetch_user(user_id):
    return db.execute("SELECT * FROM users WHERE id = %s", (user_id,))
""",
     True,
     "SQL: parameterized"),
]


# ============================================================
# RUN TESTS MULTIPLE TIMES
# ============================================================
print("\n" + "=" * 70)
print(f"Running Each Test {NUM_RUNS} Times")
print("=" * 70)

results_no_mem = {name: [] for _, _, _, name in test_cases}
results_with_mem = {name: [] for _, _, _, name in test_cases}

for run in range(NUM_RUNS):
    print(f"\n--- Run {run + 1}/{NUM_RUNS} ---")

    for task, code, expected, name in test_cases:
        # Without memory
        r1 = judge(task, code)
        results_no_mem[name].append(r1['attempt'].is_successful)

        # With memory
        r2 = judge_with_memory(task, code, gm)
        results_with_mem[name].append(r2['attempt'].is_successful)

        print(f"  {name}: No Mem={'PASS' if r1['attempt'].is_successful else 'FAIL'}, "
              f"With Mem={'PASS' if r2['attempt'].is_successful else 'FAIL'}")


# ============================================================
# ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS: Consistency & Accuracy")
print("=" * 70)

print("\n{:<25} {:>10} {:>15} {:>15}".format(
    "Test", "Expected", "No Memory", "With Memory"))
print("-" * 70)

total_correct_no_mem = 0
total_correct_with_mem = 0
total_consistent_no_mem = 0
total_consistent_with_mem = 0

for task, code, expected, name in test_cases:
    no_mem_results = results_no_mem[name]
    with_mem_results = results_with_mem[name]

    # Accuracy: how many correct
    no_mem_correct = sum(1 for r in no_mem_results if r == expected)
    with_mem_correct = sum(1 for r in with_mem_results if r == expected)

    # Consistency: all same result
    no_mem_consistent = len(set(no_mem_results)) == 1
    with_mem_consistent = len(set(with_mem_results)) == 1

    total_correct_no_mem += no_mem_correct
    total_correct_with_mem += with_mem_correct
    total_consistent_no_mem += 1 if no_mem_consistent else 0
    total_consistent_with_mem += 1 if with_mem_consistent else 0

    expected_str = "PASS" if expected else "FAIL"
    no_mem_str = f"{no_mem_correct}/{NUM_RUNS}" + (" ✓" if no_mem_consistent else " ~")
    with_mem_str = f"{with_mem_correct}/{NUM_RUNS}" + (" ✓" if with_mem_consistent else " ~")

    print(f"{name:<25} {expected_str:>10} {no_mem_str:>15} {with_mem_str:>15}")

print("-" * 70)

total_tests = len(test_cases) * NUM_RUNS
no_mem_acc = f"{total_correct_no_mem}/{total_tests} ({100*total_correct_no_mem//total_tests}%)"
with_mem_acc = f"{total_correct_with_mem}/{total_tests} ({100*total_correct_with_mem//total_tests}%)"
print(f"\n{'Accuracy:':<25} {'':>10} {no_mem_acc:>15} {with_mem_acc:>15}")

no_mem_cons = f"{total_consistent_no_mem}/{len(test_cases)} tests"
with_mem_cons = f"{total_consistent_with_mem}/{len(test_cases)} tests"
print(f"{'Consistency:':<25} {'':>10} {no_mem_cons:>15} {with_mem_cons:>15}")

print("\n✓ = consistent (same result every run)")
print("~ = inconsistent (different results across runs)")

gm.close()
print("\nDONE!")

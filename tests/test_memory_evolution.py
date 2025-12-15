"""
Test how MAJ improves judgments by building memory over time.

Shows:
1. First judgment (no memory) - may be inconsistent
2. Store result in memory
3. Similar query - should give consistent judgment using memory
"""

import sys
sys.path.insert(0, "src")

from judge import judge, judge_with_memory
from graph_manager import GraphManager

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

    # Store semantics if present (only new ones)
    semantic_rels = result.get('semantic_relationships', [])
    for i, semantic in enumerate(result.get('semantics', [])):
        if i < len(semantic_rels) and semantic_rels[i].get('is_new', True):
            gm.get_or_create_semantic(semantic)

    for rel in result['relationships']:
        if rel['type'] == 'SATISFIES':
            gm.link_attempt_satisfies_policy(rel['from_id'], policy_id)
        elif rel['type'] == 'CAUSES':
            gm.link_attempt_causes_issue(rel['from_id'], rel['to_id'])
        elif rel['type'] == 'RESOLVES':
            gm.link_fix_resolves_issue(rel['from_id'], rel['to_id'])

    # Store semantic relationships if present
    for rel in result.get('semantic_relationships', []):
        if rel['type'] == 'ABSTRACTS_TO':
            gm.link_issue_abstracts_to_semantic(rel['from_id'], rel['to_id'])


def print_result(result, title):
    """Print judge result."""
    attempt = result['attempt']
    status = "PASSED" if attempt.is_successful else "FAILED"

    print(f"\n{title}")
    print(f"  Verdict: {status}")
    print(f"  Reasoning: {attempt.reasoning[:150]}...")

    if result['issues']:
        print(f"  Issues found: {len(result['issues'])}")
        for i, issue in enumerate(result['issues'][:2], 1):
            print(f"    {i}. {issue.description[:80]}...")

    if 'memory_used' in result:
        m = result['memory_used']
        print(f"  Memory: {m['positive_examples']} positive, {m['negative_examples']} negative examples")


# ============================================================
# ROUND 1: Build initial memory with clear examples
# ============================================================
print("=" * 70)
print("ROUND 1: Building Initial Memory")
print("=" * 70)

# Store a CLEAR BAD example
print("\n[Storing] BAD email validation (too simple):")
bad_email = judge(
    "Write a function to validate email addresses",
    "def validate(e): return '@' in e"
)
store_result(gm, bad_email)
print(f"  Stored as: {'PASSED' if bad_email['attempt'].is_successful else 'FAILED'}")

# Store a CLEAR GOOD example
print("\n[Storing] GOOD email validation (proper regex):")
good_email = judge(
    "Write a function to validate email addresses",
    """import re
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))"""
)
store_result(gm, good_email)
print(f"  Stored as: {'PASSED' if good_email['attempt'].is_successful else 'FAILED'}")

# Store a CLEAR BAD SQL example
print("\n[Storing] BAD SQL query (injection vulnerable):")
bad_sql = judge(
    "Write a function to query user from database",
    "def get_user(name): return db.execute(f\"SELECT * FROM users WHERE name='{name}'\")"
)
store_result(gm, bad_sql)
print(f"  Stored as: {'PASSED' if bad_sql['attempt'].is_successful else 'FAILED'}")

# Store a CLEAR GOOD SQL example
print("\n[Storing] GOOD SQL query (parameterized):")
good_sql = judge(
    "Write a function to query user from database",
    "def get_user(name): return db.execute('SELECT * FROM users WHERE name=?', (name,))"
)
store_result(gm, good_sql)
print(f"  Stored as: {'PASSED' if good_sql['attempt'].is_successful else 'FAILED'}")


# ============================================================
# ROUND 2: Test WITHOUT memory (stateless)
# ============================================================
print("\n" + "=" * 70)
print("ROUND 2: Testing WITHOUT Memory (Stateless)")
print("=" * 70)

# Test 1: Slightly different email validation (still bad)
print("\n" + "-" * 50)
print("TEST 1: Email validation - checks @ and . (should FAIL)")
print("-" * 50)
test1_code = """
def check_email(email):
    if '@' not in email:
        return False
    if '.' not in email:
        return False
    return True
"""
result1_no_mem = judge(
    "Write a function to validate email addresses",
    test1_code
)
print_result(result1_no_mem, "WITHOUT Memory:")

result1 = judge_with_memory(
    "Write a function to validate email addresses",
    test1_code,
    gm
)
print_result(result1, "WITH Memory:")


# Test 2: Another bad email validation variant
print("\n" + "-" * 50)
print("TEST 2: Email validation - split by @ (should FAIL)")
print("-" * 50)
test2_code = """
def is_valid_email(email):
    parts = email.split('@')
    return len(parts) == 2
"""
result2_no_mem = judge(
    "Write a function to validate email addresses",
    test2_code
)
print_result(result2_no_mem, "WITHOUT Memory:")

result2 = judge_with_memory(
    "Write a function to validate email addresses",
    test2_code,
    gm
)
print_result(result2, "WITH Memory:")


# Test 3: Good email validation (different style)
print("\n" + "-" * 50)
print("TEST 3: Email validation - proper regex (should PASS)")
print("-" * 50)
test3_code = """
import re

def validate_email(email):
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
"""
result3_no_mem = judge(
    "Write a function to validate email addresses",
    test3_code
)
print_result(result3_no_mem, "WITHOUT Memory:")

result3 = judge_with_memory(
    "Write a function to validate email addresses",
    test3_code,
    gm
)
print_result(result3, "WITH Memory:")


# Test 4: SQL with string concatenation (should fail)
print("\n" + "-" * 50)
print("TEST 4: SQL query - string concat (should FAIL)")
print("-" * 50)
test4_code = """
def fetch_user(user_id):
    sql = "SELECT * FROM users WHERE id = " + str(user_id)
    return database.query(sql)
"""
result4_no_mem = judge(
    "Write a function to get user by id from database",
    test4_code
)
print_result(result4_no_mem, "WITHOUT Memory:")

result4 = judge_with_memory(
    "Write a function to get user by id from database",
    test4_code,
    gm
)
print_result(result4, "WITH Memory:")


# Test 5: SQL with parameterized query (should pass)
print("\n" + "-" * 50)
print("TEST 5: SQL query - parameterized (should PASS)")
print("-" * 50)
test5_code = """
def fetch_user(user_id):
    return db.execute("SELECT * FROM users WHERE id = %s", (user_id,))
"""
result5_no_mem = judge(
    "Write a function to get user by id from database",
    test5_code
)
print_result(result5_no_mem, "WITHOUT Memory:")

result5 = judge_with_memory(
    "Write a function to get user by id from database",
    test5_code,
    gm
)
print_result(result5, "WITH Memory:")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: WITHOUT Memory vs WITH Memory")
print("=" * 70)

results = [
    ("Email: @ and . check", result1_no_mem, result1, False),
    ("Email: split by @", result2_no_mem, result2, False),
    ("Email: proper regex", result3_no_mem, result3, True),
    ("SQL: string concat", result4_no_mem, result4, False),
    ("SQL: parameterized", result5_no_mem, result5, True),
]

print("\n{:<25} {:>10} {:>12} {:>10}".format("Test", "Expected", "No Memory", "With Memory"))
print("-" * 60)

correct_no_mem = 0
correct_with_mem = 0

for name, res_no_mem, res_with_mem, expected_pass in results:
    expected = "PASS" if expected_pass else "FAIL"
    no_mem = "PASS" if res_no_mem['attempt'].is_successful else "FAIL"
    with_mem = "PASS" if res_with_mem['attempt'].is_successful else "FAIL"

    no_mem_match = "✓" if (res_no_mem['attempt'].is_successful == expected_pass) else "✗"
    with_mem_match = "✓" if (res_with_mem['attempt'].is_successful == expected_pass) else "✗"

    print(f"{name:<25} {expected:>10} {no_mem_match} {no_mem:>10} {with_mem_match} {with_mem:>10}")

    if res_no_mem['attempt'].is_successful == expected_pass:
        correct_no_mem += 1
    if res_with_mem['attempt'].is_successful == expected_pass:
        correct_with_mem += 1

print("-" * 60)
print(f"{'Accuracy':<25} {'':>10} {correct_no_mem}/{len(results):>10} {correct_with_mem}/{len(results):>10}")

gm.close()
print("\nDONE!")

"""
Test: Hard Edge Cases

These are tricky cases where memory should help:
- Subtle security issues
- Borderline implementations
- Cases that look good but aren't (or vice versa)
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


# ============================================================
# HARD EDGE CASES
# ============================================================
edge_cases = [
    # --- XSS Cases ---
    ("Write a function to display user comment on webpage",
     """
def show_comment(comment):
    return f"<div>{comment}</div>"
""",
     False,
     "XSS #1: Direct HTML injection"),

    ("Write a function to display user comment on webpage",
     """
import html
def show_comment(comment):
    return f"<div>{html.escape(comment)}</div>"
""",
     True,
     "XSS #2: HTML escaped"),

    ("Write a function to display user comment on webpage",
     """
def show_comment(comment):
    # Remove script tags
    clean = comment.replace("<script>", "").replace("</script>", "")
    return f"<div>{clean}</div>"
""",
     False,
     "XSS #3: Weak sanitization (bypass: <scr<script>ipt>)"),

    # --- Password Cases ---
    ("Write a function to store user password",
     """
def store_password(password):
    return password  # Store as-is
""",
     False,
     "Password #1: Plain text storage"),

    ("Write a function to store user password",
     """
import hashlib
def store_password(password):
    return hashlib.md5(password.encode()).hexdigest()
""",
     False,
     "Password #2: MD5 (weak, no salt)"),

    ("Write a function to store user password",
     """
import hashlib
import os
def store_password(password):
    salt = os.urandom(32)
    hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt + hash
""",
     True,
     "Password #3: PBKDF2 with salt"),

    # --- Path Traversal ---
    ("Write a function to read file from uploads directory",
     """
def read_upload(filename):
    return open(f"/uploads/{filename}").read()
""",
     False,
     "Path #1: Traversal vulnerable (../../../etc/passwd)"),

    ("Write a function to read file from uploads directory",
     """
import os
def read_upload(filename):
    if ".." in filename:
        raise ValueError("Invalid filename")
    return open(f"/uploads/{filename}").read()
""",
     False,
     "Path #2: Weak check (....// bypass)"),

    ("Write a function to read file from uploads directory",
     """
import os
def read_upload(filename):
    base = "/uploads"
    path = os.path.normpath(os.path.join(base, filename))
    if not path.startswith(base):
        raise ValueError("Invalid path")
    return open(path).read()
""",
     True,
     "Path #3: Proper path validation"),

    # --- Rate Limiting ---
    ("Write a function to check if user can make API request",
     """
def can_request(user_id):
    return True  # Allow all
""",
     False,
     "Rate #1: No rate limiting"),

    ("Write a function to check if user can make API request",
     """
request_count = {}
def can_request(user_id):
    count = request_count.get(user_id, 0)
    if count >= 100:
        return False
    request_count[user_id] = count + 1
    return True
""",
     False,
     "Rate #2: No time window (counter never resets)"),

    ("Write a function to check if user can make API request",
     """
import time
request_times = {}
def can_request(user_id, limit=100, window=60):
    now = time.time()
    times = request_times.get(user_id, [])
    times = [t for t in times if now - t < window]
    if len(times) >= limit:
        return False
    times.append(now)
    request_times[user_id] = times
    return True
""",
     True,
     "Rate #3: Sliding window"),

    # --- Subtle SQL Cases ---
    ("Write a function to search products by name",
     """
def search(name):
    return db.execute(f"SELECT * FROM products WHERE name LIKE '%{name}%'")
""",
     False,
     "SQL #1: LIKE injection"),

    ("Write a function to search products by name",
     """
def search(name):
    return db.execute("SELECT * FROM products WHERE name LIKE ?", (f"%{name}%",))
""",
     True,
     "SQL #2: Parameterized LIKE"),

    # --- Integer Overflow ---
    ("Write a function to calculate total price",
     """
def total_price(quantity, unit_price):
    return quantity * unit_price
""",
     False,
     "Overflow #1: No bounds check (huge quantity = negative total)"),

    ("Write a function to calculate total price",
     """
def total_price(quantity, unit_price):
    if quantity < 0 or quantity > 10000:
        raise ValueError("Invalid quantity")
    if unit_price < 0:
        raise ValueError("Invalid price")
    return quantity * unit_price
""",
     True,
     "Overflow #2: Bounds validated"),
]


# ============================================================
# RUN 1: Without Memory
# ============================================================
print("=" * 70)
print("RUN 1: WITHOUT MEMORY (Stateless)")
print("=" * 70)

results_no_mem = []

for task, code, expected, label in edge_cases:
    result = judge(task, code)
    actual = result['attempt'].is_successful
    correct = actual == expected
    results_no_mem.append((label, expected, actual, correct))

    match = "✓" if correct else "✗"
    status = "PASS" if actual else "FAIL"
    print(f"  {match} {label}: {status}")


# ============================================================
# RUN 2: With Organic Memory
# ============================================================
print("\n" + "=" * 70)
print("RUN 2: WITH ORGANIC MEMORY")
print("=" * 70)

results_with_mem = []
memory_size = 0

for task, code, expected, label in edge_cases:
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

    store_result(gm, result)
    memory_size += 1


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: EDGE CASES")
print("=" * 70)

print("\n{:<45} {:>8} {:>10} {:>10}".format(
    "Test", "Expected", "No Mem", "With Mem"))
print("-" * 75)

correct_no_mem = 0
correct_with_mem = 0

for i in range(len(edge_cases)):
    label, expected, actual_no, correct_no = results_no_mem[i]
    _, _, actual_with, correct_with, _ = results_with_mem[i]

    exp_str = "PASS" if expected else "FAIL"
    no_str = "✓" if correct_no else "✗"
    with_str = "✓" if correct_with else "✗"

    print(f"{label:<45} {exp_str:>8} {no_str:>10} {with_str:>10}")

    if correct_no:
        correct_no_mem += 1
    if correct_with:
        correct_with_mem += 1

print("-" * 75)
total = len(edge_cases)
print(f"{'ACCURACY':<45} {'':>8} {correct_no_mem}/{total} ({100*correct_no_mem//total}%) {correct_with_mem}/{total} ({100*correct_with_mem//total}%)")

# Category breakdown
print("\n" + "=" * 70)
print("BY CATEGORY")
print("=" * 70)

categories = {}
for i, (task, code, expected, label) in enumerate(edge_cases):
    cat = label.split("#")[0].strip()
    if cat not in categories:
        categories[cat] = {"no_mem": 0, "with_mem": 0, "total": 0}
    categories[cat]["total"] += 1
    if results_no_mem[i][3]:
        categories[cat]["no_mem"] += 1
    if results_with_mem[i][3]:
        categories[cat]["with_mem"] += 1

for cat, stats in categories.items():
    print(f"  {cat:<15}: No Mem {stats['no_mem']}/{stats['total']}, With Mem {stats['with_mem']}/{stats['total']}")

gm.close()
print("\nDONE!")

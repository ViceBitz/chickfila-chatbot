import json
import time
from pathlib import Path

from django.core.management.base import BaseCommand

from chat.views import reload_knowledge_base, create_rag_chain_with_sources

TEST_CASES_FILE = Path(__file__).resolve().parent.parent.parent.parent / "data" / "test_cases.json"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
BOLD = "\033[1m"
RESET = "\033[0m"


class Command(BaseCommand):
    help = "Run accuracy evaluation against known test questions"

    def add_arguments(self, parser):
        parser.add_argument(
            "--topic",
            type=str,
            default=None,
            help="Filter by topic (nutrition, menu, location)",
        )

    def handle(self, *args, **options):
        topic_filter = options["topic"]

        test_cases = json.loads(TEST_CASES_FILE.read_text())
        if topic_filter:
            test_cases = [t for t in test_cases if t.get("topic") == topic_filter]

        if not test_cases:
            self.stdout.write(self.style.WARNING("No test cases found."))
            return

        self.stdout.write(f"\n{BOLD}Loading knowledge base...{RESET}")
        reload_knowledge_base()
        self.stdout.write(f"{BOLD}Running {len(test_cases)} test(s){f' [{topic_filter}]' if topic_filter else ''}...{RESET}\n")

        passed = 0
        results = []

        for case in test_cases:
            question = case["question"]
            expected = [kw.lower() for kw in case["expected_keywords"]]

            start = time.time()
            try:
                result = create_rag_chain_with_sources(question)
                answer = result["answer"].lower()
                elapsed = (time.time() - start) * 1000

                missing = [kw for kw in expected if kw not in answer]
                ok = len(missing) == 0
            except Exception as e:
                answer = f"ERROR: {e}"
                elapsed = (time.time() - start) * 1000
                missing = expected
                ok = False

            status = PASS if ok else FAIL
            passed += ok
            results.append((case, ok, answer, missing, elapsed))

            self.stdout.write(f"  [{status}] {case['id']} — {question}")
            if not ok:
                self.stdout.write(f"         Missing keywords: {missing}")
            self.stdout.write(f"         Answer: {result.get('answer', answer)[:1000] if isinstance(result, dict) else answer[:1000]}")
            self.stdout.write(f"         ({elapsed:.0f}ms)\n")

        total = len(test_cases)
        pct = (passed / total) * 100
        bar = ("█" * passed) + ("░" * (total - passed))
        colour = self.style.SUCCESS if pct == 100 else (self.style.WARNING if pct >= 75 else self.style.ERROR)

        self.stdout.write(f"{BOLD}Results: {passed}/{total} passed ({pct:.0f}%){RESET}")
        self.stdout.write(colour(f"  [{bar}]"))

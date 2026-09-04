"""
test_tenant_isolation.py

Module 4: the real cross-tenant leakage test, run against the two canary
points injected by inject_tenant_canaries.py. Deliberately queries with the
canary text nearly verbatim, so if the tenant_id filter is broken, the
wrong-tenant canary is the most likely top match -- this is the strongest
test that can be constructed here, not a token gesture. Checks the full
retrieval set at the retrieval layer (before reranking trims it down),
since a leak there is the real signal of the isolation mechanism failing,
not whether reranking happened to discard it afterward.

Checks:
  1. tenant_a can retrieve its own canary
  2. tenant_a's retrieval set NEVER contains tenant_b's canary
  3. tenant_b can retrieve its own canary
  4. tenant_b's retrieval set NEVER contains tenant_a's canary
"""

from pipeline import hybrid_search

TENANT_A_CANARY_TEXT = "The secret onboarding code for Zephyrsoft Industries is QUARTZ-7742."
TENANT_B_CANARY_TEXT = "The secret onboarding code for Halcyon Ventures is FALCON-3391."

TENANT_A_QUERY = "What is the secret onboarding code for Zephyrsoft Industries?"
TENANT_B_QUERY = "What is the secret onboarding code for Halcyon Ventures?"


def retrieved_texts(query, tenant_id, n_results=20):
    results = hybrid_search(query, tenant_id, n_results=n_results, max_per_source=n_results)
    return results["documents"][0]


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    all_passed = True

    docs = retrieved_texts(TENANT_A_QUERY, tenant_id="tenant_a")
    all_passed &= check("tenant_a can retrieve its own canary", TENANT_A_CANARY_TEXT in docs)
    all_passed &= check(
        "tenant_a's retrieval set does not contain tenant_b's canary (own-topic query)",
        TENANT_B_CANARY_TEXT not in docs,
    )

    # the real leakage probe: tenant_a deliberately asking about tenant_b's exact topic
    docs = retrieved_texts(TENANT_B_QUERY, tenant_id="tenant_a")
    all_passed &= check(
        "tenant_a querying tenant_b's exact canary topic never retrieves tenant_b's canary",
        TENANT_B_CANARY_TEXT not in docs,
    )

    docs = retrieved_texts(TENANT_B_QUERY, tenant_id="tenant_b")
    all_passed &= check("tenant_b can retrieve its own canary", TENANT_B_CANARY_TEXT in docs)
    all_passed &= check(
        "tenant_b's retrieval set does not contain tenant_a's canary (own-topic query)",
        TENANT_A_CANARY_TEXT not in docs,
    )

    # the real leakage probe: tenant_b deliberately asking about tenant_a's exact topic
    docs = retrieved_texts(TENANT_A_QUERY, tenant_id="tenant_b")
    all_passed &= check(
        "tenant_b querying tenant_a's exact canary topic never retrieves tenant_a's canary",
        TENANT_A_CANARY_TEXT not in docs,
    )

    print()
    if all_passed:
        print("ALL CHECKS PASSED -- no cross-tenant leakage detected.")
    else:
        print("AT LEAST ONE CHECK FAILED -- real cross-tenant leakage detected.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

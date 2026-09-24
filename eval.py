"""
eval.py
-------
Evaluates RETRIEVAL quality (and optionally GENERATION quality) for your multi-document
RAG pipeline across all ingested documents in the vectorstore.

Documents Evaluated:
- employee_handbook.txt (HR policies, working hours, benefits, PTO, leave)
- it_policy.txt (security, passwords, device encryption, VPN, incident reporting)
- engineering_guide.txt (PR reviews, testing coverage, deployments, on-call, tech stack)
- financial_report.txt (revenue, net profit, segment breakdown, cash position, metrics)
- policy.txt (customer privacy, GDPR rights, retention periods, international transfers)

Key Metrics Reported:
1. Hit Rate@k: Percentage of queries where at least one chunk from the expected
   document appears in the top-k retrieved results.
2. Precision@k: Percentage of retrieved chunks that belong to the expected document.
3. MRR (Mean Reciprocal Rank): Rewards retrieving the relevant chunk earlier
   (rank 1 is scored 1.0, rank 2 is 0.5, rank 3 is 0.33, etc.).
4. Keyword Coverage: Percentage of required factual keywords found in the retrieved text.
5. Distractor Analysis: Shows which other documents were retrieved alongside the target document.
6. Out-of-Scope Handling: Tests how the pipeline responds to queries not covered by any document.

Usage & Commands:
Run this script from the terminal to evaluate the pipeline.
Available arguments:
  --k <int>       : Number of chunks to retrieve (default: 4). Example: python eval.py --k 6
  --verbose       : Print detailed logs for every query (retrieved chunks, distractors).
  --hard          : Run the HARD_EVAL_SET (challenging queries) instead of standard set.
  --set <name>    : Specify an eval set to run ('standard', 'hard', 'all'). Example: --set all

Examples:
    python eval.py
    python eval.py --k 5
    python eval.py --with-generation
    python eval.py --verbose
    python eval.py --hard
    python eval.py --set all/hard/standard
"""

import argparse
import os
import sys
from collections import defaultdict
from rag_chain import load_retriever, VECTORSTORE_DIR

# ---------------------------------------------------------------------------
# Comprehensive Multi-Document Evaluation Set
# ---------------------------------------------------------------------------
EVAL_SET = [
    # =======================================================================
    # 1. EMPLOYEE HANDBOOK (HR, PTO, Benefits, Hours)
    # =======================================================================
    {
        "question": "How many days of paid time off (PTO) do employees get per year, and how much can be carried over?",
        "expected_sources": ["employee_handbook.txt"],
        "expected_keywords": ["22 days", "5 days", "carried over", "accrued"],
    },
    {
        "question": "How many days per week can employees work remotely, and what are the core working hours?",
        "expected_sources": ["employee_handbook.txt"],
        "expected_keywords": ["3 days", "11:00 am", "4:00 pm", "slack"],
    },
    {
        "question": "What is the parental leave duration for primary and secondary caregivers?",
        "expected_sources": ["employee_handbook.txt"],
        "expected_keywords": ["16 weeks", "6 weeks", "parental leave"],
    },
    {
        "question": "What is the company retirement 401k match and annual learning development budget?",
        "expected_sources": ["employee_handbook.txt"],
        "expected_keywords": ["6%", "1,500", "retirement"],
    },

    # =======================================================================
    # 2. IT SECURITY POLICY (Passwords, Devices, VPN, Incident Reporting)
    # =======================================================================
    {
        "question": "What are the password requirements and how often must passwords be changed?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["14 characters", "90 days", "mfa"],
    },
    {
        "question": "Within what time frame must a lost or stolen company device be reported to IT Security?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["2 hours", "lost or stolen"],
    },
    {
        "question": "What VPN is required for remote access and what is the rule for public Wi-Fi?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["globalprotect", "vpn", "public wi-fi"],
    },
    {
        "question": "What endpoint protection software and encryption must be enabled on company laptops?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["crowdstrike falcon", "full-disk encryption"],
    },

    # =======================================================================
    # 3. ENGINEERING PRACTICES GUIDE (Code Reviews, CI/CD, On-Call, Tech Stack)
    # =======================================================================
    {
        "question": "How many reviewer approvals are required for a pull request and what is the target size limit?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["2 reviewers", "400 lines", "senior engineer"],
    },
    {
        "question": "What is the minimum unit test coverage required by the CI pipeline before merging?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["80%", "unit test", "coverage"],
    },
    {
        "question": "How does the production deployment canary release process work?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["5%", "30 minutes", "canary"],
    },
    {
        "question": "What is the target response time for a Sev-1 critical incident page during on-call rotation?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["10 minutes", "sev-1", "pagerduty"],
    },
    {
        "question": "What tech stack and database standards are specified for backend services and caching?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["python", "fastapi", "postgresql", "redis", "kafka"],
    },

    # =======================================================================
    # 4. FINANCIAL REPORT Q3 FY2026 (Revenue, Profit, Margins, Cash Position)
    # =======================================================================
    {
        "question": "What was NovaTech's total revenue, net profit, and YoY revenue growth in Q3 FY2026?",
        "expected_sources": ["financial_report.txt"],
        "expected_keywords": ["48.2 million", "6.7 million", "22%"],
    },
    {
        "question": "What percentage of total revenue came from Enterprise Cloud Services versus SMB licensing?",
        "expected_sources": ["financial_report.txt"],
        "expected_keywords": ["57%", "27.4 million", "25%", "12.1 million"],
    },
    {
        "question": "What was the company's cash position, debt status, and free cash flow as of September 30, 2026?",
        "expected_sources": ["financial_report.txt"],
        "expected_keywords": ["84.3 million", "9.1 million", "long-term debt"],
    },
    {
        "question": "What were the customer churn rate and net revenue retention metrics reported in Q3?",
        "expected_sources": ["financial_report.txt"],
        "expected_keywords": ["2.1%", "118%", "churn"],
    },

    # =======================================================================
    # 5. PRIVACY POLICY (GDPR, Data Retention, International Transfers)
    # =======================================================================
    {
        "question": "How long are transaction records retained for legal, tax, and accounting requirements?",
        "expected_sources": ["policy.txt"],
        "expected_keywords": ["7 years", "tax", "accounting"],
    },
    {
        "question": "Do you sell personal customer data to third parties for marketing purposes?",
        "expected_sources": ["policy.txt"],
        "expected_keywords": ["not sell", "third parties", "marketing"],
    },
    {
        "question": "What is the policy regarding children under 13 years of age?",
        "expected_sources": ["policy.txt"],
        "expected_keywords": ["under 13", "knowingly collect"],
    },

    # =======================================================================
    # 6. NEGATIVE / OUT-OF-SCOPE QUERIES (Should NOT match any corporate document)
    # =======================================================================
    {
        "question": "What is the warranty and return policy for purchasing a replacement laptop battery?",
        "expected_sources": [],
        "expected_keywords": [],
    },
    {
        "question": "Who won the men's singles tennis championship at Wimbledon in 2024?",
        "expected_sources": [],
        "expected_keywords": [],
    },
    {
        "question": "What are the daily lunch specials and vegetarian meal options in the cafeteria?",
        "expected_sources": [],
        "expected_keywords": [],
    },
]

STANDARD_EVAL_SET = EVAL_SET

# ---------------------------------------------------------------------------
# Hard / Adversarial Evaluation Set: Cross-Document Competition & Paraphrasing
# ---------------------------------------------------------------------------
HARD_EVAL_SET = [
    # --- Cross-Document Ambiguity / Competition ---
    {
        "question": "How fast do we have to respond to an incident?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["30 minutes", "it security"],
    },
    {
        "question": "When do promotion and salary review decisions happen?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["march", "september", "promotion committee"],
    },
    {
        "question": "What is the procedure when an employee leaves or resigns from the company?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["4 hours", "offboarding", "return"],
    },
    {
        "question": "Where can customer PII and sensitive payment records be stored?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["restricted", "encrypted"],
    },

    # --- Paraphrased / Zero-Keyword Queries ---
    {
        "question": "Can I take Monday off if production broke overnight and woke me up twice?",
        "expected_sources": ["engineering_guide.txt"],
        "expected_keywords": ["monday off", "woken up"],
    },
    {
        "question": "Is my daughter or spouse covered if they visit the hospital?",
        "expected_sources": ["employee_handbook.txt"],
        "expected_keywords": ["dependents", "50%"],
    },
    {
        "question": "Can I keep my company MacBook or laptop after my last day?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["returned", "5 business days"],
    },
    {
        "question": "Which cloud provider contract was renegotiated in summer 2026 to cut hosting expenses?",
        "expected_sources": ["financial_report.txt"],
        "expected_keywords": ["aws", "june 2026"],
    },
    {
        "question": "Can a newly hired employee be dismissed with only one week notice?",
        "expected_sources": ["employee_handbook.txt"],
        "expected_keywords": ["probationary", "1 week"],
    },
    {
        "question": "What are the rules regarding watching Netflix or torrenting in the office?",
        "expected_sources": ["it_policy.txt"],
        "expected_keywords": ["streaming", "torrenting"],
    },
]


def evaluate(eval_set=EVAL_SET, k=3, with_generation=False, verbose=False):
    """
    Runs retrieval (and optionally generation) evaluation across the eval set.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        print(f"\n❌ Error: Vectorstore directory '{VECTORSTORE_DIR}' not found.")
        print("Please ingest your documents first using ingest.py or the Streamlit app.")
        sys.exit(1)

    print(f"\nInitializing retriever with k={k}...")
    retriever = load_retriever(k=k)

    ask_fn = None
    if with_generation:
        print("Loading generation pipeline (rag_chain.ask)...")
        from rag_chain import ask
        ask_fn = ask

    results = []

    for item in eval_set:
        question = item["question"]
        expected_sources = item.get("expected_sources", [])
        expected_keywords = item.get("expected_keywords", [])
        is_negative = len(expected_sources) == 0

        docs = retriever.invoke(question)
        retrieved_sources = [doc.metadata.get("source", "") for doc in docs]
        retrieved_filenames = [os.path.basename(s) for s in retrieved_sources]
        retrieved_texts = [doc.page_content for doc in docs]
        combined_text = " ".join(t.lower() for t in retrieved_texts)

        # --- Retrieval Hit & Rank Logic ---
        hit = False
        rank = None
        precision = None

        if not is_negative:
            num_relevant = sum(1 for fn in retrieved_filenames if any(exp in fn for exp in expected_sources))
            precision = num_relevant / len(retrieved_filenames) if retrieved_filenames else 0.0

            for i, src in enumerate(retrieved_sources):
                if any(exp in src for exp in expected_sources):
                    hit = True
                    rank = i + 1  # 1-indexed rank
                    break
        else:
            hit = None
            rank = None
            precision = None

        # --- Keyword Coverage (for positive queries) ---
        found_keywords = [kw for kw in expected_keywords if kw.lower() in combined_text]
        missing_keywords = [kw for kw in expected_keywords if kw.lower() not in combined_text]
        keyword_coverage = (
            len(found_keywords) / len(expected_keywords) if expected_keywords else None
        )

        # --- Distractors ---
        # Chunks retrieved that do NOT match the expected document
        distractors = [fn for fn in retrieved_filenames if not any(exp in fn for exp in expected_sources)]

        # --- Optional Generation ---
        generation_answer = None
        if ask_fn:
            gen_res = ask_fn(question)
            generation_answer = gen_res.get("answer", "")

        results.append({
            "question": question,
            "is_negative": is_negative,
            "hit": hit,
            "rank": rank,
            "precision": precision,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "retrieved_filenames": retrieved_filenames,
            "distractors": distractors,
            "retrieved_texts": retrieved_texts,
            "expected_keywords": expected_keywords,
            "found_keywords": found_keywords,
            "missing_keywords": missing_keywords,
            "keyword_coverage": keyword_coverage,
            "generation_answer": generation_answer,
        })

    return results


def summarize(results, k=3, verbose=False):
    """
    Displays granular per-document results, distractor analysis, and summary metrics.
    """
    positive_results = [r for r in results if not r["is_negative"]]
    negative_results = [r for r in results if r["is_negative"]]

    n_pos = len(positive_results)
    hits = [r for r in positive_results if r["hit"]]
    hit_rate = len(hits) / n_pos if n_pos else 0

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = [1 / r["rank"] for r in positive_results if r["hit"] and r["rank"]]
    mrr = sum(reciprocal_ranks) / n_pos if n_pos else 0

    # Average Keyword Coverage
    coverages = [r["keyword_coverage"] for r in positive_results if r["keyword_coverage"] is not None]
    avg_keyword_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    # Average Precision@k
    precisions = [r["precision"] for r in positive_results if r["precision"] is not None]
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    print("\n" + "=" * 75)
    print(f"MULTI-DOCUMENT RETRIEVAL EVALUATION RESULTS (Top-k = {k})")
    print("=" * 75)

    # Breakdown by expected document
    doc_stats = defaultdict(lambda: {"total": 0, "hits": 0, "ranks": []})
    for r in positive_results:
        doc_key = r["expected_sources"][0] if r["expected_sources"] else "Unknown"
        doc_stats[doc_key]["total"] += 1
        if r["hit"]:
            doc_stats[doc_key]["hits"] += 1
            doc_stats[doc_key]["ranks"].append(r["rank"])

    print("\n--- IN-DOMAIN RETRIEVAL RESULTS ---")
    for r in positive_results:
        status = "✅ HIT " if r["hit"] else "❌ MISS"
        rank_str = f"Rank {r['rank']}" if r["rank"] else "No hit"
        cov_pct = f"{r['keyword_coverage']:.0%}" if r["keyword_coverage"] is not None else "N/A"
        print(f"\n{status} | {rank_str} | Keyword Coverage: {cov_pct}")
        print(f"Q: {r['question']}")
        print(f"   Expected:  {r['expected_sources']}")
        print(f"   Retrieved: {r['retrieved_filenames']}")

        if r["distractors"]:
            print(f"   Distractor chunks: {r['distractors']}")
        if r["found_keywords"]:
            print(f"   Matched keywords:  {r['found_keywords']}")
        if r["missing_keywords"]:
            print(f"   Missing keywords:  {r['missing_keywords']}")

        if r["generation_answer"]:
            preview = r['generation_answer'].replace("\n", " ")[:200]
            print(f"   🤖 Answer: {preview}...")

        if verbose:
            print("   Top retrieved snippet:")
            snippet = r['retrieved_texts'][0].replace('\n', ' ')[:140]
            print(f"   > \"{snippet}...\"")

    if negative_results:
        print("\n--- OUT-OF-SCOPE / NEGATIVE QUERIES ---")
        for r in negative_results:
            print(f"\n🔍 Negative Test | Q: {r['question']}")
            print(f"   Nearest retrieved chunks: {r['retrieved_filenames']}")
            if r["generation_answer"]:
                refusal_keywords = [
                    "don't have enough information",
                    "not mentioned",
                    "not provided",
                    "does not contain",
                ]
                correct_refusal = any(kw in r["generation_answer"].lower() for kw in refusal_keywords)
                refusal_badge = "✅ Refused correctly" if correct_refusal else "⚠️ Check hallucination"
                print(f"   {refusal_badge} | 🤖 Answer: {r['generation_answer'][:150]}")
            else:
                print("   Nearest-neighbor chunks returned. Run with --with-generation to test LLM refusal.")

    print("\n" + "=" * 75)
    print("PER-DOCUMENT RETRIEVAL PERFORMANCE")
    print("=" * 75)
    for doc_name, s in sorted(doc_stats.items()):
        doc_hit_rate = s["hits"] / s["total"] if s["total"] else 0
        avg_rank = sum(s["ranks"]) / len(s["ranks"]) if s["ranks"] else 0
        print(f"📄 {doc_name:<26} : Hit Rate = {doc_hit_rate:.1%} ({s['hits']}/{s['total']}), Avg Rank = {avg_rank:.2f}")

    print("\n" + "=" * 75)
    print("OVERALL SUMMARY METRICS")
    print("=" * 75)
    print(f"Total In-Domain Questions:  {n_pos}")
    print(f"Hit Rate@{k}:                {hit_rate:.2%} ({len(hits)}/{n_pos} found)")
    print(f"Precision@{k}:               {avg_precision:.2%}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.3f}")
    print(f"Average Keyword Coverage:   {avg_keyword_coverage:.2%}")
    if negative_results:
        print(f"Negative / Out-of-Scope:    {len(negative_results)} tested")
    print("=" * 75)

    if hit_rate < 0.8:
        print("\n⚠️  Hit rate is below 80% -- suggestions:")
        print("  - Increase k: python eval.py --k 5")
        print("  - Check chunk_size and chunk_overlap in ingest.py")
        print("  - Inspect distractor chunks to see what is confusing the retriever")
    elif mrr < 0.7:
        print("\n💡 Tip: Chunks are retrieved but often at rank 2 or 3 rather than rank 1.")
        print("  - Consider adding a reranker (e.g., Cohere/BGE reranker) or optimizing chunk overlap.")
    else:
        print("\n🎉 Outstanding multi-document retrieval performance across the corpus!")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-document RAG pipeline retrieval and generation.")
    parser.add_argument(
        "-k", "--k", type=int, default=3, help="Number of chunks to retrieve per query (default: 3)"
    )
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Also run generation via ChatGroq and display answers/citations",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display top retrieved text snippets for each query",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Run the hard/adversarial eval set (cross-document ambiguity & zero-keyword queries)",
    )
    parser.add_argument(
        "--set",
        choices=["standard", "hard", "all"],
        default="standard",
        help="Choose evaluation set to run: standard (default), hard, or all",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Determine which eval set to run
    if args.hard or args.set == "hard":
        active_eval_set = HARD_EVAL_SET
        set_name = "HARD / ADVERSARIAL EVALUATION SET"
    elif args.set == "all":
        active_eval_set = STANDARD_EVAL_SET + HARD_EVAL_SET
        set_name = "COMPLETE EVALUATION SET (Standard + Hard)"
    else:
        active_eval_set = STANDARD_EVAL_SET
        set_name = "STANDARD MULTI-DOCUMENT EVALUATION SET"

    print(f"\nRunning {set_name} ({len(active_eval_set)} total queries)")

    results = evaluate(
        eval_set=active_eval_set,
        k=args.k,
        with_generation=args.with_generation,
        verbose=args.verbose,
    )
    summarize(results, k=args.k, verbose=args.verbose)

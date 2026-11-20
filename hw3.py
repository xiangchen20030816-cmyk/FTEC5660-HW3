#!/usr/bin/env python3
"""FTEC5660 HW3 student starter: an agent that applies 3 of Fisher's 15 points using tools."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sqlite3
import threading
import zlib
from pathlib import Path
from typing import Any


MODEL_NAME = "deepseek-v4-flash"
INTERVIEW_LIMIT = 10
MAX_SQL_ROWS = 50

# Philip Fisher, "Common Stocks and Uncommon Profits", Part One, Chapter 3.
# For every point, PASS means the company looks good on it. You choose 3
# points and decide yourself what PASS means for each of them.
FISHER_POINTS = {
    1: "Does the company have products or services with sufficient market potential to make possible a sizable increase in sales for at least several years?",
    2: "Does the management have a determination to continue to develop products or processes that will still further increase total sales potentials when the growth potentials of currently attractive product lines have largely been exploited?",
    3: "How effective are the company's research and development efforts in relation to its size?",
    4: "Does the company have an above-average sales organization?",
    5: "Does the company have a worthwhile profit margin?",
    6: "What is the company doing to maintain or improve profit margins?",
    7: "Does the company have outstanding labor and personnel relations?",
    8: "Does the company have outstanding executive relations?",
    9: "Does the company have depth to its management?",
    10: "How good are the company's cost analysis and accounting controls?",
    11: "Are there other aspects of the business, somewhat peculiar to the industry involved, which will give the investor important clues as to how outstanding the company may be in relation to its competition?",
    12: "Does the company have a short-range or long-range outlook in regard to profits?",
    13: "In the foreseeable future will the growth of the company require sufficient equity financing so that the larger number of shares then outstanding will largely cancel the existing stockholders' benefit from this anticipated growth?",
    14: "Does the management talk freely to investors about its affairs when things are going well but 'clam up' when troubles and disappointments occur?",
    15: "Does the company have a management of unquestionable integrity?",
}
VERDICTS = ("PASS", "FAIL")


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Choose any 3 of the 15 points, e.g. CHOSEN_POINTS = [1, 7, 15]. Your agent
# gives a PASS or FAIL verdict on these 3 points only, with your own rule for each.
CHOSEN_POINTS: list[int] = [4, 7, 11]  ### YOUR CODE HERE


def build_agent() -> Any:
    """Create and return your model or agent once.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env. Each company has its own tools (see ``analyze_companies``), so
    you will probably create one tool-calling agent per company there.
    """
    ### YOUR CODE HERE
    from langchain_deepseek import ChatDeepSeek

    return ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=1500,
        timeout=60,
        max_retries=2,
    )
    return None

#使用工具分析公司
async def analyze_companies(agent: Any, companies: dict[str, list[Any]]) -> dict[str, dict[str, Any]]:
    """Run your agent and return its verdicts on CHOSEN_POINTS for every company.

    ``companies`` maps each company folder name to its four LangChain tools:

        list_documents()             names and dates of the company's documents
        read_document(name)          the text of one document
        sql_query(query)             one read-only SQLite query on company.db
        interview(role, topic)       one interview answer (INTERVIEW_LIMIT per company)

    Give the tools to a tool-calling agent, e.g.
    ``create_agent(model, tools=companies[name], system_prompt=...)``, or call
    them yourself with ``tool.invoke({...})``. The README explains the
    documents, the database tables and the interview roles and topics.

    Return one dictionary per company folder name, e.g. with
    CHOSEN_POINTS = [1, 7, 15]:

        {
            "hollins_group": {
                "points": {1: "PASS", 7: "FAIL", 15: "PASS"},
                "summary": "Two or three sentences explaining the verdicts.",
            },
            ...
        }

    Each chosen point needs a verdict ("PASS" or "FAIL"; True/False also
    works). Other points are ignored.

    Catch errors per company (e.g. a failed API call) and still return an
    entry for it: an exception here means no results.csv, which scores zero.
    You may analyse companies in parallel (e.g. ``asyncio.gather``).
    """
    ### YOUR CODE HERE
    from langchain_core.messages import HumanMessage, SystemMessage

    # (topic, primary interviewee, backup interviewee)
    interview_plan = {
        4: ("sales_team", "customer", "competitor"),
        7: ("working_conditions", "current_employee", "former_employee"),
        11: ("competitive_edge", "competitor", "customer"),
    }
    
    rules = {
        4: "PASS when sales staff are capable, consultative, responsive and valuable; "
           "FAIL for weak, passive or ineffective sales support.",
        7: "PASS for healthy morale, respectful treatment and good working conditions; "
           "FAIL for serious personnel, morale or working-condition problems.",
        11: "PASS for a meaningful competitive advantage such as differentiated "
            "expertise, products, service or loyalty; FAIL when readily replaceable "
            "or behind competitors.",
    }

    unknown_points = set(CHOSEN_POINTS) - set(interview_plan)
    if unknown_points:
        raise ValueError(
            f"This interview-only workflow does not cover points {sorted(unknown_points)}"
        )

    criteria = "\n".join(
        f"POINT {p}: {FISHER_POINTS[p]}\nRule: {rules[p]}"
        for p in CHOSEN_POINTS
    )
    format_example = {
        "points": {str(p): "PASS" for p in CHOSEN_POINTS},
        "summary": "2-3 short English sentences justifying the verdicts.",
        "needs_followup": [],
    }

    system_prompt = (
        "You are a careful analyst applying Philip Fisher's stock-selection principles.\n"
        "Use ONLY the interview evidence provided for the target company.\n"
        "Judge the target company, NOT the interviewee or the interviewee's employer.\n"
        "Consider each point independently. Handle negation and comparisons accurately.\n"
        "A clearly negative interview is sufficient evidence for FAIL; do NOT request "
        "more interviews just because the verdict is FAIL.\n"
        "Ask for another interview only when the existing quotation is missing, "
        "non-informative, internally ambiguous, or insufficient to determine a verdict.\n"
        "Do not fabricate facts. Make the best supported provisional PASS/FAIL judgment "
        "even when evidence is incomplete.\n\n"
        f"{criteria}\n\n"
        "Return ONLY JSON with fields: points, summary, needs_followup.\n"
        "points must map each chosen point number to PASS or FAIL; summary is 2-3 "
        "short English sentences; needs_followup is a list of point numbers needing "
        "a second interview, or [] if all are sufficiently supported.\n"
        "Example FORMAT only (not the correct answers):\n"
        f"{json.dumps(format_example)}"
    )

    def parse_answer(content: Any) -> dict[str, Any] | None:
        if not isinstance(content, str):
            return None
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end < start:
            return None
        try:
            obj = json.loads(content[start:end + 1])
            raw_points = obj["points"]
            points: dict[int, str] = {}
            for point in CHOSEN_POINTS:
                val = raw_points.get(str(point), raw_points.get(point))
                if not isinstance(val, str) or val.strip().upper() not in VERDICTS:
                    return None
                points[point] = val.strip().upper()
            summary = obj["summary"]
            if not isinstance(summary, str) or not summary.strip():
                return None
            needs_raw = obj.get("needs_followup", [])
            if not isinstance(needs_raw, list):
                return None
            needs_followup: set[int] = set()
            for val in needs_raw:
                if not isinstance(val, (int, str)) or str(val) not in {
                    str(p) for p in CHOSEN_POINTS
                }:
                    return None
                needs_followup.add(int(val))
            return {
                "points": points,
                "summary": summary.strip(),
                "needs_followup": needs_followup,
            }
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    def informative(quote: str) -> bool:
        if not isinstance(quote, str) or not quote.strip():
            return False
        low = quote.lower()
        return not any(s in low for s in (
            "interview limit reached", "i can't comment", "i can’t comment",
            "cannot comment", "no comment on that", "unknown role", "unknown topic",
            "interview error:",
        ))

    semaphore = asyncio.Semaphore(4)

    async def ask_model(name: str, evidence: dict[str, Any]):
        user_prompt = (
            f"Target company: {name}\n"
            "Assess these interview quotations. If multiple interviews cover a point, "
            "consider both.\n"
            + json.dumps(evidence, ensure_ascii=False)
        )
        async with semaphore:
            last_error = "unknown error"
            for attempt in range(3):
                try:
                    reply = await agent.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ])
                    result = parse_answer(reply.content)
                    if result is not None:
                        return result, ""
                    last_error = "invalid model JSON/verdicts"
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                if attempt < 2:
                    await asyncio.sleep(1 + 2 * attempt)
            return None, last_error

    async def analyze_one(name: str, tools: list[Any]):
        try:
            interview_tool = tools[3]
            evidence: dict[str, Any] = {}
            backup_used: set[int] = set()

            def obtain(point: int, backup: bool) -> None:
                topic, primary, alternate = interview_plan[point]
                role = alternate if backup else primary
                try:
                    quote = interview_tool.invoke({"role": role, "topic": topic})
                except Exception as exc:
                    quote = f"Interview error: {type(exc).__name__}: {exc}"
                evidence[str(point)]["interviews"].append({
                    "role": role, "quote": str(quote)
                })
                if backup:
                    backup_used.add(point)

            # Stage 1: one interview per point. Use a backup immediately if invalid.
            for point in CHOSEN_POINTS:
                evidence[str(point)] = {
                    "question": FISHER_POINTS[point], "interviews": []
                }
                obtain(point, backup=False)
                first_quote = evidence[str(point)]["interviews"][0]["quote"]
                if not informative(first_quote):
                    obtain(point, backup=True)

            result, error = await ask_model(name, evidence)
            if result is None:
                print(f"[ERROR] {name}: {error}", flush=True)
                return name, {"points": {}, "summary": f"Analysis failed: {error}"}

            # Stage 2: only interview another knowledgeable person when needed.
            followups = result["needs_followup"] - backup_used
            for point in CHOSEN_POINTS:
                if point in followups:
                    obtain(point, backup=True)

            if followups:
                second_result, error = await ask_model(name, evidence)
                if second_result is not None:
                    result = second_result
                else:
                    print(f"[WARN] {name}: follow-up analysis failed ({error}); "
                          "using first verdict", flush=True)

            final_result = {
                "points": result["points"], "summary": result["summary"]
            }
            count = sum(len(x["interviews"]) for x in evidence.values())
            print(f"[OK] {name}: {final_result['points']} "
                  f"(interviews={count})", flush=True)
            return name, final_result
        except Exception as exc:
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}", flush=True)
            return name, {
                "points": {}, "summary": "Evidence collection or analysis failed."
            }

    items = await asyncio.gather(
        *(analyze_one(name, tools) for name, tools in companies.items())
    )
    return dict(items)

    _ = agent
    return {name: {"points": {}, "summary": ""} for name in companies}


# Everything below is provided runner/scoring code. No edits are needed.
# Your code must read company data only through these tools: do not open
# company.db or interviews.dat directly.

INTERVIEWS_USED: dict[str, int] = {}


def company_folders(folder: Path) -> list[Path]:
    """Return one sub-folder per company directly inside *folder*, sorted by name."""
    return sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))


def make_tools(company_folder: Path) -> list[Any]:
    """Build the four tools for one company."""
    from langchain_core.tools import tool

    name = company_folder.name
    db_uri = f"{(company_folder / 'company.db').resolve().as_uri()}?mode=ro"
    answers = json.loads(zlib.decompress((company_folder / "interviews.dat").read_bytes()))
    documents = {p.name: p for p in sorted(company_folder.glob("*.md"))}
    lock = threading.Lock()
    INTERVIEWS_USED[name] = 0

    @tool
    def list_documents() -> str:
        """List this company's documents (annual letter, earnings calls, news) with their dates."""
        lines = []
        for doc_name, path in documents.items():
            text = path.read_text(encoding="utf-8").splitlines()
            dated = next((line for line in text if line.startswith("Date:")), "")
            lines.append(f"{doc_name}: {text[0].lstrip('# ')} ({dated or 'undated'})")
        return "\n".join(lines)

    @tool
    def read_document(name: str) -> str:
        """Return the full text of one document named by list_documents, e.g. 'annual_letter_2025.md'."""
        path = documents.get(name.strip())
        if path is None:
            return f"No document named {name!r}. Available: {', '.join(documents)}"
        return path.read_text(encoding="utf-8")

    @tool
    def sql_query(query: str) -> str:
        """Run one read-only SQLite query on this company's database and return at most 50 rows.

        Amounts are in USD thousands. Dates are 'YYYY-MM-DD', months 'YYYY-MM', quarters like '2025Q3'.
        Tables:
          company(name, industry)
          industry_medians(industry, revenue_growth_2y_pct, new_product_share_pct, new_product_sales_per_rd, operating_margin_pct)
          orders(order_id, order_date, product, customer, amount, note)
          products(product, launch_date)
          monthly_costs(month, cost_of_sales, selling_admin, research_development)
          quarterly_cash(quarter, operating_cash_flow, capex, cash_end)
          capex_plan(year, planned_capex)
          returns(month, units_shipped, units_returned)
          insider_trades(trade_date, person, title, side, shares, price)
        Use SUM, GROUP BY and WHERE to compute totals rather than reading raw rows.
        """
        try:
            con = sqlite3.connect(db_uri, uri=True)
            try:
                cursor = con.execute(query)
                columns = [c[0] for c in cursor.description or []]
                rows = cursor.fetchmany(MAX_SQL_ROWS + 1)
            finally:
                con.close()
        except sqlite3.Error as exc:
            return f"SQL error: {exc}"
        if not columns:
            return "Query returned no columns."
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)  # noqa: E731
        out = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:MAX_SQL_ROWS]]
        if len(rows) > MAX_SQL_ROWS:
            out.append(f"... more rows not shown (limit {MAX_SQL_ROWS}); aggregate or add LIMIT/OFFSET.")
        if not rows:
            out.append("(no rows)")
        return "\n".join(out)

    @tool
    def interview(role: str, topic: str) -> str:
        """Interview one person about the company (at most 10 interviews per company).

        role: customer, current_employee, former_employee, competitor or supplier.
        topic: sales_team, working_conditions, top_management, decision_making or competitive_edge.
        People who would not know about a topic say so; the README lists who knows what.
        """
        role, topic = role.strip().lower(), topic.strip().lower()
        if role not in answers:
            return f"Unknown role {role!r}. Roles: {', '.join(answers)}. (Not counted.)"
        if topic not in answers[role]:
            return f"Unknown topic {topic!r}. Topics: {', '.join(answers[role])}. (Not counted.)"
        with lock:
            if INTERVIEWS_USED[name] >= INTERVIEW_LIMIT:
                return f"Interview limit reached: {INTERVIEW_LIMIT} interviews per company."
            INTERVIEWS_USED[name] += 1
        return answers[role][topic]

    return [list_documents, read_document, sql_query, interview]


def normalize_verdict(value: Any) -> str | None:
    """Accept PASS/FAIL (any case) or a boolean."""
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    text = str(getattr(value, "content", value)).strip().upper()
    return text if text in VERDICTS else None


def check_chosen_points(chosen: Any) -> list[int]:
    """CHOSEN_POINTS must name 3 different points between 1 and 15."""
    try:
        points = sorted({int(p) for p in chosen})
    except (TypeError, ValueError):
        points = []
    if len(points) != 3 or len(chosen) != 3 or not all(p in FISHER_POINTS for p in points):
        raise SystemExit("Set CHOSEN_POINTS at the top of hw3.py to 3 different point numbers "
                         f"between 1 and 15, e.g. [1, 7, 15] (now: {chosen!r}).")
    return points


def normalize_points(points: Any, chosen: list[int]) -> dict[int, str | None]:
    """Map each chosen point number to PASS/FAIL, or None when missing/invalid."""
    points = points if isinstance(points, dict) else {}
    by_number: dict[int, Any] = {}
    for key, value in points.items():
        try:
            by_number[int(key)] = value
        except (TypeError, ValueError):
            continue
    return {p: normalize_verdict(by_number.get(p)) for p in chosen}


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read the expected 15 point verdicts per company."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def point_errors(points: dict[int, str | None], expected: dict[str, Any] | None) -> tuple[int, str]:
    """Count correct point verdicts and describe the wrong ones."""
    if expected is None:
        return 0, ""
    wrong = []
    for p in points:
        truth = expected["points"][str(p)]
        if points[p] != truth:
            wrong.append(f"{p}: expected {truth}, predicted {points[p] or 'missing'}")
    if all(v is None for v in points.values()):
        return 0, "all point verdicts missing"
    return len(points) - len(wrong), "; ".join(wrong)


def balanced_point_accuracy(
    analyses: dict[str, Any], truth: dict[str, dict[str, Any]], names: list[str], chosen: list[int]
) -> float:
    """Mean of the accuracy on true-PASS points and on true-FAIL points.

    Most verdicts are PASS, so answering PASS everywhere gets a high plain
    accuracy but only 50% balanced accuracy. Grading uses the balanced number.
    """
    hits = {"PASS": [0, 0], "FAIL": [0, 0]}
    for name in names:
        if name not in truth:
            continue
        entry = analyses.get(name)
        points = normalize_points(entry.get("points") if isinstance(entry, dict) else None, chosen)
        for p in chosen:
            expected = truth[name]["points"][str(p)]
            hits[expected][0] += points[p] == expected
            hits[expected][1] += 1
    rates = [h / n for h, n in hits.values() if n]
    return sum(rates) / len(rates) if rates else 0.0


def write_results(
    names: list[str], analyses: dict[str, Any], truth: dict[str, dict[str, Any]], chosen: list[int]
) -> tuple[Path, int]:
    """Write results.csv and return the number of correct point verdicts."""
    output = Path("results.csv")
    correct_points = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["company", "points", "points_correct", "point_errors", "interviews_used", "summary"])
        for name in names:
            entry = analyses.get(name)
            entry = entry if isinstance(entry, dict) else {}
            points = normalize_points(entry.get("points"), chosen)
            expected = truth.get(name)
            n_points, errors = point_errors(points, expected)
            correct_points += n_points
            writer.writerow([
                name,
                " ".join(f"{p}:{points[p] or 'missing'}" for p in chosen),
                f"{n_points}/{len(chosen)}" if expected else "",
                errors,
                INTERVIEWS_USED.get(name, 0),
                str(entry.get("summary", "")).strip(),
            ])
    return output, correct_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW3 on company data folders")
    parser.add_argument(
        "--data-folder",
        required=True,
        type=Path,
        help="folder containing one sub-folder per company",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    folders = company_folders(folder)
    if not folders:
        raise SystemExit(f"no company folders found in {folder}")

    chosen = check_chosen_points(CHOSEN_POINTS)
    load_env_file()
    companies = {path.name: make_tools(path) for path in folders}
    agent = build_agent()
    analyses = await analyze_companies(agent, companies)
    if not isinstance(analyses, dict):
        raise TypeError("analyze_companies() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct_points = write_results(list(companies), analyses, truth, chosen)
    summary = f" Your points: {', '.join(map(str, chosen))}."
    if truth:
        summary += (
            f" Point accuracy: {correct_points}/{len(companies) * len(chosen)}"
            f" (balanced: {balanced_point_accuracy(analyses, truth, list(companies), chosen):.1%})."
        )
    print(f"Processed {len(companies)} compan{'y' if len(companies) == 1 else 'ies'}. Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.data_folder.is_dir():
        raise SystemExit(f"not a folder: {args.data_folder}")
    return asyncio.run(run(args.data_folder))


if __name__ == "__main__":
    raise SystemExit(main())

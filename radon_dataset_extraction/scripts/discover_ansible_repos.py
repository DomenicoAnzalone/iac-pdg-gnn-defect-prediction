import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "output" / "discovery" / "ansible_candidate_repos.txt"
DEFAULT_SELECTED_REPORT = PROJECT_ROOT / "output" / "discovery" / "ansible_candidate_repos_selected.csv"
DEFAULT_EXCLUDED_REPORT = PROJECT_ROOT / "output" / "discovery" / "ansible_candidate_repos_excluded.csv"
DEFAULT_METHODOLOGY_REPORT = PROJECT_ROOT / "output" / "discovery" / "ansible_discovery_methodology.md"

GITHUB_API_URL = "https://api.github.com/search/repositories"

SEARCH_STRATEGIES = [
    {
        "name": "ansible-role",
        "query": "ansible-role in:name,description,readme",
        "reason": "name/description/readme contains ansible-role",
    },
    {
        "name": "ansible role",
        "query": "\"ansible role\" in:name,description,readme",
        "reason": "name/description/readme contains ansible role",
    },
    {
        "name": "ansible-collection",
        "query": "ansible-collection in:name,description,readme",
        "reason": "repository appears to be an Ansible collection",
    },
    {
        "name": "ansible collection",
        "query": "\"ansible collection\" in:name,description,readme",
        "reason": "repository appears to be an Ansible collection",
    },
    {
        "name": "playbook",
        "query": "ansible playbook in:name,description,readme",
        "reason": "repository matches Ansible playbook query",
    },
    {
        "name": "molecule",
        "query": "ansible molecule in:name,description,readme",
        "reason": "repository matches Ansible Molecule query",
    },
    {
        "name": "infrastructure automation",
        "query": "ansible \"infrastructure automation\" in:name,description,readme",
        "reason": "repository matches infrastructure automation Ansible query",
    },
    {
        "name": "hardening",
        "query": "ansible hardening in:name,description,readme",
        "reason": "repository matches hardening Ansible query",
    },
    {
        "name": "security",
        "query": "ansible security in:name,description,readme",
        "reason": "repository matches security Ansible query",
    },
    {
        "name": "monitoring",
        "query": "ansible monitoring in:name,description,readme",
        "reason": "repository matches monitoring Ansible query",
    },
    {
        "name": "prometheus",
        "query": "ansible prometheus in:name,description,readme",
        "reason": "repository matches prometheus Ansible query",
    },
    {
        "name": "grafana",
        "query": "ansible grafana in:name,description,readme",
        "reason": "repository matches grafana Ansible query",
    },
    {
        "name": "docker",
        "query": "ansible docker in:name,description,readme",
        "reason": "repository matches docker Ansible query",
    },
    {
        "name": "nginx",
        "query": "ansible nginx in:name,description,readme",
        "reason": "repository matches nginx Ansible query",
    },
    {
        "name": "mysql",
        "query": "ansible mysql in:name,description,readme",
        "reason": "repository matches mysql Ansible query",
    },
    {
        "name": "postgresql",
        "query": "ansible postgresql in:name,description,readme",
        "reason": "repository matches postgresql Ansible query",
    },
    {
        "name": "kubernetes",
        "query": "ansible kubernetes in:name,description,readme",
        "reason": "repository matches kubernetes Ansible query",
    },
    {
        "name": "openstack",
        "query": "ansible openstack in:name,description,readme",
        "reason": "repository matches openstack Ansible query",
    },
    {
        "name": "topic ansible",
        "query": "topic:ansible",
        "reason": "repository has ansible-related topics",
    },
    {
        "name": "topic ansible-role",
        "query": "topic:ansible-role",
        "reason": "repository has ansible-role topic",
    },
    {
        "name": "topic ansible-collection",
        "query": "topic:ansible-collection",
        "reason": "repository has ansible-collection topic",
    },
]

EXCLUSION_KEYWORDS = [
    "awesome",
    "tutorial",
    "example",
    "examples",
    "sample",
    "samples",
    "workshop",
    "course",
    "exercise",
    "exercises",
    "interview",
    "interview-questions",
    "training",
    "cheatsheet",
    "cheat-sheet",
    "demo",
    "learn",
    "learning",
    "guide",
    "handbook",
]

ANSIBLE_SIGNALS = [
    "ansible",
    "ansible-role",
    "ansible role",
    "ansible-collection",
    "ansible collection",
    "playbook",
    "molecule",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Discover GitHub repositories that are likely Ansible RADON candidates "
            "and write a repo_url,branch input file plus discovery reports."
        )
    )
    parser.add_argument("--max-repos", type=int, default=100)
    parser.add_argument("--min-stars", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument(
        "--strategies",
        default=None,
        help=(
            "Optional comma-separated strategy names to run. "
            "By default all configured strategies are used."
        ),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--selected-report", default=str(DEFAULT_SELECTED_REPORT))
    parser.add_argument("--excluded-report", default=str(DEFAULT_EXCLUDED_REPORT))
    parser.add_argument("--methodology-report", default=str(DEFAULT_METHODOLOGY_REPORT))
    parser.add_argument(
        "--token",
        default=None,
        help="GitHub token. If omitted, GITHUB_TOKEN or GH_TOKEN is used when available.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between GitHub API requests.",
    )
    return parser.parse_args()


def selected_strategies(args):
    if not args.strategies:
        return SEARCH_STRATEGIES

    requested = {
        value.strip().lower()
        for value in args.strategies.split(",")
        if value.strip()
    }
    strategies = [
        strategy
        for strategy in SEARCH_STRATEGIES
        if strategy["name"].lower() in requested
    ]
    missing = sorted(requested - {strategy["name"].lower() for strategy in strategies})

    if missing:
        raise SystemExit(
            "Unknown discovery strategies: {}\nAvailable strategies: {}".format(
                ", ".join(missing),
                ", ".join(strategy["name"] for strategy in SEARCH_STRATEGIES),
            )
        )

    return strategies


def github_headers(token):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "radon-ansible-discovery",
    }

    if token:
        headers["Authorization"] = "Bearer " + token

    return headers


def request_json(url, headers):
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        payload = response.read().decode("utf-8")
        remaining = response.headers.get("X-RateLimit-Remaining", "")
        reset = response.headers.get("X-RateLimit-Reset", "")
        return json.loads(payload), remaining, reset


def search_repositories(strategy, page, per_page, min_stars, headers):
    query = strategy["query"]
    if min_stars > 0:
        query = query + " stars:>=" + str(min_stars)

    params = urlencode({
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": per_page,
        "page": page,
    })
    url = GITHUB_API_URL + "?" + params
    return request_json(url, headers)


def text_blob(repo):
    values = [
        repo.get("name") or "",
        repo.get("full_name") or "",
        repo.get("description") or "",
        " ".join(repo.get("topics") or []),
    ]
    return " ".join(values).lower()


def is_likely_tutorial(repo):
    blob = text_blob(repo)
    return any(keyword in blob for keyword in EXCLUSION_KEYWORDS)


def has_ansible_signal(repo):
    blob = text_blob(repo)
    return any(signal in blob for signal in ANSIBLE_SIGNALS)


def collect_selection_reason(repo, matched_reasons):
    reasons = list(dict.fromkeys(matched_reasons))
    topics = [topic.lower() for topic in (repo.get("topics") or [])]
    blob = text_blob(repo)

    if any(topic.startswith("ansible") for topic in topics):
        reasons.append("repository has ansible-related topics")
    if "ansible-role" in blob or "ansible role" in blob:
        reasons.append("name/description contains ansible-role")
    if "ansible-collection" in blob or "ansible collection" in blob:
        reasons.append("repository appears to be an Ansible collection")

    reasons.append("passed minimum stars and activity filters")
    reasons.append("default branch detected from GitHub metadata")
    return "; ".join(dict.fromkeys(reasons))


def pushed_days_ago(repo):
    pushed_at = repo.get("pushed_at")
    if not pushed_at:
        return 3650

    try:
        pushed = datetime.strptime(pushed_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max((now - pushed).days, 0)
    except Exception:
        return 3650


def priority_score(repo, matched_queries_count):
    stars = int(repo.get("stargazers_count") or 0)
    forks = int(repo.get("forks_count") or 0)
    days = pushed_days_ago(repo)
    recency = max(0, 3650 - days) / 3650.0

    score = 0.0
    score += min(stars, 5000) * 1.0
    score += min(forks, 1000) * 0.5
    score += matched_queries_count * 100.0
    score += recency * 250.0

    if has_ansible_signal(repo):
        score += 250.0
    if any((topic or "").lower().startswith("ansible") for topic in (repo.get("topics") or [])):
        score += 200.0

    return round(score, 3)


def repo_row(repo, selection_reason, score, matched_queries):
    return {
        "repo_url": repo.get("html_url") or "",
        "branch": repo.get("default_branch") or "",
        "full_name": repo.get("full_name") or "",
        "stars": repo.get("stargazers_count") or 0,
        "forks": repo.get("forks_count") or 0,
        "watchers": repo.get("watchers_count") or 0,
        "open_issues": repo.get("open_issues_count") or 0,
        "language": repo.get("language") or "",
        "created_at": repo.get("created_at") or "",
        "updated_at": repo.get("updated_at") or "",
        "pushed_at": repo.get("pushed_at") or "",
        "description": repo.get("description") or "",
        "topics": ";".join(repo.get("topics") or []),
        "archived": repo.get("archived"),
        "disabled": repo.get("disabled"),
        "fork": repo.get("fork"),
        "size": repo.get("size") or 0,
        "license": ((repo.get("license") or {}).get("spdx_id") or ""),
        "matched_queries": ";".join(matched_queries),
        "selection_reason": selection_reason,
        "priority_score": score,
    }


def excluded_row(repo, reason, matched_query):
    return {
        "repo_url": repo.get("html_url") or "",
        "branch": repo.get("default_branch") or "",
        "full_name": repo.get("full_name") or "",
        "stars": repo.get("stargazers_count") or 0,
        "forks": repo.get("forks_count") or 0,
        "language": repo.get("language") or "",
        "pushed_at": repo.get("pushed_at") or "",
        "updated_at": repo.get("updated_at") or "",
        "description": repo.get("description") or "",
        "topics": ";".join(repo.get("topics") or []),
        "archived": repo.get("archived"),
        "disabled": repo.get("disabled"),
        "matched_query": matched_query,
        "exclusion_reason": reason,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_input_file(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write("{},{}\n".format(row["repo_url"], row["branch"]))


def write_methodology_report(path, args, strategies, selected_rows, excluded_rows, errors, token_used):
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Ansible repository discovery report",
        "",
        "Generated at: {}".format(datetime.now(timezone.utc).isoformat()),
        "",
        "## Configuration",
        "",
        "- max_repos: {}".format(args.max_repos),
        "- min_stars: {}".format(args.min_stars),
        "- max_pages_per_query: {}".format(args.max_pages),
        "- per_page: {}".format(args.per_page),
        "- token_used: {}".format("yes" if token_used else "no"),
        "- output_input_file: `{}`".format(Path(args.output)),
        "- selected_report: `{}`".format(Path(args.selected_report)),
        "- excluded_report: `{}`".format(Path(args.excluded_report)),
        "",
        "## Search strategies",
        "",
    ]

    for strategy in strategies:
        lines.append("- `{}`: `{}`".format(strategy["name"], strategy["query"]))

    lines.extend([
        "",
        "## Inclusion criteria",
        "",
        "- Repository returned by at least one configured GitHub search strategy.",
        "- Repository metadata contains an Ansible signal in name, description, or topics.",
        "- Repository has at least the configured minimum number of stars.",
        "- Repository has a non-empty GitHub `html_url` and `default_branch`.",
        "",
        "## Exclusion criteria",
        "",
        "- Archived repositories.",
        "- Disabled repositories.",
        "- Repositories below the configured minimum stars threshold.",
        "- Repositories likely to be tutorials, examples, workshops, courses, cheatsheets, demos, or generic learning material.",
        "- Repositories without clear Ansible-related metadata.",
        "- Duplicates already discovered by an earlier strategy.",
        "",
        "## Deduplication",
        "",
        "Repositories are deduplicated by GitHub `full_name`. When the same repository is found by multiple strategies, the selected report keeps one row and records all matched strategy names.",
        "",
        "## Prioritization",
        "",
        "Candidates are sorted by a computed priority score that combines stars, forks, number of matched strategies, recent push activity, and explicit Ansible signals in metadata or topics.",
        "",
        "## Results",
        "",
        "- selected_repositories: {}".format(len(selected_rows)),
        "- excluded_or_duplicate_repositories: {}".format(len(excluded_rows)),
        "- api_errors: {}".format(len(errors)),
    ])

    if errors:
        lines.extend(["", "## API errors", ""])
        for error in errors:
            lines.append("- {}".format(error))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    strategies = selected_strategies(args)
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = github_headers(token)

    discovered = {}
    excluded_rows = []
    errors = []

    for strategy in strategies:
        for page in range(1, args.max_pages + 1):
            try:
                payload, remaining, reset = search_repositories(
                    strategy=strategy,
                    page=page,
                    per_page=args.per_page,
                    min_stars=args.min_stars,
                    headers=headers,
                )
            except HTTPError as exc:
                errors.append(
                    "{} page {} failed with HTTP {}: {}".format(
                        strategy["name"], page, exc.code, exc.reason
                    )
                )
                break
            except URLError as exc:
                errors.append("{} page {} failed: {}".format(strategy["name"], page, exc.reason))
                break

            items = payload.get("items") or []
            if not items:
                break

            print(
                "strategy={} page={} items={} rate_remaining={}".format(
                    strategy["name"], page, len(items), remaining
                )
            )

            for repo in items:
                full_name = (repo.get("full_name") or "").lower()
                if not full_name:
                    excluded_rows.append(excluded_row(repo, "missing or invalid metadata", strategy["name"]))
                    continue

                if full_name in discovered:
                    discovered[full_name]["matched_queries"].append(strategy["name"])
                    discovered[full_name]["matched_reasons"].append(strategy["reason"])
                    excluded_rows.append(excluded_row(repo, "duplicate repository", strategy["name"]))
                    continue

                if repo.get("archived"):
                    excluded_rows.append(excluded_row(repo, "archived repository", strategy["name"]))
                    continue

                if repo.get("disabled"):
                    excluded_rows.append(excluded_row(repo, "disabled repository", strategy["name"]))
                    continue

                if int(repo.get("stargazers_count") or 0) < args.min_stars:
                    excluded_rows.append(excluded_row(repo, "below minimum stars", strategy["name"]))
                    continue

                if not repo.get("html_url") or not repo.get("default_branch"):
                    excluded_rows.append(excluded_row(repo, "missing or invalid metadata", strategy["name"]))
                    continue

                if is_likely_tutorial(repo):
                    excluded_rows.append(excluded_row(repo, "likely tutorial/example/workshop", strategy["name"]))
                    continue

                if not has_ansible_signal(repo):
                    excluded_rows.append(excluded_row(repo, "not clearly Ansible-related", strategy["name"]))
                    continue

                discovered[full_name] = {
                    "repo": repo,
                    "matched_queries": [strategy["name"]],
                    "matched_reasons": [strategy["reason"]],
                }

            if len(items) < args.per_page:
                break

            reset_timestamp = int(reset) if str(reset).isdigit() else 0
            if remaining == "0" and reset_timestamp > 0:
                sleep_for = max(reset_timestamp - int(time.time()) + 5, 1)
                print("GitHub rate limit reached; sleeping {} seconds".format(sleep_for))
                time.sleep(sleep_for)
            elif args.sleep > 0:
                time.sleep(args.sleep)

    selected_rows = []
    for item in discovered.values():
        repo = item["repo"]
        matched_queries = list(dict.fromkeys(item["matched_queries"]))
        matched_reasons = list(dict.fromkeys(item["matched_reasons"]))
        score = priority_score(repo, len(matched_queries))
        reason = collect_selection_reason(repo, matched_reasons)
        selected_rows.append(repo_row(repo, reason, score, matched_queries))

    selected_rows.sort(
        key=lambda row: (
            float(row["priority_score"]),
            int(row["stars"]),
            row["pushed_at"],
        ),
        reverse=True,
    )
    selected_rows = selected_rows[: args.max_repos]

    selected_fieldnames = [
        "repo_url",
        "branch",
        "full_name",
        "stars",
        "forks",
        "watchers",
        "open_issues",
        "language",
        "created_at",
        "updated_at",
        "pushed_at",
        "description",
        "topics",
        "archived",
        "disabled",
        "fork",
        "size",
        "license",
        "matched_queries",
        "selection_reason",
        "priority_score",
    ]
    excluded_fieldnames = [
        "repo_url",
        "branch",
        "full_name",
        "stars",
        "forks",
        "language",
        "pushed_at",
        "updated_at",
        "description",
        "topics",
        "archived",
        "disabled",
        "matched_query",
        "exclusion_reason",
    ]

    write_input_file(Path(args.output), selected_rows)
    write_csv(Path(args.selected_report), selected_rows, selected_fieldnames)
    write_csv(Path(args.excluded_report), excluded_rows, excluded_fieldnames)
    write_methodology_report(
        Path(args.methodology_report),
        args,
        strategies,
        selected_rows,
        excluded_rows,
        errors,
        bool(token),
    )

    print("=" * 80)
    print("DISCOVERY COMPLETED")
    print("=" * 80)
    print("Selected repositories: {}".format(len(selected_rows)))
    print("Excluded or duplicate repositories: {}".format(len(excluded_rows)))
    print("Input file: {}".format(Path(args.output).resolve()))
    print("Selected report: {}".format(Path(args.selected_report).resolve()))
    print("Excluded report: {}".format(Path(args.excluded_report).resolve()))
    print("Methodology report: {}".format(Path(args.methodology_report).resolve()))


if __name__ == "__main__":
    main()

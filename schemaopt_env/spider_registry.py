"""Spider dataset registry and weighted workload sampling utilities.

Phase 1 scope:
- Parse Spider split files and schema metadata.
- Build per-database query indexes from text-to-SQL pairs.
- Provide weighted sampling that emphasizes complex query patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

_SPIDER_ROOT = Path(__file__).resolve().parent / "spider_data"
_SPLIT_FILES = {
    "train_spider": "train_spider.json",
    "train_others": "train_others.json",
    "dev": "dev.json",
    "test": "test.json",
}

_TABLE_REF_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*|\"[^\"]+\")", re.IGNORECASE)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def _strip_identifier_quotes(value: str) -> str:
    return value.strip().strip('"').strip("`").strip("[").strip("]").lower()


def _extract_table_refs(sql: str) -> tuple[str, ...]:
    refs = {_strip_identifier_quotes(match.group(1)) for match in _TABLE_REF_RE.finditer(sql)}
    return tuple(sorted(item for item in refs if item))


def _extract_operation_patterns(sql: str) -> tuple[str, ...]:
    lowered = _normalize_sql(sql)
    patterns: set[str] = set()
    if " join " in f" {lowered} ":
        patterns.add("join")
    if " left join " in f" {lowered} ":
        patterns.add("left_join")
    if " right join " in f" {lowered} ":
        patterns.add("right_join")
    if " where " in f" {lowered} ":
        patterns.add("where")
    if " group by " in f" {lowered} ":
        patterns.add("group_by")
    if " having " in f" {lowered} ":
        patterns.add("having")
    if " order by " in f" {lowered} ":
        patterns.add("order_by")
    if " limit " in f" {lowered} ":
        patterns.add("limit")
    if " distinct " in f" {lowered} ":
        patterns.add("distinct")
    if " union " in f" {lowered} ":
        patterns.add("union")
    if " intersect " in f" {lowered} ":
        patterns.add("intersect")
    if " except " in f" {lowered} ":
        patterns.add("except")
    if re.search(r"\b(count|sum|avg|min|max)\s*\(", lowered):
        patterns.add("aggregate")
    if re.search(r"\bselect\b.*\bselect\b", lowered):
        patterns.add("subquery")
    return tuple(sorted(patterns))


@dataclass(frozen=True)
class SpiderQuerySpec:
    query_id: str
    db_id: str
    split: str
    sql: str
    question: str
    normalized_sql: str
    tables: tuple[str, ...]
    operation_patterns: tuple[str, ...]
    historical_frequency: int
    complexity_score: float
    sampling_weight: float

    def summary(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "db_id": self.db_id,
            "split": self.split,
            "question": self.question,
            "tables": list(self.tables),
            "operation_patterns": list(self.operation_patterns),
            "historical_frequency": self.historical_frequency,
            "complexity_score": round(self.complexity_score, 6),
            "sampling_weight": round(self.sampling_weight, 6),
        }


@dataclass(frozen=True)
class SpiderDatabaseSpec:
    db_id: str
    table_names: tuple[str, ...]
    sqlite_path: str
    schema_source: str

    def summary(self) -> Dict[str, Any]:
        return {
            "db_id": self.db_id,
            "table_count": len(self.table_names),
            "table_names": list(self.table_names),
            "sqlite_path": self.sqlite_path,
            "schema_source": self.schema_source,
        }


@dataclass(frozen=True)
class SpiderRegistry:
    spider_root: str
    databases: Dict[str, SpiderDatabaseSpec]
    queries_by_split: Dict[str, tuple[SpiderQuerySpec, ...]]
    queries_by_db: Dict[str, tuple[SpiderQuerySpec, ...]]

    def list_databases(self) -> List[Dict[str, Any]]:
        return [item.summary() for item in sorted(self.databases.values(), key=lambda value: value.db_id)]

    def list_db_ids(self) -> List[str]:
        return sorted(self.databases.keys())

    def db_summary(self, db_id: str) -> Dict[str, Any]:
        if db_id not in self.databases:
            raise KeyError(f"Unknown db_id: {db_id}")
        queries = self.queries_by_db.get(db_id, ())
        split_counts: Dict[str, int] = {}
        for query in queries:
            split_counts[query.split] = split_counts.get(query.split, 0) + 1
        return {
            **self.databases[db_id].summary(),
            "query_count": len(queries),
            "split_counts": dict(sorted(split_counts.items())),
            "top_patterns": _top_patterns(queries),
        }

    def sample_queries(
        self,
        db_id: str,
        sample_size: int,
        *,
        split: Optional[str] = None,
        seed: Optional[int] = None,
        min_complexity: float = 0.0,
    ) -> List[SpiderQuerySpec]:
        if sample_size <= 0:
            return []
        if db_id not in self.databases:
            raise KeyError(f"Unknown db_id: {db_id}")

        candidates = [
            query
            for query in self.queries_by_db.get(db_id, ())
            if (split is None or query.split == split) and query.complexity_score >= min_complexity
        ]
        if not candidates:
            return []

        selected_count = min(sample_size, len(candidates))
        rng = random.Random(seed)
        pool = list(candidates)
        selected: List[SpiderQuerySpec] = []
        for _ in range(selected_count):
            weights = [max(1e-9, query.sampling_weight) for query in pool]
            index = _weighted_index(weights, rng)
            selected.append(pool.pop(index))
        return selected


def _weighted_index(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    draw = rng.uniform(0.0, total)
    running = 0.0
    for idx, weight in enumerate(weights):
        running += weight
        if draw <= running:
            return idx
    return len(weights) - 1


def _safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_sqlite_path(spider_root: Path, db_id: str) -> tuple[str, str]:
    train_path = spider_root / "database" / db_id / f"{db_id}.sqlite"
    test_path = spider_root / "test_database" / db_id / f"{db_id}.sqlite"
    if train_path.exists():
        return str(train_path.resolve()), "tables.json"
    if test_path.exists():
        return str(test_path.resolve()), "test_tables.json"
    if (spider_root / "database" / db_id).exists():
        return str(train_path.resolve()), "tables.json"
    return str(test_path.resolve()), "test_tables.json"


def _load_schema_entries(spider_root: Path) -> Dict[str, SpiderDatabaseSpec]:
    by_db: Dict[str, SpiderDatabaseSpec] = {}
    for source_name, file_name in (("tables.json", "tables.json"), ("test_tables.json", "test_tables.json")):
        payload = _safe_load_json(spider_root / file_name) or []
        for item in payload:
            db_id = str(item.get("db_id") or "").strip()
            if not db_id or db_id in by_db:
                continue
            tables = tuple(_strip_identifier_quotes(value) for value in item.get("table_names_original", []))
            sqlite_path, schema_source = _resolve_sqlite_path(spider_root, db_id)
            by_db[db_id] = SpiderDatabaseSpec(
                db_id=db_id,
                table_names=tables,
                sqlite_path=sqlite_path,
                schema_source=schema_source if source_name == schema_source else source_name,
            )
    return by_db


def _build_pattern_idf(queries: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for query in queries:
        total += 1
        for pattern in set(query.get("patterns", ())):
            counts[pattern] = counts.get(pattern, 0) + 1
    if total == 0:
        return {}
    return {pattern: (1.0 + (total / max(1, count))) for pattern, count in counts.items()}


def _compute_complexity(patterns: Sequence[str], sql: str) -> float:
    lowered = _normalize_sql(sql)
    join_count = lowered.count(" join ")
    nesting_count = lowered.count("select") - 1
    set_ops = sum(lowered.count(token) for token in (" union ", " intersect ", " except "))
    return 1.0 + 0.8 * len(patterns) + 0.5 * join_count + 0.7 * max(0, nesting_count) + 0.9 * set_ops


def _top_patterns(queries: Sequence[SpiderQuerySpec], limit: int = 8) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for query in queries:
        for pattern in query.operation_patterns:
            counts[pattern] = counts.get(pattern, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:limit])


def load_spider_registry(spider_root: str | Path = _SPIDER_ROOT) -> SpiderRegistry:
    root = Path(spider_root)
    databases = _load_schema_entries(root)

    raw_records: List[Dict[str, Any]] = []
    by_split: Dict[str, List[SpiderQuerySpec]] = {split: [] for split in _SPLIT_FILES}
    by_db: Dict[str, List[SpiderQuerySpec]] = {}
    normalized_frequency: Dict[tuple[str, str], int] = {}

    staged_rows: List[Dict[str, Any]] = []
    for split_name, split_file in _SPLIT_FILES.items():
        payload = _safe_load_json(root / split_file) or []
        for idx, item in enumerate(payload):
            db_id = str(item.get("db_id") or "").strip()
            sql = str(item.get("query") or "").strip()
            question = str(item.get("question") or "").strip()
            if not db_id or not sql:
                continue
            normalized_sql = _normalize_sql(sql)
            normalized_frequency[(db_id, normalized_sql)] = normalized_frequency.get((db_id, normalized_sql), 0) + 1
            patterns = _extract_operation_patterns(sql)
            tables = _extract_table_refs(sql)
            row = {
                "split": split_name,
                "db_id": db_id,
                "query_id": f"{split_name}:{db_id}:{idx:05d}",
                "sql": sql,
                "question": question,
                "normalized_sql": normalized_sql,
                "tables": tables,
                "patterns": patterns,
            }
            staged_rows.append(row)
            raw_records.append(row)

    pattern_idf = _build_pattern_idf(raw_records)
    for row in staged_rows:
        history = normalized_frequency[(row["db_id"], row["normalized_sql"])]
        complexity = _compute_complexity(row["patterns"], row["sql"])
        pattern_bias = sum(pattern_idf.get(pattern, 1.0) for pattern in row["patterns"]) if row["patterns"] else 1.0
        sampling_weight = max(1e-9, complexity * (1.0 + 0.15 * history) * (1.0 + 0.03 * pattern_bias))
        spec = SpiderQuerySpec(
            query_id=row["query_id"],
            db_id=row["db_id"],
            split=row["split"],
            sql=row["sql"],
            question=row["question"],
            normalized_sql=row["normalized_sql"],
            tables=row["tables"],
            operation_patterns=row["patterns"],
            historical_frequency=history,
            complexity_score=complexity,
            sampling_weight=sampling_weight,
        )
        by_split[row["split"]].append(spec)
        by_db.setdefault(row["db_id"], []).append(spec)
        if row["db_id"] not in databases:
            sqlite_path, schema_source = _resolve_sqlite_path(root, row["db_id"])
            databases[row["db_id"]] = SpiderDatabaseSpec(
                db_id=row["db_id"],
                table_names=(),
                sqlite_path=sqlite_path,
                schema_source=schema_source,
            )

    return SpiderRegistry(
        spider_root=str(root.resolve()),
        databases=databases,
        queries_by_split={key: tuple(value) for key, value in by_split.items()},
        queries_by_db={key: tuple(value) for key, value in by_db.items()},
    )


SPIDER_REGISTRY = load_spider_registry()


def list_spider_database_summaries() -> List[Dict[str, Any]]:
    return SPIDER_REGISTRY.list_databases()


def get_spider_database_summary(db_id: str) -> Dict[str, Any]:
    return SPIDER_REGISTRY.db_summary(db_id)


def get_spider_database_spec(db_id: str) -> SpiderDatabaseSpec:
    if db_id not in SPIDER_REGISTRY.databases:
        raise KeyError(f"Unknown db_id: {db_id}")
    return SPIDER_REGISTRY.databases[db_id]


def sample_spider_queries_for_db(
    db_id: str,
    sample_size: int,
    *,
    split: Optional[str] = None,
    seed: Optional[int] = None,
    min_complexity: float = 0.0,
) -> List[SpiderQuerySpec]:
    return SPIDER_REGISTRY.sample_queries(
        db_id=db_id,
        sample_size=sample_size,
        split=split,
        seed=seed,
        min_complexity=min_complexity,
    )


def spider_queries_for_db(db_id: str, *, split: Optional[str] = None) -> List[SpiderQuerySpec]:
    if db_id not in SPIDER_REGISTRY.databases:
        raise KeyError(f"Unknown db_id: {db_id}")
    queries = SPIDER_REGISTRY.queries_by_db.get(db_id, ())
    if split is None:
        return list(queries)
    return [query for query in queries if query.split == split]

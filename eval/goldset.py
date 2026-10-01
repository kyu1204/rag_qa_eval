"""골드셋 로드·검증. 문서 무관: 근거는 (document_id, page?, quote?)로만 가리킨다.

골드셋 = 디렉터리 하나:
  manifest.json  {"name", "documents": [{"id", "sha256"}], "files": [...jsonl]}
  *.jsonl        문항. type별 필수 필드는 validate() 참조. status가 approved인 것만 평가한다.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

NEG_TYPES = {"unanswerable", "partial", "false_premise", "confusion", "temporal"}
STATUSES = {"draft", "approved", "rejected"}


@dataclass
class GoldSet:
    name: str
    path: Path
    documents: list[dict]  # [{"id", "sha256"}]
    items: list[dict]  # approved만
    hash: str  # manifest와 문항 파일 바이트의 sha256 앞 12자리
    skipped: int  # approved가 아닌 문항 수


def group(item: dict) -> str:
    return {"quant": "정량", "qual": "정성"}.get(item["type"], "답 없음·함정")


def _evidence_errors(where: str, evidence: list) -> list[str]:
    if not evidence:
        return [f"{where}: evidence 없음"]
    return [f"{where}: evidence에 document_id와 page 또는 quote가 필요" for e in evidence
            if not e.get("document_id") or (e.get("page") is None and not e.get("quote"))]


def validate(item: dict) -> list[str]:
    qid = item.get("id", "<id 없음>")
    errors = [f"{qid}: {k} 없음" for k in ("id", "type", "question", "status", "provenance") if not item.get(k)]
    if item.get("status") and item["status"] not in STATUSES:
        errors.append(f"{qid}: status는 {sorted(STATUSES)} 중 하나")
    kind = item.get("type")
    if kind == "quant":
        if not item.get("answer"):
            errors.append(f"{qid}: answer 없음")
        errors += _evidence_errors(qid, item.get("evidence", []))
    elif kind == "qual":
        points = item.get("must_include") or []
        if not points:
            errors.append(f"{qid}: must_include 없음")
        for i, p in enumerate(points):
            if not p.get("facts"):
                errors.append(f"{qid} 포인트 {i}: facts 없음")
            errors += _evidence_errors(f"{qid} 포인트 {i}", p.get("evidence", []))
    elif kind in NEG_TYPES:
        scoring = item.get("scoring") or {}
        if not {"2", "0"} <= set(scoring) <= {"2", "1", "0"}:
            errors.append(f"{qid}: scoring은 2와 0을 포함한 2/1/0 기준")
        if item.get("expected_status") not in {"answered", "insufficient_context"}:
            errors.append(f"{qid}: expected_status 없음")
    else:
        errors.append(f"{qid}: 알 수 없는 type {kind}")
    return errors


def load(path: str | Path = "eval/gold") -> GoldSet:
    path = Path(path)
    manifest_bytes = (path / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes)
    items = []
    for name in manifest["files"]:
        raw = (path / name).read_bytes()
        digest.update(raw)
        items += [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    errors = [e for item in items for e in validate(item)]
    ids = [item.get("id") for item in items]
    errors += [f"{i}: id 중복" for i in sorted({i for i in ids if ids.count(i) > 1})]
    if errors:
        raise ValueError("골드셋 검증 실패:\n" + "\n".join(errors))
    approved = [item for item in items if item["status"] == "approved"]
    return GoldSet(manifest["name"], path, manifest["documents"], approved, digest.hexdigest()[:12], len(items) - len(approved))


def document_problems(gold: GoldSet, conn) -> list[str]:
    """골드셋이 가리키는 문서가 DB에 같은 내용(해시)으로 적재돼 있는지. 다르면 근거 쪽이 어긋난다."""
    problems = []
    for doc in gold.documents:
        row = conn.execute("SELECT content_hash FROM documents WHERE id = %s", (doc["id"],)).fetchone()
        if row is None:
            problems.append(f"문서 미적재: {doc['id']}")
        elif row[0] != doc["sha256"]:
            problems.append(f"문서 내용이 골드셋과 다름: {doc['id']} (DB {row[0][:8]} != 골드셋 {doc['sha256'][:8]})")
    return problems

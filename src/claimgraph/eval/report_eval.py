"""Citation validity of stored reports: does every citation resolve to a real claim?"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ..db import init_db, session_scope
from ..models_db import ClaimRow, Report
from ..report.citation_check import check_citations
from .common import save_result, waiting


def main() -> Path:
    init_db()
    with session_scope() as session:
        reports = session.scalars(select(Report)).all()
        known = set(session.scalars(select(ClaimRow.claim_id)).all())
        if not reports:
            return waiting("report", "No reports yet. Open a topic's report first.", "waiting_for_reports")
        total_cited = unresolved = sentences_removed = from_template = uncited_after_check = 0
        for report in reports:
            total_cited += len(report.cited_claim_ids or [])
            unresolved += sum(1 for claim_id in report.cited_claim_ids or [] if claim_id not in known)
            sentences_removed += len(report.removed_sentences or [])
            from_template += report.model == "template"
            uncited_after_check += len(check_citations(report.text, known).removed)
        return save_result(
            "report",
            {
                "status": "measured",
                "reports": len(reports),
                "citations": total_cited,
                "unresolved_citations": unresolved,
                "model_sentences_removed": sentences_removed,
                "reports_using_template": from_template,
                "sentences_failing_check_after_cleanup": uncited_after_check,
            },
        )


if __name__ == "__main__":
    print(main())

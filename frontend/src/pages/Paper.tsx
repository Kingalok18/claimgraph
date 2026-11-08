import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/TextLayer.css";
import "react-pdf/dist/Page/AnnotationLayer.css";
import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import { api, Claim } from "../api";
import { directionWords, Empty, ErrorNote, Skeleton, SkeletonRows } from "../components/ui";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

const normal = (text: string) => text.normalize("NFKC").toLowerCase().replace(/[\s-]+/g, "");

/**
 * Which characters of which text pieces belong to a quote. Quotes are matched across the whole page,
 * because a PDF line can hold the end of one sentence and the start of the next.
 */
function markRanges(items: string[], quotes: string[]): Map<number, [number, number][]> {
  let joined = "";
  const owner: [number, number][] = [];
  items.forEach((text, item) => {
    for (let index = 0; index < text.length; index++) {
      const piece = normal(text[index]);
      for (let unit = 0; unit < piece.length; unit++) owner.push([item, index]);
      joined += piece;
    }
  });
  const ranges = new Map<number, [number, number][]>();
  const add = (item: number, start: number, end: number) => ranges.set(item, [...(ranges.get(item) ?? []), [start, end]]);
  for (const quote of quotes) {
    const position = quote ? joined.indexOf(quote) : -1;
    if (position < 0) {
      // The quote continues on another page: mark the pieces of it found on this one.
      items.forEach((text, item) => {
        const piece = normal(text);
        if (piece.length >= 10 && quote.includes(piece)) add(item, 0, text.length);
      });
      continue;
    }
    const spans = new Map<number, [number, number]>();
    for (let index = position; index < position + quote.length; index++) {
      const [item, character] = owner[index];
      const span = spans.get(item);
      spans.set(item, span ? [span[0], character + 1] : [character, character + 1]);
    }
    spans.forEach(([start, end], item) => add(item, start, end));
  }
  return ranges;
}

function withMarks(text: string, ranges: [number, number][]): string {
  let output = "";
  let cursor = 0;
  for (const [start, end] of [...ranges].sort((a, b) => a[0] - b[0])) {
    if (end <= cursor) continue;
    const from = Math.max(start, cursor);
    output += escapeHtml(text.slice(cursor, from)) + `<mark>${escapeHtml(text.slice(from, end))}</mark>`;
    cursor = end;
  }
  return output + escapeHtml(text.slice(cursor));
}
const escapeHtml = (text: string) => text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

function ClaimCard({ claim, selected, onClick }: { claim: Claim; selected: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      data-testid="claim-card"
      className="block w-full border-b border-line px-3 py-2.5 text-left transition-colors hover:bg-subtle"
      style={selected ? { background: "var(--subtle)" } : undefined}
    >
      <div className="text-[13.5px]">{claim.text}</div>
      <div className="tabular mt-1 flex flex-wrap gap-x-2 text-[12px] text-muted">
        {claim.page && <span>Page {claim.page}</span>}
        <span>{directionWords[claim.direction] ?? claim.direction}</span>
        {claim.hedged && <span>hedged</span>}
      </div>
    </button>
  );
}

function ClaimList({ claims, selected, onSelect }: { claims: Claim[]; selected: string | null; onSelect: (claim: Claim) => void }) {
  const scroller = useRef<HTMLDivElement>(null);
  const virtual = useVirtualizer({ count: claims.length, getScrollElement: () => scroller.current, estimateSize: () => 84, overscan: 6 });
  return (
    <div ref={scroller} className="min-h-0 flex-1 overflow-auto">
      <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
        {virtual.getVirtualItems().map((row) => (
          <div key={claims[row.index].claim_id} ref={virtual.measureElement} data-index={row.index} className="absolute left-0 top-0 w-full" style={{ transform: `translateY(${row.start}px)` }}>
            <ClaimCard claim={claims[row.index]} selected={selected === claims[row.index].claim_id} onClick={() => onSelect(claims[row.index])} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default function PaperPage() {
  const { id = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const query = useQuery({ queryKey: ["paper", id], queryFn: () => api.paper(id) });
  const [pages, setPages] = useState(0);
  const [width, setWidth] = useState(640);
  const [selected, setSelected] = useState<string | null>(params.get("claim"));
  const viewer = useRef<HTMLDivElement>(null);
  const page = Math.max(1, Number(params.get("page")) || 1);

  const setPage = useCallback(
    (next: number) => {
      setParams(
        (current) => {
          current.set("page", String(next));
          return current;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  useEffect(() => {
    if (!viewer.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.min(Math.max(entry.contentRect.width - 32, 320), 900)));
    observer.observe(viewer.current);
    return () => observer.disconnect();
  }, [query.data?.has_pdf]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT") return;
      if (event.key === "ArrowRight" && pages) setPage(Math.min(page + 1, pages));
      if (event.key === "ArrowLeft") setPage(Math.max(page - 1, 1));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [page, pages, setPage]);

  const claims = useMemo(() => query.data?.claims_list ?? [], [query.data]);
  // Compare text with spaces and hyphens removed: the PDF breaks lines differently from the stored quote.
  const quotes = useMemo(() => claims.filter((claim) => claim.page === page).map((claim) => normal(claim.quote)), [claims, page]);
  const [pageText, setPageText] = useState<{ page: number; items: string[] } | null>(null);
  const onText = useCallback(
    ({ items }: { items: object[] }) => setPageText({ page, items: items.map((item) => ("str" in item ? String(item.str) : "")) }),
    [page],
  );
  const marks = useMemo(() => (pageText?.page === page ? markRanges(pageText.items, quotes) : new Map<number, [number, number][]>()), [pageText, page, quotes]);
  const highlight = useCallback(
    ({ str, itemIndex }: { str: string; itemIndex: number }) => {
      const ranges = marks.get(itemIndex);
      return ranges ? withMarks(str, ranges) : escapeHtml(str);
    },
    [marks],
  );
  const file = useMemo(() => `/papers/${id}/pdf`, [id]);

  if (query.isError) return <ErrorNote error={query.error} />;
  const paper = query.data;

  return (
    <div className="grid h-full grid-cols-[220px_minmax(0,1fr)_320px] max-lg:grid-cols-[minmax(0,1fr)_280px]">
      <aside className="flex min-h-0 flex-col overflow-auto border-r border-line bg-surface px-3 py-4 max-lg:hidden">
        <Link to="/library" className="mb-3 inline-flex items-center gap-1 text-[13px] text-muted hover:text-ink">
          <ArrowLeft size={14} /> Library
        </Link>
        {!paper ? (
          <SkeletonRows rows={8} height="h-5" />
        ) : (
          <>
            <div className="text-[13.5px] font-medium leading-snug">{paper.title}</div>
            <div className="tabular mt-1 text-[12.5px] text-muted">{[paper.year, paper.venue].filter(Boolean).join(", ")}</div>
            {paper.authors && <div className="mt-1 text-[12.5px] text-muted">{paper.authors}</div>}
            <div className="mb-1 mt-4 text-[11.5px] font-medium uppercase tracking-wide text-muted">Outline</div>
            {paper.sections.length === 0 && <div className="text-[13px] text-muted">No headings were found.</div>}
            {paper.sections.map((section) => (
              <button
                key={section.char_start}
                className="block w-full truncate rounded px-1.5 py-1 text-left text-[13px] transition-colors hover:bg-subtle"
                title={section.heading ?? ""}
                onClick={() => section.page_start && setPage(section.page_start)}
              >
                {section.heading}
              </button>
            ))}
          </>
        )}
      </aside>

      <section ref={viewer} className="flex min-h-0 flex-col overflow-auto" data-testid="pdf-viewer">
        <div className="sticky top-0 z-10 flex flex-none items-center justify-center gap-2 border-b border-line bg-bg py-1.5 text-[13px]">
          <button className="btn !px-1.5 !py-1" aria-label="Previous page" disabled={page <= 1} onClick={() => setPage(page - 1)}>
            <ChevronLeft size={16} />
          </button>
          <span className="tabular w-24 text-center text-muted">
            Page {page}
            {pages ? ` of ${pages}` : ""}
          </span>
          <button className="btn !px-1.5 !py-1" aria-label="Next page" disabled={!!pages && page >= pages} onClick={() => setPage(page + 1)}>
            <ChevronRight size={16} />
          </button>
        </div>
        <div className="flex flex-1 justify-center p-4">
          {paper && !paper.has_pdf ? (
            <Empty title="The PDF file is not available">It may have been moved. Its claims are still listed here.</Empty>
          ) : (
            <Document file={file} onLoadSuccess={({ numPages }) => setPages(numPages)} loading={<Skeleton className="h-[800px] w-[620px]" />} error={<Empty title="This PDF could not be opened" />}>
              <div className="border border-line bg-white">
                <Page pageNumber={page} width={width} renderAnnotationLayer={false} onGetTextSuccess={onText} customTextRenderer={highlight} loading={<Skeleton className="h-[800px] w-[620px]" />} />
              </div>
            </Document>
          )}
        </div>
      </section>

      <aside className="flex min-h-0 flex-col border-l border-line bg-surface">
        <div className="flex-none border-b border-line px-3 py-2.5 text-[13px] font-medium">
          Claims <span className="tabular font-normal text-muted">{claims.length}</span>
          {paper && paper.claims_dropped > 0 && (
            <span className="tabular ml-1 font-normal text-muted" title="Proposed claims whose quote could not be found in the paper">
              ({paper.claims_dropped} dropped)
            </span>
          )}
        </div>
        {!paper ? (
          <div className="p-3">
            <SkeletonRows rows={6} height="h-16" />
          </div>
        ) : claims.length === 0 ? (
          <Empty title={paper.extract_status === "done" ? "No claims with verified quotes" : "Not analysed yet"} />
        ) : (
          <ClaimList
            claims={claims}
            selected={selected}
            onSelect={(claim) => {
              setSelected(claim.claim_id);
              if (claim.page) setPage(claim.page);
            }}
          />
        )}
        {selected && claims.find((claim) => claim.claim_id === selected) && (
          <blockquote className="quote flex-none border-t border-line px-3 py-2.5 !text-[14px]">{claims.find((claim) => claim.claim_id === selected)!.quote}</blockquote>
        )}
      </aside>
    </div>
  );
}

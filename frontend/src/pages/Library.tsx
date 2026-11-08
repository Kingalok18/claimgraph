import { useCallback, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useNavigate } from "react-router-dom";
import { FileUp, Play, RefreshCw } from "lucide-react";
import { api, PaperSummary } from "../api";
import { toast } from "../components/Status";
import { Empty, ErrorNote, SkeletonRows, useListKeys } from "../components/ui";

function statusText(paper: PaperSummary): { text: string; bad?: boolean } {
  if (paper.missing) return { text: "File missing", bad: true };
  if (paper.parse_status === "failed") return { text: "Could not read", bad: true };
  if (paper.extract_status === "done") return { text: "Analysed" };
  if (paper.chunks_done > 0) return { text: `In progress, part ${paper.chunks_done} of ${paper.chunks_total}` };
  return { text: "New" };
}

const GRID = "grid grid-cols-[minmax(0,1fr)_64px_180px_64px_96px] items-center gap-3";

function canAnalyse(paper: PaperSummary): boolean {
  return paper.parse_status === "parsed" && paper.extract_status === "pending" && !paper.missing;
}

export default function LibraryPage() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const library = useQuery({ queryKey: ["library"], queryFn: api.library });
  const [dragging, setDragging] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const papers = useMemo(() => library.data?.papers ?? [], [library.data]);

  const virtual = useVirtualizer({ count: papers.length, getScrollElement: () => scroller.current, estimateSize: () => 41, overscan: 12 });

  const open = useCallback((index: number) => papers[index] && navigate(`/paper/${papers[index].id}`), [papers, navigate]);
  const move = useCallback((index: number) => virtual.scrollToIndex(index, { align: "auto" }), [virtual]);
  const active = useListKeys(papers.length, open, move);

  const upload = useMutation({
    mutationFn: async (files: File[]) => {
      let added = 0;
      for (const file of files) added += (await api.upload(file)).new;
      return added;
    },
    onSuccess: (added) => {
      toast(added ? `${added} ${added === 1 ? "paper" : "papers"} added to the library.` : "These papers are already in the library.");
      client.invalidateQueries({ queryKey: ["library"] });
    },
    onError: (error) => toast(error instanceof Error ? error.message : "The upload failed.", true),
  });

  const scan = useMutation({
    mutationFn: api.scan,
    onSuccess: () => client.invalidateQueries({ queryKey: ["activeJob"] }),
  });

  const analyse = useMutation({
    mutationFn: (limit?: number) => api.analyze(limit),
    onSuccess: () => client.invalidateQueries({ queryKey: ["activeJob"] }),
    onError: (error) => toast(error instanceof Error ? error.message : "The analysis could not start.", true),
  });

  const analyseOne = useMutation({
    mutationFn: (paper: PaperSummary) => api.analyze(undefined, [paper.id]),
    onSuccess: (_, paper) => {
      toast(`Analysing "${paper.title ?? paper.id}". Progress shows at the top.`);
      client.invalidateQueries({ queryKey: ["activeJob"] });
    },
    onError: (error) => toast(error instanceof Error ? error.message : "The analysis could not start.", true),
  });

  const activeJob = useQuery({ queryKey: ["activeJob"], queryFn: api.activeJob, staleTime: 0 });
  const busy = analyse.isPending || analyseOne.isPending || activeJob.data?.job?.kind === "analyze";

  const pending = library.data?.pending ?? 0;

  const onFiles = (list: FileList | null) => {
    const files = Array.from(list ?? []).filter((file) => file.name.toLowerCase().endsWith(".pdf"));
    if (files.length) upload.mutate(files);
    else if (list?.length) toast("Only PDF files can be added.", true);
  };

  return (
    <div className="mx-auto flex h-full max-w-6xl flex-col px-6 py-5">
      <div className="flex flex-none flex-wrap items-center gap-2">
        <h1 className="mr-auto text-[17px] font-semibold">Library</h1>
        <span className="tabular text-[13px] text-muted" data-testid="library-count">
          {papers.length} {papers.length === 1 ? "paper" : "papers"}, {pending} new
        </span>
        <button className="btn" onClick={() => scan.mutate()} title="Look for new PDFs in the papers folder">
          <RefreshCw size={14} /> Check folder
        </button>
        {pending > 5 && (
          <button className="btn" onClick={() => analyse.mutate(5)} data-testid="analyse-five">
            Analyse 5
          </button>
        )}
        <button className="btn btn-primary" disabled={pending === 0 || busy} onClick={() => analyse.mutate(undefined)} data-testid="analyse-all">
          Analyse new papers
        </button>
      </div>

      <label
        className="mt-3 flex flex-none cursor-pointer items-center justify-center gap-2 rounded border border-dashed py-3 text-[13.5px] text-muted transition-colors hover:bg-surface"
        style={{ borderColor: dragging ? "var(--accent)" : "var(--border)", background: dragging ? "var(--subtle)" : undefined }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          onFiles(event.dataTransfer.files);
        }}
      >
        <FileUp size={16} />
        <span>{upload.isPending ? "Adding papers" : "Drop PDFs here or add them to the papers folder"}</span>
        <input type="file" accept=".pdf,application/pdf" multiple className="sr-only" data-testid="upload" onChange={(event) => onFiles(event.target.files)} />
      </label>

      <div className="card mt-3 flex min-h-0 flex-col overflow-hidden">
        <div className={`${GRID} flex-none border-b border-line px-3 py-2 text-[12px] font-medium text-muted`}>
          <span>Title</span>
          <span>Year</span>
          <span>Status</span>
          <span className="text-right">Claims</span>
          <span />
        </div>
        {library.isError ? (
          <ErrorNote error={library.error} />
        ) : library.isLoading ? (
          <div className="p-3">
            <SkeletonRows rows={10} />
          </div>
        ) : papers.length === 0 ? (
          <Empty title="No papers yet">Drop PDFs above, or add them to {library.data?.folder}.</Empty>
        ) : (
          <div ref={scroller} className="min-h-0 flex-initial overflow-auto" data-testid="paper-list">
            <div style={{ height: virtual.getTotalSize(), position: "relative" }}>
              {virtual.getVirtualItems().map((row) => {
                const paper = papers[row.index];
                const status = statusText(paper);
                return (
                  <div
                    key={paper.id}
                    data-testid="paper-row"
                    onClick={() => navigate(`/paper/${paper.id}`)}
                    className={`${GRID} absolute left-0 top-0 w-full cursor-pointer border-b border-line px-3 transition-colors hover:bg-subtle`}
                    style={{ height: row.size, transform: `translateY(${row.start}px)`, background: row.index === active ? "var(--subtle)" : undefined }}
                  >
                    <span className="truncate text-[13.5px]" title={paper.title ?? ""}>
                      {paper.title}
                      {paper.source === "online" && <span className="ml-2 text-[11.5px] text-muted">added from search</span>}
                    </span>
                    <span className="tabular text-[13px] text-muted">{paper.year ?? ""}</span>
                    <span className="truncate text-[13px]" style={{ color: status.bad ? "var(--contradicts)" : "var(--text-2)" }} title={paper.parse_error ?? undefined}>
                      {status.text}
                    </span>
                    <span className="tabular text-right text-[13px]">{paper.extract_status === "done" || paper.claims ? paper.claims : ""}</span>
                    <span className="flex justify-end">
                      {canAnalyse(paper) && (
                        <button
                          className="btn h-7 px-2 text-[12.5px]"
                          data-testid="analyse-one"
                          disabled={busy}
                          title={busy ? "Wait for the current analysis to finish" : "Analyse only this paper"}
                          onClick={(event) => {
                            event.stopPropagation();
                            analyseOne.mutate(paper);
                          }}
                        >
                          <Play size={12} /> Analyse
                        </button>
                      )}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

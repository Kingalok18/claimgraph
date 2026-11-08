import { ReactNode, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import * as Dialog from "@radix-ui/react-dialog";
import { ArrowLeft, Download, RefreshCw, X } from "lucide-react";
import { api, Claim, Report } from "../api";
import { toast } from "../components/Status";
import { Empty, ErrorNote, lastTopic, rememberTopic, Skeleton, SkeletonRows } from "../components/ui";

const CITATION = /\[([A-Za-z0-9]+_c\d+)\]/g;

function withCitations(text: string, numbers: Map<string, number>, open: (id: string) => void): ReactNode[] {
  const output: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const index = match.index ?? 0;
    output.push(text.slice(last, index).replace(/\s+$/, ""));
    const id = match[1];
    if (!numbers.has(id)) numbers.set(id, numbers.size + 1);
    output.push(
      <a key={`${id}-${index}`} className="cite tabular" href="#" data-testid="citation" onClick={(event) => {
        event.preventDefault();
        open(id);
      }}>
        {numbers.get(id)}
      </a>,
    );
    last = index + match[0].length;
  }
  output.push(text.slice(last));
  return output;
}

function ReportBody({ report, onOpen }: { report: Report; onOpen: (id: string) => void }) {
  const content = useMemo(() => {
    const numbers = new Map<string, number>();
    const blocks: ReactNode[] = [];
    report.text.split("\n").forEach((line, index) => {
      const stripped = line.trim();
      if (!stripped) return;
      if (stripped.startsWith("#")) blocks.push(<h2 key={index}>{stripped.replace(/^#+\s*/, "")}</h2>);
      else blocks.push(<p key={index}>{withCitations(stripped.replace(/^[-*]\s+/, ""), numbers, onOpen)}</p>);
    });
    return blocks;
  }, [report, onOpen]);
  return <div className="report-body font-serif text-[16.5px] leading-[1.65]" data-testid="report-body">{content}</div>;
}

function ClaimDialog({ claim, onClose }: { claim: Claim | null; onClose: () => void }) {
  return (
    <Dialog.Root open={!!claim} onOpenChange={(open) => !open && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/20" />
        <Dialog.Content className="card fixed left-1/2 top-1/2 z-50 w-[min(560px,92vw)] -translate-x-1/2 -translate-y-1/2 p-5" data-testid="claim-dialog">
          <div className="flex items-start gap-3">
            <Dialog.Title className="flex-1 text-[14px] font-medium">{claim?.paper_title}{claim?.year ? `, ${claim.year}` : ""}</Dialog.Title>
            <Dialog.Close className="text-muted hover:text-ink" aria-label="Close">
              <X size={16} />
            </Dialog.Close>
          </div>
          <Dialog.Description asChild>
            <blockquote className="quote mt-3 border-l-2 border-line pl-3">{claim?.quote}</blockquote>
          </Dialog.Description>
          <div className="mt-3 text-[13px]">
            <Link className="text-accent hover:underline" to={`/paper/${claim?.paper_id}?page=${claim?.page ?? 1}&claim=${claim?.claim_id}`}>
              Open page {claim?.page ?? ""}
            </Link>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function TopicPicker() {
  const topics = useQuery({ queryKey: ["topics"], queryFn: api.topics });
  return (
    <div className="mx-auto h-full max-w-2xl overflow-auto px-6 py-5">
      <h1 className="text-[17px] font-semibold">Report</h1>
      <p className="mb-3 mt-1 text-[13.5px] text-muted">Choose a topic to read its evidence map.</p>
      {topics.isLoading ? (
        <SkeletonRows />
      ) : !topics.data?.topics.length ? (
        <Empty title="No topics yet">Topics appear after papers are analysed in the Library.</Empty>
      ) : (
        <div className="card overflow-hidden">
          {topics.data.topics.map((topic) => (
            <Link key={topic.id} to={`/report/${topic.id}`} className="block border-b border-line px-4 py-2.5 text-[14px] transition-colors last:border-b-0 hover:bg-subtle">
              {topic.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

export default function ReportPage() {
  const { id } = useParams();
  const remembered = lastTopic();
  if (!id && !remembered) return <TopicPicker />;
  if (!id) return <TopicPicker />;
  return <ReportView topicId={Number(id)} />;
}

function ReportView({ topicId }: { topicId: number }) {
  const client = useQueryClient();
  const [claimId, setClaimId] = useState<string | null>(null);
  useEffect(() => rememberTopic(topicId), [topicId]);
  const query = useQuery({ queryKey: ["report", topicId], queryFn: () => api.report(topicId), staleTime: Infinity, retry: 0 });
  const rewrite = useMutation({
    mutationFn: () => api.report(topicId, true),
    onSuccess: (data) => {
      client.setQueryData(["report", topicId], data);
      toast("The report was rewritten.");
    },
    onError: (error) => toast(error instanceof Error ? error.message : "The report could not be rewritten.", true),
  });

  const report = query.data;
  const busy = query.isLoading || rewrite.isPending;

  return (
    <div className="h-full overflow-auto">
      <article className="mx-auto max-w-[70ch] px-6 py-8">
        <div className="no-print mb-6 flex flex-wrap items-center gap-2">
          <Link to={`/topics/${topicId}`} className="mr-auto inline-flex items-center gap-1 text-[13px] text-muted hover:text-ink">
            <ArrowLeft size={14} /> Back to the graph
          </Link>
          <button className="btn" disabled={busy} onClick={() => rewrite.mutate()}>
            <RefreshCw size={14} /> Rewrite
          </button>
          <a className="btn" href={`/topics/${topicId}/report.md`} data-testid="export-md">
            <Download size={14} /> Markdown
          </a>
          <a className="btn" href={`/topics/${topicId}/report.pdf`} data-testid="export-pdf">
            <Download size={14} /> PDF
          </a>
        </div>

        {query.isError ? (
          <ErrorNote error={query.error} />
        ) : busy || !report ? (
          <div role="status">
            <Skeleton className="mb-5 h-7 w-3/4" />
            <p className="mb-5 text-[13.5px] text-muted">Writing the report. On this computer it can take a few minutes.</p>
            <SkeletonRows rows={6} height="h-5" />
          </div>
        ) : (
          <>
            <h1 className="text-[22px] font-semibold leading-snug" data-testid="report-title">{report.topic_label}</h1>
            <p className="tabular mb-2 mt-1 text-[12.5px] text-muted">
              {report.cited_claim_ids.length} claims cited
              {report.model === "template" ? ", assembled directly from the stored relations" : ""}
              {report.removed_count > 0 ? `, ${report.removed_count} unsupported ${report.removed_count === 1 ? "sentence" : "sentences"} removed` : ""}
            </p>
            <ReportBody report={report} onOpen={setClaimId} />
          </>
        )}
      </article>
      <ClaimDialog claim={claimId && report ? (report.claims[claimId] ?? null) : null} onClose={() => setClaimId(null)} />
    </div>
  );
}

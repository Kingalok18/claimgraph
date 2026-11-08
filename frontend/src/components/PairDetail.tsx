import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { Check } from "lucide-react";
import { api, Pair, PairSide, Relation, RELATIONS } from "../api";
import { directionWords, ErrorNote, RelationBadge, Skeleton } from "./ui";
import { toast } from "./Status";

const FIELD_LABELS: Record<string, string> = { dataset: "Dataset", model: "Model or population", metric: "Metric", setting: "Setting" };

function Side({ side, name }: { side: PairSide; name: string }) {
  return (
    <div className="min-w-0 flex-1" data-testid={`pair-side-${name}`}>
      <blockquote className="quote border-l-2 border-line pl-3">{side.quote}</blockquote>
      <div className="mt-2 pl-3 text-[13px] text-muted">
        <div className="text-ink">{side.paper_title}</div>
        <div className="tabular mt-0.5 flex flex-wrap items-center gap-x-2">
          {side.year && <span>{side.year}</span>}
          <span>{directionWords[side.direction] ?? side.direction}</span>
          {side.hedged && <span>hedged wording</span>}
          <Link className="text-accent hover:underline" to={`/paper/${side.paper_id}?page=${side.page ?? 1}&claim=${side.claim_id}`}>
            Open page{side.page ? ` ${side.page}` : ""}
          </Link>
        </div>
      </div>
    </div>
  );
}

export function explainRelation(pair: Pair): string {
  if (pair.reason) return pair.reason;
  if (pair.relation === "supports") return "The two findings say the same thing.";
  return "There is not enough here to decide.";
}

function reasonLine(pair: Pair): string {
  const nli = pair.nli_label && pair.nli_prob != null ? `The comparison model read "${pair.nli_label}" (${Math.round(pair.nli_prob * 100)}%).` : "";
  return `${explainRelation(pair)} ${nli}`.trim();
}

export function PairDetail({ pairId, onLabelled }: { pairId: number; onLabelled?: () => void }) {
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["pair", pairId], queryFn: () => api.pair(pairId) });
  const label = useMutation({
    mutationFn: (relation: Relation) => api.labelPair(pairId, relation),
    onMutate: async (relation) => {
      // The label shows immediately; a failure puts the old one back.
      await client.cancelQueries({ queryKey: ["pair", pairId] });
      const previous = client.getQueryData<Pair>(["pair", pairId]);
      if (previous) client.setQueryData(["pair", pairId], { ...previous, user_label: relation });
      return { previous };
    },
    onError: (error, _relation, context) => {
      if (context?.previous) client.setQueryData(["pair", pairId], context.previous);
      toast(error instanceof Error ? error.message : "The label was not saved.", true);
    },
    onSuccess: (data) => {
      client.setQueryData(["pair", pairId], data);
      client.invalidateQueries({ queryKey: ["graph", data.topic_id] });
      client.invalidateQueries({ queryKey: ["topics"] });
      client.invalidateQueries({ queryKey: ["labelStats"] });
      onLabelled?.();
    },
  });

  if (query.isError) return <ErrorNote error={query.error} />;
  if (!query.data) {
    return (
      <div className="space-y-4" role="status" aria-label="Loading">
        <Skeleton className="h-6 w-40" />
        <div className="flex gap-6">
          <Skeleton className="h-32 flex-1" />
          <Skeleton className="h-32 flex-1" />
        </div>
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  const pair = query.data;
  const shown = pair.user_label ?? pair.relation;
  const fields = Object.keys(FIELD_LABELS);

  return (
    <div data-testid="pair-detail">
      <div className="flex flex-wrap items-center gap-2">
        <RelationBadge relation={shown} />
        {pair.user_label && pair.user_label !== pair.relation && <span className="text-[12.5px] text-muted">Corrected by you</span>}
        {pair.user_label && pair.user_label === pair.relation && (
          <span className="inline-flex items-center gap-1 text-[12.5px] text-muted">
            <Check size={14} /> Confirmed by you
          </span>
        )}
      </div>
      <p className="mt-1.5 text-[13.5px] text-muted" data-testid="pair-reason">
        {reasonLine(pair)}
      </p>

      <div className="mt-4 flex flex-col gap-6 md:flex-row">
        <Side side={pair.a} name="a" />
        <Side side={pair.b} name="b" />
      </div>

      <table className="mt-5 w-full border-collapse text-[13.5px]" data-testid="condition-table">
        <thead>
          <tr className="border-b border-line text-left text-[12px] text-muted">
            <th className="w-1/4 py-1.5 pr-3 font-medium">Condition</th>
            <th className="w-[37.5%] py-1.5 pr-3 font-medium">First finding</th>
            <th className="py-1.5 font-medium">Second finding</th>
          </tr>
        </thead>
        <tbody>
          {fields.map((field) => {
            const differs = pair.differing_fields.includes(field);
            return (
              <tr key={field} className="border-b border-line" style={differs ? { background: "var(--diff)" } : undefined} data-differs={differs || undefined}>
                <td className="py-1.5 pl-1 pr-3 text-muted">
                  {FIELD_LABELS[field]}
                  {differs && <span className="ml-1.5 text-[11.5px] font-medium text-ink">differs</span>}
                </td>
                <td className="py-1.5 pr-3">{pair.a.conditions[field] ?? <span className="text-muted">not stated</span>}</td>
                <td className="py-1.5">{pair.b.conditions[field] ?? <span className="text-muted">not stated</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="mt-5">
        <div className="mb-1.5 text-[12.5px] text-muted">Is this right? Confirm it, or choose the correct relation.</div>
        <div className="flex flex-wrap gap-1.5">
          {RELATIONS.map((item) => (
            <button
              key={item.key}
              className="btn"
              aria-pressed={shown === item.key}
              style={shown === item.key ? { borderColor: item.colour, color: item.ink } : undefined}
              onClick={() => label.mutate(item.key)}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

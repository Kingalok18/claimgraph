import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import * as Tabs from "@radix-ui/react-tabs";
import { api, Relation, RELATIONS } from "../api";
import { PairDetail } from "../components/PairDetail";
import { toast } from "../components/Status";
import { Empty, SkeletonRows } from "../components/ui";

function PairLabelling() {
  const client = useQueryClient();
  const queue = useQuery({ queryKey: ["pairQueue"], queryFn: api.pairQueue });
  const current = queue.data?.items[0]?.id ?? null;

  const refresh = () => {
    client.invalidateQueries({ queryKey: ["pairQueue"] });
    client.invalidateQueries({ queryKey: ["labelStats"] });
  };

  useEffect(() => {
    if (current === null) return;
    const handler = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).tagName === "INPUT" || event.ctrlKey || event.metaKey) return;
      const choice = RELATIONS[Number(event.key) - 1];
      if (!choice) return;
      api
        .labelPair(current, choice.key as Relation)
        .then((data) => {
          client.setQueryData(["pair", current], data);
          refresh();
        })
        .catch((error) => toast(error.message, true));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [current]); // eslint-disable-line react-hooks/exhaustive-deps

  if (queue.isLoading) return <SkeletonRows rows={5} height="h-16" />;
  if (current === null) return <Empty title="Every pair is labelled">New pairs appear after more papers are analysed.</Empty>;
  return (
    <div className="card p-5">
      <PairDetail key={current} pairId={current} onLabelled={refresh} />
      <p className="mt-4 text-[12.5px] text-muted">
        Keys: {RELATIONS.map((item, index) => `${index + 1} ${item.label.toLowerCase()}`).join(", ")}
      </p>
    </div>
  );
}

const VERDICTS = [
  { key: "correct", label: "Correct" },
  { key: "wrong", label: "Wrong" },
  { key: "not_a_finding", label: "Not a finding" },
];

function ClaimLabelling() {
  const client = useQueryClient();
  const queue = useQuery({ queryKey: ["claimQueue"], queryFn: api.claimQueue });
  const claim = queue.data?.items[0];

  const submit = (verdict: string) => {
    if (!claim) return;
    api
      .labelClaim(claim.claim_id, verdict)
      .then(() => {
        client.invalidateQueries({ queryKey: ["claimQueue"] });
        client.invalidateQueries({ queryKey: ["labelStats"] });
      })
      .catch((error) => toast(error.message, true));
  };

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement).tagName === "INPUT" || event.ctrlKey || event.metaKey) return;
      const choice = VERDICTS[Number(event.key) - 1];
      if (choice) submit(choice.key);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  if (queue.isLoading) return <SkeletonRows rows={5} height="h-16" />;
  if (!claim) return <Empty title="Every claim is checked">New claims appear after more papers are analysed.</Empty>;
  return (
    <div className="card p-5" data-testid="claim-check">
      <div className="text-[12.5px] text-muted">{claim.paper_title}</div>
      <p className="mt-2 text-[14.5px] font-medium">{claim.text}</p>
      <blockquote className="quote mt-3 border-l-2 border-line pl-3">{claim.quote}</blockquote>
      <p className="mt-3 text-[13px] text-muted">Does the claim say what the quote says, and is it a finding of this paper?</p>
      <div className="mt-2 flex gap-1.5">
        {VERDICTS.map((item, index) => (
          <button key={item.key} className="btn" onClick={() => submit(item.key)}>
            <span className="tabular text-muted">{index + 1}</span> {item.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function LabellingPage() {
  const stats = useQuery({ queryKey: ["labelStats"], queryFn: api.labelStats });
  const [tab, setTab] = useState("pairs");
  const triggerClass =
    "rounded px-2.5 py-1 text-[13.5px] text-muted transition-colors data-[state=active]:bg-subtle data-[state=active]:font-medium data-[state=active]:text-ink";
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-5xl px-6 py-5">
        <h1 className="text-[17px] font-semibold">Labelling</h1>
        <p className="mt-1 text-[13.5px] text-muted">Your labels are the reference for measuring accuracy. They are never used to change a number by hand.</p>
        <Tabs.Root value={tab} onValueChange={setTab} className="mt-3">
          <Tabs.List className="mb-3 flex items-center gap-1">
            <Tabs.Trigger value="pairs" className={triggerClass}>
              Claim pairs
            </Tabs.Trigger>
            <Tabs.Trigger value="claims" className={triggerClass}>
              Claim checks
            </Tabs.Trigger>
            <span className="tabular ml-auto text-[12.5px] text-muted" data-testid="label-stats">
              {stats.data
                ? tab === "pairs"
                  ? `${stats.data.pair_labels} of ${stats.data.pairs_total} labelled`
                  : `${stats.data.claim_labels} of ${stats.data.claims_total} checked`
                : ""}
            </span>
          </Tabs.List>
          <Tabs.Content value="pairs">
            <PairLabelling />
          </Tabs.Content>
          <Tabs.Content value="claims">
            <ClaimLabelling />
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
}

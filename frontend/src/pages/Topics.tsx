import { useCallback, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api, RELATIONS, TopicSummary } from "../api";
import { toast } from "../components/Status";
import { Empty, ErrorNote, SkeletonRows, useListKeys } from "../components/ui";

function StackedBar({ topic }: { topic: TopicSummary }) {
  const parts = RELATIONS.map((item) => ({ ...item, count: topic[item.key] }));
  const total = parts.reduce((sum, part) => sum + part.count, 0);
  const words = parts.filter((part) => part.count).map((part) => `${part.count} ${part.label.toLowerCase()}`).join(", ");
  if (total === 0) return <div className="h-2 w-full rounded bg-subtle" title="No claim pairs" />;
  return (
    <div className="flex h-2 w-full gap-px overflow-hidden rounded" role="img" aria-label={words} title={words}>
      {parts
        .filter((part) => part.count)
        .map((part) => (
          <div key={part.key} style={{ width: `${(part.count / total) * 100}%`, background: part.colour }} />
        ))}
    </div>
  );
}

export default function TopicsPage() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const query = useQuery({ queryKey: ["topics"], queryFn: api.topics });
  const [question, setQuestion] = useState("");
  const topics = query.data?.topics ?? [];

  const open = useCallback((index: number) => topics[index] && navigate(`/topics/${topics[index].id}`), [topics, navigate]);
  const active = useListKeys(topics.length, open);

  const add = useMutation({
    mutationFn: () => api.addQuestion(question.trim()),
    onSuccess: () => {
      setQuestion("");
      client.invalidateQueries({ queryKey: ["activeJob"] });
    },
    onError: (error) => toast(error instanceof Error ? error.message : "The question could not be added.", true),
  });

  return (
    <div className="mx-auto h-full max-w-4xl overflow-auto px-6 py-5">
      <h1 className="text-[17px] font-semibold">Topics</h1>
      <form
        className="mt-3 flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (question.trim().length >= 3) add.mutate();
        }}
      >
        <input
          className="input flex-1"
          value={question}
          data-testid="question-input"
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Add a research question, for example: Does adversarial training reduce clean accuracy?"
          aria-label="Add a research question"
        />
        <button className="btn" disabled={question.trim().length < 3 || add.isPending}>
          Add question
        </button>
      </form>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-muted">
        {RELATIONS.map((item) => (
          <span key={item.key} className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: item.colour }} />
            {item.label}
          </span>
        ))}
      </div>

      <div className="mt-3">
        {query.isError ? (
          <ErrorNote error={query.error} />
        ) : query.isLoading ? (
          <SkeletonRows rows={6} height="h-16" />
        ) : topics.length === 0 ? (
          <Empty title="No topics yet">Topics appear after papers are analysed in the Library.</Empty>
        ) : (
          <div className="card overflow-hidden">
            {topics.map((topic, index) => (
              <button
                key={topic.id}
                data-testid="topic-row"
                className="block w-full border-b border-line px-4 py-3 text-left transition-colors last:border-b-0 hover:bg-subtle"
                style={index === active ? { background: "var(--subtle)" } : undefined}
                onClick={() => navigate(`/topics/${topic.id}`)}
              >
                <div className="flex items-baseline gap-3">
                  <span className="min-w-0 flex-1 truncate text-[14px] font-medium">{topic.label}</span>
                  {topic.source === "question" && <span className="text-[12px] text-muted">your question</span>}
                  <span className="tabular flex-none text-[12.5px] text-muted">
                    {topic.claims} {topic.claims === 1 ? "claim" : "claims"}
                  </span>
                </div>
                <div className="mt-2">
                  <StackedBar topic={topic} />
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

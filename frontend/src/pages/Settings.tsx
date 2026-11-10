import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { ErrorNote, SkeletonRows } from "../components/ui";
import { getTheme, setTheme, ThemeChoice } from "../theme";

function Row({ name, ok, children }: { name: string; ok?: boolean; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[170px_70px_minmax(0,1fr)] items-baseline gap-3 border-b border-line px-4 py-2.5 text-[13.5px] last:border-b-0" data-testid="setting-row">
      <span className="text-muted">{name}</span>
      <span className="text-[12.5px] font-medium" style={{ color: ok === undefined ? "transparent" : ok ? "var(--supports)" : "var(--contradicts)" }}>
        {ok === undefined ? "" : ok ? "Ready" : "Needs setup"}
      </span>
      <span className="min-w-0 break-words">{children}</span>
    </div>
  );
}

export default function SettingsPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: 5000 });
  const [theme, chooseTheme] = useState<ThemeChoice>(getTheme());

  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-3xl px-6 py-5">
        <h1 className="text-[17px] font-semibold">Settings</h1>

        <h2 className="mb-2 mt-5 text-[13.5px] font-semibold">Appearance</h2>
        <div className="flex gap-1.5" role="radiogroup" aria-label="Theme">
          {(["light", "dark", "system"] as ThemeChoice[]).map((choice) => (
            <button
              key={choice}
              role="radio"
              aria-checked={theme === choice}
              data-testid={`theme-${choice}`}
              className="btn capitalize"
              style={theme === choice ? { borderColor: "var(--accent)", color: "var(--accent)" } : undefined}
              onClick={() => {
                setTheme(choice);
                chooseTheme(choice);
              }}
            >
              {choice}
            </button>
          ))}
        </div>

        <h2 className="mb-2 mt-6 text-[13.5px] font-semibold">Models and services</h2>
        {health.isError ? (
          <ErrorNote error={health.error} />
        ) : !health.data ? (
          <SkeletonRows rows={5} />
        ) : (
          <div className="card">
            <Row name="Database" ok={health.data.services.database.ok}>
              {health.data.services.database.kind === "mysql" ? "MySQL" : health.data.services.database.kind}. {health.data.services.database.ok ? "" : health.data.services.database.detail}
            </Row>
            <Row name="Language model" ok={health.data.services.llm.ok}>
              {health.data.services.llm.model}. {health.data.services.llm.ok ? "Runs on this computer." : health.data.services.llm.detail}
            </Row>
            <Row name="Embedding model" ok={health.data.services.embeddings.ok}>
              {health.data.services.embeddings.model}. {health.data.services.embeddings.ok ? "" : "Downloads on first use, or run: python scripts/run.py setup"}
            </Row>
            <Row name="Comparison model" ok={health.data.services.nli.ok}>
              {health.data.services.nli.finetuned ? "Fine-tuned on SciFact." : `${health.data.services.nli.model}, not fine-tuned.`}{" "}
              {health.data.services.nli.ok ? "" : "Downloads on first use, or run: python scripts/run.py setup"}
            </Row>
            <Row name="Claim index" ok={health.data.services.vector_store.ok}>
              Qdrant, {health.data.services.vector_store.mode === "local" ? "inside the project folder" : "server"}.
            </Row>
          </div>
        )}

      </div>
    </div>
  );
}

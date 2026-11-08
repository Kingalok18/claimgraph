import { useEffect, useState } from "react";
import { relationInfo } from "../api";

const LAST_TOPIC = "claimgraph-last-topic";

export function rememberTopic(id: number | string) {
  try {
    localStorage.setItem(LAST_TOPIC, String(id));
  } catch {
    /* a convenience only */
  }
}

export function lastTopic(): string | null {
  try {
    return localStorage.getItem(LAST_TOPIC);
  } catch {
    return null;
  }
}

/** Relation badge. It always carries the word, because colour must not be the only signal. */
export function RelationBadge({ relation }: { relation: string | null | undefined }) {
  const info = relationInfo(relation);
  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded border border-line bg-surface px-2 py-0.5 text-[12.5px] font-medium"
      style={{ color: info.ink }}
    >
      <svg width="16" height="4" aria-hidden="true">
        <line x1="0" y1="2" x2="16" y2="2" stroke={info.colour} strokeWidth="2" strokeDasharray={info.line === "dashed" ? "4 3" : info.line === "dotted" ? "1.5 2.5" : undefined} />
      </svg>
      {info.label}
    </span>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

export function SkeletonRows({ rows = 6, height = "h-9" }: { rows?: number; height?: string }) {
  return (
    <div className="space-y-2" role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} className={`${height} w-full`} />
      ))}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="mx-auto max-w-md py-16 text-center">
      <div className="font-medium">{title}</div>
      {children && <div className="mt-1 text-[13.5px] text-muted">{children}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: unknown }) {
  return (
    <div className="card mx-auto my-8 max-w-lg px-4 py-3 text-[13.5px]" role="alert">
      <span className="font-medium text-contradicts">This could not be loaded. </span>
      {error instanceof Error ? error.message : "Check that the app is still running."}
    </div>
  );
}

/** j and k move through a list, Enter opens the current row. */
export function useListKeys(count: number, onOpen: (index: number) => void, onMove?: (index: number) => void) {
  const [active, setActive] = useState(-1);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key === "j" || event.key === "k") {
        event.preventDefault();
        setActive((current) => {
          const next = event.key === "j" ? Math.min(current + 1, count - 1) : Math.max(current - 1, 0);
          onMove?.(next);
          return next;
        });
      } else if (event.key === "Enter" && active >= 0 && target.tagName !== "BUTTON" && target.tagName !== "A") {
        onOpen(active);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [count, active, onOpen, onMove]);
  return active;
}

export const directionWords: Record<string, string> = {
  increase: "increases",
  decrease: "decreases",
  no_effect: "no effect",
  mixed: "mixed",
};

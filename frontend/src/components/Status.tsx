import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "framer-motion";
import { api, Job } from "../api";

type ToastMessage = { id: number; text: string; bad?: boolean };
let nextToast = 1;
const listeners = new Set<(toast: ToastMessage) => void>();

export function toast(text: string, bad = false) {
  const message = { id: nextToast++, text, bad };
  listeners.forEach((listener) => listener(message));
}

export function Toasts() {
  const [items, setItems] = useState<ToastMessage[]>([]);
  useEffect(() => {
    const listener = (message: ToastMessage) => {
      setItems((current) => [...current, message]);
      window.setTimeout(() => setItems((current) => current.filter((item) => item.id !== message.id)), message.bad ? 9000 : 4500);
    };
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2" role="status" aria-live="polite">
      <AnimatePresence>
        {items.map((item) => (
          <motion.div
            key={item.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16, ease: "easeOut" }}
            className="card pointer-events-auto px-3 py-2 text-[13.5px]"
            style={item.bad ? { borderColor: "var(--contradicts)" } : undefined}
          >
            {item.text}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

function finishedText(job: Job): string {
  const summary = (job.progress.summary ?? {}) as Record<string, any>;
  if (job.kind === "scan") {
    const added = summary.new ?? 0;
    return added ? `${added} ${added === 1 ? "paper" : "papers"} added to the library.` : "The library is up to date.";
  }
  if (job.kind === "analyze") {
    const kept = summary.extraction?.kept ?? 0;
    return `Analysis finished. ${kept} new ${kept === 1 ? "claim" : "claims"} found.`;
  }
  if (job.kind === "question") return summary.topic_id ? "Your research question is ready." : (summary.message ?? "No claims matched.");
  return "Finished.";
}

/** The slim strip at the top of the window, shown only while something runs. */
export function ProgressStrip() {
  const client = useQueryClient();
  const seen = useRef<number | null>(null);
  const active = useQuery({
    queryKey: ["activeJob"],
    queryFn: api.activeJob,
    refetchInterval: (query) => (query.state.data?.job ? 1500 : 4000),
    staleTime: 0,
  });
  const job = active.data?.job ?? null;

  useEffect(() => {
    if (job) {
      seen.current = job.id;
      if (job.kind === "scan" || job.progress.step === "claims") {
        client.invalidateQueries({ queryKey: ["library"] });
      }
      return;
    }
    if (seen.current !== null) {
      const finished = seen.current;
      seen.current = null;
      api
        .job(finished)
        .then((result) => {
          if (result.status === "failed") toast(result.error ?? "The job did not finish.", true);
          else if (result.status === "cancelled") toast("Stopped. Finished work was kept.");
          else toast(finishedText(result));
        })
        .catch(() => undefined);
      client.invalidateQueries();
    }
  }, [job, client]);

  const total = job?.progress.total ?? 0;
  const done = job?.progress.done ?? 0;
  const percent = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null;

  return (
    <div className="flex-none no-print" style={{ height: job ? undefined : 0, overflow: "hidden" }}>
      {job && (
        <div className="border-b border-line bg-surface">
          <div className="relative h-[3px] overflow-hidden bg-subtle" role="progressbar" aria-valuenow={percent ?? undefined} aria-valuemin={0} aria-valuemax={100}>
            {percent !== null ? (
              <div className="h-full bg-accent transition-[width] duration-200" style={{ width: `${percent}%` }} />
            ) : (
              <div className="strip-indeterminate h-full bg-accent" />
            )}
          </div>
          <div className="flex items-center gap-3 px-4 py-1 text-[12.5px] text-muted">
            <span data-testid="job-label" className="tabular font-medium text-ink">
              {job.progress.label ?? "Working"}
            </span>
            {job.progress.current && <span className="truncate">{job.progress.current}</span>}
            {job.kind !== "scan" && (
              <button className="ml-auto flex-none hover:text-ink" onClick={() => api.cancelJob(job.id).then(() => toast("Stopping after the current step."))}>
                Stop
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Search } from "lucide-react";
import { api } from "../api";

export function SearchBox() {
  const [value, setValue] = useState("");
  const [term, setTerm] = useState("");
  const [open, setOpen] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const timer = window.setTimeout(() => setTerm(value.trim()), 180);
    return () => window.clearTimeout(timer);
  }, [value]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (event.key === "/" && target.tagName !== "INPUT" && target.tagName !== "TEXTAREA") {
        event.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const results = useQuery({ queryKey: ["search", term], queryFn: () => api.search(term), enabled: term.length >= 2, staleTime: 5000 });
  const papers = results.data?.papers ?? [];
  const claims = results.data?.claims ?? [];
  const hasResults = papers.length + claims.length > 0;

  const go = (path: string) => {
    setOpen(false);
    setValue("");
    input.current?.blur();
    navigate(path);
  };

  return (
    <div className="relative">
      <Search size={15} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-muted" />
      <input
        ref={input}
        value={value}
        data-testid="search"
        onChange={(event) => {
          setValue(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setOpen(false);
            input.current?.blur();
          }
          if (event.key === "Enter") {
            if (papers[0]) go(`/paper/${papers[0].id}`);
            else if (claims[0]) go(`/paper/${claims[0].paper_id}?page=${claims[0].page ?? 1}`);
          }
        }}
        placeholder="Search claims and papers"
        aria-label="Search claims and papers"
        className="input w-64 pl-8 pr-7 lg:w-80"
      />
      <kbd className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-muted">/</kbd>
      {open && term.length >= 2 && (
        <div className="card absolute right-0 top-full z-40 mt-1 max-h-[26rem] w-[30rem] max-w-[90vw] overflow-auto py-1">
          {!hasResults && <div className="px-3 py-2 text-[13.5px] text-muted">{results.isLoading ? "Searching" : "Nothing found."}</div>}
          {papers.length > 0 && <div className="px-3 pb-0.5 pt-1.5 text-[11.5px] font-medium uppercase tracking-wide text-muted">Papers</div>}
          {papers.map((paper) => (
            <button key={paper.id} className="block w-full px-3 py-1.5 text-left text-[13.5px] hover:bg-subtle" onMouseDown={() => go(`/paper/${paper.id}`)}>
              {paper.title}
              {paper.year && <span className="tabular ml-1.5 text-muted">{paper.year}</span>}
            </button>
          ))}
          {claims.length > 0 && <div className="px-3 pb-0.5 pt-1.5 text-[11.5px] font-medium uppercase tracking-wide text-muted">Claims</div>}
          {claims.map((claim) => (
            <button
              key={claim.claim_id}
              className="block w-full px-3 py-1.5 text-left text-[13.5px] hover:bg-subtle"
              onMouseDown={() => go(`/paper/${claim.paper_id}?page=${claim.page ?? 1}`)}
            >
              {claim.text}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

import { useEffect, useRef, useState } from "react";
import { NavLink, Route, Routes, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle } from "lucide-react";
import { api } from "./api";
import { ProgressStrip, Toasts } from "./components/Status";
import { SearchBox } from "./components/SearchBox";
import { lastTopic } from "./components/ui";
import LibraryPage from "./pages/Library";
import PaperPage from "./pages/Paper";
import TopicsPage from "./pages/Topics";
import TopicGraphPage from "./pages/TopicGraph";
import PairPage from "./pages/Pair";
import ReportPage from "./pages/Report";
import LabellingPage from "./pages/Labelling";
import SettingsPage from "./pages/Settings";

const NAV = [
  { to: "/library", label: "Library" },
  { to: "/topics", label: "Topics" },
  { to: "/report", label: "Report" },
  { to: "/labelling", label: "Labelling" },
  { to: "/settings", label: "Settings" },
];

function isTyping(target: EventTarget | null) {
  const element = target as HTMLElement | null;
  return !!element && (element.tagName === "INPUT" || element.tagName === "TEXTAREA" || element.isContentEditable);
}

/** "g" then a letter goes to a screen. Lists handle j, k, Enter and Esc themselves. */
function useGoShortcuts() {
  const navigate = useNavigate();
  const waiting = useRef(false);
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (isTyping(event.target) || event.ctrlKey || event.metaKey || event.altKey) return;
      if (waiting.current) {
        waiting.current = false;
        if (event.key === "l") navigate("/library");
        if (event.key === "t") navigate("/topics");
        if (event.key === "r") navigate(lastTopic() ? `/report/${lastTopic()}` : "/report");
        return;
      }
      if (event.key === "g") {
        waiting.current = true;
        window.setTimeout(() => (waiting.current = false), 1200);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [navigate]);
}

export default function App() {
  useGoShortcuts();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, staleTime: 30_000, refetchInterval: 60_000 });
  const [dismissed, setDismissed] = useState(false);
  const blocking = health.data?.setup_problems.filter((problem) => problem.blocking) ?? [];
  const name = health.data?.app_name ?? "ClaimGraph";

  useEffect(() => {
    document.title = name;
  }, [name]);

  return (
    <div className="flex h-full flex-col">
      <ProgressStrip />
      <header className="flex h-12 flex-none items-center gap-6 border-b border-line bg-surface px-4">
        <span className="text-[14.5px] font-semibold tracking-tight">{name}</span>
        <nav className="flex items-center gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded px-2.5 py-1 text-[13.5px] transition-colors ${isActive ? "bg-subtle font-medium text-ink" : "text-muted hover:text-ink"}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="ml-auto">
          <SearchBox />
        </div>
      </header>

      {blocking.length > 0 && !dismissed && (
        <div className="flex flex-none items-start gap-2 border-b border-line bg-subtle px-4 py-2 text-[13.5px]" role="alert">
          <AlertCircle size={16} className="mt-0.5 flex-none text-contradicts" />
          <div className="flex-1">
            {blocking.map((problem) => (
              <div key={problem.service}>
                <span className="font-medium capitalize">{problem.service}: </span>
                {problem.detail}
              </div>
            ))}
          </div>
          <NavLink to="/settings" className="text-accent hover:underline">
            Setup
          </NavLink>
          <button className="text-muted hover:text-ink" onClick={() => setDismissed(true)}>
            Hide
          </button>
        </div>
      )}

      <main className="min-h-0 flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<LibraryPage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/paper/:id" element={<PaperPage />} />
          <Route path="/topics" element={<TopicsPage />} />
          <Route path="/topics/:id" element={<TopicGraphPage />} />
          <Route path="/pair/:id" element={<PairPage />} />
          <Route path="/report" element={<ReportPage />} />
          <Route path="/report/:id" element={<ReportPage />} />
          <Route path="/labelling" element={<LabellingPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
      <Toasts />
    </div>
  );
}

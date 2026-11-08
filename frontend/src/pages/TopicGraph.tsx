import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import cytoscape, { Core } from "cytoscape";
import fcose from "cytoscape-fcose";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowLeft, FileText, X } from "lucide-react";
import { api, GraphData, relationInfo, RELATIONS } from "../api";
import { PairDetail } from "../components/PairDetail";
import { Empty, ErrorNote, rememberTopic, Skeleton } from "../components/ui";

cytoscape.use(fcose);
const COLLAPSE_ABOVE = 300;

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function graphStyle(): cytoscape.StylesheetCSS[] {
  const relationEdge = (key: string, variable: string, line: string) => ({
    selector: `edge[relation = "${key}"]`,
    css: { "line-color": cssVar(variable), "line-style": line } as any,
  });
  return [
    {
      selector: "node",
      css: {
        "background-color": "data(colour)",
        width: 16,
        height: 16,
        label: "data(label)",
        "font-family": "Inter Variable, sans-serif",
        "font-size": 9,
        color: cssVar("--text-2"),
        "text-wrap": "ellipsis",
        "text-max-width": "130px",
        "text-valign": "bottom",
        "text-margin-y": 4,
        "transition-property": "width height",
        "transition-duration": 150,
      } as any,
    },
    { selector: "node:selected", css: { width: 22, height: 22, "border-width": 2, "border-color": cssVar("--accent"), color: cssVar("--text") } as any },
    { selector: "edge", css: { width: 1.6, "curve-style": "bezier", opacity: 0.85 } as any },
    { selector: "edge:selected", css: { width: 3, opacity: 1 } as any },
    { selector: "edge.hover", css: { width: 3, opacity: 1 } as any },
    relationEdge("supports", "--supports", "solid"),
    relationEdge("contradicts", "--contradicts", "solid"),
    relationEdge("different_setting", "--setting", "dashed"),
    relationEdge("unclear", "--unclear", "dotted"),
  ];
}

/** Very large graphs start as one node per paper, with edges totalled between papers. */
function collapseByPaper(data: GraphData): cytoscape.ElementDefinition[] {
  const counts = new Map<string, number>();
  data.elements.nodes.forEach((node) => counts.set(node.data.paper_id, (counts.get(node.data.paper_id) ?? 0) + 1));
  const nodes = data.papers.map((paper) => ({
    data: { id: paper.paper_id, label: `${paper.title} (${counts.get(paper.paper_id) ?? 0})`, colour: paper.colour },
  }));
  const owner = new Map(data.elements.nodes.map((node) => [node.data.id, node.data.paper_id]));
  const edges = new Map<string, { source: string; target: string; relation: string }>();
  for (const edge of data.elements.edges) {
    const [source, target] = [owner.get(edge.data.source)!, owner.get(edge.data.target)!].sort();
    if (source === target) continue;
    const key = `${source}|${target}|${edge.data.relation}`;
    if (!edges.has(key)) edges.set(key, { source, target, relation: edge.data.relation });
  }
  return [...nodes, ...[...edges.entries()].map(([id, item]) => ({ data: { id, ...item } }))];
}

export default function TopicGraphPage() {
  const params = useParams();
  const topicId = Number(params.id);
  const query = useQuery({ queryKey: ["graph", topicId], queryFn: () => api.graph(topicId) });
  const box = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const [pairId, setPairId] = useState<number | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const [expanded, setExpanded] = useState(false);
  const data = query.data;
  const nodeCount = data?.elements.nodes.length ?? 0;
  const collapsed = nodeCount > COLLAPSE_ABOVE && !expanded;

  useEffect(() => rememberTopic(topicId), [topicId]);

  const elements = useMemo(() => {
    if (!data) return [];
    return collapsed ? collapseByPaper(data) : [...data.elements.nodes, ...data.elements.edges];
  }, [data, collapsed]);

  useEffect(() => {
    if (!box.current || !elements.length) return;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const instance = cytoscape({
      container: box.current,
      elements,
      style: graphStyle(),
      layout: { name: "fcose", animate: !reduced, animationDuration: 400, randomize: true, nodeRepulsion: 9000, idealEdgeLength: 170, padding: 40 } as any,
      minZoom: 0.15,
      // Fitting a small graph to the window would blow the labels up, so the first view stops near
      // actual size. The reader can zoom in further once the layout has settled.
      maxZoom: 1.25,
      wheelSensitivity: 0.25,
    });
    instance.one("layoutstop", () => instance.maxZoom(3));
    cy.current = instance;
    instance.on("tap", "edge", (event) => {
      const id = event.target.data("pair_id");
      if (id) setPairId(id);
    });
    instance.on("tap", "node", (event) => {
      instance.animate({ center: { eles: event.target }, duration: reduced ? 0 : 200, easing: "ease-out" });
    });
    instance.on("mouseover", "edge", (event) => {
      event.target.addClass("hover");
      const point = event.renderedPosition;
      setHover({ x: point.x, y: point.y, text: relationInfo(event.target.data("relation")).label });
    });
    instance.on("mouseout", "edge", (event) => {
      event.target.removeClass("hover");
      setHover(null);
    });
    instance.on("mouseover", "node", (event) => {
      const point = event.target.renderedPosition();
      const node = event.target.data();
      setHover({ x: point.x, y: point.y, text: node.text ? `${node.text} (${node.paper_title})` : node.label });
    });
    instance.on("mouseout", "node", () => setHover(null));
    const restyle = () => instance.style(graphStyle() as any);
    window.addEventListener("claimgraph-theme", restyle);
    return () => {
      window.removeEventListener("claimgraph-theme", restyle);
      instance.destroy();
      cy.current = null;
    };
  }, [elements]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setPairId(null);
        cy.current?.elements().unselect();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  if (query.isError) return <ErrorNote error={query.error} />;

  return (
    <div className="relative flex h-full flex-col">
      <div className="flex flex-none flex-wrap items-center gap-3 border-b border-line bg-surface px-4 py-2">
        <Link to="/topics" className="inline-flex items-center gap-1 text-[13px] text-muted hover:text-ink">
          <ArrowLeft size={14} /> Topics
        </Link>
        {data ? <h1 className="text-[14.5px] font-semibold" data-testid="topic-title">{data.topic.label}</h1> : <Skeleton className="h-5 w-64" />}
        <span className="tabular text-[12.5px] text-muted">
          {data ? `${nodeCount} claims, ${data.elements.edges.length} relations` : ""}
        </span>
        {nodeCount > COLLAPSE_ABOVE && (
          <button className="btn" onClick={() => setExpanded((value) => !value)}>
            {collapsed ? "Show every claim" : "Group by paper"}
          </button>
        )}
        <Link to={`/report/${topicId}`} className="btn ml-auto" data-testid="open-report">
          <FileText size={14} /> Report
        </Link>
      </div>

      <div className="relative min-h-0 flex-1">
        {!data ? (
          <div className="p-6">
            <Skeleton className="h-full min-h-[400px] w-full" />
          </div>
        ) : nodeCount === 0 ? (
          <Empty title="This topic has no claims" />
        ) : (
          <div ref={box} className="absolute inset-0" data-testid="graph" aria-label="Claim graph" />
        )}

        {hover && (
          <div className="card pointer-events-none absolute z-10 max-w-xs px-2 py-1 text-[12.5px]" style={{ left: hover.x + 12, top: hover.y + 12 }}>
            {hover.text}
          </div>
        )}

        {data && nodeCount > 0 && (
          <div className="card absolute bottom-3 left-3 z-10 px-3 py-2 text-[12.5px] text-muted">
            {RELATIONS.map((item) => (
              <div key={item.key} className="flex items-center gap-2">
                <svg width="22" height="6" aria-hidden="true">
                  <line x1="0" y1="3" x2="22" y2="3" stroke={item.colour} strokeWidth="2" strokeDasharray={item.line === "dashed" ? "5 3" : item.line === "dotted" ? "1.5 3" : undefined} />
                </svg>
                {item.label}
              </div>
            ))}
            <div className="mt-1 text-[11.5px]">Each dot colour is one paper. Select a line to compare its two claims.</div>
          </div>
        )}

        <AnimatePresence>
          {pairId !== null && (
            <motion.aside
              key="pair"
              initial={{ x: 24, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 24, opacity: 0 }}
              transition={{ duration: 0.18, ease: "easeOut" }}
              className="absolute bottom-0 right-0 top-0 z-20 w-[min(720px,92%)] overflow-auto border-l border-line bg-surface p-5"
              data-testid="pair-panel"
            >
              <div className="mb-3 flex items-center justify-between">
                <Link to={`/pair/${pairId}`} className="text-[13px] text-accent hover:underline">
                  Open full view
                </Link>
                <button className="text-muted hover:text-ink" aria-label="Close" onClick={() => setPairId(null)}>
                  <X size={16} />
                </button>
              </div>
              <PairDetail pairId={pairId} />
            </motion.aside>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

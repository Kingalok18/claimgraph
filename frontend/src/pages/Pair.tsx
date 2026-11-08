import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { api } from "../api";
import { PairDetail } from "../components/PairDetail";

export default function PairPage() {
  const pairId = Number(useParams().id);
  const pair = useQuery({ queryKey: ["pair", pairId], queryFn: () => api.pair(pairId) });
  return (
    <div className="h-full overflow-auto">
      <div className="mx-auto max-w-5xl px-6 py-5">
        <Link to={pair.data ? `/topics/${pair.data.topic_id}` : "/topics"} className="mb-4 inline-flex items-center gap-1 text-[13px] text-muted hover:text-ink">
          <ArrowLeft size={14} /> Back to the graph
        </Link>
        <h1 className="mb-3 text-[17px] font-semibold">Claim pair</h1>
        <div className="card p-5">
          <PairDetail pairId={pairId} />
        </div>
      </div>
    </div>
  );
}

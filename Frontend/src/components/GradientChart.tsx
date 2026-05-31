import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { GradientItem } from "../types";

interface Props {
  items: GradientItem[];
  /** "signed": TF-IDF attribution (pos=injection, neg=safe). "magnitude": gradient norm (always +). */
  chartType: "signed" | "magnitude";
}

const MAX_ITEMS = 25;

function truncate(s: string, n = 18) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function itemColor(score: number, chartType: "signed" | "magnitude"): string {
  if (chartType === "magnitude") {
    // Blue shade scaled by relative magnitude — handled via opacity via Cell fill
    return "#4f8ef7";
  }
  return score >= 0 ? "#f87171" : "#4ade80";
}

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: GradientItem }[];
}) => {
  if (!active || !payload?.length) return null;
  const { word, score } = payload[0].payload;
  return (
    <div
      style={{
        background: "#1e2535",
        border: "1px solid #2d3748",
        borderRadius: 6,
        padding: "6px 10px",
        fontSize: 12,
        color: "#e2e8f0",
      }}
    >
      <strong>{word}</strong>
      <br />
      {score.toFixed(4)}
    </div>
  );
};

export default function GradientChart({ items, chartType }: Props) {
  const sorted =
    chartType === "signed"
      ? [...items]
          .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
          .slice(0, MAX_ITEMS)
      : [...items].sort((a, b) => b.score - a.score).slice(0, MAX_ITEMS);

  const data = sorted.map((d) => ({
    ...d,
    word: truncate(d.word),
    fullWord: d.word,
  }));

  const barHeight = 22;
  const chartHeight = Math.max(120, data.length * barHeight + 40);

  return (
    <ResponsiveContainer width="100%" height={chartHeight}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
      >
        <XAxis
          type="number"
          tick={{ fill: "#64748b", fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: "#2d3748" }}
          tickFormatter={(v: number) => v.toFixed(2)}
        />
        <YAxis
          type="category"
          dataKey="word"
          width={110}
          tick={{ fill: "#94a3b8", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
        {chartType === "signed" && <ReferenceLine x={0} stroke="#2d3748" />}
        <Bar dataKey="score" radius={[0, 3, 3, 0]} maxBarSize={18}>
          {data.map((entry, i) => (
            <Cell key={i} fill={itemColor(entry.score, chartType)} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

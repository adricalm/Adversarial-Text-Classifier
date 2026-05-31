import { ModelResult } from "../types";
import GradientChart from "./GradientChart";

interface Props {
  title: string;
  subtitle: string;
  result: ModelResult | null;
  loading: boolean;
  chartType: "signed" | "magnitude";
}

export default function ModelPanel({ title, subtitle, result, loading, chartType }: Props) {
  const cls = result?.label === "INJECTION" ? "injection" : "safe";

  return (
    <div className="model-panel">
      <div className="panel-header">
        <div className="panel-title">{title}</div>
        <div className="panel-subtitle">{subtitle}</div>
      </div>

      {loading ? (
        <div className="panel-placeholder">
          <div className="loading-dots">
            <span /><span /><span />
          </div>
        </div>
      ) : result === null ? (
        <div className="panel-placeholder">Enter text and click Predict</div>
      ) : (
        <>
          <div className="prediction">
            <span className={`badge ${cls}`}>{result.label}</span>
          </div>

          <div className="confidence-row">
            <span>Confidence</span>
            <div className="conf-bar-track">
              <div
                className={`conf-bar-fill ${cls}`}
                style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
              />
            </div>
            <span className="conf-pct">{(result.confidence * 100).toFixed(1)}%</span>
          </div>

          {result.items.length > 0 && (
            <div className="chart-section">
              <h4>
                {"Embedding gradient sensitivity (L2 norm)"}
              </h4>
              <GradientChart items={result.items} chartType={chartType} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

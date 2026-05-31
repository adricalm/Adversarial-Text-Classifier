export interface GradientItem {
  word: string;
  score: number;
}

export interface ModelResult {
  label: "INJECTION" | "SAFE";
  confidence: number;
  items: GradientItem[];
}

export interface PredictResponse {
  tfidf: ModelResult;
  pretrained: ModelResult;
  finetuned: ModelResult;
}

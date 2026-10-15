import type { components } from "./generated";
/** API fields are generated from the backend OpenAPI contract. */
export type ApiHealth = components["schemas"]["Health"];
export type SearchResult = components["schemas"]["SearchResult"];
export type ApiQueryRequest = components["schemas"]["QueryRequest"];
export type Paper = components["schemas"]["Paper"];
export type Citation = components["schemas"]["Citation"];
export type Claim = components["schemas"]["Claim"];
export type AnswerResult = Omit<components["schemas"]["AnswerResult"], "source" | "verification"> & {
  verification: { status: "demo" | "verified" | "skipped" | "unavailable" };
  source: "demo" | "api";
};
export type AnswerRequest = { query: string; paper_ids: string[] };
export type AnswerJob = Omit<components["schemas"]["Job"], "result" | "error"> & {
  result: AnswerResult | SearchResult | null;
  error: components["schemas"]["ApiError"] | null;
};
export type QuestionRecord = {
  id: string;
  query: string;
  paper_ids: string[];
  result: AnswerResult;
  created_at: string;
  dataset_id: string;
};

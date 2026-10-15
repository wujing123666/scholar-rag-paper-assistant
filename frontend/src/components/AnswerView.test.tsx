import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { answerDemo, exampleQueries } from "../mocks/demo";
import AnswerView from "./AnswerView";

afterEach(cleanup);
describe("final answer and source display", () => {
  it("opens the exact citation without generating a new answer", () => {
    const result = answerDemo({ query: exampleQueries[0].query, paper_ids: [] });
    const onCitation = vi.fn();
    render(<AnswerView result={result} onCitation={onCitation} />);
    fireEvent.click(screen.getAllByRole("button", { name: "查看引用 C1，第1页" })[0]);
    expect(onCitation).toHaveBeenCalledExactlyOnceWith(result.citations[0]);
    expect(screen.getByText(/未调用生成模型或 Claim 评审/)).toBeTruthy();
  });
  it("renders claim text literally without interpreting HTML", () => {
    const result = answerDemo({ query: exampleQueries[0].query, paper_ids: [] });
    result.claims[0].text = '<img src=x onerror="alert(1)">';
    const { container } = render(<AnswerView result={result} onCitation={() => {}} />);
    expect(container.querySelector("img")).toBeNull();
    expect(screen.getByText(/<img src=x/)).toBeTruthy();
  });
});

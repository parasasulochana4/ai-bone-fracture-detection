import { createGoogleGenerativeAI } from "@ai-sdk/google";

export function getAiModel() {
  const apiKey = process.env["GEMINI_API_KEY"];
  if (!apiKey) throw new Error("AI is not configured (missing GEMINI_API_KEY)");
  const google = createGoogleGenerativeAI({ apiKey });
  return google(process.env["GEMINI_MODEL"] || "gemini-flash-latest");
}

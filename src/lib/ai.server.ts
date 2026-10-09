import { createOpenAI } from "@ai-sdk/openai";

// Google Gemini via its OpenAI-compatible endpoint, using your own GEMINI_API_KEY.
const GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai";
export const AI_MODEL = "gemini-2.5-flash";

export function createAiProvider(_request?: Request) {
  const apiKey = process.env["GEMINI_API_KEY"];
  if (!apiKey) throw new Error("AI is not configured (missing GEMINI_API_KEY)");
  const provider = createOpenAI({ baseURL: GEMINI_URL, apiKey });
  return { provider };
}

export const RESPONSES_PROVIDER_OPTIONS = {} as const;

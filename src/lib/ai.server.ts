import { createGoogleGenerativeAI } from "@ai-sdk/google";

export const AI_MODEL = "gemini-2.5-flash";

/**
 * Google Gemini provider for X-ray analysis and doctor chat.
 * Reads GOOGLE_API_KEY from the environment (server-side only).
 */
export function createAiProvider() {
  const apiKey = process.env["GOOGLE_API_KEY"];
  if (!apiKey) throw new Error("AI is not configured (missing GOOGLE_API_KEY)");
  const provider = createGoogleGenerativeAI({ apiKey });
  return { provider };
}

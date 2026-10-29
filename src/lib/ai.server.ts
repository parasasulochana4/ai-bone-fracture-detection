import { createGoogleGenerativeAI } from "@ai-sdk/google";

export const AI_MODEL = "gemini-3.8-flash";

/**
 * Fallback models used only when the primary model is temporarily
 * unavailable (overloaded / rate-limited). Ordered by preference.
 */
export const FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.6-flash"];

/** Errors that mean "try again / try another model", not "bad input". */
export function isTransientError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const message = `${error.name} ${error.message}`;
  return (
    /AI_RetryError/i.test(message) ||
    /high demand|overloaded|rate.?limit|too many requests|unavailable|internal error|timeout|aborted/i.test(
      message,
    )
  );
}

/**
 * Google Gemini provider for X-ray analysis and doctor chat.
 * Reads GOOGLE_API_KEY from the environment (server-side only).
 */
export function createAiProvider() {
  const apiKey = process.env["GOOGLE_API_KEY"];
  if (!apiKey) throw new Error("AI is not configured (missing GOOGLE_API_KEY)");
  const provider = createGoogleGenerativeAI({ apiKey });
  return { provider, modelChain: [AI_MODEL, ...FALLBACK_MODELS] };
}

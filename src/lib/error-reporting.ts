/**
 * Client-side error reporting hook.
 * Called from the root error boundary; logs to the browser console.
 */
export function reportError(error: unknown, context: Record<string, unknown> = {}) {
  if (typeof window === "undefined") return;
  const message =
    error instanceof Response
      ? `Response ${error.status}${error.url ? ` at ${error.url}` : ""}`
      : error instanceof Error
        ? error.message
        : String(error);
  const stack = error instanceof Error ? error.stack : undefined;
  console.error("[FractureAI] Unhandled error:", message, { ...context, stack });
}

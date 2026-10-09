<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

# Architecture rules

- AI calls go through Lovable AI Gateway (`openai/gpt-6-astra`) via `src/lib/ai.server.ts`; never expose `LOVABLE_API_KEY` client-side.
- X-ray analysis is a one-shot `createServerFn` in `src/lib/analyze.functions.ts` returning a structured zod object (fracture verdict, confidence, hotspots, treatment plan).
- The "heatmap" is a client-side Grad-CAM-style overlay (`src/components/HeatmapOverlay.tsx`) drawn from AI-returned normalized hotspot coordinates — it is not true Grad-CAM from a trained model.
- AI doctor chat streams through `src/routes/api/chat.ts` (AI SDK `useChat` + `DefaultChatTransport`); the analysis result is injected into the system prompt as context. No chat persistence.
- PDF reports are generated client-side with jsPDF in `src/lib/report.ts`.
- Dark clinical theme (teal primary) is defined as `:root` tokens in `src/styles.css`; use semantic tokens, never hardcoded colors.

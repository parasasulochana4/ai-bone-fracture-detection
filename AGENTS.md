# Architecture rules

- Fracture detection runs a trained DenseNet121 classifier **in the browser** via `onnxruntime-web` (`src/lib/model.ts`, model at `public/models/fracture-densenet121.onnx`, WASM runtime in `public/ort/`). The model outputs `fracture_probability` and a 7×7 class activation map (`cam`).
- The training pipeline lives in `ml/` (see `ml/README.md`); measured metrics are in `ml/metrics.json` and are what the site displays. Never hand-edit accuracy numbers.
- Report text (region, type, findings, treatment plan) and the doctor chat use Google Gemini via `src/lib/ai.server.ts`; needs `GEMINI_API_KEY` server-side (optional `GEMINI_MODEL`). Gemini never overrides the classifier verdict.
- X-ray analysis server fn: `src/lib/analyze.functions.ts`. AI doctor chat streams through `src/routes/api/chat.ts`. No chat persistence.
- The heatmap (`src/components/HeatmapOverlay.tsx`) is drawn from the model's real CAM.
- PDF reports are generated client-side with jsPDF in `src/lib/report.ts`.
- Dark clinical theme (teal primary) is defined as `:root` tokens in `src/styles.css`; use semantic tokens, never hardcoded colors.

# FractureAI

AI-assisted bone fracture detection from X-ray images: a trained DenseNet121 classifier with a
class-activation heatmap, a Gemini-written report and treatment plan, an AI doctor chat, and a
downloadable PDF report.

## How it works

- **Detection** — a DenseNet121 fine-tuned on bone X-rays (`ml/`) runs in the browser with
  `onnxruntime-web`. It returns the fracture probability and a class activation map, which is
  drawn as the heatmap. The accuracy shown on the site is measured on a leak-free held-out test
  set (see `ml/README.md`).
- **Report and chat** — Google Gemini writes the findings and treatment plan (consistent with
  the model's verdict) and powers the doctor chat.

This is a screening aid, not a medical diagnosis.

## Run locally

```bash
bun install
echo "GEMINI_API_KEY=your-key" > .env.local   # optional: GEMINI_MODEL=gemini-flash-latest
bun run dev                                    # http://localhost:8080
```

## Deploy

`bun run build` produces `.output/` via Nitro. Run it with `bun .output/server/index.mjs`, or
build on Vercel / Netlify / Cloudflare where Nitro picks the right preset automatically
(`NITRO_PRESET` overrides). Set `GEMINI_API_KEY` in the host's environment variables.

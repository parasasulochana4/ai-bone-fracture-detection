import { createServerFn } from "@tanstack/react-start";
import { generateText, Output, type ModelMessage } from "ai";
import { z } from "zod";
import { getAiModel } from "./ai.server";
import type { Hotspot, ModelPrediction } from "./model";

const reportSchema = z.object({
  isXray: z.boolean().describe("true if the image appears to be a bone X-ray"),
  boneRegion: z.string().describe("anatomical region, e.g. 'distal radius (wrist)'"),
  fractureType: z.string().describe("e.g. 'transverse', 'hairline', 'comminuted', or 'none'"),
  severity: z.string().describe("'none', 'mild', 'moderate', or 'severe'"),
  summary: z.string().describe("2-3 sentence plain-language summary of the assessment"),
  findings: z.array(z.string()).describe("key radiological observations"),
  treatmentPlan: z.array(z.string()).describe("ordered treatment and care steps"),
  urgency: z.string().describe("'routine', 'soon', or 'urgent'"),
});

export type FractureAnalysis = z.infer<typeof reportSchema> & {
  fractureDetected: boolean;
  confidence: number;
  hotspots: Hotspot[];
};

const SYSTEM_PROMPT = `You are a radiology assistant writing the report for a bone fracture screening app.
A trained DenseNet121 classifier has already decided whether a fracture is present. Its verdict is final:
do not contradict it. Your job is to describe the image consistently with that verdict.
Rules:
- If the image is not a bone X-ray, set isXray to false.
- Identify the anatomical region and, if fractured, the most likely fracture type and severity; otherwise use 'none'.
- List concrete radiological findings (cortical disruption, lucent lines, displacement, angulation, periosteal reaction, etc.).
- Provide a practical, ordered treatment plan (immobilization, imaging follow-up, orthopedic referral, pain management, rehabilitation).
- Set urgency: 'urgent' for displaced/open/severe fractures, 'soon' for stable fractures, 'routine' otherwise.
- Be honest about uncertainty. This is a screening aid, not a medical diagnosis.`;

export const analyzeXray = createServerFn({ method: "POST" })
  .inputValidator((data: { image: string; prediction: ModelPrediction }) => {
    if (!data?.image || typeof data.image !== "string" || !data.image.startsWith("data:image/")) {
      throw new Error("A valid X-ray image is required");
    }
    if (data.image.length > 14_000_000) {
      throw new Error("Image is too large — please upload an image under 10 MB");
    }
    const p = data.prediction;
    if (!p || typeof p.fractureDetected !== "boolean" || typeof p.confidence !== "number") {
      throw new Error("Model prediction is required");
    }
    return { image: data.image, prediction: p };
  })
  .handler(async ({ data }): Promise<FractureAnalysis> => {
    const { prediction } = data;
    const mediaType = data.image.slice(5, data.image.indexOf(";")) || "image/jpeg";
    const verdict = prediction.fractureDetected
      ? `FRACTURE DETECTED (model confidence ${(prediction.confidence * 100).toFixed(1)}%)`
      : `NO FRACTURE (model confidence ${(prediction.confidence * 100).toFixed(1)}%)`;
    const messages: ModelMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `Classifier verdict: ${verdict}. Write the structured report for this X-ray.`,
          },
          { type: "file", data: data.image, mediaType },
        ],
      },
    ];
    const { output } = await generateText({
      model: getAiModel(),
      system: SYSTEM_PROMPT,
      messages,
      output: Output.object({ schema: reportSchema }),
    });
    if (!output) throw new Error("The AI could not analyze this image. Try a clearer X-ray.");
    return {
      ...output,
      fractureDetected: prediction.fractureDetected,
      confidence: prediction.confidence,
      hotspots: prediction.fractureDetected ? prediction.hotspots : [],
    };
  });

import { createServerFn } from "@tanstack/react-start";
import { streamText, Output, type ModelMessage } from "ai";
import { z } from "zod";
import { AI_MODEL, createAiProvider, RESPONSES_PROVIDER_OPTIONS } from "./ai.server";

const hotspotSchema = z.object({
  x: z.number().describe("horizontal center of the region, 0-1 relative to image width"),
  y: z.number().describe("vertical center of the region, 0-1 relative to image height"),
  radius: z.number().describe("radius of the region, 0-1 relative to image size"),
  intensity: z.number().describe("attention intensity 0-1, 1 = strongest"),
});

const analysisSchema = z.object({
  isXray: z.boolean().describe("true if the image appears to be a bone X-ray"),
  fractureDetected: z.boolean(),
  confidence: z.number().describe("confidence 0-1 in the fracture assessment"),
  boneRegion: z.string().describe("anatomical region, e.g. 'distal radius (wrist)'"),
  fractureType: z.string().describe("e.g. 'transverse', 'hairline', 'comminuted', or 'none'"),
  severity: z.string().describe("'none', 'mild', 'moderate', or 'severe'"),
  summary: z.string().describe("2-3 sentence plain-language summary of the assessment"),
  findings: z.array(z.string()).describe("key radiological observations"),
  treatmentPlan: z.array(z.string()).describe("ordered treatment and care steps"),
  urgency: z.string().describe("'routine', 'soon', or 'urgent'"),
  hotspots: z.array(hotspotSchema).describe("image regions the assessment focused on"),
});

export type FractureAnalysis = z.infer<typeof analysisSchema>;

const SYSTEM_PROMPT = `You are a radiology assistant specializing in bone fracture detection on X-ray images.
Analyze the uploaded image and assess whether a bone fracture is present.
Rules:
- If the image is not a bone X-ray, set isXray to false and fractureDetected to false.
- Estimate a calibrated confidence score (0-1) for your fracture assessment.
- Identify the anatomical region and, if fractured, the fracture type and severity.
- List concrete radiological findings (cortical disruption, lucent lines, displacement, angulation, periosteal reaction, etc.).
- Provide a practical, ordered treatment plan (immobilization, imaging follow-up, orthopedic referral, pain management, rehabilitation).
- Set urgency: 'urgent' for displaced/open/severe fractures, 'soon' for stable fractures, 'routine' otherwise.
- Provide hotspots: normalized (0-1) image regions that drove your assessment — the suspected fracture site(s) with high intensity, or the most scrutinized bone areas when no fracture is found.
- Be honest about uncertainty. This is a screening aid, not a medical diagnosis.`;

export const analyzeXray = createServerFn({ method: "POST" })
  .inputValidator((data: { image: string }) => {
    if (!data?.image || typeof data.image !== "string" || !data.image.startsWith("data:image/")) {
      throw new Error("A valid X-ray image is required");
    }
    if (data.image.length > 14_000_000) {
      throw new Error("Image is too large — please upload an image under 10 MB");
    }
    return { image: data.image };
  })
  .handler(async ({ data }): Promise<FractureAnalysis> => {
    const { provider } = createAiProvider();
    const mediaType = data.image.slice(5, data.image.indexOf(";")) || "image/jpeg";
    const messages: ModelMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "Assess this X-ray for bone fractures and return the structured analysis." },
          { type: "file", data: data.image, mediaType },
        ],
      },
    ];
    const result = streamText({
      model: provider.chat(AI_MODEL),
      system: SYSTEM_PROMPT,
      messages,
      output: Output.object({ schema: analysisSchema }),
      providerOptions: RESPONSES_PROVIDER_OPTIONS,
    });
    const analysis = await result.output;
    if (!analysis) throw new Error("The AI could not analyze this image. Try a clearer X-ray.");
    return analysis;
  });

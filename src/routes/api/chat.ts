import { createFileRoute } from "@tanstack/react-router";
import { convertToModelMessages, streamText, type UIMessage } from "ai";
import { getAiModel } from "@/lib/ai.server";

const DOCTOR_PROMPT = `You are "FractureAI Doctor", an AI orthopedic consultation assistant inside a bone-fracture X-ray analysis app.
You are chatting with a patient (or caregiver) after their X-ray was analyzed.

Behavior:
- Answer questions about the analysis, the fracture, the treatment plan, recovery timelines, warning signs, and general bone health.
- Use clear, compassionate, plain language. Avoid jargon; explain terms when you must use them.
- Keep answers concise (a short paragraph or a few bullet points) unless the user asks for detail.
- Always remind the user, when giving medical guidance, that you are an AI screening aid and they should consult a licensed physician — but do not repeat the disclaimer on every single message.
- Never prescribe medication dosages. For pain, suggest discussing options with their doctor or pharmacist.
- If the user describes emergency symptoms (numbness, blue/cold limb, severe swelling, open wound, deformity), tell them to seek emergency care immediately.

The patient's X-ray analysis result is provided below as context. Refer to it naturally in conversation.`;

export const Route = createFileRoute("/api/chat")({
  server: {
    handlers: {
      POST: async ({ request }) => {
        let body: { messages?: UIMessage[]; analysisContext?: string };
        try {
          body = await request.json();
        } catch {
          return Response.json({ error: "Invalid request body" }, { status: 400 });
        }
        if (!Array.isArray(body.messages)) {
          return Response.json({ error: "messages are required" }, { status: 400 });
        }
        const modelMessages = await convertToModelMessages(body.messages);
        const system = body.analysisContext
          ? `${DOCTOR_PROMPT}\n\n--- X-RAY ANALYSIS CONTEXT ---\n${body.analysisContext}\n--- END CONTEXT ---`
          : DOCTOR_PROMPT;
        const result = streamText({
          model: getAiModel(),
          system,
          messages: modelMessages,
          abortSignal: request.signal,
        });
        return result.toUIMessageStreamResponse({ sendReasoning: false });
      },
    },
  },
});

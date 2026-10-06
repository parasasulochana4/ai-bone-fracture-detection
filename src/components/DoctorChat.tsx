import { useEffect, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { Bot, Send, Square, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { FractureAnalysis } from "@/lib/analyze.functions";

function buildContext(a: FractureAnalysis): string {
  return [
    `Fracture detected: ${a.fractureDetected ? "yes" : "no"}`,
    `Confidence: ${Math.round(a.confidence * 100)}%`,
    `Region: ${a.boneRegion}`,
    `Fracture type: ${a.fractureType}`,
    `Severity: ${a.severity}`,
    `Urgency: ${a.urgency}`,
    `Summary: ${a.summary}`,
    `Findings: ${a.findings.join("; ")}`,
    `Treatment plan: ${a.treatmentPlan.join("; ")}`,
  ].join("\n");
}

/** Minimal markdown: **bold**, - bullets, line breaks. */
function renderMarkdown(text: string) {
  return text.split("\n").map((line, i) => {
    const parts = line.split(/(\*\*[^*]+\*\*)/g).map((seg, j) =>
      seg.startsWith("**") && seg.endsWith("**") ? (
        <strong key={j}>{seg.slice(2, -2)}</strong>
      ) : (
        <span key={j}>{seg}</span>
      ),
    );
    return (
      <span key={i}>
        {parts}
        {i < text.split("\n").length - 1 && <br />}
      </span>
    );
  });
}

export function DoctorChat({ analysis }: { analysis: FractureAnalysis }) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const { messages, sendMessage, stop, status, error } = useChat({
    transport: new DefaultChatTransport({
      api: "/api/chat",
      body: { analysisContext: buildContext(analysis) },
    }),
  });
  const busy = status === "submitted" || status === "streaming";

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, status]);

  const submit = () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    void sendMessage({ text });
  };

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="rounded-xl border border-border bg-muted/40 p-4 text-sm text-muted-foreground">
            <p className="font-medium text-foreground">Ask me anything about your result</p>
            <ul className="mt-2 list-disc space-y-1 pl-5">
              <li>How serious is this fracture?</li>
              <li>How long will recovery take?</li>
              <li>What should I avoid doing?</li>
            </ul>
          </div>
        )}
        {messages.map((m) => {
          const isUser = m.role === "user";
          const text = m.parts
            .filter((p) => p.type === "text")
            .map((p) => p.text)
            .join("");
          if (!text) return null;
          return (
            <div key={m.id} className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
              <div
                className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${
                  isUser ? "bg-primary text-primary-foreground" : "bg-accent text-accent-foreground"
                }`}
              >
                {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
              </div>
              <div
                className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                  isUser
                    ? "bg-primary text-primary-foreground"
                    : "border border-border bg-card text-card-foreground"
                }`}
              >
                {text}
              </div>
            </div>
          );
        })}
        {busy && (
          <div className="flex gap-3">
            <div className="mt-0.5 flex h-7 w-7 items-center justify-center rounded-full bg-accent text-accent-foreground">
              <Bot className="h-4 w-4" />
            </div>
            <div className="rounded-2xl border border-border bg-card px-4 py-2.5 text-sm text-muted-foreground">
              Thinking…
            </div>
          </div>
        )}
        {error && (
          <p className="text-center text-xs text-destructive">
            Something went wrong — please try sending your message again.
          </p>
        )}
      </div>

      <div className="border-t border-border p-3">
        <div className="flex items-end gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask the AI doctor…"
            rows={1}
            className="max-h-32 min-h-10 resize-none"
          />
          {busy ? (
            <Button size="icon" variant="outline" onClick={() => void stop()} aria-label="Stop">
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button size="icon" onClick={submit} disabled={!input.trim()} aria-label="Send">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <p className="mt-2 text-[11px] text-muted-foreground">
          AI screening aid — always confirm with a licensed physician.
        </p>
      </div>
    </div>
  );
}

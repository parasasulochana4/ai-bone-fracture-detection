import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bone,
  CheckCircle2,
  Download,
  Flame,
  Loader2,
  MessageSquareText,
  ScanLine,
  Stethoscope,
  Trophy,
  Upload,
  X,
} from "lucide-react";
import { analyzeXray, type FractureAnalysis } from "@/lib/analyze.functions";
import { predictFracture } from "@/lib/model";
import modelMetrics from "@/lib/model-metrics.json";
import { downloadReport } from "@/lib/report";
import { HeatmapOverlay } from "@/components/HeatmapOverlay";
import { DoctorChat } from "@/components/DoctorChat";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "FractureAI — Bone Fracture X-Ray Analysis" },
      {
        name: "description",
        content:
          "Upload a bone X-ray and get an AI fracture assessment with confidence score, attention heatmap, treatment plan, AI doctor chat, and a downloadable report.",
      },
      { property: "og:title", content: "FractureAI — Bone Fracture X-Ray Analysis" },
      {
        property: "og:description",
        content:
          "AI-powered bone fracture detection from X-rays with confidence scores, heatmaps, treatment plans, and an AI doctor consultation.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

interface ModelStat {
  name: string;
  accuracy: number; // 0-100
  note: string;
}

// Measured on the held-out test split by ml/finetune.py — regenerate, never hand-edit.
const MODELS: ModelStat[] = modelMetrics.models;
const bestModel = MODELS.reduce((a, b) => (b.accuracy > a.accuracy ? b : a));

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Could not read the file"));
    reader.readAsDataURL(file);
  });
}

function Index() {
  const [image, setImage] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [analysis, setAnalysis] = useState<FractureAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File | undefined | null) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG, …).");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("Image is too large — please keep it under 10 MB.");
      return;
    }
    setError(null);
    setAnalysis(null);
    setLoading(true);
    try {
      const dataUrl = await readAsDataUrl(file);
      setImage(dataUrl);
      setFileName(file.name);
      const prediction = await predictFracture(dataUrl);
      const result = await analyzeXray({ data: { image: dataUrl, prediction } });
      if (!result.isXray) {
        setError("This doesn't look like a bone X-ray. Please upload a valid X-ray image.");
        setImage(null);
        setLoading(false);
        return;
      }
      setAnalysis(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed — please try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  const reset = () => {
    setImage(null);
    setAnalysis(null);
    setError(null);
    setFileName("");
  };

  const confidencePct = analysis ? Math.round(analysis.confidence * 100) : 0;

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/60">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-4 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/15 text-primary">
              <Bone className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-foreground">FractureAI</h1>
              <p className="text-xs text-muted-foreground">Bone Fracture X-Ray Diagnostics</p>
            </div>
          </div>
          <Badge variant="outline" className="hidden gap-1.5 sm:inline-flex">
            <Activity className="h-3 w-3 text-primary" />
            AI screening aid — not a medical diagnosis
          </Badge>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6">
        {!image && !loading && (
          <div className="mx-auto max-w-3xl">
            <div className="mb-8 text-center">
              <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                Upload an X-ray. Get answers in seconds.
              </h2>
              <p className="mt-3 text-muted-foreground">
                Fracture detection with confidence scoring, an attention heatmap, a treatment plan,
                and an AI doctor you can chat with.
              </p>
            </div>

            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                void handleFile(e.dataTransfer.files?.[0]);
              }}
              className={`flex w-full flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-6 py-16 transition-colors ${
                dragOver
                  ? "border-primary bg-primary/10"
                  : "border-border bg-card/50 hover:border-primary/60 hover:bg-card"
              }`}
            >
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                <Upload className="h-7 w-7" />
              </div>
              <div className="text-center">
                <p className="font-semibold text-foreground">
                  Drop your X-ray here, or click to browse
                </p>
                <p className="mt-1 text-sm text-muted-foreground">JPEG or PNG, up to 10 MB</p>
              </div>
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => void handleFile(e.target.files?.[0])}
            />

            <div className="mt-10 grid gap-4 sm:grid-cols-3">
              {[
                {
                  icon: ScanLine,
                  title: "Detection + confidence",
                  text: "Fracture or no fracture, with a calibrated confidence score.",
                },
                {
                  icon: Flame,
                  title: "Attention heatmap",
                  text: "See exactly which regions of the X-ray drove the assessment.",
                },
                {
                  icon: MessageSquareText,
                  title: "AI doctor chat",
                  text: "Ask follow-up questions and download a full PDF report.",
                },
              ].map((f) => (
                <div key={f.title} className="rounded-xl border border-border bg-card/50 p-5">
                  <f.icon className="h-5 w-5 text-primary" />
                  <p className="mt-3 font-semibold text-foreground">{f.title}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{f.text}</p>
                </div>
              ))}
            </div>

            {/* Models compared */}
            <div className="mt-12 text-left">
              <div className="mb-5 text-center">
                <h3 className="text-xl font-bold tracking-tight text-foreground">
                  Models compared
                </h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Measured fracture-detection accuracy of each trained model — the best performer is
                  highlighted.
                </p>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {MODELS.map((m) => {
                  const isBest = m.name === bestModel.name;
                  return (
                    <div
                      key={m.name}
                      className={`rounded-xl border p-5 ${
                        isBest ? "border-primary/60 bg-primary/10" : "border-border bg-card/50"
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <p className="font-semibold text-foreground">{m.name}</p>
                        {isBest && (
                          <Badge className="gap-1 bg-primary text-primary-foreground hover:bg-primary">
                            <Trophy className="h-3 w-3" />
                            Best model
                          </Badge>
                        )}
                      </div>
                      <p className="mt-0.5 text-xs text-muted-foreground">{m.note}</p>
                      <div className="mt-3 mb-1.5 flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Accuracy</span>
                        <span
                          className={`font-bold ${isBest ? "text-primary" : "text-foreground"}`}
                        >
                          {m.accuracy.toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={m.accuracy} className="h-2" />
                    </div>
                  );
                })}
              </div>
              <p className="mt-4 text-center text-xs text-muted-foreground">
                Accuracy on {modelMetrics.testSize.toLocaleString()} held-out test X-rays never seen
                in training ({modelMetrics.dataset}). Best model: sensitivity{" "}
                {modelMetrics.sensitivity.toFixed(1)}%, specificity{" "}
                {modelMetrics.specificity.toFixed(1)}%. This app runs the best model in your
                browser.
              </p>
            </div>
          </div>
        )}

        {error && (
          <div className="mx-auto mb-6 flex max-w-3xl items-start gap-3 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <div className="flex-1">{error}</div>
            <button onClick={() => setError(null)} aria-label="Dismiss">
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        {(image || loading) && (
          <div className="grid gap-6 lg:grid-cols-5">
            {/* Left: image + verdict */}
            <div className="space-y-6 lg:col-span-3">
              <div className="overflow-hidden rounded-2xl border border-border bg-card">
                <div className="flex items-center justify-between border-b border-border px-4 py-3">
                  <div className="flex min-w-0 items-center gap-2 text-sm text-muted-foreground">
                    <ScanLine className="h-4 w-4 shrink-0" />
                    <span className="truncate">{fileName || "X-ray"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {analysis?.fractureDetected && (
                      <Button
                        size="sm"
                        variant={showHeatmap ? "default" : "outline"}
                        onClick={() => setShowHeatmap((v) => !v)}
                        className="gap-1.5"
                      >
                        <Flame className="h-3.5 w-3.5" />
                        Heatmap {showHeatmap ? "on" : "off"}
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" onClick={reset} className="gap-1.5">
                      <X className="h-3.5 w-3.5" />
                      New scan
                    </Button>
                  </div>
                </div>
                <div className="relative flex items-center justify-center bg-black/60 p-4">
                  {image && (
                    <div className="relative inline-block max-w-full">
                      <img
                        src={image}
                        alt="Uploaded X-ray"
                        className="max-h-[520px] w-auto rounded-lg"
                      />
                      {analysis?.fractureDetected && (
                        <HeatmapOverlay hotspots={analysis.hotspots} visible={showHeatmap} />
                      )}
                    </div>
                  )}
                  {loading && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/70 backdrop-blur-sm">
                      <Loader2 className="h-8 w-8 animate-spin text-primary" />
                      <p className="text-sm font-medium text-foreground">Analyzing your X-ray…</p>
                      <p className="text-xs text-muted-foreground">
                        Detecting fractures, scoring confidence, mapping attention
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {analysis && (
                <div className="rounded-2xl border border-border bg-card p-6">
                  <div className="flex flex-wrap items-center gap-3">
                    {analysis.fractureDetected ? (
                      <Badge className="gap-1.5 bg-destructive/15 text-destructive hover:bg-destructive/15">
                        <AlertTriangle className="h-3.5 w-3.5" />
                        Fracture detected
                      </Badge>
                    ) : (
                      <Badge className="gap-1.5 bg-emerald-500/15 text-emerald-400 hover:bg-emerald-500/15">
                        <CheckCircle2 className="h-3.5 w-3.5" />
                        No fracture detected
                      </Badge>
                    )}
                    <Badge variant="outline">{analysis.boneRegion}</Badge>
                    <Badge variant="outline">Type: {analysis.fractureType}</Badge>
                    <Badge variant="outline">Severity: {analysis.severity}</Badge>
                    <Badge
                      variant="outline"
                      className={
                        analysis.urgency === "urgent"
                          ? "border-destructive/50 text-destructive"
                          : ""
                      }
                    >
                      Urgency: {analysis.urgency}
                    </Badge>
                  </div>

                  <div className="mt-5">
                    <div className="mb-1.5 flex items-center justify-between text-sm">
                      <span className="font-medium text-foreground">Confidence</span>
                      <span className="font-bold text-primary">{confidencePct}%</span>
                    </div>
                    <Progress value={confidencePct} className="h-2" />
                  </div>

                  <p className="mt-5 text-sm leading-relaxed text-muted-foreground">
                    {analysis.summary}
                  </p>

                  {analysis.urgency === "urgent" && (
                    <div className="mt-4 flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
                      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                      This may need prompt medical attention. Please see a doctor as soon as
                      possible.
                    </div>
                  )}

                  <Button className="mt-6 gap-2" onClick={() => downloadReport(analysis, image)}>
                    <Download className="h-4 w-4" />
                    Download PDF report
                  </Button>
                </div>
              )}
            </div>

            {/* Right: findings, plan, chat */}
            <div className="space-y-6 lg:col-span-2">
              {analysis && (
                <>
                  <div className="rounded-2xl border border-border bg-card p-6">
                    <h3 className="flex items-center gap-2 font-semibold text-foreground">
                      <Stethoscope className="h-4 w-4 text-primary" />
                      Key findings
                    </h3>
                    <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                      {analysis.findings.map((f, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                          {f}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="rounded-2xl border border-border bg-card p-6">
                    <h3 className="flex items-center gap-2 font-semibold text-foreground">
                      <Activity className="h-4 w-4 text-primary" />
                      Treatment plan
                    </h3>
                    <ol className="mt-3 space-y-2.5 text-sm text-muted-foreground">
                      {analysis.treatmentPlan.map((t, i) => (
                        <li key={i} className="flex gap-3">
                          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-primary/15 text-[11px] font-bold text-primary">
                            {i + 1}
                          </span>
                          {t}
                        </li>
                      ))}
                    </ol>
                  </div>

                  <div className="flex h-[460px] flex-col overflow-hidden rounded-2xl border border-border bg-card">
                    <div className="border-b border-border px-4 py-3">
                      <h3 className="flex items-center gap-2 font-semibold text-foreground">
                        <MessageSquareText className="h-4 w-4 text-primary" />
                        AI doctor consultation
                      </h3>
                    </div>
                    <div className="min-h-0 flex-1">
                      <DoctorChat analysis={analysis} />
                    </div>
                  </div>
                </>
              )}
              {!analysis && !loading && (
                <div className="rounded-2xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
                  Results will appear here after analysis.
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

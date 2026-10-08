import type { Hotspot } from "@/components/HeatmapOverlay";

export type { Hotspot };

export interface ModelPrediction {
  fractureDetected: boolean;
  /** Confidence (0-1) in the predicted class. */
  confidence: number;
  /** Raw model probability (0-1) that a fracture is present. */
  fractureProbability: number;
  /** Class-activation-map regions, normalized to the image (0-1). */
  hotspots: Hotspot[];
}

const MODEL_URL = "/models/fracture-densenet121.onnx";
const INPUT_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
export const DECISION_THRESHOLD = 0.5;

type Ort = typeof import("onnxruntime-web");
let sessionPromise: Promise<{
  ort: Ort;
  session: import("onnxruntime-web").InferenceSession;
}> | null = null;

function getSession() {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import("onnxruntime-web");
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ["wasm"],
      });
      return { ort, session };
    })().catch((e) => {
      sessionPromise = null;
      throw e;
    });
  }
  return sessionPromise;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not decode the image"));
    img.src = src;
  });
}

async function toTensorData(dataUrl: string): Promise<Float32Array> {
  const img = await loadImage(dataUrl);
  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is not supported in this browser");
  ctx.drawImage(img, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    // X-rays are grayscale; the model was trained on grayscale replicated to 3 channels.
    const gray =
      (0.299 * (data[i * 4] ?? 0) +
        0.587 * (data[i * 4 + 1] ?? 0) +
        0.114 * (data[i * 4 + 2] ?? 0)) /
      255;
    for (let c = 0; c < 3; c++) out[c * plane + i] = (gray - (MEAN[c] ?? 0)) / (STD[c] ?? 1);
  }
  return out;
}

function camToHotspots(cam: Float32Array, size: number): Hotspot[] {
  let min = Infinity;
  let max = -Infinity;
  for (const v of cam) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  const range = max - min || 1;
  const cells = Array.from(cam, (v, i) => ({ i, v: (v - min) / range }))
    .filter((c) => c.v >= 0.5)
    .sort((a, b) => b.v - a.v)
    .slice(0, 6);
  return cells.map(({ i, v }) => ({
    x: ((i % size) + 0.5) / size,
    y: (Math.floor(i / size) + 0.5) / size,
    radius: 1.1 / size,
    intensity: v,
  }));
}

/** Runs the trained DenseNet121 fracture classifier locally in the browser. */
export async function predictFracture(dataUrl: string): Promise<ModelPrediction> {
  const [{ ort, session }, input] = await Promise.all([getSession(), toTensorData(dataUrl)]);
  const feeds = { image: new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]) };
  const out = await session.run(feeds);
  const probTensor = out["fracture_probability"];
  const camTensor = out["cam"];
  if (!probTensor || !camTensor) throw new Error("Unexpected model output");
  const prob = (probTensor.data as Float32Array)[0] ?? 0;
  const camSize = camTensor.dims[camTensor.dims.length - 1] ?? 7;
  const fractureDetected = prob >= DECISION_THRESHOLD;
  return {
    fractureDetected,
    confidence: fractureDetected ? prob : 1 - prob,
    fractureProbability: prob,
    hotspots: camToHotspots(camTensor.data as Float32Array, camSize),
  };
}

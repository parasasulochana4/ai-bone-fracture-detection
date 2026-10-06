import { useEffect, useRef } from "react";

export interface Hotspot {
  x: number;
  y: number;
  radius: number;
  intensity: number;
}

/** Draws a Grad-CAM-style attention heatmap over the X-ray using radial gradients. */
export function HeatmapOverlay({ hotspots, visible }: { hotspots: Hotspot[]; visible: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const rect = parent.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, rect.width, rect.height);
      if (!visible) return;

      for (const h of hotspots) {
        const cx = h.x * rect.width;
        const cy = h.y * rect.height;
        const r = Math.max(12, h.radius * Math.min(rect.width, rect.height));
        const alpha = Math.min(1, Math.max(0.15, h.intensity));
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g.addColorStop(0, `rgba(255, 45, 45, ${0.75 * alpha})`);
        g.addColorStop(0.35, `rgba(255, 140, 0, ${0.5 * alpha})`);
        g.addColorStop(0.7, `rgba(255, 220, 0, ${0.25 * alpha})`);
        g.addColorStop(1, "rgba(0, 120, 255, 0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [hotspots, visible]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full transition-opacity duration-500"
      style={{ opacity: visible ? 1 : 0, mixBlendMode: "screen" }}
      aria-hidden
    />
  );
}

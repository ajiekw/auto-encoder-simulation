export function paintMap(canvas, values, width, height, options = {}) {
  const { stretch = false, grid = false, tint = null } = options;
  const scale = Math.max(2, Math.round(144 / Math.max(width, height)));
  canvas.width = width * scale;
  canvas.height = height * scale;
  const ctx = canvas.getContext('2d');
  let lo = 0;
  let hi = 1;
  if (stretch) {
    const sorted = Array.from(values).sort((a, b) => a - b);
    lo = sorted[Math.floor((sorted.length - 1) * 0.05)] ?? 0;
    hi = sorted[Math.ceil((sorted.length - 1) * 0.95)] ?? 1;
    if (!(hi > lo)) {
      lo = sorted[0] ?? 0;
      hi = (sorted[sorted.length - 1] ?? 1) + 1e-4;
    }
  }
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let v = values[y * width + x];
      v = stretch ? (v - lo) / (hi - lo) : v;
      v = Math.min(1, Math.max(0, v));
      let r;
      let g;
      let b;
      if (tint) {
        r = Math.round(8 + v * (tint[0] - 8));
        g = Math.round(10 + v * (tint[1] - 10));
        b = Math.round(14 + v * (tint[2] - 14));
      } else {
        const tone = Math.round(255 * v);
        r = tone;
        g = tone;
        b = tone;
      }
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x * scale, y * scale, scale, scale);
    }
  }
  if (grid) {
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= width; i += 1) {
      const p = Math.min(canvas.width - 1, i * scale) + 0.5;
      ctx.beginPath();
      ctx.moveTo(p, 0);
      ctx.lineTo(p, canvas.height);
      ctx.stroke();
    }
    for (let j = 0; j <= height; j += 1) {
      const p = Math.min(canvas.height - 1, j * scale) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, p);
      ctx.lineTo(canvas.width, p);
      ctx.stroke();
    }
  }
}

export function hexToRgb(hex) {
  return [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
}

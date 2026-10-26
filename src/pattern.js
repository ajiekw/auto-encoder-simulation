/** Pola kalibrasi 32×32, nilai piksel di [0, 1]. Dipakai sebagai tensor masukan. */
export function calibrationGrid(n = 32) {
  const data = new Float32Array(n * n);
  const cx = (n - 1) / 2;
  const cy = (n - 1) / 2;
  for (let y = 0; y < n; y += 1) {
    for (let x = 0; x < n; x += 1) {
      const dx = x - cx;
      const dy = y - cy;
      const r = Math.hypot(dx, dy);
      const ring = r > 8.15 && r < 11.35;
      const stem = Math.abs(dx) <= 1.25 && dy > -1.5 && dy < 12.2;
      const hub = r <= 2.15;
      let v = 0.06;
      if (ring) v = 0.92;
      if (stem) v = 0.74;
      if (hub) v = 1;
      data[y * n + x] = v;
    }
  }
  return data;
}

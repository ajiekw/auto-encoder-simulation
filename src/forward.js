import { STAGES, convTOut, numel } from './architecture.js';
import { calibrationGrid } from './pattern.js';

function mulberry32(seed) {
  let a = seed >>> 0;
  return function rng() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  const u = Math.max(rng(), 1e-8);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function volume(c, h, w, data) {
  return { kind: 'volume', c, h, w, data };
}

function vector(n, data) {
  return { kind: 'vector', vector: n, data };
}

export function summarize(tensor) {
  const data = tensor.data;
  const n = data.length;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sum2 = 0;
  let zeros = 0;
  for (let i = 0; i < n; i += 1) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sum2 += v * v;
    if (v === 0) zeros += 1;
  }
  const mean = sum / n;
  const variance = Math.max(0, sum2 / n - mean * mean);
  return { n, min, max, mean, std: Math.sqrt(variance), zeroFrac: zeros / n };
}

function conv2d(input, weight, bias, stride, pad) {
  const { c: cin, h, w, data } = input;
  const { cout, k, data: wdata } = weight;
  const hout = Math.floor((h + 2 * pad - k) / stride) + 1;
  const wout = Math.floor((w + 2 * pad - k) / stride) + 1;
  const out = new Float32Array(cout * hout * wout);
  for (let oc = 0; oc < cout; oc += 1) {
    for (let oy = 0; oy < hout; oy += 1) {
      for (let ox = 0; ox < wout; ox += 1) {
        let acc = bias[oc];
        for (let ic = 0; ic < cin; ic += 1) {
          for (let kh = 0; kh < k; kh += 1) {
            const iy = oy * stride + kh - pad;
            if (iy < 0 || iy >= h) continue;
            for (let kw = 0; kw < k; kw += 1) {
              const ix = ox * stride + kw - pad;
              if (ix < 0 || ix >= w) continue;
              const iv = data[(ic * h + iy) * w + ix];
              const wv = wdata[((oc * cin + ic) * k + kh) * k + kw];
              acc += iv * wv;
            }
          }
        }
        out[(oc * hout + oy) * wout + ox] = acc;
      }
    }
  }
  return volume(cout, hout, wout, out);
}

function convTranspose2d(input, weight, bias, stride, pad, outPad) {
  const { c: cin, h: hin, w: win, data } = input;
  const { cout, k, data: wdata } = weight;
  const hout = convTOut(hin, k, stride, pad, outPad);
  const wout = convTOut(win, k, stride, pad, outPad);
  const out = new Float32Array(cout * hout * wout);
  for (let oc = 0; oc < cout; oc += 1) {
    for (let i = 0; i < hout * wout; i += 1) out[oc * hout * wout + i] = bias[oc];
  }
  for (let ic = 0; ic < cin; ic += 1) {
    for (let ih = 0; ih < hin; ih += 1) {
      for (let iw = 0; iw < win; iw += 1) {
        const iv = data[(ic * hin + ih) * win + iw];
        for (let oc = 0; oc < cout; oc += 1) {
          for (let kh = 0; kh < k; kh += 1) {
            const oh = ih * stride - pad + kh;
            if (oh < 0 || oh >= hout) continue;
            for (let kw = 0; kw < k; kw += 1) {
              const ow = iw * stride - pad + kw;
              if (ow < 0 || ow >= wout) continue;
              const wv = wdata[((ic * cout + oc) * k + kh) * k + kw];
              out[(oc * hout + oh) * wout + ow] += iv * wv;
            }
          }
        }
      }
    }
  }
  return volume(cout, hout, wout, out);
}

function linear(inputData, nin, nout, weight, bias) {
  const out = new Float32Array(nout);
  for (let o = 0; o < nout; o += 1) {
    let acc = bias[o];
    const row = o * nin;
    for (let i = 0; i < nin; i += 1) acc += weight[row + i] * inputData[i];
    out[o] = acc;
  }
  return out;
}

function mapActivation(tensor, activation) {
  if (activation === 'none' || activation === 'linear') return tensor;
  const data = new Float32Array(tensor.data.length);
  if (activation === 'relu') {
    for (let i = 0; i < data.length; i += 1) data[i] = tensor.data[i] > 0 ? tensor.data[i] : 0;
  } else if (activation === 'sigmoid') {
    for (let i = 0; i < data.length; i += 1) data[i] = 1 / (1 + Math.exp(-tensor.data[i]));
  } else {
    throw new Error(`Aktivasi tidak dikenal: ${activation}`);
  }
  return { ...tensor, data };
}

function makeArray(length, fanIn, rng, activation) {
  const data = new Float32Array(length);
  const he = activation === 'sigmoid' || activation === 'none' || activation === 'linear' ? 1 : Math.sqrt(2);
  const gain = activation === 'sigmoid' ? 3.4 : 1;
  const scale = (gain * he) / Math.sqrt(fanIn);
  for (let i = 0; i < length; i += 1) data[i] = randn(rng) * scale;
  return data;
}

function applyOp(op, tensor, rng) {
  if (op.type === 'conv') {
    const weight = {
      cout: op.cout,
      cin: op.cin,
      k: op.k,
      data: makeArray(op.cout * op.cin * op.k * op.k, op.cin * op.k * op.k, rng, op.activation),
    };
    const bias = new Float32Array(op.cout);
    const out = conv2d(tensor, weight, bias, op.s, op.p);
    return { tensor: mapActivation(out, op.activation), paramCount: weight.data.length + bias.length };
  }
  if (op.type === 'convT') {
    const weight = {
      cout: op.cout,
      cin: op.cin,
      k: op.k,
      data: makeArray(op.cin * op.cout * op.k * op.k, op.cin * op.k * op.k, rng, op.activation),
    };
    const bias = new Float32Array(op.cout);
    const out = convTranspose2d(tensor, weight, bias, op.s, op.p, op.op);
    return { tensor: mapActivation(out, op.activation), paramCount: weight.data.length + bias.length };
  }
  if (op.type === 'linear') {
    const source = op.flatten ? tensor.data : tensor.data;
    if (source.length !== op.nin) {
      throw new Error(`Linear mengharapkan ${op.nin} nilai, menerima ${source.length}`);
    }
    const weight = makeArray(op.nout * op.nin, op.nin, rng, op.activation);
    const bias = new Float32Array(op.nout);
    const data = linear(source, op.nin, op.nout, weight, bias);
    let next = op.reshape
      ? volume(op.reshape.c, op.reshape.h, op.reshape.w, data)
      : vector(op.nout, data);
    if (op.reshape && numel(op.reshape) !== op.nout) {
      throw new Error('Reshape tidak sesuai jumlah keluaran linear');
    }
    next = mapActivation(next, op.activation);
    return { tensor: next, paramCount: weight.length + bias.length };
  }
  throw new Error(`Operasi tidak dikenal: ${op.type}`);
}

function sameShape(tensor, shape) {
  if (shape.vector) return tensor.kind === 'vector' && tensor.vector === shape.vector;
  return tensor.kind === 'volume' && tensor.c === shape.c && tensor.h === shape.h && tensor.w === shape.w;
}

export function simulate(seed = 7) {
  const rng = mulberry32(seed);
  const pixels = calibrationGrid();
  let tensor = volume(1, 32, 32, pixels);
  const stages = {};
  for (const stage of STAGES) {
    if (!stage.op) {
      if (!sameShape(tensor, stage.output)) {
        throw new Error(`Bentuk ${stage.id} tidak sesuai spesifikasi`);
      }
      stages[stage.id] = { tensor, stats: summarize(tensor), paramCount: 0 };
      continue;
    }
    const applied = applyOp(stage.op, tensor, rng);
    if (applied.paramCount !== stage.params) {
      throw new Error(`Jumlah parameter ${stage.id} tidak sesuai: ${applied.paramCount} vs ${stage.params}`);
    }
    tensor = applied.tensor;
    if (!sameShape(tensor, stage.output)) {
      const got = tensor.kind === 'vector' ? `vektor ${tensor.vector}` : `${tensor.c}×${tensor.h}×${tensor.w}`;
      throw new Error(`Keluaran ${stage.id} berbentuk ${got}, spesifikasi ${JSON.stringify(stage.output)}`);
    }
    for (let i = 0; i < tensor.data.length; i += 1) {
      if (!Number.isFinite(tensor.data[i])) throw new Error(`Nilai tidak hingga di ${stage.id}`);
    }
    stages[stage.id] = { tensor, stats: summarize(tensor), paramCount: applied.paramCount };
  }
  return { seed, stages };
}

export function channelValues(tensor, channel) {
  const { h, w, data } = tensor;
  return data.subarray(channel * h * w, (channel + 1) * h * w);
}

export function meanValues(tensor) {
  const { c, h, w, data } = tensor;
  const out = new Float32Array(h * w);
  for (let i = 0; i < out.length; i += 1) {
    let sum = 0;
    for (let ch = 0; ch < c; ch += 1) sum += data[ch * h * w + i];
    out[i] = sum / c;
  }
  return out;
}

function assertClose(actual, expected, label) {
  if (Math.abs(actual - expected) > 1e-4) {
    throw new Error(`${label}: ${actual} ≠ ${expected}`);
  }
}

export function selfTest() {
  const input = volume(1, 2, 2, Float32Array.from([1, 2, 3, 4]));
  const out = conv2d(
    input,
    { cout: 1, cin: 1, k: 1, data: Float32Array.from([5]) },
    Float32Array.from([0.5]),
    1,
    0,
  );
  assertClose(out.data[0], 5.5, 'conv00');
  assertClose(out.data[1], 10.5, 'conv01');
  assertClose(out.data[2], 15.5, 'conv10');
  assertClose(out.data[3], 20.5, 'conv11');

  const impulse = volume(1, 1, 1, Float32Array.from([1]));
  const taps = convTranspose2d(
    impulse,
    { cout: 1, cin: 1, k: 3, data: Float32Array.from([1, 1, 1, 1, 1, 1, 1, 1, 1]) },
    Float32Array.from([0]),
    2,
    1,
    1,
  );
  if (taps.h !== 2 || taps.w !== 2) throw new Error(`ConvTranspose impulse shape ${taps.h}×${taps.w}`);
  let tapSum = 0;
  for (let i = 0; i < taps.data.length; i += 1) tapSum += taps.data[i];
  assertClose(tapSum, 4, 'transpose taps');

  const sim = simulate(7);
  if (sim.stages.dec3.tensor.h !== 32 || sim.stages.dec3.tensor.c !== 1) {
    throw new Error('Forward pass tidak kembali ke 1×32×32');
  }
  if (sim.stages.latent.tensor.vector !== 32) throw new Error('Bottleneck bukan ℝ^32');
  if (sim.stages.input.stats.n !== 1024) throw new Error('Masukan bukan 1024 nilai');
}

const entry = typeof process !== 'undefined' ? process.argv?.[1] : '';
if (entry && entry.endsWith('forward.js')) {
  selfTest();
}

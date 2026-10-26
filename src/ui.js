import katex from 'katex';
import 'katex/dist/katex.min.css';
import { GROUPS, OVERVIEW, STAGES, TOTAL_PARAMS, numel, shapeLabel } from './architecture.js';
import { channelValues, meanValues } from './forward.js';
import { hexToRgb, paintMap } from './paint.js';

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function fmt(n) {
  if (!Number.isFinite(n)) return '—';
  const abs = Math.abs(n);
  if (abs !== 0 && (abs < 0.001 || abs >= 100000)) return n.toExponential(2);
  return n.toLocaleString('id-ID', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
}

function fmtInt(n) {
  return n.toLocaleString('id-ID');
}

function renderTex(target, tex) {
  katex.render(tex, target, { throwOnError: false, displayMode: true });
}

function previousStage(stage) {
  const index = STAGES.findIndex((item) => item.id === stage.id);
  return index > 0 ? STAGES[index - 1] : null;
}

function glyphValues(tensor) {
  if (tensor.kind === 'vector') return null;
  if (tensor.c === 1) return channelValues(tensor, 0);
  return meanValues(tensor);
}

function paintGlyph(canvas, tensor, tint, { grid = false, stretch = false } = {}) {
  if (tensor.kind === 'vector') return false;
  const values = glyphValues(tensor);
  paintMap(canvas, values, tensor.w, tensor.h, { tint, grid, stretch });
  return true;
}

function vectorBars(tensor, color) {
  const wrap = el('div', 'bars');
  const data = tensor.data;
  let max = 1e-6;
  for (let i = 0; i < data.length; i += 1) max = Math.max(max, Math.abs(data[i]));
  const step = Math.ceil(data.length / 28);
  for (let i = 0; i < data.length; i += step) {
    const bar = document.createElement('i');
    const t = Math.abs(data[i]) / max;
    bar.style.height = `${Math.round(16 + t * 84)}%`;
    bar.style.background = color;
    wrap.appendChild(bar);
  }
  return wrap;
}

function shapeFigure(tensor, color, options) {
  const fig = el('div', 'glyph');
  if (tensor.kind === 'vector') {
    fig.appendChild(vectorBars(tensor, color));
    return fig;
  }
  const canvas = document.createElement('canvas');
  paintGlyph(canvas, tensor, options.tint, options);
  fig.appendChild(canvas);
  return fig;
}

function stat(label, value) {
  const node = el('div', 'stat');
  node.append(el('span', null, label), el('b', null, value));
  return node;
}

export function createUi(root, handlers) {
  const inspector = root.querySelector('#inspector');
  const timeline = root.querySelector('#timeline');
  const playButton = root.querySelector('#play');
  const prevButton = root.querySelector('#prev');
  const nextButton = root.querySelector('#next');
  const reseedButton = root.querySelector('#reseed');
  const status = root.querySelector('#status');

  let sim = null;
  let selectedId = 'overview';
  let mode = 'idle';

  playButton.addEventListener('click', () => handlers.onPlay());
  prevButton.addEventListener('click', () => handlers.onStep(-1));
  nextButton.addEventListener('click', () => handlers.onStep(1));
  reseedButton.addEventListener('click', () => handlers.onReseed());

  const overviewPill = el('button', 'pill is-active');
  overviewPill.type = 'button';
  overviewPill.textContent = 'Mesin';
  overviewPill.addEventListener('click', () => handlers.onSelect('overview'));
  timeline.appendChild(overviewPill);

  const pills = new Map([['overview', overviewPill]]);
  for (const stage of STAGES) {
    const pill = el('button', 'pill');
    pill.type = 'button';
    pill.style.setProperty('--accent', GROUPS[stage.group].color);
    pill.innerHTML = `<small>${stage.index}</small>${stage.short}`;
    pill.title = stage.name;
    pill.addEventListener('click', () => handlers.onSelect(stage.id));
    timeline.appendChild(pill);
    pills.set(stage.id, pill);
  }

  function setSim(next) {
    sim = next;
  }

  function setMode(next) {
    mode = next;
    playButton.textContent = mode === 'playing' ? 'Jeda' : mode === 'paused' ? 'Lanjut' : mode === 'done' ? 'Putar lagi' : 'Alirkan sinyal';
    status.textContent = mode === 'playing'
      ? 'Sinyal bergerak antar blok'
      : mode === 'paused'
        ? 'Aliran dijeda'
        : 'Seret untuk mengorbit · gulir untuk zoom · klik blok';
  }

  function select(id) {
    selectedId = id ?? 'overview';
    pills.forEach((pill, key) => pill.classList.toggle('is-active', key === selectedId));
    const active = pills.get(selectedId);
    active?.scrollIntoView({ inline: 'nearest', block: 'nearest' });
    render();
  }

  function render() {
    inspector.replaceChildren();
    inspector.style.setProperty('--accent', selectedId === 'overview' ? '#d5dee8' : GROUPS[stageById(selectedId).group].color);
    if (selectedId === 'overview') renderOverview();
    else renderStage(stageById(selectedId));
  }

  function renderOverview() {
    inspector.append(
      el('p', 'eyebrow', 'Seluruh mesin'),
      el('h2', null, OVERVIEW.name),
      el('p', 'summary', OVERVIEW.summary),
    );

    const eq = el('div', 'equations');
    const primary = el('div', 'equation');
    renderTex(primary, String.raw`z = E(x), \quad \hat{x} = D(z)`);
    const map = el('div', 'equation');
    renderTex(map, String.raw`E:\mathbb{R}^{1024}\rightarrow\mathbb{R}^{32},\quad D:\mathbb{R}^{32}\rightarrow\mathbb{R}^{1024}`);
    eq.append(primary, map);
    inspector.append(eq);

    inspector.append(el('h3', null, 'Jumlah angka di tiap tahap'));
    const chart = el('div', 'chart');
    const max = Math.max(...STAGES.map((stage) => numel(stage.output)));
    for (const stage of STAGES) {
      const count = numel(stage.output);
      const row = el('button', 'bar-row');
      row.type = 'button';
      row.append(el('span', 'bar-name', stage.short));
      const track = el('span', 'bar-track');
      const fill = el('span', 'bar-fill');
      fill.style.width = `${Math.max(1.5, (count / max) * 100)}%`;
      fill.style.background = GROUPS[stage.group].color;
      track.appendChild(fill);
      row.append(track, el('span', 'bar-value', fmtInt(count)));
      row.addEventListener('click', () => handlers.onSelect(stage.id));
      chart.appendChild(row);
    }
    inspector.append(chart);
    inspector.append(el('p', 'note', 'Batang menunjukkan banyaknya bilangan pada keluaran tiap blok. Penyusutan spasial di encoder tidak selalu mengurangi jumlah angka; kompresi yang sebenarnya terjadi di z, dari 1.024 menjadi 32.'));

    const facts = el('div', 'stats');
    facts.append(
      stat('Parameter', fmtInt(TOTAL_PARAMS)),
      stat('Blok', String(STAGES.length)),
      stat('Kode laten', '32'),
      stat('Seed bobot', sim ? String(sim.seed) : '—'),
    );
    inspector.append(facts);
    inspector.append(el('p', 'note', 'Forward pass memakai bobot acak tetap (inisialisasi He / Xavier). Bentuk tensor dan angkanya nyata. Citra keluaran belum mirip masukan karena jaringan belum dilatih.'));
  }

  function renderStage(stage) {
    const group = GROUPS[stage.group];
    inspector.append(
      el('p', 'eyebrow', `${group.label} · ${stage.index}`),
      el('h2', null, stage.name),
      el('p', 'role', stage.role),
      el('p', 'summary', stage.summary),
    );

    const shapes = el('div', 'shapes');
    const prev = previousStage(stage);
    if (prev && sim) {
      shapes.append(
        shapeCard('Masuk', sim.stages[prev.id].tensor, prev, false),
        el('div', 'shape-arrow', '→'),
        shapeCard('Keluar', sim.stages[stage.id].tensor, stage, true),
      );
    } else if (sim) {
      shapes.append(shapeCard('Bentuk x', sim.stages[stage.id].tensor, stage, true));
    }
    inspector.append(shapes);

    const equations = el('div', 'equations');
    for (const equation of stage.equations) {
      const block = el('div', 'equation');
      block.appendChild(el('span', 'eq-label', equation.label));
      const math = el('div', 'math');
      renderTex(math, equation.tex);
      block.appendChild(math);
      equations.appendChild(block);
    }
    inspector.append(equations);

    inspector.append(el('p', 'calc', stage.calcPlain), el('p', 'weight', stage.weightText));

    const facts = el('div', 'stats');
    facts.append(
      stat('Parameter', fmtInt(stage.params)),
      stat('Elemen keluar', fmtInt(numel(stage.output))),
      stat('Bentuk keluar', shapeLabel(stage.output)),
      stat('Aktivasi', activationLabel(stage.activation)),
    );
    inspector.append(facts);

    if (sim) {
      const stats = sim.stages[stage.id].stats;
      inspector.append(el('h3', null, 'Nilai forward pass'));
      const live = el('div', 'stats');
      live.append(
        stat('Min', fmt(stats.min)),
        stat('Max', fmt(stats.max)),
        stat('Rata-rata', fmt(stats.mean)),
        stat('Simpangan', fmt(stats.std)),
      );
      inspector.append(live);
      if (stage.activation === 'relu') {
        inspector.append(el('p', 'calc', `${(stats.zeroFrac * 100).toLocaleString('id-ID', { maximumFractionDigits: 1 })}% sel bernilai nol setelah ReLU.`));
      }
      inspector.append(el('p', 'note', simNote(stage)));
    }

    inspector.append(el('p', 'note', stage.note));
  }

  function shapeCard(title, tensor, stage, outgoing) {
    const card = el('div', 'shape-card');
    card.append(
      el('span', 'shape-k', title),
      shapeFigure(tensor, GROUPS[stage.group].color, glyphOptions(tensor, stage, outgoing)),
      el('strong', null, shapeLabel(tensor.kind === 'vector' ? { vector: tensor.vector } : tensor)),
      el('em', null, `${fmtInt(tensor.data.length)} nilai`),
    );
    return card;
  }

  return { setSim, select, setMode, render };
}

function stageById(id) {
  return STAGES.find((stage) => stage.id === id);
}

function activationLabel(activation) {
  if (activation === 'relu') return 'ReLU';
  if (activation === 'sigmoid') return 'Sigmoid';
  if (activation === 'linear' || activation === 'none') return 'Linear';
  return activation;
}

function glyphOptions(tensor, stage, outgoing) {
  const tint = hexToRgb(GROUPS[stage.group].hex);
  const mono = tensor.kind === 'volume' && tensor.c === 1;
  const stretch = !mono && outgoing && stage.id !== 'input';
  return {
    tint: mono ? null : tint,
    grid: mono,
    stretch: tensor.kind === 'volume' && !mono,
  };
}

function simNote(stage) {
  if (stage.id === 'input') return 'Pola cincin adalah data masukan. Skala warna mengikuti nilai piksel 0–1.';
  if (stage.id === 'dec3') return 'x̂ ditampilkan pada skala sigmoid 0–1, tanpa peregangan kontras. Kisi menandai resolusi 32×32.';
  if (stage.output.vector) return 'Tiap batang adalah satu komponen z. Tinggi mengikuti nilai mutlak, dinormalisasi di dalam vektor ini.';
  return 'Peta menampilkan rata-rata kanal, dengan kontras diregangkan per tahap supaya bentuk spasialnya terbaca. Di dalam blok 3D, lembaran adalah kanal-kanal pertama.';
}

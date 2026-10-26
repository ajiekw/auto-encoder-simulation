export const K = 3;
export const STRIDE = 2;
export const PAD = 1;
export const OUT_PAD = 1;
export const Z_DIM = 32;
export const INPUT_SIZE = 32;

export const GROUPS = {
  input: { id: 'input', label: 'Masukan', color: '#d5dee8', hex: 0xd5dee8 },
  encoder: { id: 'encoder', label: 'Encoder', color: '#3ddec4', hex: 0x3ddec4 },
  latent: { id: 'latent', label: 'Latent', color: '#f0c36a', hex: 0xf0c36a },
  decoder: { id: 'decoder', label: 'Decoder', color: '#ff7a59', hex: 0xff7a59 },
};

export function convOut(h, k = K, s = STRIDE, p = PAD) {
  return Math.floor((h + 2 * p - k) / s) + 1;
}

export function convTOut(h, k = K, s = STRIDE, p = PAD, op = OUT_PAD) {
  return (h - 1) * s - 2 * p + (k - 1) + op + 1;
}

export function convParams(cin, cout, k = K) {
  return cout * (cin * k * k + 1);
}

export function linearParams(nin, nout) {
  return nout * nin + nout;
}

export function numel(shape) {
  if (!shape) return 0;
  if (shape.vector) return shape.vector;
  return shape.c * shape.h * shape.w;
}

export function shapeLabel(shape) {
  if (!shape) return '—';
  if (shape.vector) return `ℝ^${shape.vector}`;
  return `${shape.c} × ${shape.h} × ${shape.w}`;
}

function volume(c, h, w) {
  return { c, h, w };
}

function buildStages() {
  const x = volume(1, INPUT_SIZE, INPUT_SIZE);
  const e1 = volume(16, convOut(x.h), convOut(x.w));
  const e2 = volume(32, convOut(e1.h), convOut(e1.w));
  const e3 = volume(64, convOut(e2.h), convOut(e2.w));
  const flat = numel(e3);
  const z = { vector: Z_DIM };
  const restored = volume(e3.c, e3.h, e3.w);
  const d1 = volume(32, convTOut(restored.h), convTOut(restored.w));
  const d2 = volume(16, convTOut(d1.h), convTOut(d1.w));
  const d3 = volume(1, convTOut(d2.h), convTOut(d2.w));

  if (e1.h !== 16 || e2.h !== 8 || e3.h !== 4 || flat !== 1024 || d1.h !== 8 || d2.h !== 16 || d3.h !== 32 || d3.c !== 1) {
    throw new Error('Rantai ukuran autoencoder tidak konsisten.');
  }

  const convSizeTex = (hin, hout) =>
    `H_{out}=\\left\\lfloor\\frac{${hin}+2\\cdot${PAD}-${K}}{${STRIDE}}\\right\\rfloor+1=${hout}`;

  const convTSizeTex = (hin, hout) =>
    `H_{out}=(${hin}-1)\\cdot${STRIDE}-2\\cdot${PAD}+(${K}-1)+${OUT_PAD}+1=${hout}`;

  const convPlain = (hin) => {
    const numer = hin + 2 * PAD - K;
    return `⌊${numer} / ${STRIDE}⌋ + 1 = ${convOut(hin)}`;
  };

  const convTPlain = (hin) =>
    `(${hin}−1)·${STRIDE} − 2·${PAD} + (${K}−1) + ${OUT_PAD} + 1 = ${convTOut(hin)}`;

  return [
    {
      id: 'input',
      index: '01',
      short: 'x',
      name: 'Masukan',
      group: 'input',
      housing: 'block',
      activation: 'none',
      params: 0,
      input: null,
      output: x,
      role: 'Tensor citra yang masuk ke mesin',
      summary:
        'Pelat masukan memegang satu citra kanal tunggal. Tidak ada bobot di sini. Pola cincin adalah sampel kalibrasi 32×32: setiap sel adalah satu bilangan, jadi tensor x berisi 1.024 nilai.',
      equations: [
        {
          label: 'Tensor',
          tex: 'x \\in \\mathbb{R}^{1 \\times 32 \\times 32}',
        },
      ],
      calcPlain: '1 kanal × 32 tinggi × 32 lebar = 1.024 nilai',
      weightText: 'Tidak ada parameter',
      note: 'Urutan sumbu yang dipakai di seluruh mesin adalah C × H × W: kanal, tinggi, lalu lebar.',
    },
    {
      id: 'enc1',
      index: '02',
      short: 'E1',
      name: 'Encoder 1',
      group: 'encoder',
      housing: 'block',
      activation: 'relu',
      params: convParams(x.c, e1.c),
      input: x,
      output: e1,
      op: { type: 'conv', cin: x.c, cout: e1.c, k: K, s: STRIDE, p: PAD, activation: 'relu' },
      role: 'Konvolusi 3×3, stride 2, ReLU',
      summary:
        'Tahap pertama membagi tinggi dan lebar menjadi dua, dari 32 menjadi 16, sambil membuka 16 kanal. Stride 2 mengambil sampel setiap dua piksel. ReLU membuang aktivasi negatif. Jumlah angka naik menjadi 4.096 karena kanal bertambah lebih cepat daripada penyusutan bidang.',
      equations: [
        {
          label: 'Operasi',
          tex: 'h^{(1)}=\\mathrm{ReLU}\\!\\left(W^{(1)} * x+b^{(1)}\\right)',
        },
        {
          label: 'Satu sel',
          tex: 'h_{c,i,j}=\\mathrm{ReLU}\\!\\left(b_c+\\sum_{c\'}\\sum_{a=0}^{2}\\sum_{b=0}^{2} W_{c,c\',a,b}\\, x_{c\',\\,is+a,\\,js+b}\\right)',
        },
        {
          label: 'ReLU',
          tex: '\\mathrm{ReLU}(u)=\\max(0,u)',
        },
        { label: 'Ukuran', tex: convSizeTex(x.h, e1.h) },
      ],
      calcPlain: `Tinggi dan lebar: ${convPlain(x.h)}. Kanal: ${x.c} → ${e1.c}.`,
      weightText: `W ∈ ℝ^${e1.c}×${x.c}×${K}×${K}, bias ∈ ℝ^${e1.c}`,
      note: 'Jendela yang jatuh di luar citra diisi nol (padding 1), jadi sudut tetap terdefinisi.',
    },
    {
      id: 'enc2',
      index: '03',
      short: 'E2',
      name: 'Encoder 2',
      group: 'encoder',
      housing: 'block',
      activation: 'relu',
      params: convParams(e1.c, e2.c),
      input: e1,
      output: e2,
      op: { type: 'conv', cin: e1.c, cout: e2.c, k: K, s: STRIDE, p: PAD, activation: 'relu' },
      role: 'Konvolusi 3×3, stride 2, ReLU',
      summary:
        'Bidang 16×16 menyusut menjadi 8×8 dan kanal naik dari 16 menjadi 32. Tiap lokasi kini merangkum wilayah yang lebih luas dari citra asal. Persamaan sama dengan Encoder 1; yang berubah adalah ukuran tensor bobot.',
      equations: [
        {
          label: 'Operasi',
          tex: 'h^{(2)}=\\mathrm{ReLU}\\!\\left(W^{(2)} * h^{(1)}+b^{(2)}\\right)',
        },
        { label: 'Ukuran', tex: convSizeTex(e1.h, e2.h) },
      ],
      calcPlain: `Tinggi dan lebar: ${convPlain(e1.h)}. Kanal: ${e1.c} → ${e2.c}.`,
      weightText: `W ∈ ℝ^${e2.c}×${e1.c}×${K}×${K}, bias ∈ ℝ^${e2.c}`,
      note: 'Penyusutan spasial bersifat dyadic: 32 → 16 → 8 → 4. Itu sebabnya stride 2 dipakai di setiap encoder.',
    },
    {
      id: 'enc3',
      index: '04',
      short: 'E3',
      name: 'Encoder 3',
      group: 'encoder',
      housing: 'block',
      activation: 'relu',
      params: convParams(e2.c, e3.c),
      input: e2,
      output: e3,
      op: { type: 'conv', cin: e2.c, cout: e3.c, k: K, s: STRIDE, p: PAD, activation: 'relu' },
      role: 'Konvolusi 3×3, stride 2, ReLU',
      summary:
        'Volume terakhir encoder berukuran 64×4×4. Bidang spasialnya kecil, tetapi tiap sel membawa 64 ciri. Jumlah angkanya kembali 1.024 — sama dengan piksel masukan — sebelum bottleneck memaksa representasi yang jauh lebih pendek.',
      equations: [
        {
          label: 'Operasi',
          tex: 'h^{(3)}=\\mathrm{ReLU}\\!\\left(W^{(3)} * h^{(2)}+b^{(3)}\\right)',
        },
        { label: 'Ukuran', tex: convSizeTex(e2.h, e3.h) },
      ],
      calcPlain: `Tinggi dan lebar: ${convPlain(e2.h)}. Kanal: ${e2.c} → ${e3.c}. Elemen: ${numel(e3).toLocaleString('id-ID')}.`,
      weightText: `W ∈ ℝ^${e3.c}×${e2.c}×${K}×${K}, bias ∈ ℝ^${e3.c}`,
      note: 'Blok ini terlihat pendek dan dalam: muka kecil mengikuti 4×4, ketebalan mengikuti 64 kanal.',
    },
    {
      id: 'latent',
      index: '05',
      short: 'z',
      name: 'Bottleneck',
      group: 'latent',
      housing: 'core',
      activation: 'linear',
      params: linearParams(flat, Z_DIM),
      input: e3,
      output: z,
      op: { type: 'linear', nin: flat, nout: Z_DIM, activation: 'none', flatten: true },
      role: 'Rata lalu proyeksi linear ke 32 dimensi',
      summary:
        'Volume 64×4×4 diratakan menjadi vektor 1.024, lalu lapisan linear menulisnya menjadi z berisi 32 bilangan. Tidak ada ReLU di sini supaya kode laten boleh negatif. Inilah kompresi yang sebenarnya: seluruh citra diwakili 32 angka, rasio 32 banding 1.',
      equations: [
        {
          label: 'Rata',
          tex: 'v=\\mathrm{flatten}\\!\\left(h^{(3)}\\right)\\in\\mathbb{R}^{1024}',
        },
        {
          label: 'Kode laten',
          tex: 'z=W_{z} v+b_{z},\\quad z\\in\\mathbb{R}^{32}',
        },
      ],
      calcPlain: `${e3.c}×${e3.h}×${e3.w} = ${flat.toLocaleString('id-ID')} → ${Z_DIM}. Rasio kompresi ${flat / Z_DIM}×.`,
      weightText: `W_z ∈ ℝ^${Z_DIM}×${flat}, bias ∈ ℝ^${Z_DIM}`,
      note: 'Reshape tidak menambah parameter. Seluruh parameter tahap ini ada pada perkalian matriks.',
    },
    {
      id: 'expand',
      index: '06',
      short: 'V',
      name: 'Proyeksi balik',
      group: 'latent',
      housing: 'block',
      activation: 'relu',
      params: linearParams(Z_DIM, flat),
      input: z,
      output: restored,
      op: {
        type: 'linear',
        nin: Z_DIM,
        nout: flat,
        activation: 'relu',
        reshape: { c: restored.c, h: restored.h, w: restored.w },
      },
      role: 'Linear, ReLU, lalu reshape ke volume',
      summary:
        'Decoder konvolusi membutuhkan volume, bukan vektor. Lapisan linear mengembangkan 32 angka menjadi 1.024, ReLU memotong nilai negatif, lalu hasilnya disusun kembali menjadi 64×4×4 — pasangan dari keluaran Encoder 3.',
      equations: [
        {
          label: 'Operasi',
          tex: 'u=\\mathrm{ReLU}(W_{u} z+b_{u})',
        },
        {
          label: 'Bentuk ulang',
          tex: 'h^{(u)}=\\mathrm{reshape}(u)\\in\\mathbb{R}^{64\\times 4\\times 4}',
        },
      ],
      calcPlain: `${Z_DIM} → ${flat.toLocaleString('id-ID')} nilai, disusun sebagai ${shapeLabel(restored)}.`,
      weightText: `W_u ∈ ℝ^${flat}×${Z_DIM}, bias ∈ ℝ^${flat}`,
      note: 'Tahap ini mengembalikan bentuk spasial, tetapi informasi yang tersedia tetap hanya 32 bilangan di z.',
    },
    {
      id: 'dec1',
      index: '07',
      short: 'D1',
      name: 'Decoder 1',
      group: 'decoder',
      housing: 'block',
      activation: 'relu',
      params: convParams(restored.c, d1.c),
      input: restored,
      output: d1,
      op: { type: 'convT', cin: restored.c, cout: d1.c, k: K, s: STRIDE, p: PAD, op: OUT_PAD, activation: 'relu' },
      role: 'ConvTranspose stride 2, ReLU',
      summary:
        'Konvolusi transpos memperbesar bidang. Dengan stride 2, padding 1, dan output padding 1, peta 4×4 tumbuh menjadi 8×8. Kanal turun dari 64 menjadi 32. Hyperparameter ini dipilih supaya ukuran kembali tepat pada kelipatan dua.',
      equations: [
        {
          label: 'Operasi',
          tex: 'h^{(4)}=\\mathrm{ReLU}\\!\\left(\\mathrm{ConvTranspose}\\!\\left(h^{(u)}, W^{(d1)}, b^{(d1)}\\right)\\right)',
        },
        { label: 'Ukuran', tex: convTSizeTex(restored.h, d1.h) },
      ],
      calcPlain: `Tinggi dan lebar: ${convTPlain(restored.h)}. Kanal: ${restored.c} → ${d1.c}.`,
      weightText: `W ∈ ℝ^${restored.c}×${d1.c}×${K}×${K}, bias ∈ ℝ^${d1.c}`,
      note: 'Pada ConvTranspose, sumbu bobot dibalik dibanding konvolusi biasa: kanal masuk di depan, kanal keluar di belakang.',
    },
    {
      id: 'dec2',
      index: '08',
      short: 'D2',
      name: 'Decoder 2',
      group: 'decoder',
      housing: 'block',
      activation: 'relu',
      params: convParams(d1.c, d2.c),
      input: d1,
      output: d2,
      op: { type: 'convT', cin: d1.c, cout: d2.c, k: K, s: STRIDE, p: PAD, op: OUT_PAD, activation: 'relu' },
      role: 'ConvTranspose stride 2, ReLU',
      summary:
        'Bidang tumbuh dari 8×8 menjadi 16×16 dan kanal turun menjadi 16. Bentuk spasial sudah mendekati citra asal. Isi tiap sel masih campuran ciri, belum satu kanal piksel.',
      equations: [
        {
          label: 'Operasi',
          tex: 'h^{(5)}=\\mathrm{ReLU}\\!\\left(\\mathrm{ConvTranspose}\\!\\left(h^{(4)}, W^{(d2)}, b^{(d2)}\\right)\\right)',
        },
        { label: 'Ukuran', tex: convTSizeTex(d1.h, d2.h) },
      ],
      calcPlain: `Tinggi dan lebar: ${convTPlain(d1.h)}. Kanal: ${d1.c} → ${d2.c}.`,
      weightText: `W ∈ ℝ^${d1.c}×${d2.c}×${K}×${K}, bias ∈ ℝ^${d2.c}`,
      note: 'Tiap piksel keluaran menerima sumbangan dari jendela kernel yang “disebar” oleh stride.',
    },
    {
      id: 'dec3',
      index: '09',
      short: 'x̂',
      name: 'Rekonstruksi',
      group: 'decoder',
      housing: 'block',
      activation: 'sigmoid',
      params: convParams(d2.c, d3.c),
      input: d2,
      output: d3,
      op: { type: 'convT', cin: d2.c, cout: d3.c, k: K, s: STRIDE, p: PAD, op: OUT_PAD, activation: 'sigmoid' },
      role: 'ConvTranspose lalu sigmoid',
      summary:
        'Tahap terakhir mengembalikan satu kanal 32×32, sama persis dengan masukan. Sigmoid memetakan setiap nilai ke interval (0, 1), sesuai piksel yang dinormalisasi. Karena bobot masih acak, x̂ belum meniru x. Galat rekonstruksi mengukur selisih itu setelah jaringan dilatih.',
      equations: [
        {
          label: 'Rekonstruksi',
          tex: '\\hat{x}=\\sigma\\!\\left(\\mathrm{ConvTranspose}\\!\\left(h^{(5)}, W^{(d3)}, b^{(d3)}\\right)\\right)',
        },
        {
          label: 'Sigmoid',
          tex: '\\sigma(u)=\\frac{1}{1+e^{-u}}',
        },
        {
          label: 'Galat',
          tex: '\\mathcal{L}=\\frac{1}{N}\\sum_{i}(x_i-\\hat{x}_i)^2',
        },
        { label: 'Ukuran', tex: convTSizeTex(d2.h, d3.h) },
      ],
      calcPlain: `Tinggi dan lebar: ${convTPlain(d2.h)}. Kanal: ${d2.c} → ${d3.c}. Bentuk akhir sama dengan x.`,
      weightText: `W ∈ ℝ^${d2.c}×${d3.c}×${K}×${K}, bias ∈ ℝ^${d3.c}`,
      note: 'Kisi pada pelat keluaran adalah x̂ yang benar-benar dihitung, diskalakan apa adanya pada 0–1. Bobot belum dilatih, jadi polanya belum kembali menjadi cincin.',
    },
  ];
}

export const STAGES = buildStages();

export const TOTAL_PARAMS = STAGES.reduce((sum, stage) => sum + stage.params, 0);

export function getStage(id) {
  return STAGES.find((stage) => stage.id === id) ?? null;
}

export const OVERVIEW = {
  id: 'overview',
  name: 'Autoencoder',
  summary:
    'Autoencoder memampatkan data menjadi kode laten, lalu membangun kembali bentuk asalnya. Mesin ini adalah autoencoder konvolusi untuk satu citra 32×32: tiga tahap encoder, bottleneck 32 dimensi, proyeksi balik, lalu tiga tahap decoder. Pilih sebuah blok untuk melihat persamaan, perubahan bentuk, dan nilai tensor dari forward pass.',
};

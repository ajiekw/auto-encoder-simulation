import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { GROUPS, STAGES } from './architecture.js';
import { channelValues, meanValues } from './forward.js';
import { hexToRgb, paintMap } from './paint.js';

const BASE_TOP = 0.28;
const FIT = 0.12;
const GAP = 0.78;

function measure(stage) {
  if (stage.housing === 'core') return { width: 1.08, height: 1.42, depth: 1.08 };
  const shape = stage.output;
  const face = 0.58 + shape.h * 0.02;
  const depth = 0.52 + Math.log2(Math.max(shape.c, 1)) * 0.13;
  return { width: face, height: face * 0.92, depth };
}

function metalMaterial(hex, roughness = 0.32) {
  const mat = new THREE.MeshStandardMaterial({
    color: hex,
    metalness: 0.86,
    roughness,
    emissive: hex,
    emissiveIntensity: 0.03,
  });
  return mat;
}

function makeTexture(renderer, canvas) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  texture.userData.canvas = canvas;
  return texture;
}

function sheetCount(stage) {
  if (stage.output.vector) return 0;
  return Math.min(4, stage.output.c);
}

export function createEngine(scene, renderer, sim) {
  const root = new THREE.Group();
  root.name = 'engine';
  scene.add(root);

  const pickRoot = new THREE.Group();
  const pipeRoot = new THREE.Group();
  root.add(pipeRoot, pickRoot);

  const placed = [];
  let cursor = 0;
  for (const stage of STAGES) {
    const size = measure(stage);
    const x = cursor + size.width / 2;
    cursor += size.width + GAP;
    placed.push({ stage, size, x });
  }
  const spanMid = (placed[0].x + placed[placed.length - 1].x) / 2;
  for (const item of placed) item.x -= spanMid;

  const blocks = placed.map((item) => buildBlock(item, pickRoot, renderer, sim));

  const pipeMat = new THREE.MeshStandardMaterial({
    color: 0x7d8794,
    metalness: 0.92,
    roughness: 0.28,
  });
  const segments = [];
  for (let i = 0; i < blocks.length - 1; i += 1) {
    const a = blocks[i].rightTip;
    const b = blocks[i + 1].leftTip;
    const mid = a.clone().lerp(b, 0.5);
    mid.y += 0.16;
    const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
    const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 20, 0.042, 12, false), pipeMat);
    tube.castShadow = true;
    pipeRoot.add(tube);
    segments.push({ curve, toId: blocks[i + 1].id });
  }

  buildBase(root, blocks);

  const packet = new THREE.Mesh(
    new THREE.SphereGeometry(0.07, 20, 20),
    new THREE.MeshStandardMaterial({
      color: 0xffffff,
      emissive: 0xffffff,
      emissiveIntensity: 1.6,
      roughness: 0.25,
      metalness: 0.15,
    }),
  );
  const packetLight = new THREE.PointLight(0xffffff, 6, 2.4, 2);
  packet.add(packetLight);
  packet.visible = false;
  root.add(packet);

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let selectedId = null;
  let hoverId = null;

  function paintBlock(block) {
    const record = sim.stages[block.id];
    const tint = hexToRgb(GROUPS[block.stage.group].hex);
    if (block.sheets) {
      block.sheets.forEach((sheet, index) => {
        const values = sheet.useMean
          ? meanValues(record.tensor)
          : channelValues(record.tensor, index);
        paintMap(sheet.texture.userData.canvas, values, record.tensor.w, record.tensor.h, {
          stretch: block.stage.id !== 'input' && block.stage.id !== 'dec3',
          grid: block.stage.id === 'dec3' || block.stage.id === 'input',
          tint: block.stage.id === 'input' || block.stage.id === 'dec3' ? null : tint,
        });
        sheet.texture.needsUpdate = true;
      });
    }
    if (block.bars && record.tensor.kind === 'vector') {
      const data = record.tensor.data;
      let max = 1e-6;
      for (let i = 0; i < data.length; i += 1) max = Math.max(max, Math.abs(data[i]));
      block.bars.forEach((bar, i) => {
        const t = Math.abs(data[i]) / max;
        bar.scale.y = 0.08 + t * 0.92;
        bar.position.y = -0.42 + bar.scale.y * 0.5;
      });
    }
  }

  blocks.forEach(paintBlock);

  function setSelected(id) {
    selectedId = id;
    blocks.forEach((block) => {
      const active = block.id === id;
      const quiet = Boolean(id) && !active;
      block.label.classList.toggle('is-active', active);
      block.label.classList.toggle('is-quiet', quiet);
      block.label.style.zIndex = active ? '3' : '1';
    });
  }

  function setHover(id) {
    hoverId = id;
    blocks.forEach((block) => {
      block.label.classList.toggle('is-hover', block.id === id && block.id !== selectedId);
    });
  }

  function pulse(id) {
    const block = blocks.find((item) => item.id === id);
    if (block) block.pulse = 1;
  }

  function update(dt) {
    blocks.forEach((block) => {
      const active = block.id === selectedId;
      const hovered = block.id === hoverId;
      const goal = active ? 1 : selectedId ? 0.34 : hovered ? 0.7 : 0.62;
      block.focus += (goal - block.focus) * Math.min(1, dt * 7);
      block.rail.material.emissiveIntensity = 0.12 + block.focus * 1.05;
      block.led.material.emissiveIntensity = 0.35 + block.focus * 1.3 + block.pulse * 2.2;
      block.ring.material.opacity = active ? 0.95 : 0;
      block.shells.forEach((mat) => {
        mat.emissiveIntensity = 0.02 + block.focus * 0.06;
      });
      block.pulse *= Math.exp(-dt * 2.6);
      if (block.spin && !reduceMotion) {
        block.spin.forEach((mesh, index) => {
          mesh.rotation.y += dt * (0.35 + index * 0.22);
          mesh.rotation.x += dt * 0.12 * (index % 2 === 0 ? 1 : -1);
        });
      }
    });
  }

  function refresh(nextSim) {
    sim = nextSim;
    blocks.forEach(paintBlock);
  }

  const minX = blocks[0].x - blocks[0].size.width / 2;
  const maxX = blocks[blocks.length - 1].x + blocks[blocks.length - 1].size.width / 2;

  return {
    root,
    pickRoot,
    blocks,
    segments,
    packet,
    bounds: { minX, maxX, midY: 0.85 },
    setSelected,
    setHover,
    pulse,
    update,
    refresh,
    frameTarget(id, aspect = 1.6, frame = null) {
      const block = blocks.find((item) => item.id === id);
      if (!block) return wideShot(minX, maxX, aspect, frame);
      const fov = (CAMERA_FOV * Math.PI) / 180;
      const distance = 3.7;
      const halfW = distance * Math.tan(fov / 2) * Math.max(aspect, 0.75);
      const centerNdc = frame?.centerNdc ?? 0;
      const lookX = block.x - centerNdc * halfW;
      const lookY = BASE_TOP + block.size.height * 0.42;
      return {
        target: new THREE.Vector3(lookX, lookY, 0),
        position: new THREE.Vector3(lookX + 0.28, lookY + 0.62, distance),
      };
    },
  };
}

export const CAMERA_FOV = 40;

function wideShot(minX, maxX, aspect, frame) {
  const span = maxX - minX + 1.35;
  const fov = (CAMERA_FOV * Math.PI) / 180;
  const usable = frame?.usable ?? 0.7;
  const centerNdc = frame?.centerNdc ?? 0;
  const dist = (span * 0.5 * 1.02) / (Math.tan(fov / 2) * Math.max(aspect, 0.75) * usable);
  const halfW = dist * Math.tan(fov / 2) * Math.max(aspect, 0.75);
  const shift = -centerNdc * halfW;
  return {
    target: new THREE.Vector3(shift, 1.05, 0),
    position: new THREE.Vector3(shift + 0.1, Math.max(2.2, dist * 0.11), dist),
  };
}

function buildBase(root, blocks) {
  const minX = blocks[0].x - blocks[0].size.width / 2 - 0.7;
  const maxX = blocks[blocks.length - 1].x + blocks[blocks.length - 1].size.width / 2 + 0.7;
  const width = maxX - minX;
  const mid = (minX + maxX) / 2;
  const slab = new THREE.Mesh(
    new THREE.BoxGeometry(width, 0.22, 3.15),
    new THREE.MeshStandardMaterial({ color: 0x1a212b, metalness: 0.72, roughness: 0.42 }),
  );
  slab.position.set(mid, 0.11, 0);
  slab.receiveShadow = true;
  slab.castShadow = true;
  root.add(slab);

  const plinth = new THREE.Mesh(
    new THREE.BoxGeometry(width + 0.45, 0.08, 3.55),
    new THREE.MeshStandardMaterial({ color: 0x10151c, metalness: 0.6, roughness: 0.55 }),
  );
  plinth.position.set(mid, 0.04, 0);
  plinth.receiveShadow = true;
  root.add(plinth);

  const boltGeo = new THREE.CylinderGeometry(0.045, 0.045, 0.05, 10);
  const boltMat = new THREE.MeshStandardMaterial({ color: 0x8d97a3, metalness: 0.9, roughness: 0.3 });
  for (const x of [minX + 0.28, maxX - 0.28]) {
    for (const z of [-1.35, 1.35]) {
      const bolt = new THREE.Mesh(boltGeo, boltMat);
      bolt.position.set(x, 0.24, z);
      bolt.castShadow = true;
      root.add(bolt);
    }
  }
}

function buildBlock(item, parent, renderer, sim) {
  const { stage, size, x } = item;
  const group = stage.housing === 'core'
    ? buildCore(stage, size, renderer, sim)
    : buildHousing(stage, size, renderer, sim);
  group.position.set(x, BASE_TOP + size.height / 2, 0);
  group.userData.stageId = stage.id;
  parent.add(group);

  const y = BASE_TOP + size.height / 2;
  const label = document.createElement('div');
  label.className = 'block-tag';
  label.style.setProperty('--accent', GROUPS[stage.group].color);
  const name = document.createElement('b');
  name.textContent = stage.short;
  const shape = document.createElement('span');
  shape.textContent = stage.output.vector
    ? `ℝ^${stage.output.vector}`
    : `${stage.output.c}×${stage.output.h}×${stage.output.w}`;
  label.append(name, shape);
  const tag = new CSS2DObject(label);
  tag.position.set(0, size.height / 2 + 0.26, 0);
  group.add(tag);

  return {
    id: stage.id,
    stage,
    size,
    x,
    group,
    label,
    rail: group.userData.rail,
    led: group.userData.led,
    ring: group.userData.ring,
    shells: group.userData.shells,
    sheets: group.userData.sheets,
    bars: group.userData.bars,
    spin: group.userData.spin,
    leftTip: new THREE.Vector3(x - size.width / 2 - FIT, y, 0),
    rightTip: new THREE.Vector3(x + size.width / 2 + FIT, y, 0),
    focus: 0.62,
    pulse: 0,
  };
}

function buildHousing(stage, size, renderer, sim) {
  const { width: w, height: h, depth: d } = size;
  const group = new THREE.Group();
  const accent = GROUPS[stage.group].hex;
  const shellColor = stage.group === 'decoder' ? 0xb7a59e : stage.group === 'latent' ? 0xb7ae96 : stage.group === 'encoder' ? 0xa7b4b6 : 0xb4bcc6;
  const wall = 0.055;
  const shells = [];
  const addShell = (geometry, x, y, z) => {
    const mat = metalMaterial(shellColor);
    shells.push(mat);
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.userData.stageId = stage.id;
    group.add(mesh);
    return mesh;
  };

  addShell(new THREE.BoxGeometry(w, h, wall), 0, 0, -d / 2 + wall / 2);
  addShell(new THREE.BoxGeometry(wall, h, d), -w / 2 + wall / 2, 0, 0);
  addShell(new THREE.BoxGeometry(wall, h, d), w / 2 - wall / 2, 0, 0);
  addShell(new THREE.BoxGeometry(w, wall, d), 0, h / 2 - wall / 2, 0);
  addShell(new THREE.BoxGeometry(w, wall, d), 0, -h / 2 + wall / 2, 0);

  const lip = 0.07;
  const lipDepth = 0.05;
  const lipMat = metalMaterial(0x8e98a4, 0.38);
  shells.push(lipMat);
  const lips = [
    [w, lip, lipDepth, 0, h / 2 - lip / 2, d / 2],
    [w, lip, lipDepth, 0, -h / 2 + lip / 2, d / 2],
    [lip, h - lip * 2, lipDepth, -w / 2 + lip / 2, 0, d / 2],
    [lip, h - lip * 2, lipDepth, w / 2 - lip / 2, 0, d / 2],
  ];
  for (const [lw, lh, ld, lx, ly, lz] of lips) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(lw, lh, ld), lipMat);
    mesh.position.set(lx, ly, lz);
    mesh.castShadow = true;
    mesh.userData.stageId = stage.id;
    group.add(mesh);
  }

  const railMat = new THREE.MeshStandardMaterial({
    color: accent,
    emissive: accent,
    emissiveIntensity: 0.4,
    metalness: 0.55,
    roughness: 0.35,
  });
  const rail = new THREE.Mesh(new THREE.BoxGeometry(w * 0.72, 0.035, 0.06), railMat);
  rail.position.set(0, h / 2 + 0.02, d * 0.12);
  group.add(rail);

  const ledMat = new THREE.MeshStandardMaterial({
    color: accent,
    emissive: accent,
    emissiveIntensity: 0.8,
    roughness: 0.4,
  });
  const led = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 12), ledMat);
  led.position.set(w * 0.32, h / 2 + 0.02, d * 0.2);
  group.add(led);

  const back = new THREE.Mesh(
    new THREE.PlaneGeometry(w - 0.16, h - 0.16),
    new THREE.MeshStandardMaterial({ color: 0x0c1016, roughness: 0.92, metalness: 0.05 }),
  );
  back.position.z = -d / 2 + wall + 0.02;
  back.userData.stageId = stage.id;
  group.add(back);

  const boltGeo = new THREE.CylinderGeometry(0.028, 0.028, 0.03, 8);
  const boltMat = metalMaterial(0x9aa3ae, 0.25);
  for (const [bx, by] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    const bolt = new THREE.Mesh(boltGeo, boltMat);
    bolt.rotation.x = Math.PI / 2;
    bolt.position.set(bx * (w / 2 - 0.11), by * (h / 2 - 0.11), d / 2 + 0.02);
    bolt.userData.stageId = stage.id;
    group.add(bolt);
  }

  const finMat = metalMaterial(0x66707c, 0.45);
  for (let i = 0; i < 4; i += 1) {
    const fin = new THREE.Mesh(new THREE.BoxGeometry(w * 0.62, 0.02, 0.07), finMat);
    fin.position.set(0, (i - 1.5) * (h / 5.2), -d / 2 - 0.035);
    group.add(fin);
  }

  addFittings(group, stage, size);

  const sheets = [];
  const record = sim.stages[stage.id];
  const planes = sheetCount(stage);
  const face = Math.min(w, h) * 0.62;
  for (let i = 0; i < planes; i += 1) {
    const canvas = document.createElement('canvas');
    const texture = makeTexture(renderer, canvas);
    const mat = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.62,
      metalness: 0.04,
      emissive: 0x111111,
      emissiveIntensity: 0.18,
    });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(face, face), mat);
    mesh.position.z = d * 0.2 - i * 0.055;
    mesh.userData.stageId = stage.id;
    group.add(mesh);
    sheets.push({ texture, useMean: planes === 1 });
    if (record.tensor.kind !== 'volume') continue;
  }

  const ring = selectionRing(stage, size);
  group.add(ring);

  group.userData.rail = rail;
  group.userData.led = led;
  group.userData.ring = ring;
  group.userData.shells = shells;
  group.userData.sheets = sheets;
  group.userData.bars = null;
  group.userData.spin = null;
  return group;
}

function buildCore(stage, size, renderer, sim) {
  const group = new THREE.Group();
  const accent = GROUPS.latent.hex;
  const shells = [];
  const gap = 1.15;
  const cageMat = metalMaterial(0xc2b48a, 0.3);
  shells.push(cageMat);
  const cage = new THREE.Mesh(
    new THREE.CylinderGeometry(0.46, 0.46, 1.22, 40, 1, true, gap / 2, Math.PI * 2 - gap),
    cageMat,
  );
  cage.castShadow = true;
  cage.userData.stageId = stage.id;
  group.add(cage);

  const capMat = metalMaterial(0x8d8472, 0.35);
  shells.push(capMat);
  for (const y of [0.62, -0.62]) {
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.08, 28), capMat);
    cap.position.y = y;
    cap.castShadow = true;
    cap.userData.stageId = stage.id;
    group.add(cap);
  }

  const spin = [];
  for (const [radius, tube, y] of [[0.34, 0.012, 0.28], [0.3, 0.01, -0.05], [0.26, 0.012, -0.32]]) {
    const mat = new THREE.MeshStandardMaterial({
      color: accent,
      emissive: accent,
      emissiveIntensity: 0.7,
      metalness: 0.4,
      roughness: 0.3,
    });
    const torus = new THREE.Mesh(new THREE.TorusGeometry(radius, tube, 8, 28), mat);
    torus.rotation.x = Math.PI / 2;
    torus.position.y = y;
    group.add(torus);
    spin.push(torus);
  }

  const record = sim.stages[stage.id];
  const bars = [];
  const count = record.tensor.data.length;
  const barMat = new THREE.MeshStandardMaterial({
    color: accent,
    emissive: accent,
    emissiveIntensity: 0.55,
    roughness: 0.4,
    metalness: 0.1,
  });
  for (let i = 0; i < count; i += 1) {
    const bar = new THREE.Mesh(new THREE.BoxGeometry(0.012, 1, 0.012), barMat);
    const span = 0.78;
    bar.position.set(-span / 2 + (i + 0.5) * (span / count), -0.2, 0.12);
    bar.userData.stageId = stage.id;
    group.add(bar);
    bars.push(bar);
  }

  const railMat = new THREE.MeshStandardMaterial({
    color: accent,
    emissive: accent,
    emissiveIntensity: 0.5,
    metalness: 0.45,
    roughness: 0.32,
  });
  const rail = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.035, 0.08), railMat);
  rail.position.set(0, 0.72, 0.18);
  group.add(rail);

  const led = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 12, 12),
    new THREE.MeshStandardMaterial({ color: accent, emissive: accent, emissiveIntensity: 0.9, roughness: 0.35 }),
  );
  led.position.set(0.34, 0.7, 0.22);
  group.add(led);

  addFittings(group, stage, size);
  const ring = selectionRing(stage, size);
  group.add(ring);

  group.userData.rail = rail;
  group.userData.led = led;
  group.userData.ring = ring;
  group.userData.shells = shells;
  group.userData.sheets = null;
  group.userData.bars = bars;
  group.userData.spin = spin;
  void renderer;
  void sim;
  return group;
}

function addFittings(group, stage, size) {
  const geo = new THREE.CylinderGeometry(0.065, 0.065, FIT, 12);
  const mat = metalMaterial(0x5e6772, 0.34);
  for (const side of [-1, 1]) {
    const fit = new THREE.Mesh(geo, mat);
    fit.rotation.z = Math.PI / 2;
    fit.position.x = side * (size.width / 2 + FIT / 2);
    fit.castShadow = true;
    fit.userData.stageId = stage.id;
    group.add(fit);
  }
}

function selectionRing(stage, size) {
  const radius = Math.max(size.width, size.depth) * 0.48;
  const mat = new THREE.MeshBasicMaterial({
    color: GROUPS[stage.group].hex,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(radius * 0.86, radius, 40), mat);
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = -size.height / 2 + 0.025;
  return ring;
}

export function stageIdFrom(object) {
  let current = object;
  while (current) {
    if (current.userData?.stageId) return current.userData.stageId;
    current = current.parent;
  }
  return null;
}

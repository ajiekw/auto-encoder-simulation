import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { GROUPS, STAGES } from './architecture.js';
import { CAMERA_FOV, createEngine, stageIdFrom } from './engine.js';
import { simulate } from './forward.js';
import { createUi } from './ui.js';
import './style.css';

const view = document.querySelector('#view');
const overlay = document.querySelector('#overlay');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
scene.fog = new THREE.Fog(0x07090d, 22, 48);

const camera = new THREE.PerspectiveCamera(CAMERA_FOV, 1, 0.1, 80);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
view.appendChild(renderer.domElement);

const labels = new CSS2DRenderer();
labels.domElement.style.position = 'absolute';
labels.domElement.style.inset = '0';
labels.domElement.style.pointerEvents = 'none';
view.appendChild(labels.domElement);

try {
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.05).texture;
  pmrem.dispose();
  scene.environmentIntensity = 0.7;
} catch (error) {
  console.warn('Environment map dilewati', error);
}

const hemi = new THREE.HemisphereLight(0xc5d4e4, 0x14181e, 0.55);
scene.add(hemi);
const key = new THREE.DirectionalLight(0xfff4e5, 2.4);
key.position.set(6, 11, 8);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.near = 1;
key.shadow.camera.far = 40;
key.shadow.camera.left = -14;
key.shadow.camera.right = 14;
key.shadow.camera.top = 14;
key.shadow.camera.bottom = -14;
key.shadow.bias = -0.0004;
key.shadow.normalBias = 0.03;
scene.add(key);
const rim = new THREE.DirectionalLight(0x8eb4d6, 1.15);
rim.position.set(-8, 5, -6);
scene.add(rim);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(18, 64),
  new THREE.MeshStandardMaterial({ color: 0x0c1016, metalness: 0.35, roughness: 0.88 }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(28, 28, 0x243041, 0x18202b);
grid.position.y = 0.005;
const gridMats = Array.isArray(grid.material) ? grid.material : [grid.material];
gridMats.forEach((mat) => {
  mat.transparent = true;
  mat.opacity = 0.45;
});
scene.add(grid);

let seed = 7;
let sim = simulate(seed);
const engine = createEngine(scene, renderer, sim);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = 1.28;
controls.minDistance = 2.1;
controls.maxDistance = 36;
controls.target.set(0, 0.8, 0);

const rig = {
  t: 1,
  fromP: new THREE.Vector3(),
  toP: new THREE.Vector3(),
  fromT: new THREE.Vector3(),
  toT: new THREE.Vector3(),
};

const play = { mode: 'idle', seg: 0, phase: 'hold', hold: 0, t: 0 };
const packetColor = new THREE.Color();
let selected = 'overview';

const ui = createUi(overlay, {
  onSelect: (id) => select(id, { fly: true }),
  onPlay: togglePlay,
  onReseed: reseed,
  onStep: step,
});
ui.setSim(sim);

function resize() {
  const width = view.clientWidth || window.innerWidth;
  const height = view.clientHeight || window.innerHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
  labels.setSize(width, height);
}

function flyTo(position, target) {
  rig.fromP.copy(camera.position);
  rig.toP.copy(position);
  rig.fromT.copy(controls.target);
  rig.toT.copy(target);
  rig.t = 0;
}

function updateRig(dt) {
  if (rig.t >= 1) return;
  rig.t = Math.min(1, rig.t + dt / 0.85);
  const k = rig.t * rig.t * (3 - 2 * rig.t);
  camera.position.lerpVectors(rig.fromP, rig.toP, k);
  controls.target.lerpVectors(rig.fromT, rig.toT, k);
}

function viewFrame() {
  const inspector = document.querySelector('#inspector');
  const rect = inspector.getBoundingClientRect();
  const coversRight = rect.top < 120 && rect.left > window.innerWidth * 0.45;
  if (!coversRight) return { usable: 0.9, centerNdc: 0 };
  const left = 20;
  const right = Math.max(left + 240, rect.left - 20);
  const usable = Math.max(0.42, (right - left) / window.innerWidth);
  const centerNdc = (((left + right) / 2) / window.innerWidth) * 2 - 1;
  return { usable, centerNdc };
}

function shotFor(id) {
  const frame = viewFrame();
  return engine.frameTarget(id && id !== 'overview' ? id : null, camera.aspect, frame);
}

function select(id, { fly = true, fromPlay = false } = {}) {
  if (!fromPlay && play.mode === 'playing') {
    play.mode = 'paused';
    ui.setMode(play.mode);
  }
  const next = id && id !== 'overview' ? id : 'overview';
  selected = next;
  engine.setSelected(next === 'overview' ? null : next);
  ui.select(next);
  if (!fly) return;
  const shot = shotFor(next);
  flyTo(shot.position, shot.target);
}

function togglePlay() {
  if (play.mode === 'playing') {
    play.mode = 'paused';
    ui.setMode(play.mode);
    return;
  }
  if (play.mode === 'paused') {
    play.mode = 'playing';
    ui.setMode(play.mode);
    return;
  }
  play.mode = 'playing';
  play.seg = 0;
  play.phase = 'hold';
  play.hold = 0.45;
  play.t = 0;
  engine.packet.visible = true;
  engine.packet.position.copy(engine.segments[0].curve.getPoint(0));
  tintPacket(STAGES[0].id);
  engine.pulse(STAGES[0].id);
  ui.setMode(play.mode);
  select(STAGES[0].id, { fly: false, fromPlay: true });
  const shot = shotFor(null);
  flyTo(shot.position, shot.target);
}

function tintPacket(id) {
  const stage = STAGES.find((item) => item.id === id);
  packetColor.set(GROUPS[stage.group].hex);
  engine.packet.material.color.copy(packetColor);
  engine.packet.material.emissive.copy(packetColor);
}

function updatePlay(dt) {
  if (play.mode !== 'playing') return;
  if (play.phase === 'hold') {
    play.hold -= dt;
    if (play.hold > 0) return;
    if (play.seg >= engine.segments.length) {
      play.mode = 'done';
      engine.packet.visible = false;
      ui.setMode(play.mode);
      return;
    }
    play.phase = 'travel';
    play.t = 0;
    tintPacket(engine.segments[play.seg].toId);
    return;
  }
  play.t += dt / 0.7;
  const segment = engine.segments[play.seg];
  const k = Math.min(1, play.t);
  const eased = k < 0.5 ? 2 * k * k : 1 - ((-2 * k + 2) ** 2) / 2;
  engine.packet.position.copy(segment.curve.getPoint(eased));
  if (k < 1) return;
  play.seg += 1;
  select(segment.toId, { fly: false, fromPlay: true });
  engine.pulse(segment.toId);
  play.phase = 'hold';
  play.hold = 0.38;
  if (play.mode === 'playing') ui.setMode('playing');
}

function step(direction) {
  const order = ['overview', ...STAGES.map((stage) => stage.id)];
  let index = order.indexOf(selected);
  if (index < 0) index = 0;
  index = Math.min(order.length - 1, Math.max(0, index + direction));
  select(order[index], { fly: true });
}

function reseed() {
  if (play.mode === 'playing') {
    play.mode = 'paused';
    ui.setMode(play.mode);
  }
  seed = (seed * 17 + 3) % 10000;
  sim = simulate(seed);
  engine.refresh(sim);
  ui.setSim(sim);
  ui.render();
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let press = null;

function setPointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

renderer.domElement.addEventListener('pointerdown', (event) => {
  press = { x: event.clientX, y: event.clientY };
  if (rig.t < 1) rig.t = 1;
});

renderer.domElement.addEventListener('pointerup', (event) => {
  if (!press) return;
  const moved = Math.hypot(event.clientX - press.x, event.clientY - press.y);
  press = null;
  if (moved > 5) return;
  setPointer(event);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(engine.pickRoot.children, true);
  const id = hits.length ? stageIdFrom(hits[0].object) : null;
  select(id ?? 'overview', { fly: true });
});

renderer.domElement.addEventListener('pointermove', (event) => {
  setPointer(event);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(engine.pickRoot.children, true);
  const id = hits.length ? stageIdFrom(hits[0].object) : null;
  engine.setHover(id);
  renderer.domElement.style.cursor = id ? 'pointer' : '';
});

window.addEventListener('keydown', (event) => {
  const tag = document.activeElement?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  if (event.key === 'ArrowRight') {
    event.preventDefault();
    step(1);
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault();
    step(-1);
  } else if (event.key === 'Escape') {
    select('overview', { fly: true });
  } else if (event.key === ' ') {
    event.preventDefault();
    togglePlay();
  }
});

window.addEventListener('resize', resize);
resize();
select('overview', { fly: false });
const opening = shotFor(null);
camera.position.copy(opening.position);
controls.target.copy(opening.target);
camera.lookAt(controls.target);

const clock = new THREE.Clock();
function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.05);
  updatePlay(dt);
  updateRig(dt);
  engine.update(dt);
  controls.update();
  renderer.render(scene, camera);
  labels.render(scene, camera);
}
frame();

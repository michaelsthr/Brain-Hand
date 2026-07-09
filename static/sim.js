import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import load_mujoco from '/static/wasm/mujoco_wasm.js';

const OPEN = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.6981,0];
const FIST = [0,0,0,1.571,1.571,1.571,0,1.571,1.571,1.571,0,1.571,1.571,1.571,
              0,0,1.571,1.571,1.571,0.17,1.2,0,0.61,0.52];
const NQ_HAND = OPEN.length;

function getPosition(buffer, index, target) {
  return target.set(
     buffer[(index * 3) + 0],
     buffer[(index * 3) + 2],
    -buffer[(index * 3) + 1]);
}
function getQuaternion(buffer, index, target) {
  return target.set(
    -buffer[(index * 4) + 1],
    -buffer[(index * 4) + 3],
     buffer[(index * 4) + 2],
    -buffer[(index * 4) + 0]);
}

let mujoco, model, data, bodies = {};
let scene, camera, renderer, controls;
let hemiLight, keyLight, rimLight;

const SCENE_THEMES = {
  dark: {
    bg: 0x0d0f14,
    hemiSky: 0x8899bb, hemiGround: 0x223344, hemiIntensity: 1.1,
    key: 0xfff0d8, keyIntensity: 1.6,
    rim: 0x556699, rimIntensity: 0.8,
  },
  light: {
    bg: 0xf4f6fa,
    hemiSky: 0xffffff, hemiGround: 0xc9d2df, hemiIntensity: 1.25,
    key: 0xffffff, keyIntensity: 1.3,
    rim: 0x9aa8bd, rimIntensity: 0.5,
  },
};

function applySceneTheme(mode) {
  const t = SCENE_THEMES[mode] || SCENE_THEMES.dark;
  scene.background = new THREE.Color(t.bg);
  hemiLight.color.setHex(t.hemiSky);
  hemiLight.groundColor.setHex(t.hemiGround);
  hemiLight.intensity = t.hemiIntensity;
  keyLight.color.setHex(t.key);
  keyLight.intensity = t.keyIntensity;
  rimLight.color.setHex(t.rim);
  rimLight.intensity = t.rimIntensity;
}

let leftT = 1.0, rightT = 1.0;
let anim = { type: 'idle' };

function tickAnim(nowMS) {
  const now = nowMS / 1000;
  if (anim.type === 'openclose') {
    const HALF = 0.875;
    const e = now - anim.start;
    let t;
    if (e < HALF)          t = 1.0 - (e / HALF);
    else if (e < 2 * HALF) t = (e - HALF) / HALF;
    else { t = 1.0; anim = { type: 'idle' }; }
    if (anim.which === 'left') { leftT = t; rightT = 1.0; }
    else                       { rightT = t; leftT = 1.0; }
  } else if (anim.type === 'oscillate') {
    const v = (Math.sin((now - anim.start) * 1.5) + 1.0) / 2.0;
    leftT = v; rightT = 1.0 - v;
  } else if (anim.type === 'easeTo') {
    const a = Math.min((now - anim.start) / anim.dur, 1.0);
    leftT  = anim.l0 + a * (anim.lT - anim.l0);
    rightT = anim.r0 + a * (anim.rT - anim.r0);
    if (a >= 1.0) anim = { type: 'idle' };
  }
}

function applyPose() {
  const q = data.qpos;
  for (let i = 0; i < NQ_HAND; i++) {
    q[i]           = OPEN[i] + leftT  * (FIST[i] - OPEN[i]);
    q[NQ_HAND + i] = OPEN[i] + rightT * (FIST[i] - OPEN[i]);
  }
  mujoco.mj_forward(model, data);
}

function buildScene() {
  const decoder = new TextDecoder('utf-8');
  const names = decoder.decode(model.names).split('\0');

  const root = new THREE.Group();
  root.name = 'MuJoCo Root';
  scene.add(root);

  bodies = {};
  const meshes = {};

  for (let g = 0; g < model.ngeom; g++) {
    if (!(model.geom_group[g] < 3)) continue;

    const b = model.geom_bodyid[g];
    const type = model.geom_type[g];
    if (type === mujoco.mjtGeom.mjGEOM_PLANE.value) continue;

    const size = [model.geom_size[g*3], model.geom_size[g*3+1], model.geom_size[g*3+2]];

    if (!(b in bodies)) {
      bodies[b] = new THREE.Group();
      bodies[b].name = names[model.name_bodyadr[b]];
      bodies[b].bodyID = b;
    }

    let geometry = new THREE.SphereGeometry(size[0] * 0.5);
    if (type === mujoco.mjtGeom.mjGEOM_SPHERE.value) {
      geometry = new THREE.SphereGeometry(size[0]);
    } else if (type === mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
      geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
    } else if (type === mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
      geometry = new THREE.SphereGeometry(1);
    } else if (type === mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
      geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
    } else if (type === mujoco.mjtGeom.mjGEOM_BOX.value) {
      geometry = new THREE.BoxGeometry(size[0]*2, size[2]*2, size[1]*2);
    } else if (type === mujoco.mjtGeom.mjGEOM_MESH.value) {
      const meshID = model.geom_dataid[g];
      if (!(meshID in meshes)) {
        geometry = new THREE.BufferGeometry();
        const va = model.mesh_vertadr[meshID], vn = model.mesh_vertnum[meshID];
        const vert = model.mesh_vert.subarray(va * 3, (va + vn) * 3).slice();
        for (let v = 0; v < vert.length; v += 3) {
          const t = vert[v+1]; vert[v+1] = vert[v+2]; vert[v+2] = -t;
        }
        const fa = model.mesh_faceadr[meshID], fn = model.mesh_facenum[meshID];
        const face = model.mesh_face.subarray(fa * 3, (fa + fn) * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(vert, 3));
        geometry.setIndex(Array.from(face));
        geometry.computeVertexNormals();
        meshes[meshID] = geometry;
      } else {
        geometry = meshes[meshID];
      }
    }

    let color = [model.geom_rgba[g*4], model.geom_rgba[g*4+1],
                 model.geom_rgba[g*4+2], model.geom_rgba[g*4+3]];
    const matId = model.geom_matid[g];
    if (matId !== -1) {
      color = [model.mat_rgba[matId*4], model.mat_rgba[matId*4+1],
               model.mat_rgba[matId*4+2], model.mat_rgba[matId*4+3]];
    }
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(color[0], color[1], color[2]),
      transparent: color[3] < 1.0,
      opacity: color[3],
      roughness: 0.7,
      metalness: 0.1,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.bodyID = b;
    bodies[b].add(mesh);
    getPosition(model.geom_pos, g, mesh.position);
    getQuaternion(model.geom_quat, g, mesh.quaternion);
    if (type === mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) mesh.scale.set(size[0], size[2], size[1]);
  }

  for (let b = 0; b < model.nbody; b++) {
    if (!bodies[b]) { bodies[b] = new THREE.Group(); bodies[b].bodyID = b; }
    root.add(bodies[b]);
  }
}

function updateBodies() {
  for (let b = 0; b < model.nbody; b++) {
    if (!bodies[b]) continue;
    getPosition(data.xpos, b, bodies[b].position);
    getQuaternion(data.xquat, b, bodies[b].quaternion);
  }
}

function renderLoop(t) {
  requestAnimationFrame(renderLoop);
  tickAnim(t);
  applyPose();
  updateBodies();
  controls.update();
  renderer.render(scene, camera);
}

async function init() {
  mujoco = await load_mujoco();

  mujoco.FS.mkdir('/working');
  const files = ['scene.xml',
    'f_distal_pst.obj','f_knuckle.obj','f_middle.obj','f_proximal.obj',
    'forearm_0.obj','forearm_1.obj','forearm_collision.obj','lf_metacarpal.obj',
    'mounting_plate.obj','palm.obj','th_distal_pst.obj','th_middle.obj',
    'th_proximal.obj','wrist.obj'];
  await Promise.all(files.map(async (f) => {
    const buf = await (await fetch('/static/model/' + f)).arrayBuffer();
    mujoco.FS.writeFile('/working/' + f, new Uint8Array(buf));
  }));

  model = mujoco.MjModel.loadFromXML('/working/scene.xml');
  data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);

  const container = document.getElementById('simView');
  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.01, 100);
  camera.position.set(0.55, 0.5, 0.75);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0.2, 0.0, 0.0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  hemiLight = new THREE.HemisphereLight(0x8899bb, 0x223344, 1.1);
  scene.add(hemiLight);
  keyLight = new THREE.DirectionalLight(0xfff0d8, 1.6);
  keyLight.position.set(-0.4, 1.2, 0.6);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  keyLight.shadow.camera.near = 0.1; keyLight.shadow.camera.far = 5;
  scene.add(keyLight);
  rimLight = new THREE.DirectionalLight(0x556699, 0.8);
  rimLight.position.set(0.4, 0.6, -0.8);
  scene.add(rimLight);

  applySceneTheme(document.documentElement.dataset.theme);

  buildScene();
  applyPose();

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  requestAnimationFrame(renderLoop);
}

window.sim = {
  ready: init(),
  setBlend(l, r) {
    anim = { type: 'idle' };
    leftT  = Math.min(Math.max(l, 0), 1);
    rightT = Math.min(Math.max(r, 0), 1);
  },
  predict(which) {
    leftT = 1.0; rightT = 1.0;
    anim = { type: 'openclose', which, start: performance.now() / 1000 };
  },
  oscillate(on) {
    anim = on ? { type: 'oscillate', start: performance.now() / 1000 }
              : { type: 'idle' };
  },
  reset() {
    anim = { type: 'easeTo', l0: leftT, r0: rightT, lT: 1.0, rT: 1.0,
             dur: 1.0, start: performance.now() / 1000 };
  },
  setTheme(mode) {
    if (scene) applySceneTheme(mode);
  },
};

import {
  type GPUApp, type GPUAppPipeline, initWebGPU,
  initRenderPipeline, buildSceneBindGroups, render,
} from "./renderer";

import type { Material } from "./types";
import { type Vec3 } from "./math";
import { Camera } from "./camera";
import { Scene } from "./scene";
import { type MeshInstance, createBox, createQuad, createSphere, createCylinder } from "./mesh";

const state = {
  scene: null as unknown as Scene,
  app: null as unknown as GPUApp
};

const ui = {
  canvas: document.querySelector("canvas") as HTMLCanvasElement,
  errorScreen: document.querySelector("#errorScreen") as HTMLDivElement,
  overlayShell: document.querySelector(".overlay-shell") as HTMLDivElement,
  raytracingCheck: document.querySelector("#raytracingCheckbox") as HTMLInputElement,
  albedoPicker: document.querySelector("#diffuseAlbedo") as HTMLInputElement,
  roughnessSlider: document.querySelector("#roughness") as HTMLInputElement,
  metalnessSlider: document.querySelector("#metalness") as HTMLInputElement,
  toneMappingCheck: document.querySelector("#toneMappingCheckbox") as HTMLInputElement,
  restirCheck: document.querySelector("#restirCheckbox") as HTMLInputElement,
  restirBiasedCheck: document.querySelector("#restirBiasedCheckbox") as HTMLInputElement,
  temporalReuseCheck: document.querySelector("#temporalReuseCheckbox") as HTMLInputElement,
  spatialReuseCheck: document.querySelector("#spatialReuseCheckbox") as HTMLInputElement,
  risOnBouncesCheck: document.querySelector("#risOnBouncesCheckbox") as HTMLInputElement,
  accumulationCheck: document.querySelector("#accumulationCheckbox") as HTMLInputElement,
  sceneSelect: document.getElementById("sceneSelect") as HTMLSelectElement,
  viewBvhCheck: document.querySelector("#viewBvh") as HTMLInputElement,
  maxDepthSlider: document.querySelector("#rayDepth") as HTMLInputElement,
  stratifiedGridSlider: document.querySelector("#stratifiedGridN") as HTMLInputElement,
  sppText: document.querySelector("#spp") as HTMLSpanElement,
  depthText: document.querySelector("#depthText") as HTMLSpanElement,
  fpsText: document.querySelector("#fpsCounter") as HTMLSpanElement,
  frameTimeText: document.querySelector("#frameTime") as HTMLSpanElement,
};

const keys: Record<string, boolean> = {
  w: false, a: false, s: false, d: false,
  ArrowUp: false, ArrowDown: false, ArrowLeft: false, ArrowRight: false,
  e: false, q: false
};

function hexToSRGB(hex: string): Vec3 {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

function initEvents() {
  if (!state.app || !state.scene) return;

  ui.canvas.addEventListener("mousedown", e => {
    state.scene.camera.lastX = e.clientX;
    state.scene.camera.lastY = e.clientY;
    if (e.button === 0) state.scene.camera.dragging = true;
  });

  window.addEventListener("mouseup", () => {
    state.scene.camera.dragging = false;
  });

  ui.canvas.addEventListener("mousemove", e => {
    if (!state.scene.camera.dragging) return;

    const dx = e.clientX - state.scene.camera.lastX;
    const dy = e.clientY - state.scene.camera.lastY;
    state.scene.camera.lastX = e.clientX;
    state.scene.camera.lastY = e.clientY;

    state.scene.camera.processMouseMovement(dx, dy);
    state.scene.frameCount = 0.0;
  });

 window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (keys.hasOwnProperty(k)) keys[k] = true;
    if (keys.hasOwnProperty(e.key)) keys[e.key] = true;
  });

  window.addEventListener("keyup", (e) => {
    const k = e.key.toLowerCase();
    if (keys.hasOwnProperty(k)) keys[k] = false;
    if (keys.hasOwnProperty(e.key)) keys[e.key] = false;
  });

  ui.canvas.addEventListener("contextmenu", e => e.preventDefault());

  ui.albedoPicker.addEventListener("input", () => {
    state.scene.frameCount = 0.0;
    state.scene.materials[3].albedo = hexToSRGB(ui.albedoPicker.value);
    state.scene.updateMaterials(state.app);
  });

  ui.roughnessSlider.addEventListener("input", () => {
    const val = parseFloat(ui.roughnessSlider.value);

    state.scene.frameCount = 0.0;
    state.scene.materials[3].roughness = val;
    state.scene.updateMaterials(state.app);
  });

  ui.metalnessSlider.addEventListener("input", () => {
    const val = parseFloat(ui.metalnessSlider.value);

    state.scene.frameCount = 0.0;
    state.scene.materials[3].metalness = val;
    state.scene.updateMaterials(state.app);
  });

  ui.raytracingCheck.addEventListener("input", () => {
    state.scene.frameCount = 0.0;
    state.scene.absoluteFrameCount = 0.0;
  });

  ui.toneMappingCheck.addEventListener("input", () => {
    state.scene.toneMappingEnabled = ui.toneMappingCheck.checked;
  });

  ui.restirCheck.addEventListener("input", () => {
    state.scene.restirEnabled = ui.restirCheck.checked;

    if (ui.restirCheck.checked) {
      ui.stratifiedGridSlider.value = "1";
      state.scene.stratifiedGridSize = 1;
      ui.sppText.innerText = "1";

      state.scene.absoluteFrameCount = 0.0;
    }

    state.scene.frameCount = 0.0;
  });

  ui.restirBiasedCheck.addEventListener("input", () => {
    state.scene.restirBiased = ui.restirBiasedCheck.checked;
    state.scene.frameCount = 0.0;
    state.scene.absoluteFrameCount = 0.0;
  });

  ui.temporalReuseCheck.addEventListener("input", () => {
    state.scene.temporalReuseEnabled = ui.temporalReuseCheck.checked;
    state.scene.frameCount = 0.0;
  });

  ui.spatialReuseCheck.addEventListener("input", () => {
    state.scene.spatialReuseEnabled = ui.spatialReuseCheck.checked;
    state.scene.frameCount = 0.0;
  });

  ui.risOnBouncesCheck.addEventListener("input", () => {
    state.scene.useRISOnBounces = ui.risOnBouncesCheck.checked;
    state.scene.frameCount = 0.0;
  });

  ui.accumulationCheck.addEventListener("input", () => {
    state.scene.accumulationEnabled = ui.accumulationCheck.checked;
    state.scene.frameCount = 0.0;
  });

  ui.viewBvhCheck.addEventListener("input", () => {
    state.scene.viewBvh = ui.viewBvhCheck.checked;
  });

  ui.maxDepthSlider.addEventListener("input", () => {
    const val = parseInt(ui.maxDepthSlider.value);

    state.scene.frameCount = 0.0;
    state.scene.maxRayDepth = val;
    ui.depthText.innerText = `${val}`;
  });

  ui.stratifiedGridSlider.addEventListener("input", () => {
    const val = parseInt(ui.stratifiedGridSlider.value);

    state.scene.frameCount = 0.0;
    state.scene.stratifiedGridSize = val;
    ui.sppText.innerText = `${val * val}`;
  });
}

function handleCameraMovement() {
  if (!state.scene) return;

  let moveForward = 0;
  let moveRight = 0;
  let moveUp = 0;

  if (keys.w || keys.ArrowUp) moveForward += 1;
  if (keys.s || keys.ArrowDown) moveForward -= 1;
  if (keys.d || keys.ArrowRight) moveRight += 1;
  if (keys.a || keys.ArrowLeft) moveRight -= 1;
  if (keys.e) moveUp += 1;
  if (keys.q) moveUp -= 1;

  if (moveForward !== 0 || moveRight !== 0 || moveUp !== 0) {
    state.scene.camera.processKeyboard(moveForward, moveRight, moveUp);
    state.scene.frameCount = 0.0;
  }
}

let lastTime = performance.now();
let accumulatedTime = 0;
let framesThisInterval = 0;

function updateStats() {
  const now = performance.now();
  const dt = now - lastTime;
  lastTime = now;

  accumulatedTime += dt;
  framesThisInterval++;

  if (accumulatedTime >= 500) {
    const avgFrameTime = accumulatedTime / framesThisInterval;
    const fps = 1000 / avgFrameTime;

    ui.fpsText.textContent = fps.toFixed(1);
    ui.frameTimeText.textContent = `${avgFrameTime.toFixed(2)} ms`;

    accumulatedTime = 0;
    framesThisInterval = 0;
  }
}

type SceneData = [Camera, Material[], MeshInstance[]];

function createCornellBox(): SceneData {
  const camAspect = ui.canvas.width / ui.canvas.height;
  const camera = new Camera([0.0, 0.6, 1.75], camAspect);

  const materials: Material[] = [
    { albedo: [0.9, 0.9, 0.9], roughness: 1.0, metalness: 0.0, emissionStrength: 0 }, // white wall
    { albedo: [0.9, 0.0, 0.0], roughness: 1.0, metalness: 0.0, emissionStrength: 0 }, // red wall
    { albedo: [0.0, 0.9, 0.0], roughness: 1.0, metalness: 0.0, emissionStrength: 0 }, // green wall
    { albedo: hexToSRGB(ui.albedoPicker.value), roughness: 1.0, metalness: 0.0, emissionStrength: 0 },
    { albedo: [1.0, 1.0, 1.0], roughness: 0.0, metalness: 0.0, emissionStrength: 50 }, // area light
  ];

  const instances: MeshInstance[] = [
    { mesh: createQuad([-0.5, 0.0, -0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), materialIndex: 0 }, // floor
    { mesh: createQuad([-0.5, 1.0, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), materialIndex: 0 }, // ceiling
    { mesh: createQuad([-0.5, 0.0, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), materialIndex: 0 }, // back wall
    { mesh: createQuad([-0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), materialIndex: 1 }, // left wall
    { mesh: createQuad([0.5, 0.0, -0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]), materialIndex: 2 },  // right wall
    { mesh: createSphere([-0.3, 0.15, 0.3], 0.15, 32, 32), materialIndex: 3 },
    { mesh: createBox([-0.15, 0.0, -0.35], 0.3, 0.575, 0.3, Math.PI / 3), materialIndex: 3 },
    { mesh: createBox([0.1, 0.0, -0.05], 0.3, 0.3, 0.3, Math.PI / 9), materialIndex: 3 },
    { mesh: createQuad([-0.1, 0.99, -0.1], [0.2, 0.0, 0.0], [0.0, 0.0, 0.2]), materialIndex: 4 },
  ];

  return [camera, materials, instances];
}

function createLightingShowcase(): SceneData {
  const corridorWidth = 3.2;
  const corridorHeight = 2.8;
  const corridorLength = 18.0;

  const halfW = corridorWidth * 0.5;
  const camAspect = ui.canvas.width / ui.canvas.height;
  const camera = new Camera([0.0, 1.15, 16.0], camAspect);

  const materials: Material[] = [
    { albedo: [0.94, 0.95, 0.98], roughness: 0.7, metalness: 0.0, emissionStrength: 0.0 }, // white shell
    { albedo: [0.17, 0.18, 0.21], roughness: 0.5, metalness: 0.0, emissionStrength: 0.0 }, // dark structural parts
    { albedo: [0.34, 0.35, 0.39], roughness: 0.75, metalness: 0.0, emissionStrength: 0.0 }, // neutral trim
    {
      albedo: hexToSRGB(ui.albedoPicker.value),
      roughness: parseFloat(ui.roughnessSlider.value),
      metalness: parseFloat(ui.metalnessSlider.value),
      emissionStrength: 0.0
    },
    { albedo: [0.34, 0.35, 0.39], roughness: 0.15, metalness: 0.0, emissionStrength: 0.0 }, // glossy accents
    { albedo: [0.99, 0.99, 0.99], roughness: 0.15, metalness: 0.0, emissionStrength: 0.0 }, // glossy floor
    { albedo: [0.05, 0.05, 0.06], roughness: 0.37, metalness: 0.0, emissionStrength: 0.0 }, // black trim
  ];

  const instances: MeshInstance[] = [];

  const pushMaterial = (
    albedo: Vec3,
    roughness: number,
    metalness: number,
    emissionStrength: number,
  ): number => {
    materials.push({
      albedo: [albedo[0], albedo[1], albedo[2]],
      roughness,
      metalness,
      emissionStrength,
    });
    return materials.length - 1;
  };

  const addQuad = (origin: Vec3, edge0: Vec3, edge1: Vec3, materialIndex: number) => {
    instances.push({ mesh: createQuad(origin, edge0, edge1), materialIndex });
  };

  const addBox = (
    origin: Vec3,
    width: number,
    height: number,
    length: number,
    angle: number,
    materialIndex: number
  ) => {
    instances.push({ mesh: createBox(origin, width, height, length, angle), materialIndex });
  };

  const addCylinder = (
    origin: Vec3,
    radius: number,
    height: number,
    radialSegments: number,
    materialIndex: number
  ) => {
    instances.push({
      mesh: createCylinder(origin, radius, height, radialSegments),
      materialIndex,
    });
  };

  const addSphere = (
    center: Vec3,
    radius: number,
    latitudeRes: number,
    longitudeRes: number,
    materialIndex: number
  ) => {
    instances.push({ mesh: createSphere(center, radius, latitudeRes, longitudeRes), materialIndex });
  };

  const hsvToRgb = (h: number, s: number, v: number): Vec3 => {
    const hh = ((h % 1.0) + 1.0) % 1.0;
    const i = Math.floor(hh * 6.0);
    const f = hh * 6.0 - i;
    const p = v * (1.0 - s);
    const q = v * (1.0 - f * s);
    const t = v * (1.0 - (1.0 - f) * s);

    switch (i % 6) {
      case 0: return [v, t, p];
      case 1: return [q, v, p];
      case 2: return [p, v, t];
      case 3: return [p, q, v];
      case 4: return [t, p, v];
      default: return [v, p, q];
    }
  };

  const addEmissiveCylinderWithPreview = (
  origin: Vec3,
  radius: number,
  height: number,
  radialSegments: number,
  color: Vec3,
  emissionStrength: number,
) => {
  const emissiveMat = pushMaterial(color, 0.05, 0.0, emissionStrength);
  instances.push({ mesh: createCylinder(origin, radius, height, radialSegments), materialIndex: emissiveMat });
};

  const addEmissiveCeilingPanelWithPreview = (
    x: number,
    zCenter: number,
    width: number,
    length: number,
    yTop: number,
    color: Vec3,
    emissionStrength: number
  ) => {
    const emissiveMat = pushMaterial(color, 0.05, 0.0, emissionStrength);
    addQuad([x, yTop, zCenter - 0.5 * length], [width, 0, 0], [0, 0, length], emissiveMat);
  };

  const addLightTotem = (cx: number, z: number, color: Vec3) => {
    const baseRadius = 0.22;
    const ringRadius = 0.16;
    const postRadius = 0.02;
    const postOffset = 0.11;

    const baseH = 0.10;
    const postY = baseH;
    const postH = 0.92;

    const lowerRingY = baseH;
    const midRingY = 0.58;
    const topRingY = 1.0;

    const coreY = baseH;
    const coreH = 0.90;
    const coreW = 0.08;

    // pedestal + rings
    addCylinder([cx, 0.0, z], baseRadius, baseH, 32, 1);
    addCylinder([cx, lowerRingY, z], ringRadius, 0.03, 32, 2);
    addCylinder([cx, midRingY, z], ringRadius, 0.03, 32, 2);
    addCylinder([cx, topRingY, z], baseRadius * 0.88, 0.05, 32, 1);

    // four slim support posts
    addCylinder([cx - postOffset, postY, z - postOffset], postRadius, postH, 20, 6);
    addCylinder([cx + postOffset, postY, z - postOffset], postRadius, postH, 20, 6);
    addCylinder([cx - postOffset, postY, z + postOffset], postRadius, postH, 20, 6);
    addCylinder([cx + postOffset, postY, z + postOffset], postRadius, postH, 20, 6);

    addEmissiveCylinderWithPreview(
      [cx, coreY, z], 0.75 * coreW, coreH, 20, color, 15
    );
  };

  // corridor walls
  addQuad([-halfW, 0.0, 0.0], [0.0, 0.0, corridorLength], [corridorWidth, 0.0, 0.0], 5); // floor
  addQuad([-halfW, corridorHeight, 0.0], [corridorWidth, 0.0, 0.0], [0.0, 0.0, corridorLength], 0); // ceiling
  addQuad([-halfW, 0.0, 0.0], [0.0, corridorHeight, 0.0], [0.0, 0.0, corridorLength], 0); // left wall
  addQuad([halfW, 0.0, 0.0], [0.0, 0.0, corridorLength], [0.0, corridorHeight, 0.0], 0); // right wall
  addQuad([-halfW, 0.0, 0.0], [corridorWidth, 0.0, 0.0], [0.0, corridorHeight, 0.0], 0); // front wall
  addQuad([-halfW, 0.0, corridorLength], [0.0, corridorHeight, 0.0], [corridorWidth, 0.0, 0.0], 0); // back wall

  // floor runway / trims
  addBox([-0.40, 0.0, 0.55], 0.80, 0.05, corridorLength - 1.10, 0.0, 2);
  addBox([-0.62, 0.0, 0.35], 0.05, 0.03, corridorLength - 0.70, 0.0, 2);
  addBox([0.57, 0.0, 0.35], 0.05, 0.03, corridorLength - 0.70, 0.0, 2);
  addBox([-0.62, 0.0, 0.35], 1.19, 0.03, 0.05, 0.0, 2);
  addBox([-0.62, 0.0, corridorLength - 0.35], 1.24, 0.03, 0.05, 0.0, 2);

  // ceiling and side trims
  addBox([-0.16, corridorHeight - 0.18, 0.35], 0.32, 0.12, corridorLength - 0.70, 0.0, 1);
  addBox([-halfW + 0.05, 0.48, 0.35], 0.06, 0.08, corridorLength - 0.70, 0.0, 6);
  addBox([halfW - 0.11, 0.48, 0.35], 0.06, 0.08, corridorLength - 0.70, 0.0, 6);
  addBox([-halfW + 0.05, 2.06, 0.35], 0.06, 0.08, corridorLength - 0.70, 0.0, 6);
  addBox([halfW - 0.11, 2.06, 0.35], 0.06, 0.08, corridorLength - 0.70, 0.0, 6);

  // front and back frames
  addBox([-1.04, 0.0, 0.04], 0.20, 2.26, 0.16, 0.0, 6);
  addBox([0.84, 0.0, 0.04], 0.20, 2.26, 0.16, 0.0, 6);
  addBox([-1.04, 2.06, 0.04], 2.08, 0.20, 0.16, 0.0, 6);

  addBox([-1.04, 0.0, corridorLength - 0.20], 0.20, 2.26, 0.16, 0.0, 6);
  addBox([0.84, 0.0, corridorLength - 0.20], 0.20, 2.26, 0.16, 0.0, 6);
  addBox([-1.04, 2.06, corridorLength - 0.20], 2.08, 0.20, 0.16, 0.0, 6);

  // ceiling emissive panels
  const ceilingLightCount = 8;
  const ceilingLightStart = 1.10;
  const ceilingLightStep = (corridorLength - 2.20) / (ceilingLightCount - 1);
  const ceilingLightW = 0.48;
  const ceilingLightL = 0.42;
  const ceilingLightColor: Vec3 = [0.97, 0.98, 1.0];

  for (let i = 0; i < ceilingLightCount; i++) {
    const z = ceilingLightStart + i * ceilingLightStep;

    addEmissiveCeilingPanelWithPreview(
      -0.92, z, ceilingLightW, ceilingLightL, corridorHeight - 0.01, ceilingLightColor, 5
    );

    addEmissiveCeilingPanelWithPreview(
      0.44, z, ceilingLightW, ceilingLightL, corridorHeight - 0.01, ceilingLightColor, 5
    );
  }

  // light totems, evenly spaced through the corridor
  const pillarCount = 8;
  const pillarStart = 2.00;
  const pillarStep = 2.00;

  for (let i = 0; i < pillarCount; i++) {
    const t = i === 1 ? 0.0 : i / (pillarCount - 1);
    const z = pillarStart + i * pillarStep;
    const x = (i % 2 === 0) ? -1.1 : 1.1;
    const color = hsvToRgb(0.00 + 0.78 * t, 0.78, 1.0);

    addLightTotem(x, z, color);
  }

  // showcase objects on the center line
  const showcaseZ = [4.1, 9.0, 13.9];
  const showcaseMaterials = [3, 0, 3];

  for (let i = 0; i < showcaseZ.length; i++) {
    const z = showcaseZ[i];

    addCylinder([0.0, 0.0, z], 0.18, 0.12, 20, 1);
    addSphere([0.0, 0.40, z], 0.22, 24, 24, showcaseMaterials[i]);
  }

  // end object
  addCylinder([0.0, 0.0, corridorLength - 1.05], 0.20, 0.14, 24, 1);
  addSphere([0.0, 0.46, corridorLength - 1.05], 0.25, 16, 16, 3);

  return [camera, materials, instances];
}

function loadScene(pipelineApp: GPUAppPipeline, sceneId: string) {
  const [ camera, materials, instances ] = sceneId === "cornell" 
    ? createCornellBox() 
    : createLightingShowcase();

  state.scene = new Scene(camera, instances, materials);

  state.scene.accumulationEnabled = ui.accumulationCheck.checked;
  state.scene.restirEnabled = ui.restirCheck.checked;
  state.scene.toneMappingEnabled = ui.toneMappingCheck.checked;
  state.scene.viewBvh = ui.viewBvhCheck.checked;
  state.scene.maxRayDepth = parseInt(ui.maxDepthSlider.value);
  state.scene.stratifiedGridSize = parseInt(ui.stratifiedGridSlider.value);
  
  if (state.scene.materials.length > 3) {
    state.scene.materials[3].albedo = hexToSRGB(ui.albedoPicker.value);
    state.scene.materials[3].roughness = parseFloat(ui.roughnessSlider.value);
    state.scene.materials[3].metalness = parseFloat(ui.metalnessSlider.value);
  }

  state.scene.createBuffers(pipelineApp);
  state.app = buildSceneBindGroups(pipelineApp, state.scene);
}

async function main() {
  const baseApp = await initWebGPU(ui.canvas);
  const pipelineApp = initRenderPipeline(baseApp);

  if (!baseApp.supportsReSTIR) {
    ui.canvas.style.display = "none";
    ui.overlayShell.style.display = "none";
    ui.errorScreen.style.display = "flex";
    return;
  }

  loadScene(pipelineApp, ui.sceneSelect.value);
  initEvents();

  ui.sceneSelect.addEventListener("change", () => {
    loadScene(pipelineApp, ui.sceneSelect.value);
  });

  function frame() {
    updateStats();

    handleCameraMovement();
    state.scene.animate();
    state.scene.camera.updateCamera();
    state.scene.updateGPU(state.app);

    const raytracingEnabled = ui.raytracingCheck.checked;
    render(state.app, state.scene, raytracingEnabled);
    requestAnimationFrame(frame);
  }
  
  requestAnimationFrame(frame);
}

main();

import { vec3, type Vec3 } from "wgpu-matrix";

import { Camera } from "./camera";
import { initWebGPU, initRenderPipeline, buildSceneBindGroups, render, type GPUAppPipeline } from "./renderer";
import { Scene } from "./scene";
import { type MeshInstance, createBox, createQuad } from "./mesh";
import type { Material, LightSource } from "./types";

const ui = {
  canvas: document.querySelector("canvas") as HTMLCanvasElement,
  raytracingCheck: document.querySelector("#raytracingCheckbox") as HTMLInputElement,
  albedoPicker: document.querySelector("#diffuseAlbedo") as HTMLInputElement,
  roughnessSlider: document.querySelector("#roughness") as HTMLInputElement,
  metalnessSlider: document.querySelector("#metalness") as HTMLInputElement,
  toneMappingCheck: document.querySelector("#toneMappingCheckbox") as HTMLInputElement,
  accumulationCheck: document.querySelector("#accumulationCheckbox") as HTMLInputElement,
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
  return vec3.create(
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  );
}

function initEvents(app: GPUAppPipeline, scene: Scene) {
  ui.canvas.addEventListener("mousedown", e => {
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;
    if (e.button === 0) scene.camera.dragging = true;
  });

  window.addEventListener("mouseup", () => {
    scene.camera.dragging = false;
  });

  ui.canvas.addEventListener("mousemove", e => {
    if (!scene.camera.dragging) return;

    const dx = e.clientX - scene.camera.lastX;
    const dy = e.clientY - scene.camera.lastY;
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;

    scene.camera.processMouseMovement(dx, dy);
    scene.frameCount = 0.0;
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
    scene.frameCount = 0.0;
    scene.materials[3].albedo = hexToSRGB(ui.albedoPicker.value);
    scene.updateMaterials(app);
  });

  ui.roughnessSlider.addEventListener("input", () => {
    const val = parseFloat(ui.roughnessSlider.value);

    scene.frameCount = 0.0;
    scene.materials[3].roughness = val;
    scene.updateMaterials(app);
  });

  ui.metalnessSlider.addEventListener("input", () => {
    const val = parseFloat(ui.metalnessSlider.value);

    scene.frameCount = 0.0;
    scene.materials[3].metalness = val;
    scene.updateMaterials(app);
  });

  ui.raytracingCheck.addEventListener("input", () => { scene.frameCount = 0.0; });

  ui.toneMappingCheck.addEventListener("input", () => {
    scene.toneMappingEnabled = ui.toneMappingCheck.checked;
  });

  ui.accumulationCheck.addEventListener("input", () => {
    scene.accumulationEnabled = ui.accumulationCheck.checked;
    scene.frameCount = 0.0;
  });

  ui.maxDepthSlider.addEventListener("input", () => {
    const val = parseInt(ui.maxDepthSlider.value);

    scene.frameCount = 0.0;
    scene.maxRayDepth = val;
    ui.depthText.innerText = `${val}`;
  });

  ui.stratifiedGridSlider.addEventListener("input", () => {
    const val = parseInt(ui.stratifiedGridSlider.value);

    scene.frameCount = 0.0;
    scene.stratifiedGridSize = val;
    ui.sppText.innerText = `${val * val}`;
  });
}

function handleCameraMovement(scene: Scene) {
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
    scene.camera.processKeyboard(moveForward, moveRight, moveUp);
    scene.frameCount = 0.0;
  }
}

let lastTime = performance.now();

function updateStats() {
  const now = performance.now();
  const dt = now - lastTime;
  lastTime = now;

  ui.fpsText.textContent = (1000 / dt).toFixed(1);
  ui.frameTimeText.textContent = `${dt.toFixed(2)} ms`;
}

type SceneData = [Camera, Material[], LightSource[], MeshInstance[]];

function createCornellBox(): SceneData {
  const camAspect = ui.canvas.width / ui.canvas.height;
  const camera = new Camera(vec3.create(0.0, 0.6, 1.75), camAspect);

  const materials: Material[] = [
    { albedo: vec3.create(0.9, 0.9, 0.9), roughness: 1.0, metalness: 0.0, materialType: 0 }, // white wall
    { albedo: vec3.create(0.9, 0.0, 0.0), roughness: 1.0, metalness: 0.0, materialType: 0 }, // red wall
    { albedo: vec3.create(0.0, 0.9, 0.0), roughness: 1.0, metalness: 0.0, materialType: 0 }, // green wall
    { albedo: hexToSRGB(ui.albedoPicker.value), roughness: 1.0, metalness: 0.0, materialType: 0 } // main object material
  ];

  const lights: LightSource[] = [
    { type: "point", position: vec3.create(0.0, 0.99, -0.1), intensity: 1.5, color: vec3.create(1.0, 1.0, 1.0), rayTracedShadows: 1 },
    {
      type: "area",
      position: vec3.create(-0.1, 0.99, -0.1),
      intensity: 40,
      u: vec3.create(0.2, 0.0, 0.0),
      v: vec3.create(0.0, 0.0, 0.2),
      color: vec3.create(1.0, 1.0, 1.0),
      rayTracedShadows: 1
    },
  ];

  const instances: MeshInstance[] = [
    { mesh: createQuad([-0.5, 0.0, -0.5], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), materialIndex: 0 }, // floor
    { mesh: createQuad([-0.5, 1.0, -0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]), materialIndex: 0 }, // ceiling
    { mesh: createQuad([-0.5, 0.0, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), materialIndex: 0 }, // back wall
    { mesh: createQuad([-0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]), materialIndex: 1 }, // left wall
    { mesh: createQuad([0.5, 0.0, -0.5], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]), materialIndex: 2 },  // right wall
    { mesh: createBox([-0.15, 0.0, -0.35], 0.3, 0.575, 0.3, Math.PI / 3), materialIndex: 3 },
    { mesh: createBox([0.1, 0.0, -0.05], 0.3, 0.3, 0.3, Math.PI / 9), materialIndex: 3 },
  ];

  return [camera, materials, lights, instances];
}

async function main() {
  const baseApp = await initWebGPU(ui.canvas);
  const pipelineApp = initRenderPipeline(baseApp);

  const [ camera, materials, lights, instances ] = createCornellBox();

  const scene = new Scene(camera, instances, materials, lights);
  scene.computeBVH();
  scene.createBuffers(pipelineApp);
  initEvents(pipelineApp, scene);

  const app = buildSceneBindGroups(pipelineApp, scene);

  function frame() {
    updateStats();

    handleCameraMovement(scene);
    scene.animate();
    scene.camera.updateCamera();
    scene.updateGPU(app);

    const raytracingEnabled = ui.raytracingCheck.checked;
    render(app, scene, raytracingEnabled);
    requestAnimationFrame(frame);

  }
  
  requestAnimationFrame(frame);
}

main();

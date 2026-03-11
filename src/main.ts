import { vec3, type Vec3 } from "wgpu-matrix";

import { Camera } from "./camera";
import { type GPUApp, initWebGPU, initRenderPipeline, createSceneBindGroup, render } from "./renderer";
import { Scene } from "./scene";
import { type MeshInstance, createBox, createQuad, createSphere } from "./mesh";
import type { Material, LightSource } from "./types";

const ui = {
  canvas: document.querySelector("canvas") as HTMLCanvasElement,
  raytracingCheck: document.querySelector("#raytracingCheckbox") as HTMLInputElement,
  albedoPicker: document.querySelector("#diffuseAlbedo") as HTMLInputElement,
  roughnessSlider: document.querySelector("#roughness") as HTMLInputElement,
  metalnessSlider: document.querySelector("#metalness") as HTMLInputElement,
};

function hexToSRGB(hex: string): Vec3 {
  return vec3.create(
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  );
}

function initEvents(app: GPUApp, scene: Scene) {
  ui.canvas.addEventListener("mousedown", e => {
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;

    if (e.button === 0) scene.camera.dragging = true;
    if (e.button === 1 || e.button === 2) scene.camera.panning = true;
  });

  window.addEventListener("mouseup", () => {
    scene.camera.dragging = false;
    scene.camera.panning = false;
  });

  ui.canvas.addEventListener("mousemove", e => {
    const dx = e.clientX - scene.camera.lastX;
    const dy = e.clientY - scene.camera.lastY;
    scene.camera.lastX = e.clientX;
    scene.camera.lastY = e.clientY;

    if (scene.camera.dragging) {
      scene.camera.yaw -= dx * scene.camera.rotateSpeed;
      scene.camera.pitch += dy * scene.camera.rotateSpeed;

      const maxPitch = Math.PI / 2 - 0.01;
      scene.camera.pitch = Math.max(-maxPitch, Math.min(maxPitch, scene.camera.pitch));
    }

    if (scene.camera.panning) {
      scene.camera.pan(dx, -dy);
    }
  });

  ui.canvas.addEventListener("wheel", e => {
    e.preventDefault();
    scene.camera.radius *= 1 + e.deltaY * scene.camera.zoomSpeed;
    scene.camera.radius = Math.max(scene.camera.minRadius, Math.min(scene.camera.maxRadius, scene.camera.radius));
  }, { passive: false });

  ui.canvas.addEventListener("contextmenu", e => e.preventDefault());

  ui.albedoPicker.addEventListener("input", () => {
    scene.materials[3].albedo = hexToSRGB(ui.albedoPicker.value);
    console.log(scene.materials[3].albedo)
    scene.updateMaterials(app);
  });

  ui.roughnessSlider.addEventListener("input", () => {
    const val = parseFloat(ui.roughnessSlider.value);

    scene.materials[3].roughness = val;
    scene.updateMaterials(app);
  });

  ui.metalnessSlider.addEventListener("input", () => {
    const val = parseFloat(ui.metalnessSlider.value);

    scene.materials[3].metalness = val;
    scene.updateMaterials(app);
  });
}

async function main() {
  const baseApp = await initWebGPU(ui.canvas);
  const app = initRenderPipeline(baseApp);

  const s = 0.5;
  const camAspect = ui.canvas.width / ui.canvas.height;
  const camera = new Camera(vec3.create(0.0, s, 0.0), camAspect, 4.0 * s);

  const materials: Material[] = [
    { albedo: vec3.create(0.9, 0.9, 0.9), roughness: 1.0, metalness: 0.0, materialType: 0 }, // white wall
    { albedo: vec3.create(0.9, 0.0, 0.0), roughness: 1.0, metalness: 0.0, materialType: 0 }, // red wall
    { albedo: vec3.create(0.0, 0.9, 0.0), roughness: 1.0, metalness: 0.0, materialType: 0 }, // green wall
    { albedo: hexToSRGB(ui.albedoPicker.value), roughness: 1.0, metalness: 0.0, materialType: 0 } // main object material
  ];

  const lights: LightSource[] = [
    // { position: vec3.create(-0.75*s, 1.5*s, 1.5*s), intensity: 1.5, color: vec3.create(1.0, 0.92, 0.56), angle, spot, rayTracedShadows: 1 },
    { type: "point", position: vec3.create(0.0, 1.9*s, -0.1), intensity: 1.5, color: vec3.create(1.0, 1.0, 1.0), rayTracedShadows: 1 },
    {
      type: "area",
      position: vec3.create(-0.1, 0.99, -0.1),
      intensity: 50,
      u: vec3.create(0.2, 0.0, 0.0),
      v: vec3.create(0.0, 0.0, 0.2),
      color: vec3.create(1.0, 1.0, 1.0),
      rayTracedShadows: 1
    },
  ];

  const instances: MeshInstance[] = [
    { mesh: createQuad([-s, 0.0, -s], [0.0, 0.0, 2.0*s], [2.0*s, 0.0, 0.0]), materialIndex: 0 }, // floor
    { mesh: createQuad([-s, 2.0*s, -s], [2.0*s, 0.0, 0.0], [0.0, 0.0, 2.0*s]), materialIndex: 0 }, // ceiling
    { mesh: createQuad([-s, 0.0, -s], [2.0*s, 0.0, 0.0], [0.0, 2.0*s, 0.0]), materialIndex: 0 }, // back wall
    { mesh: createQuad([-s, 0.0, -s], [0.0, 2.0*s, 0.0], [0.0, 0.0, 2.0*s]), materialIndex: 1 }, // left wall
    { mesh: createQuad([s, 0.0, -s], [0.0, 0.0, 2.0*s], [0.0, 2.0*s, 0.0]), materialIndex: 2 },  // right wall
    // { mesh: createSphere([0.0, s, 0.0], 0.2, 32, 32), materialIndex: 3 },
    { mesh: createBox([-0.15, 0.0, -0.35], s * 0.6, 1.15 * s, s * 0.6, Math.PI / 3), materialIndex: 3 },
    { mesh: createBox([0.1, 0.0, -0.05], s * 0.6, 0.6 * s, s * 0.6, Math.PI / 9), materialIndex: 3 },
  ];

  const scene = new Scene(camera, instances, materials, lights);
  scene.computeBVH();
  scene.createBuffers(app);

  const bindGroup = createSceneBindGroup(app, scene);

  initEvents(app, scene);

  function frame() {
    const raytracingEnabled = ui.raytracingCheck.checked;

    scene.animate();
    scene.camera.updateCamera();
    scene.updateGPU(app);

    render(app, scene, bindGroup, raytracingEnabled);
    requestAnimationFrame(frame);
  }
  
  requestAnimationFrame(frame);
}

main();

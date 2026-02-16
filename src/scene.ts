import { vec3 } from "wgpu-matrix";

import { Camera } from "./camera";
import { mergeMeshes, type MeshInstance, type MergedGeometry } from "./mesh";
import type { GPUAppBase } from "./renderer";
import type { LightSource, Material } from "./types";
import { createGPUBuffer } from "./utils";

export class Scene {
  camera: Camera;
  instances: MeshInstance[];
  materials: Material[];
  lights: LightSource[];
  
  mergedGeometry?: MergedGeometry;
  time: number = 0;

  // GPU buffers
  buffersInitialized: boolean = false;
  uniformBuffer?: GPUBuffer;
  posBuffer?: GPUBuffer;
  normBuffer?: GPUBuffer;
  triBuffer?: GPUBuffer;
  instanceBuffer?: GPUBuffer;
  matBuffer?: GPUBuffer;
  lightBuffer?: GPUBuffer;

  lightDataArray?: ArrayBuffer;

  constructor(camera: Camera, instances: MeshInstance[], materials: Material[], lights: LightSource[]) {
    this.camera = camera;
    this.instances = instances;
    this.materials = materials;
    this.lights = lights;
  }

  createBuffers(app: GPUAppBase) {
    this.mergedGeometry = mergeMeshes(this.instances);
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.posBuffer = createGPUBuffer(app.device, this.mergedGeometry.positions, storageUsage);
    this.normBuffer = createGPUBuffer(app.device, this.mergedGeometry.normals, storageUsage);
    this.triBuffer = createGPUBuffer(app.device, this.mergedGeometry.indices, storageUsage);
    this.instanceBuffer = createGPUBuffer(app.device, this.mergedGeometry.instances, storageUsage);

    this.uniformBuffer = app.device.createBuffer({
      size: 88 * 4, // 88 floats in scene struct in shader
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 8 floats per material
    const matData = new Float32Array(this.materials.length * 8);
    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      
      matData.set(m.albedo, offset);
      matData[offset + 3] = m.roughness;
      matData[offset + 4] = m.metalness;
      // 5, 6, 7 are padding
    }
    this.matBuffer = createGPUBuffer(app.device, matData, storageUsage);

    // 12 floats/u32s per light
    this.lightDataArray = new ArrayBuffer(this.lights.length * 48);
    this.packLightData();
    
    this.lightBuffer = createGPUBuffer(app.device, new Uint8Array(this.lightDataArray), storageUsage);

    this.buffersInitialized = true;
  }

  private packLightData() {
    if (!this.lightDataArray) return;
    
    const lightF32 = new Float32Array(this.lightDataArray);
    const lightU32 = new Uint32Array(this.lightDataArray);

    for (let i = 0; i < this.lights.length; i++) {
      const l = this.lights[i];
      const offset = i * 12;

      lightF32.set(l.position, offset); // 0, 1, 2
      lightF32[offset + 3] = l.intensity; // 3
      lightF32.set(l.color, offset + 4); // 4, 5, 6
      lightF32[offset + 7] = l.angle; // 7
      lightF32.set(l.spot, offset + 8); // 8, 9, 10
      lightU32[offset + 11] = l.rayTracedShadows; // 11
    }
  }

  animate() {
    this.time += 1.0;
    const angle = this.time / 40.0;
    
    this.lights[this.lights.length - 1].position = vec3.create(0.5 * Math.cos (angle), 0.9, 0.5 * Math.sin (angle));
  }

  updateMaterials(queue: GPUQueue) {
    if (!this.buffersInitialized) return;

    const matData = new Float32Array(this.materials.length * 8);

    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      matData.set(m.albedo, offset);
      matData[offset + 3] = m.roughness;
      matData[offset + 4] = m.metalness;
    }

    queue.writeBuffer(this.matBuffer!, 0, matData);
  }

  updateGPU(queue: GPUQueue) {
    if (!this.buffersInitialized) return;

    const sceneData = new Float32Array(88);

    // matrices
    sceneData.set(this.camera.modelMat, 0);
    sceneData.set(this.camera.viewMat, 16);
    sceneData.set(this.camera.invViewMat, 32);
    sceneData.set(this.camera.transInvViewMat, 48);
    sceneData.set(this.camera.projMat, 64);
    
    // camera
    sceneData[80] = this.camera.fov;
    sceneData[81] = this.camera.aspect;
    
    // scene
    sceneData[84] = this.instances.length;
    sceneData[85] = this.lights.length;

    queue.writeBuffer(this.uniformBuffer!, 0, sceneData);

    // update lights
    // repack and upload it again
    this.packLightData();
    queue.writeBuffer(this.lightBuffer!, 0, this.lightDataArray!);
  }
}

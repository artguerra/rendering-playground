import type { GPUAppPipeline, GPUAppBase } from "./renderer";
import type { LightSource, Material, PointLight } from "./types";
import { type MeshInstance, type MergedGeometry, mergeMeshes } from "./mesh";
import { createGPUBuffer } from "./utils";
import { Camera } from "./camera";
import { BVHTree } from "./bvh";

interface EmissiveTriangle {
  triIdx: number;
  meshIdx: number;
}

export class Scene {
  camera: Camera;
  instances: MeshInstance[];
  materials: Material[];

  emissiveTriangles: EmissiveTriangle[] = [];
  pointLights: PointLight[] = [];

  bvh?: BVHTree;
  viewBvh: boolean = false;
  
  mergedGeometry: MergedGeometry;
  toneMappingEnabled: boolean = false;
  restirEnabled: boolean = true;
  useRISOnBounces: boolean = true;
  accumulationEnabled: boolean = true;
  stratifiedGridSize: number = 1;
  maxRayDepth: number = 3;
  frameCount: number = 0.0;

  // GPU buffers
  buffersInitialized: boolean = false;
  uniformBuffer?: GPUBuffer;
  posBuffer?: GPUBuffer;
  normBuffer?: GPUBuffer;
  triBuffer?: GPUBuffer;
  instanceBuffer?: GPUBuffer;
  matBuffer?: GPUBuffer;
  pointLightBuffer?: GPUBuffer;
  emissiveTriBuffer?: GPUBuffer;

  bvhDataArray?: ArrayBuffer;
  bvhBuffer?: GPUBuffer;

  pointLightDataArray?: ArrayBuffer;
  emissiveTriDataArray?: ArrayBuffer;

  constructor(camera: Camera, instances: MeshInstance[], materials: Material[], lights: LightSource[]) {
    this.camera = camera;
    this.instances = instances;
    this.materials = materials;

    for (const light of lights) {
      if (light.type === "point") this.pointLights.push(light);
    }

    this.mergedGeometry = mergeMeshes(this.instances);
    this.computeBVH();
    this.extractEmissiveTriangles();
  }

  private extractEmissiveTriangles() {
    this.emissiveTriangles = [];

    for (let i = 0; i < this.instances.length; i++) {
      const matIdx = this.instances[i].materialIndex;
      const mat = this.materials[matIdx];

      if (mat.emissionStrength && mat.emissionStrength > 0.0) {
        const triOffset = this.mergedGeometry.instances[i * 8 + 1];
        const triCount = this.mergedGeometry.instances[i * 8 + 2];

        for (let t = 0; t < triCount; t++) {
          this.emissiveTriangles.push({
            triIdx: triOffset + t,
            meshIdx: i,
          });
        }
      }
    }
  }

  computeBVH() {
    const allNodes: any[] = [];

    for (let i = 0; i < this.instances.length; i++) {
      const vertexOffset = this.mergedGeometry.instances[i * 8 + 0];
      const triOffset = this.mergedGeometry.instances[i * 8 + 1];
      const triCount = this.mergedGeometry.instances[i * 8 + 2];

      // build BVH for this specific mesh
      const bvh = new BVHTree(this.mergedGeometry, triOffset, triCount, vertexOffset);
      bvh.buildRecursive(bvh.rootIdx);

      // sort the indices buffer for this mesh
      bvh.reorderIndices(this.mergedGeometry.indices);

      const bvhRoot = allNodes.length;
      const flatNodes = bvh.flatten(bvhRoot);

      allNodes.push(...flatNodes);

      // set bvh root node for mesh
      this.mergedGeometry.instances[i * 8 + 4] = bvhRoot;
      this.mergedGeometry.instances[i * 8 + 5] = flatNodes.length;
    }

    // 8 bytes per node
    this.bvhDataArray = new ArrayBuffer(allNodes.length * 8 * 4);
    const f32View = new Float32Array(this.bvhDataArray);
    const u32View = new Uint32Array(this.bvhDataArray);

    for (let i = 0; i < allNodes.length; ++i) {
      const node = allNodes[i];
      const idx = i * 8;

      f32View.set(node.bounds.minCorner, idx);
      u32View[idx + 3] = node.numPrimitives !== null ? node.numPrimitives : 0;
      f32View.set(node.bounds.maxCorner, idx + 4);

      // for leaves, skipLink is the absolute triangle index. 
      // for interiors, skipLink is relative, so we add the absolute bvhRoot to make it a global pointer!
      u32View[idx + 7] = node.skipLink;
    }
  }

  createBuffers(app: GPUAppBase) {
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.posBuffer = createGPUBuffer(app.device, this.mergedGeometry.positions, storageUsage);
    this.normBuffer = createGPUBuffer(app.device, this.mergedGeometry.normals, storageUsage);
    this.triBuffer = createGPUBuffer(app.device, this.mergedGeometry.indices, storageUsage);
    this.instanceBuffer = createGPUBuffer(app.device, this.mergedGeometry.instances, storageUsage);

    if (this.bvhBuffer?.size === 0) throw new Error("BVH was not initialized.");
    this.bvhBuffer = createGPUBuffer(app.device, this.bvhDataArray!, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    this.uniformBuffer = app.device.createBuffer({
      size: 100 * 4, // 100 floats in scene struct in shader
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 8 bytes per material
    const matData = new ArrayBuffer(this.materials.length * 8 * 4);
    const matDataF32 = new Float32Array(matData);
    const matDataU32 = new Uint32Array(matData);
    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      
      matDataF32.set(m.albedo, offset);
      matDataF32[offset + 3] = m.roughness;
      matDataF32[offset + 4] = m.metalness;
      matDataU32[offset + 5] = m.materialType;
      matDataF32[offset + 6] = m.emissionStrength;
      // 7 is padding
    }
    this.matBuffer = createGPUBuffer(app.device, matData, storageUsage);

    this.packLightData();
    this.pointLightBuffer = createGPUBuffer(app.device, new Uint8Array(this.pointLightDataArray!), storageUsage);
    this.emissiveTriBuffer = createGPUBuffer(app.device, new Uint8Array(this.emissiveTriDataArray!), storageUsage);

    this.buffersInitialized = true;
  }

  private packLightData() {
    // point lights (8 floats/u32s = 32 bytes each)
    const numPointLights = Math.max(1, this.pointLights.length); // at least allocate 1
    this.pointLightDataArray = new ArrayBuffer(numPointLights * 8 * 4);
    const pF32View = new Float32Array(this.pointLightDataArray);
    const pU32View = new Uint32Array(this.pointLightDataArray);

    for (let i = 0; i < this.pointLights.length; i++) {
      const l = this.pointLights[i];
      const offset = i * 8;
      pF32View.set(l.position, offset);
      pF32View[offset + 3] = l.intensity;
      pF32View.set(l.color, offset + 4);
      pU32View[offset + 7] = l.rayTracedShadows;
    }

    // emissive triangles (2 u32s = 8 bytes each)
    const numEmissive = Math.max(1, this.emissiveTriangles.length);
    this.emissiveTriDataArray = new ArrayBuffer(numEmissive * 2 * 4);
    const eU32View = new Uint32Array(this.emissiveTriDataArray);

    for (let i = 0; i < this.emissiveTriangles.length; i++) {
      const offset = i * 2;
      eU32View[offset] = this.emissiveTriangles[i].triIdx;
      eU32View[offset + 1] = this.emissiveTriangles[i].meshIdx;
    }
  }

  animate() {
    this.frameCount += 1.0;
  }

  updateMaterials(app: GPUAppPipeline) {
    if (!this.buffersInitialized) return;

    const matData = new ArrayBuffer(this.materials.length * 8 * 4);
    const matDataF32 = new Float32Array(matData);
    const matDataU32 = new Uint32Array(matData);

    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      matDataF32.set(m.albedo, offset);
      matDataF32[offset + 3] = m.roughness;
      matDataF32[offset + 4] = m.metalness;
      matDataU32[offset + 5] = m.materialType;
      matDataF32[offset + 6] = m.emissionStrength;
    }

    app.device.queue.writeBuffer(this.matBuffer!, 0, matData);
  }

  updateGPU(app: GPUAppPipeline) {
    if (!this.buffersInitialized) return;

    const sceneData = new ArrayBuffer(100 * 4);
    const f32View = new Float32Array(sceneData);
    const u32View = new Uint32Array(sceneData);

    // matrices
    f32View.set(this.camera.modelMat, 0);
    f32View.set(this.camera.viewMat, 16);
    f32View.set(this.camera.invViewMat, 32);
    f32View.set(this.camera.transInvModelMat, 48);
    f32View.set(this.camera.projMat, 64);
    
    // camera
    f32View[80] = this.camera.fov;
    f32View[81] = this.camera.aspect;
    
    // scene
    f32View[84] = app.canvas.width;
    f32View[85] = app.canvas.height;
    f32View[86] = this.instances.length;
    u32View[87] = this.pointLights.length;
    u32View[88] = this.emissiveTriangles.length;
    u32View[89] = new Uint32Array([Date.now()])[0]; // take only LSB
    u32View[90] = this.frameCount;
    u32View[91] = +this.toneMappingEnabled;
    u32View[92] = +this.accumulationEnabled;
    u32View[93] = this.maxRayDepth;
    u32View[94] = this.stratifiedGridSize;
    u32View[95] = +this.restirEnabled;

    f32View[96] = 0.0; // pad
    f32View[97] = 0.0; // pad
    f32View[98] = 0.0; // pad

    u32View[99] = +this.useRISOnBounces;

    app.device.queue.writeBuffer(this.uniformBuffer!, 0, sceneData);

    // update lights
    // repack and upload it again
    this.packLightData();
    app.device.queue.writeBuffer(this.pointLightBuffer!, 0, this.pointLightDataArray!);
    app.device.queue.writeBuffer(this.emissiveTriBuffer!, 0, this.emissiveTriDataArray!);
  }
}

import type { GPUAppPipeline, GPUAppBase } from "./renderer";
import type { Material } from "./types";
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

  bvh?: BVHTree;
  viewBvh: boolean = false;
  
  mergedGeometry: MergedGeometry;
  
  // restir options
  toneMappingEnabled: boolean = false;
  restirEnabled: boolean = true;
  useRISOnBounces: boolean = false;
  restirBiased: boolean = true;
  temporalReuseEnabled: boolean = true;
  spatialReuseEnabled: boolean = true;
  accumulationEnabled: boolean = true;

  stratifiedGridSize: number = 1;
  maxRayDepth: number = 3;
  frameCount: number = 0.0;
  absoluteFrameCount: number = 0.0;

  // GPU buffers
  buffersInitialized: boolean = false;
  uniformBuffer?: GPUBuffer;
  posBuffer?: GPUBuffer;
  normBuffer?: GPUBuffer;
  triBuffer?: GPUBuffer;
  instanceBuffer?: GPUBuffer;
  matBuffer?: GPUBuffer;
  emissiveTriBuffer?: GPUBuffer;

  bvhDataArray?: ArrayBuffer;
  bvhBuffer?: GPUBuffer;

  emissiveTriDataArray?: ArrayBuffer;

  constructor(camera: Camera, instances: MeshInstance[], materials: Material[]) {
    this.camera = camera;
    this.instances = instances;
    this.materials = materials;

    this.mergedGeometry = mergeMeshes(this.instances);
    this.computeBVH();
    this.extractEmissiveTriangles();
  }

  private extractEmissiveTriangles() {
    this.emissiveTriangles = [];
    const instanceU32 = new Uint32Array(this.mergedGeometry.instances);

    for (let i = 0; i < this.instances.length; i++) {
      const matIdx = this.instances[i].materialIndex;
      const mat = this.materials[matIdx];

      const instanceIdx = i * 12; // stride 12

      if (mat.emissionStrength && mat.emissionStrength > 0.0) {
        const triOffset = instanceU32[instanceIdx + 1];
        const triCount = instanceU32[instanceIdx + 2];

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
    const instanceU32 = new Uint32Array(this.mergedGeometry.instances);

    for (let i = 0; i < this.instances.length; i++) {
      const instanceIdx = i * 12; // stride 12
      const vertexOffset = instanceU32[instanceIdx + 0];
      const triOffset = instanceU32[instanceIdx + 1];
      const triCount = instanceU32[instanceIdx + 2];

      // build BVH for this specific mesh
      const bvh = new BVHTree(this.mergedGeometry, triOffset, triCount, vertexOffset);
      bvh.buildRecursive(bvh.rootIdx);

      // sort the indices buffer for this mesh
      bvh.reorderIndices(this.mergedGeometry.indices);

      const bvhRoot = allNodes.length;
      const flatNodes = bvh.flatten(bvhRoot);

      allNodes.push(...flatNodes);

      // set bvh root node for mesh
      instanceU32[instanceIdx + 7] = bvhRoot;
      instanceU32[instanceIdx + 8] = flatNodes.length;
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
      size: 116 * 4, // 116 floats in scene struct in shader
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 8 bytes per material
    const matData = new ArrayBuffer(this.materials.length * 8 * 4);
    const matDataF32 = new Float32Array(matData);
    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      
      matDataF32.set(m.albedo, offset);
      matDataF32[offset + 3] = m.roughness;
      matDataF32[offset + 4] = m.metalness;
      matDataF32[offset + 5] = m.emissionStrength;
      // 6, 7 are padding
    }
    this.matBuffer = createGPUBuffer(app.device, matData, storageUsage);

    // emissive triangles (2 u32s = 8 bytes each)
    // considering lights dont move, if so have to update on updateGPU
    const numEmissive = Math.max(1, this.emissiveTriangles.length);
    console.log(`number of emissive triangles: ${numEmissive}`);

    this.emissiveTriDataArray = new ArrayBuffer(numEmissive * 2 * 4);
    const eU32View = new Uint32Array(this.emissiveTriDataArray);

    for (let i = 0; i < this.emissiveTriangles.length; i++) {
      const offset = i * 2;
      eU32View[offset] = this.emissiveTriangles[i].triIdx;
      eU32View[offset + 1] = this.emissiveTriangles[i].meshIdx;
    }
    this.emissiveTriBuffer = createGPUBuffer(app.device, new Uint8Array(this.emissiveTriDataArray!), storageUsage);

    this.buffersInitialized = true;
  }

  animate() {
    this.frameCount += 1.0;
    this.absoluteFrameCount += 1.0;
  }

  updateMaterials(app: GPUAppPipeline) {
    if (!this.buffersInitialized) return;

    const matData = new ArrayBuffer(this.materials.length * 8 * 4);
    const matDataF32 = new Float32Array(matData);

    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      matDataF32.set(m.albedo, offset);
      matDataF32[offset + 3] = m.roughness;
      matDataF32[offset + 4] = m.metalness;
      matDataF32[offset + 5] = m.emissionStrength;
      // 6, 7 are padding
    }

    app.device.queue.writeBuffer(this.matBuffer!, 0, matData);
  }

  updateGPU(app: GPUAppPipeline) {
    if (!this.buffersInitialized) return;

    const sceneData = new ArrayBuffer(116 * 4);
    const f32View = new Float32Array(sceneData);
    const u32View = new Uint32Array(sceneData);

    // matrices
    f32View.set(this.camera.modelMat, 0);
    f32View.set(this.camera.viewMat, 16);
    f32View.set(this.camera.prevViewMat, 32);
    f32View.set(this.camera.invViewMat, 48);
    f32View.set(this.camera.transInvModelMat, 64);
    f32View.set(this.camera.projMat, 80);
    
    // camera
    f32View[96] = this.camera.fov;
    f32View[97] = this.camera.aspect;
    f32View[98] = 0.0; // padding
    f32View[99] = 0.0; // padding
    
    // scene
    f32View[100] = app.canvas.width;
    f32View[101] = app.canvas.height;
    f32View[102] = this.instances.length;
    u32View[103] = this.emissiveTriangles.length;
    u32View[104] = new Uint32Array([Date.now()])[0]; // take only LSB
    u32View[105] = this.frameCount;
    u32View[106] = this.absoluteFrameCount;
    u32View[107] = +this.toneMappingEnabled;
    u32View[108] = +this.accumulationEnabled;
    u32View[109] = this.maxRayDepth;
    u32View[110] = this.stratifiedGridSize;
    u32View[111] = +this.restirEnabled;
    u32View[112] = +this.useRISOnBounces;
    u32View[113] = +this.restirBiased;
    u32View[114] = 0; // padding
    u32View[115] = 0; // padding

    app.device.queue.writeBuffer(this.uniformBuffer!, 0, sceneData);
  }
}

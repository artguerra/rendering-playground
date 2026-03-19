import mainShaders from "@shaders/main.wgsl?raw";
import pathtracerComputeShaders from "@shaders/pathtracer_compute.wgsl?raw";
import displayPathtracingShaders from "@shaders/display_pathtracing.wgsl?raw";

import { Scene } from "./scene";

export interface GPUAppBase {
  device: GPUDevice;
  adapter: GPUAdapter;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  canvasFormat: GPUTextureFormat;
}

export interface GPUAppPipeline extends GPUAppBase {
  depthTexture: GPUTexture;
  accumulationRead: GPUTexture;
  accumulationWrite: GPUTexture;
  pathtraceOutput: GPUTexture;

  primarySurfaceBuffer: GPUBuffer;
  reservoirsBufferA: GPUBuffer;
  reservoirsBufferB: GPUBuffer;

  mainShaderModule: GPUShaderModule;
  pathtracerComputeShaderModule: GPUShaderModule;
  displayPathtracingShaderModule: GPUShaderModule;

  rasterPipeline: GPURenderPipeline;
  displayPathtracingPipeline: GPURenderPipeline;
  wireframePipeline: GPURenderPipeline;

  visibilityPipeline: GPUComputePipeline;
  initialRisPipeline: GPUComputePipeline;
  visibilityReusePipeline: GPUComputePipeline;
  temporalReusePipeline: GPUComputePipeline;
  spatialReusePipeline: GPUComputePipeline;
  shadePathtracePipeline: GPUComputePipeline;

  sceneBindGroupLayout: GPUBindGroupLayout;
  geometryBindGroupLayout: GPUBindGroupLayout;
  restirBindGroupLayout: GPUBindGroupLayout;
  displayBindGroupLayout: GPUBindGroupLayout;
}

export interface GPUApp extends GPUAppPipeline {
  sceneBindGroup: GPUBindGroup,
  geometryBindGroup: GPUBindGroup,
  restirBindGroupA: GPUBindGroup,
  restirBindGroupB: GPUBindGroup,
  displayBindGroupA: GPUBindGroup,
  displayBindGroupB: GPUBindGroup,
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GPUAppBase> {
  if (!navigator.gpu) throw new Error("WebGPU is not supported on this browser.");

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("No adapter available for WebGPU.");

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage
    },
  });
  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: canvasFormat,
    alphaMode: "opaque",
  });

  return { adapter, device, canvas, context, canvasFormat };
}

export function initRenderPipeline(app: GPUAppBase): GPUAppPipeline {
  const mainShaderModule = app.device.createShaderModule({
    label: "rasterization/wireframe shaders",
    code: mainShaders,
  });

  const pathtracerComputeShaderModule = app.device.createShaderModule({
    label: "pathtracer compute shaders",
    code: pathtracerComputeShaders,
  });

  const displayPathtracingShaderModule = app.device.createShaderModule({
    label: "display pathtracing shaders",
    code: displayPathtracingShaders,
  });

  const sceneBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ]
  });

  const geometryBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ]
  });

  const restirBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: "rgba32float", access: "write-only" } },
    ]
  });

  const displayBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, storageTexture: { format: "rgba32float", access: "read-only" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, storageTexture: { format: "rgba32float", access: "read-only" } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, storageTexture: { format: "rgba32float", access: "write-only" } },
    ]
  });

  const rasterPipelineLayout = app.device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout, geometryBindGroupLayout],
  });

  const computePipelineLayout = app.device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout, geometryBindGroupLayout, restirBindGroupLayout],
  });

  const displayPipelineLayout = app.device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout, displayBindGroupLayout],
  });

  const rasterPipeline = app.device.createRenderPipeline({
    label: "rasterization pipeline",
    layout: rasterPipelineLayout,
    vertex: {
      module: mainShaderModule,
      entryPoint: "raster_vertex_main",
    },
    fragment: {
      module: mainShaderModule,
      entryPoint: "raster_fragment_main",
      targets: [{ format: app.canvasFormat }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  const wireframePipeline = app.device.createRenderPipeline({
    label: "wireframe pipeline",
    layout: rasterPipelineLayout,
    vertex: {
      module: mainShaderModule,
      entryPoint: "wireframe_vertex_main",
    },
    fragment: {
      module: mainShaderModule,
      entryPoint: "wireframe_fragment_main",
      targets: [{ format: app.canvasFormat }],
    },
    primitive: {
      topology: "line-list",
      cullMode: "none",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: false,
      depthCompare: "less-equal",
    },
  });

  const visibilityPipeline = app.device.createComputePipeline({
    label: "visibility compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: pathtracerComputeShaderModule,
      entryPoint: "visibility_main",
    },
  });

  const initialRisPipeline = app.device.createComputePipeline({
    label: "initial RIS compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: pathtracerComputeShaderModule,
      entryPoint: "initial_ris_main",
    },
  });

  const visibilityReusePipeline = app.device.createComputePipeline({
    label: "visibility reuse compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: pathtracerComputeShaderModule,
      entryPoint: "visibility_reuse_main",
    },
  });

  const temporalReusePipeline = app.device.createComputePipeline({
    label: "temporal reuse compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: pathtracerComputeShaderModule,
      entryPoint: "temporal_reuse_main",
    },
  });

  const spatialReusePipeline = app.device.createComputePipeline({
    label: "spatial reuse compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: pathtracerComputeShaderModule,
      entryPoint: "spatial_reuse_main",
    },
  });

  const shadePathtracePipeline = app.device.createComputePipeline({
    label: "shade/pathtrace compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: pathtracerComputeShaderModule,
      entryPoint: "shade_pathtrace_main",
    },
  });

  const displayPathtracingPipeline = app.device.createRenderPipeline({
    label: "display pathtracing pipeline",
    layout: displayPipelineLayout,
    vertex: {
      module: displayPathtracingShaderModule,
      entryPoint: "present_vertex_main",
    },
    fragment: {
      module: displayPathtracingShaderModule,
      entryPoint: "present_fragment_main",
      targets: [{ format: app.canvasFormat }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
  });

  const depthTexture = app.device.createTexture({
    size: [app.canvas.width, app.canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const accumulationRead = app.device.createTexture({
    size: [app.canvas.width, app.canvas.height],
    format: "rgba32float",
    usage: GPUTextureUsage.STORAGE_BINDING,
  });

  const accumulationWrite = app.device.createTexture({
    size: [app.canvas.width, app.canvas.height],
    format: "rgba32float",
    usage: GPUTextureUsage.STORAGE_BINDING,
  });

  const pathtraceOutput = app.device.createTexture({
    size: [app.canvas.width, app.canvas.height],
    format: "rgba32float",
    usage: GPUTextureUsage.STORAGE_BINDING,
  });

  const pixelCount = app.canvas.width * app.canvas.height;
  const primarySurfaceStride = 48;
  const reservoirInitialStride = 32;

  const primarySurfaceBuffer = app.device.createBuffer({
    size: pixelCount * primarySurfaceStride,
    usage: GPUBufferUsage.STORAGE,
  });

  const reservoirsBufferA = app.device.createBuffer({
    size: pixelCount * reservoirInitialStride,
    usage: GPUBufferUsage.STORAGE,
  });

  const reservoirsBufferB = app.device.createBuffer({
    size: pixelCount * reservoirInitialStride,
    usage: GPUBufferUsage.STORAGE,
  });

  return { ...app,
    mainShaderModule, pathtracerComputeShaderModule, displayPathtracingShaderModule,
    depthTexture, accumulationRead, accumulationWrite, pathtraceOutput,
    primarySurfaceBuffer, reservoirsBufferA, reservoirsBufferB,
    sceneBindGroupLayout, geometryBindGroupLayout, restirBindGroupLayout, displayBindGroupLayout,
    rasterPipeline, displayPathtracingPipeline, wireframePipeline,
    visibilityPipeline, initialRisPipeline, visibilityReusePipeline, temporalReusePipeline, spatialReusePipeline, shadePathtracePipeline,
  };
}

export function buildSceneBindGroups(app: GPUAppPipeline, scene: Scene): GPUApp {
  if (!scene.buffersInitialized) {
    throw new Error("Cannot create bind group: Scene buffers are not initialized.");
  }

  const sceneBindGroup = app.device.createBindGroup({
    label: "scene bind group",
    layout: app.sceneBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.uniformBuffer! } },
      { binding: 1, resource: { buffer: scene.matBuffer! } },
    ],
  });

  const geometryBindGroup = app.device.createBindGroup({
    label: "geometry bind group",
    layout: app.geometryBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.posBuffer! } },
      { binding: 1, resource: { buffer: scene.normBuffer! } },
      { binding: 2, resource: { buffer: scene.triBuffer! } },
      { binding: 3, resource: { buffer: scene.instanceBuffer! } },
      { binding: 4, resource: { buffer: scene.bvhBuffer! } },
      { binding: 5, resource: { buffer: scene.emissiveTriBuffer! } },
    ],
  });

  const restirBindGroupA = app.device.createBindGroup({
    label: "restir bind group A",
    layout: app.restirBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: app.primarySurfaceBuffer } },
      { binding: 1, resource: { buffer: app.reservoirsBufferA } },
      { binding: 2, resource: { buffer: app.reservoirsBufferB } },
      { binding: 3, resource: app.pathtraceOutput.createView() },
    ],
  });

  const restirBindGroupB = app.device.createBindGroup({
    label: "restir bind group B",
    layout: app.restirBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: app.primarySurfaceBuffer } },
      { binding: 1, resource: { buffer: app.reservoirsBufferB } },
      { binding: 2, resource: { buffer: app.reservoirsBufferA } },
      { binding: 3, resource: app.pathtraceOutput.createView() },
    ],
  });

  const displayBindGroupA = app.device.createBindGroup({
    label: "display bind group A",
    layout: app.displayBindGroupLayout,
    entries: [
      { binding: 0, resource: app.pathtraceOutput.createView() },
      { binding: 1, resource: app.accumulationRead.createView() },
      { binding: 2, resource: app.accumulationWrite.createView() },
    ],
  });

  const displayBindGroupB = app.device.createBindGroup({
    label: "display bind group B",
    layout: app.displayBindGroupLayout,
    entries: [
      { binding: 0, resource: app.pathtraceOutput.createView() },
      { binding: 1, resource: app.accumulationWrite.createView() },
      { binding: 2, resource: app.accumulationRead.createView() },
    ],
  });

  return { ...app, sceneBindGroup, geometryBindGroup, restirBindGroupA, restirBindGroupB,
    displayBindGroupA, displayBindGroupB };
}

export function render(app: GPUApp, scene: Scene, useRaytracing: boolean): void {
  const encoder = app.device.createCommandEncoder({ label: "renderer encoder" });

  if (useRaytracing) {
    const workgroupCountX = Math.ceil(app.canvas.width / 8);
    const workgroupCountY = Math.ceil(app.canvas.height / 8);

    const bg0 = (scene.frameCount % 2 === 0) ? app.restirBindGroupA : app.restirBindGroupB;
    const bg1 = (scene.frameCount % 2 === 0) ? app.restirBindGroupB : app.restirBindGroupA;

    const computePass = encoder.beginComputePass({ label: "pathtracing compute pass" });

    computePass.setBindGroup(0, app.sceneBindGroup);
    computePass.setBindGroup(1, app.geometryBindGroup);

    computePass.setBindGroup(2, bg0);

    computePass.setPipeline(app.visibilityPipeline);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    computePass.setPipeline(app.initialRisPipeline);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    computePass.setPipeline(app.visibilityReusePipeline);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    if (scene.temporalReuseEnabled) {
      computePass.setPipeline(app.temporalReusePipeline);
      computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
    }

    if (scene.spatialReuseEnabled) {
      // swap buffers to have the reservoirs computed so far in the reservoirs_prev storage, write to reservoirs_curr
      computePass.setBindGroup(2, bg1);
      computePass.setPipeline(app.spatialReusePipeline);
      computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

      // second pass if biased
      if (scene.restirBiased) {
        computePass.setBindGroup(2, bg0);
        computePass.setPipeline(app.spatialReusePipeline);
        computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);
      }
    }

    computePass.setPipeline(app.shadePathtracePipeline);
    computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY);

    computePass.end();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "Display pathtracing pass",
      colorAttachments: [{
        view: app.context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
    };

    const pass = encoder.beginRenderPass(renderPassDescriptor);

    pass.setPipeline(app.displayPathtracingPipeline);
    pass.setBindGroup(0, app.sceneBindGroup);
    pass.setBindGroup(1, (scene.frameCount % 2 === 0) ? app.displayBindGroupA : app.displayBindGroupB);
    pass.draw(6);

    pass.end();
  } else {
    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "rasterizer rendering pass",
      colorAttachments: [{
        view: app.context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: app.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    const pass = encoder.beginRenderPass(renderPassDescriptor);

    pass.setPipeline(app.rasterPipeline);

    pass.setBindGroup(0, app.sceneBindGroup);
    pass.setBindGroup(1, app.geometryBindGroup);
    
    for (let i = 0; i < scene.instances.length; i++) {
      const numIndices = scene.instances[i].mesh.indices.length;
      pass.draw(numIndices, 1, 0, i); 
    }

    if (scene.viewBvh && scene.bvhDataArray) {
      pass.setPipeline(app.wireframePipeline);
      const numNodes = scene.bvhDataArray.byteLength / 32;
      pass.draw(24, numNodes, 0, 0);
    }
    
    pass.end();
  }

  const commandBuffer = encoder.finish();
  app.device.queue.submit([commandBuffer]);
}
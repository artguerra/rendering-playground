import mainShaders from "@shaders/main.wgsl?raw";

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

  shaderModule: GPUShaderModule;
  rasterPipeline: GPURenderPipeline;
  raytracingPipeline: GPURenderPipeline;
  wireframePipeline: GPURenderPipeline;

  sceneBindGroupLayout: GPUBindGroupLayout;
  geometryBindGroupLayout: GPUBindGroupLayout;
  lightBindGroupLayout: GPUBindGroupLayout;
}

export interface GPUApp extends GPUAppPipeline {
  sceneBindGroup: GPUBindGroup,
  geometryBindGroupA: GPUBindGroup,
  geometryBindGroupB: GPUBindGroup,
  lightBindGroup: GPUBindGroup,
}

export async function initWebGPU(canvas: HTMLCanvasElement): Promise<GPUAppBase> {
  if (!navigator.gpu) throw new Error("WebGPU is not supported on this browser.");

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("No adapter available for WebGPU.");

  const device = await adapter.requestDevice();
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
  const shaderModule = app.device.createShaderModule({
    label: "rasterization/raytracing shaders",
    code: mainShaders,
  });

  const sceneBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ]
  });

  const geometryBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.FRAGMENT, storageTexture: { format: "rgba32float", access: "read-only" } },
      { binding: 6, visibility: GPUShaderStage.FRAGMENT, storageTexture: { format: "rgba32float", access: "write-only" } },
    ]
  });

  const lightBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
    ]
  });

  const pipelineLayout = app.device.createPipelineLayout({
    bindGroupLayouts: [sceneBindGroupLayout, geometryBindGroupLayout, lightBindGroupLayout],
  });

  const rasterPipeline = app.device.createRenderPipeline({
    label: "rasterization pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "raster_vertex_main",
    },
    fragment: {
      module: shaderModule,
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
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "wireframe_vertex_main",
    },
    fragment: {
      module: shaderModule,
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

  const raytracingPipeline = app.device.createRenderPipeline({
    label: "raytracing pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "ray_vertex_main",
    },
    fragment: {
      module: shaderModule,
      entryPoint: "ray_fragment_main",
      targets: [{ format: app.canvasFormat }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: false,
      depthCompare: "less",
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

  return { ...app, shaderModule, depthTexture, accumulationRead, accumulationWrite,
    sceneBindGroupLayout, geometryBindGroupLayout, lightBindGroupLayout,
    rasterPipeline, raytracingPipeline, wireframePipeline };
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

  const geometryBindGroupA = app.device.createBindGroup({
    label: "geometry bind group A",
    layout: app.geometryBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.posBuffer! } },
      { binding: 1, resource: { buffer: scene.normBuffer! } },
      { binding: 2, resource: { buffer: scene.triBuffer! } },
      { binding: 3, resource: { buffer: scene.instanceBuffer! } },
      { binding: 4, resource: { buffer: scene.bvhBuffer! } },
      { binding: 5, resource: app.accumulationRead.createView() },
      { binding: 6, resource: app.accumulationWrite.createView() },
    ],
  });

  const geometryBindGroupB = app.device.createBindGroup({
    label: "geometry bind group B",
    layout: app.geometryBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.posBuffer! } },
      { binding: 1, resource: { buffer: scene.normBuffer! } },
      { binding: 2, resource: { buffer: scene.triBuffer! } },
      { binding: 3, resource: { buffer: scene.instanceBuffer! } },
      { binding: 4, resource: { buffer: scene.bvhBuffer! } },
      { binding: 5, resource: app.accumulationWrite.createView() },
      { binding: 6, resource: app.accumulationRead.createView() },
    ],
  });

  const lightBindGroup = app.device.createBindGroup({
    label: "light bind group",
    layout: app.lightBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.pointLightBuffer! } },
      { binding: 1, resource: { buffer: scene.areaLightBuffer! } },
    ],
  });

  return { ...app, sceneBindGroup, geometryBindGroupA, geometryBindGroupB, lightBindGroup };
}

export function render(app: GPUApp, scene: Scene, useRaytracing: boolean): void {
  const renderPassDescriptor: GPURenderPassDescriptor = {
    label: "Main rendering pass",
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

  const encoder = app.device.createCommandEncoder({ label: "display encoder" });
  const pass = encoder.beginRenderPass(renderPassDescriptor);

  pass.setBindGroup(0, app.sceneBindGroup);
  pass.setBindGroup(1, (scene.frameCount % 2 === 0) ? app.geometryBindGroupA : app.geometryBindGroupB);
  pass.setBindGroup(2, app.lightBindGroup);

  if (useRaytracing) {
    pass.setPipeline(app.raytracingPipeline);
    pass.draw(6);
  } else {
    pass.setPipeline(app.rasterPipeline);
    
    for (let i = 0; i < scene.instances.length; i++) {
      const numIndices = scene.instances[i].mesh.indices.length;
      pass.draw(numIndices, 1, 0, i); 
    }
  }

  if (scene.viewBvh && scene.bvhDataArray) {
    pass.setPipeline(app.wireframePipeline);
    const numNodes = scene.bvhDataArray.byteLength / 32;
    pass.draw(24, numNodes, 0, 0);
  }
  
  pass.end();

  const commandBuffer = encoder.finish();
  app.device.queue.submit([commandBuffer]);
}

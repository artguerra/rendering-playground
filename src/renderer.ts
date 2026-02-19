import mainShaders from "@shaders/main.wgsl?raw";
import computeShaders from "@shaders/procedural.wgsl?raw";

import { Scene } from "./scene";

export interface GPUAppBase {
  device: GPUDevice;
  adapter: GPUAdapter;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  canvasFormat: GPUTextureFormat;
}

export interface GPUApp extends GPUAppBase {
  depthTexture: GPUTexture;
  noiseTexture: GPUTexture;
  noiseSampler: GPUSampler;

  shaderModule: GPUShaderModule;
  computeShaderModule: GPUShaderModule;

  rasterPipeline: GPURenderPipeline;
  raytracingPipeline: GPURenderPipeline;
  computePipeline: GPUComputePipeline;

  bindGroupLayout: GPUBindGroupLayout;
  computeBindGroupLayout: GPUBindGroupLayout;

  renderBindGroup?: GPUBindGroup;
  computeBindGroup?: GPUBindGroup;
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

export function initRenderPipeline(app: GPUAppBase): GPUApp {
  const shaderModule = app.device.createShaderModule({
    label: "rasterization/raytracing shaders",
    code: mainShaders,
  });

  const computeShaderModule = app.device.createShaderModule({
    label: "procedural texture shaders",
    code: computeShaders,
  });

  const noiseTexture = app.device.createTexture({
    label: "noise texture",
    size: [1024, 1024, 1],
    format:  "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });

  const noiseSampler = app.device.createSampler({
    label: "noise sampler",
    minFilter: "linear",
    magFilter: "linear",
    addressModeU: "repeat",
    addressModeV: "repeat",
  });

  const computeBindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: "rgba16float" } },
    ]
  });

  const computePipeline = app.device.createComputePipeline({
    layout: app.device.createPipelineLayout({ bindGroupLayouts: [computeBindGroupLayout] }),
    compute: {
      module: computeShaderModule,
      entryPoint: "main",
    }
  });

  const bindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 7, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { } },
      { binding: 8, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: { } },
    ]
  });

  const pipelineLayout = app.device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
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
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  const depthTexture = app.device.createTexture({
    size: [app.canvas.width, app.canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  return { ...app, shaderModule, computeShaderModule, bindGroupLayout, computeBindGroupLayout,
          rasterPipeline, raytracingPipeline, computePipeline, depthTexture, noiseTexture, noiseSampler };
}

export function createSceneBindGroup(app: GPUApp, scene: Scene) {
  if (!scene.buffersInitialized) {
    throw new Error("Cannot create bind group: Scene buffers are not initialized.");
  }

  app.renderBindGroup = app.device.createBindGroup({
    layout: app.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.uniformBuffer! } },
      { binding: 1, resource: { buffer: scene.posBuffer! } },
      { binding: 2, resource: { buffer: scene.normBuffer! } },
      { binding: 3, resource: { buffer: scene.triBuffer! } },
      { binding: 4, resource: { buffer: scene.instanceBuffer! } },
      { binding: 5, resource: { buffer: scene.matBuffer! } },
      { binding: 6, resource: { buffer: scene.lightBuffer! } },
      { binding: 7, resource: app.noiseTexture.createView() },
      { binding: 8, resource: app.noiseSampler },
    ],
  });
}

export function createComputeBindGroup(app: GPUApp, scene: Scene) {
  if (!scene.buffersInitialized) {
    throw new Error("Cannot create bind group: Scene buffers are not initialized.");
  }

  app.computeBindGroup = app.device.createBindGroup({
    layout: app.computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.noiseParamsBuffer! } },
      { binding: 1, resource: app.noiseTexture.createView() },
    ]
  });
};

export function render(app: GPUApp, scene: Scene, useRaytracing: boolean): void {
  if (!app.renderBindGroup) {
    throw new Error("Render bind group not initialized.");
  }

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

  pass.setBindGroup(0, app.renderBindGroup!);

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
  
  pass.end();

  const commandBuffer = encoder.finish();
  app.device.queue.submit([commandBuffer]);
}

export function bakeNoiseTexture(app: GPUApp) {
  if (!app.computeBindGroup) {
    throw new Error("Compute bind group not initialized.");
  }

  const encoder = app.device.createCommandEncoder({ label: "compute encoder" });

  const computePass = encoder.beginComputePass();

  computePass.setPipeline(app.computePipeline);
  computePass.setBindGroup(0, app.computeBindGroup!);
  computePass.dispatchWorkgroups(64, 64, 1); // 1024 / 64 = 64
  computePass.end();

  const commandBuffer = encoder.finish();
   app.device.queue.submit([commandBuffer]);
}


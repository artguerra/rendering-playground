import mainShaders from "@shaders/main.wgsl?raw";

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
  shaderModule: GPUShaderModule;
  rasterPipeline: GPURenderPipeline;
  raytracingPipeline: GPURenderPipeline;
  bindGroupLayout: GPUBindGroupLayout;
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

  const bindGroupLayout = app.device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
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

  return { ...app, shaderModule, bindGroupLayout, rasterPipeline, raytracingPipeline, depthTexture };
}

export function createSceneBindGroup(app: GPUApp, scene: Scene): GPUBindGroup {
  if (!scene.buffersInitialized) {
    throw new Error("Cannot create bind group: Scene buffers are not initialized.");
  }

  return app.device.createBindGroup({
    layout: app.bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: scene.uniformBuffer! } },
      { binding: 1, resource: { buffer: scene.posBuffer! } },
      { binding: 2, resource: { buffer: scene.normBuffer! } },
      { binding: 3, resource: { buffer: scene.triBuffer! } },
      { binding: 4, resource: { buffer: scene.instanceBuffer! } },
      { binding: 5, resource: { buffer: scene.matBuffer! } },
      { binding: 6, resource: { buffer: scene.lightBuffer! } },
    ],
  });
}

export function render(app: GPUApp, scene: Scene, bindGroup: GPUBindGroup, useRaytracing: boolean): void {
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

  pass.setBindGroup(0, bindGroup);

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


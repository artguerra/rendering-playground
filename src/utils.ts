export function createGPUBuffer(device: GPUDevice, data: ArrayBufferView<ArrayBuffer>, usage: GPUBufferUsageFlags) {
  const buffer = device.createBuffer({
    size: (data.byteLength + 3) & ~3, // 4-byte aligned
    usage,
  });
  device.queue.writeBuffer(buffer, 0, data);

  return buffer;
}

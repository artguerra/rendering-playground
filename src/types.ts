import type { Vec3 } from "wgpu-matrix";

export interface LightSource {
  position: Vec3;
  intensity: number;
  color: Vec3;
  angle: number;
  spot: Vec3
  rayTracedShadows: number; // 0 or 1
}

export interface Material {
  albedo: Vec3;
  roughness: number;
  metalness: number;
}

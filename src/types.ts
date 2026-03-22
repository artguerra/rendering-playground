import type { Vec3 } from "./math";

export type LightType = "point" | "area";

export interface Material {
  albedo: Vec3;
  roughness: number;
  metalness: number;
  emissionStrength: number; // 0 = not emissive, >0 emissive
}

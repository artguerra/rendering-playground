import { type Vec3, vec3Add, vec3Sub, vec3Normalize, vec3Cross } from "./math";

export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
}

export interface MeshInstance {
  mesh: Mesh;
  materialIndex: number;
}

export interface MergedGeometry {
  positions: Float32Array<ArrayBuffer>;
  normals: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
  instances: Uint32Array<ArrayBuffer>; // packed data for WGSL: array<Mesh>
}

export function mergeMeshes(meshInstances: MeshInstance[]): MergedGeometry {
  const numVertices = meshInstances.reduce((acc, cur) => acc + cur.mesh.positions.length, 0);
  const numIndices = meshInstances.reduce((acc, cur) => acc + cur.mesh.indices.length, 0);

  const merged: MergedGeometry = {
    positions: new Float32Array(numVertices),
    normals: new Float32Array(numVertices),
    indices: new Uint32Array(numIndices),
    instances: new Uint32Array(meshInstances.length * 8),
  };

  let posOffset = 0;
  let idxOffset = 0;
  let triOffset = 0;

  for (let i = 0; i < meshInstances.length; ++i) {
    const { mesh, materialIndex } = meshInstances[i];
    const triCount = mesh.indices.length / 3;

    merged.positions.set(mesh.positions, posOffset);
    merged.normals.set(mesh.normals, posOffset);
    merged.indices.set(mesh.indices, idxOffset);

    const vertexOffset = posOffset / 3;
    const instanceIdx = i * 8; // 8-slot stride
    
    merged.instances[instanceIdx + 0] = vertexOffset; // posOffset
    merged.instances[instanceIdx + 1] = triOffset; // triOffset
    merged.instances[instanceIdx + 2] = triCount; // numOfTriangles
    merged.instances[instanceIdx + 3] = materialIndex; // materialIndex
    merged.instances[instanceIdx + 4] = 0; // bvh_root, filled by bvh
    merged.instances[instanceIdx + 5] = 0; // bvh_count, filled by bvh
    merged.instances[instanceIdx + 6] = 0; // padding
    merged.instances[instanceIdx + 7] = 0; // padding

    posOffset += mesh.positions.length;
    idxOffset += mesh.indices.length;
    triOffset += triCount;
  }

  return merged;
}

// --------------------  PRIMITIVES --------------------

export function computeNormals(mesh: Mesh) {
  const numTri = mesh.indices.length / 3;
  const length = mesh.normals.length;

  for (let i = 0; i < length; ++i) {
    mesh.normals[i] = 0.0;
  }

  for (let i = 0; i < numTri; ++i) {
    const v0 = mesh.indices[3 * i];
    const v1 = mesh.indices[3 * i + 1];
    const v2 = mesh.indices[3 * i + 2];

    const p0: Vec3 = [mesh.positions[3 * v0], mesh.positions[3 * v0 + 1], mesh.positions[3 * v0 + 2]];
    const p1: Vec3 = [mesh.positions[3 * v1], mesh.positions[3 * v1 + 1], mesh.positions[3 * v1 + 2]];
    const p2: Vec3 = [mesh.positions[3 * v2], mesh.positions[3 * v2 + 1], mesh.positions[3 * v2 + 2]];

    const e01 = vec3Sub(p1, p0);
    const e12 = vec3Sub(p2, p1);
    const c = vec3Cross(e01, e12);
    const nt = vec3Normalize(c);

    mesh.normals[3 * v0] += nt[0];
    mesh.normals[3 * v0 + 1] += nt[1];
    mesh.normals[3 * v0 + 2] += nt[2];
    mesh.normals[3 * v1] += nt[0];
    mesh.normals[3 * v1 + 1] += nt[1];
    mesh.normals[3 * v1 + 2] += nt[2];
    mesh.normals[3 * v2] += nt[0];
    mesh.normals[3 * v2 + 1] += nt[1];
    mesh.normals[3 * v2 + 2] += nt[2];
  }

  for (let i = 0; i < length / 3; ++i) {
    const ni: Vec3 = [mesh.normals[3 * i], mesh.normals[3 * i + 1], mesh.normals[3 * i + 2]];
    const nni = vec3Normalize(ni);
    mesh.normals[3 * i] = nni[0];
    mesh.normals[3 * i + 1] = nni[1];
    mesh.normals[3 * i + 2] = nni[2];
  }
}

export function createQuad(origin: Vec3, edge0: Vec3, edge1: Vec3): Mesh {
  const a = vec3Add(origin, edge0);
  const b = vec3Add(a, edge1);
  const c = vec3Add(origin, edge1);
  const n = vec3Cross(edge0, edge1);
  const indices = [0, 1, 2, 0, 2, 3];
  
  return {
    positions: new Float32Array([...origin, ...a, ...b, ...c]),
    normals: new Float32Array([...n, ...n, ...n, ...n]),
    indices: new Uint32Array(indices),
  };
}

export function createBox(
  origin: Vec3, width: number, height: number, length: number, angle: number = 0.0
): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  const w = width;
  const h = height;
  const l = length;

  const c = [
    [0, 0, l], [w, 0, l], [w, h, l], [0, h, l], // front
    [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]  // back
  ];

  const faceCorners = [
    [0, 1, 2, 3], // front
    [1, 5, 6, 2], // right
    [5, 4, 7, 6], // back
    [4, 0, 3, 7], // left
    [3, 2, 6, 7], // top
    [4, 5, 1, 0]  // bottom
  ];

  // flat normals for each face
  const faceNormals = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0]
  ];

  const sinTheta = Math.sin(angle);
  const cosTheta = Math.cos(angle);
  let vOffset = 0;

  for (let f = 0; f < 6; f++) {
    const corners = faceCorners[f];
    const n = faceNormals[f];

    const nx = n[0] * cosTheta - n[2] * sinTheta;
    const ny = n[1];
    const nz = n[0] * sinTheta + n[2] * cosTheta;

    for (let i = 0; i < 4; i++) {
      const corner = c[corners[i]];
      const x = corner[0];
      const y = corner[1];
      const z = corner[2];

      const rx = x * cosTheta - z * sinTheta;
      const ry = y;
      const rz = x * sinTheta + z * cosTheta;

      positions.push(rx + origin[0], ry + origin[1], rz + origin[2]);
      normals.push(nx, ny, nz);
    }

    indices.push(
      vOffset + 0, vOffset + 1, vOffset + 2,
      vOffset + 0, vOffset + 2, vOffset + 3
    );
    vOffset += 4;
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices)
  };
}

export function createSphere(origin: Vec3, radius: number, latitudeRes: number, longitudeRes: number): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  for (let lat = 0; lat <= latitudeRes; lat++) {
    const theta = lat * Math.PI / latitudeRes;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    for (let lon = 0; lon <= longitudeRes; lon++) {
      const phi = lon * 2 * Math.PI / longitudeRes;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      const x = cosPhi * sinTheta;
      const y = cosTheta;
      const z = sinPhi * sinTheta;

      positions.push(origin[0] + radius * x, origin[1] + radius * y, origin[2] + radius * z);
      normals.push(x, y, z);
    }
  }

  for (let lat = 0; lat < latitudeRes; lat++) {
    for (let lon = 0; lon < longitudeRes; lon++) {
      const first = lat * (longitudeRes + 1) + lon;
      const second = first + longitudeRes + 1;

      indices.push(first, first + 1, second, second, first + 1, second + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices)
  };
}

export function createCylinder(
  origin: Vec3, radius: number, height: number, radialSegments: number
): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  let vOffset = 0;

  for (let i = 0; i < radialSegments; i++) {
    const theta1 = (i / radialSegments) * 2.0 * Math.PI;
    const theta2 = ((i + 1) / radialSegments) * 2.0 * Math.PI;

    const x1 = Math.cos(theta1) * radius;
    const z1 = Math.sin(theta1) * radius;
    const x2 = Math.cos(theta2) * radius;
    const z2 = Math.sin(theta2) * radius;

    positions.push(
      origin[0] + x1, origin[1], origin[2] + z1,
      origin[0] + x2, origin[1], origin[2] + z2,
      origin[0] + x2, origin[1] + height, origin[2] + z2,
      origin[0] + x1, origin[1] + height, origin[2] + z1
    );

    const nx = (x1 + x2) / 2.0;
    const nz = (z1 + z2) / 2.0;
    const n = vec3Normalize([nx, 0, nz]);
    normals.push(n[0], n[1], n[2], n[0], n[1], n[2], n[0], n[1], n[2], n[0], n[1], n[2]);

    indices.push(
      vOffset + 0, vOffset + 2, vOffset + 1,
      vOffset + 0, vOffset + 3, vOffset + 2
    );
    vOffset += 4;
  }

  // top cap
  for (let i = 0; i < radialSegments; i++) {
    const theta1 = (i / radialSegments) * 2.0 * Math.PI;
    const theta2 = ((i + 1) / radialSegments) * 2.0 * Math.PI;

    const x1 = Math.cos(theta1) * radius;
    const z1 = Math.sin(theta1) * radius;
    const x2 = Math.cos(theta2) * radius;
    const z2 = Math.sin(theta2) * radius;

    positions.push(
      origin[0], origin[1] + height, origin[2],
      origin[0] + x1, origin[1] + height, origin[2] + z1,
      origin[0] + x2, origin[1] + height, origin[2] + z2
    );

    normals.push(
      0, 1, 0,
      0, 1, 0,
      0, 1, 0
    );

    indices.push(vOffset + 0, vOffset + 2, vOffset + 1);
    vOffset += 3;
  }

  // bottom cap
  for (let i = 0; i < radialSegments; i++) {
    const theta1 = (i / radialSegments) * 2.0 * Math.PI;
    const theta2 = ((i + 1) / radialSegments) * 2.0 * Math.PI;

    const x1 = Math.cos(theta1) * radius;
    const z1 = Math.sin(theta1) * radius;
    const x2 = Math.cos(theta2) * radius;
    const z2 = Math.sin(theta2) * radius;

    positions.push(
      origin[0], origin[1], origin[2],
      origin[0] + x1, origin[1], origin[2] + z1,
      origin[0] + x2, origin[1], origin[2] + z2
    );

    normals.push(
      0, -1, 0,
      0, -1, 0,
      0, -1, 0
    );

    indices.push(vOffset + 0, vOffset + 1, vOffset + 2);
    vOffset += 3;
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices)
  };
}
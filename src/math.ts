export type Vec3 = [number, number, number];
export type Mat4 = [
  number, number, number, number,
  number, number, number, number,
  number, number, number, number,
  number, number, number, number
];

// ------------------------------ vector math ------------------------------

export function vec3Length(a: Vec3): number {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

export function vec3Add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vec3AddScaled(a: Vec3, b: Vec3, k: number): Vec3 {
  return [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
}

export function vec3Sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vec3Mul(a: Vec3, b: Vec3): Vec3 {
  return [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
}

export function vec3MulScalar(a: Vec3, k: number): Vec3 {
  return [a[0] * k, a[1] * k, a[2] * k];
}

export function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function vec3Normalize(a: Vec3): Vec3 {
  const l = vec3Length(a);
  return [a[0] / l, a[1] / l, a[2] / l];
}

export function vec3Min(a: Vec3, b: Vec3): Vec3 {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])];
}

export function vec3Max(a: Vec3, b: Vec3): Vec3 {
  return [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];
}

// ------------------------------ matrix math ------------------------------

export function mat4Identity(): Mat4 {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ];
}

export function mat4Translate(x: number, y: number, z: number): Mat4 {
  const m = mat4Identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

export function mat4Multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Array(16) as Mat4;
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

export function mat4Transpose(a: Mat4): Mat4 {
  const out = new Array(16) as Mat4;
  out[0]  = a[0];
  out[1]  = a[4];
  out[2]  = a[8];
  out[3]  = a[12];
  out[4]  = a[1];
  out[5]  = a[5];
  out[6]  = a[9];
  out[7]  = a[13];
  out[8]  = a[2];
  out[9]  = a[6];
  out[10] = a[10];
  out[11] = a[14];
  out[12] = a[3];
  out[13] = a[7];
  out[14] = a[11];
  out[15] = a[15];
  return out;
}

export function mat4Invert(a: Mat4): Mat4 | null {
  const out = new Array(16) as Mat4;

  const a00 = a[0],  a01 = a[1],  a02 = a[2],  a03 = a[3];
  const a10 = a[4],  a11 = a[5],  a12 = a[6],  a13 = a[7];
  const a20 = a[8],  a21 = a[9],  a22 = a[10], a23 = a[11];
  const a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  //determinant
  let det =
    b00 * b11 -
    b01 * b10 +
    b02 * b09 +
    b03 * b08 -
    b04 * b07 +
    b05 * b06;

  if (!det) return null;
  det = 1.0 / det;

  out[0]  = ( a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1]  = (-a01 * b11 + a02 * b10 - a03 * b09) * det;
  out[2]  = ( a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3]  = (-a21 * b05 + a22 * b04 - a23 * b03) * det;

  out[4]  = (-a10 * b11 + a12 * b08 - a13 * b07) * det;
  out[5]  = ( a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6]  = (-a30 * b05 + a32 * b02 - a33 * b01) * det;
  out[7]  = ( a20 * b05 - a22 * b02 + a23 * b01) * det;

  out[8]  = ( a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9]  = (-a00 * b10 + a01 * b08 - a03 * b06) * det;
  out[10] = ( a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * det;

  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * det;
  out[13] = ( a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * det;
  out[15] = ( a20 * b03 - a21 * b01 + a22 * b00) * det;

  return out;
}

export function mat4Perspective(fovy: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovy * 0.5);
  const nf = 1.0 / (near - far);

  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * nf, -1,
    0, 0, (near * far) * nf, 0,
  ];
}

export function lookAt(out: Mat4, eye: Vec3, target: Vec3, up: Vec3) {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];

  let len = Math.hypot(zx, zy, zz);
  zx /= len; zy /= len; zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;

  len = Math.hypot(xx, xy, xz);
  xx /= len; xy /= len; xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx; out[1] = yx; out[2] = zx; out[3] = 0;
  out[4] = xy; out[5] = yy; out[6] = zy; out[7] = 0;
  out[8] = xz; out[9] = yz; out[10] = zz; out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
}

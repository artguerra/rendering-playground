import { type Vec3, vec3, type Mat4, mat4 } from "wgpu-matrix";

export class Camera {
  // world vectors
  target: Vec3;
  up: Vec3 = vec3.create(0, 1, 0);

  // matrices
  modelMat: Mat4;
  viewMat: Mat4;
  invViewMat: Mat4;
  transInvViewMat: Mat4;
  projMat: Mat4;

  // camera parameters
  aspect: number;
  fov: number = Math.PI / 4.0;
  near: number = 0.1;
  far: number = 10000.0;
  radius: number = 1;

  // camera control
  yaw: number = 0;
  pitch: number = 0;

  rotateSpeed: number = 0.005;
  zoomSpeed: number = 0.001;
  panSpeed: number = 0.002;
  
  minRadius: number = 1;
  maxRadius: number = 10000;

  dragging: boolean  = false;
  panning: boolean = false;
  lastX: number = 0;
  lastY: number = 0;

  constructor(target: Vec3, aspectRatio: number, radius: number) {
    this.target = target;
    this.aspect = aspectRatio;
    this.radius = radius;

    this.modelMat = mat4.identity();
    this.viewMat  = mat4.identity(); 
    this.invViewMat = mat4.identity();
    this.transInvViewMat = mat4.identity();
    this.projMat = mat4.perspective(
      this.fov,
      this.aspect,
      this.near,
      this.far
    );
  }

  getCameraPosition() {
    return [
      this.target[0] + this.radius * Math.cos(this.pitch) * Math.sin(this.yaw),
      this.target[1] + this.radius * Math.sin(this.pitch),
      this.target[2] + this.radius * Math.cos(this.pitch) * Math.cos(this.yaw)
    ];
  }

  updateCamera() {
    mat4.lookAt(this.getCameraPosition(), this.target, this.up, this.viewMat);
    this.invViewMat = mat4.invert(this.viewMat);
    this.transInvViewMat = mat4.transpose(this.invViewMat);
  }

  pan(dx: number, dy: number) {
    const cosYaw = Math.cos(this.yaw);
    const sinYaw = Math.sin(this.yaw);

    // right vector
    const rightX = cosYaw;
    const rightZ = -sinYaw;

    // up vector (world up)
    const upX = 0;
    const upZ = 0;

    const scale = this.radius * this.panSpeed;

    this.target[0] -= (rightX * dx - upX * dy) * scale;
    this.target[1] -= dy * scale;
    this.target[2] -= (rightZ * dx - upZ * dy) * scale;
  }
}

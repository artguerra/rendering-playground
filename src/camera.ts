import {
  type Vec3, type Mat4, mat4Identity, mat4Invert, mat4Transpose,
  mat4Perspective, lookAt, vec3Normalize, vec3Cross, vec3Add,
  vec3AddScaled,
} from "./math";

export class Camera {
  // world vectors
  position: Vec3;
  front: Vec3 = [0, 0, -1];
  right: Vec3 = [1, 0, 0];
  worldUp: Vec3 = [0, 1, 0];

  // matrices
  modelMat: Mat4;
  viewMat: Mat4;
  invViewMat: Mat4;
  transInvModelMat: Mat4;
  projMat: Mat4;

  // camera parameters
  aspect: number;
  fov: number = Math.PI / 4.0;
  near: number = 0.1;
  far: number = 10000.0;

  // camera control
  yaw: number = 0; // start by looking down the -Z axis
  pitch: number = 0;

  rotateSpeed: number = 0.001;
  moveSpeed: number = 0.05;

  dragging: boolean = false;
  lastX: number = 0;
  lastY: number = 0;

  constructor(position: Vec3, aspectRatio: number) {
    this.position = position;
    this.aspect = aspectRatio;

    this.modelMat = mat4Identity();
    this.viewMat  = mat4Identity(); 
    this.invViewMat = mat4Identity();
    this.transInvModelMat = mat4Identity();
    this.projMat = mat4Perspective(
      this.fov,
      this.aspect,
      this.near,
      this.far
    );
    
    this.updateVectors();
  }

  updateVectors() {
    this.front[0] = -Math.cos(this.pitch) * Math.sin(this.yaw);
    this.front[1] = -Math.sin(this.pitch);
    this.front[2] = -Math.cos(this.pitch) * Math.cos(this.yaw);
    this.front = vec3Normalize(this.front);

    this.right = vec3Cross(this.front, this.worldUp);
    this.right = vec3Normalize(this.right);
  }

  updateCamera() {
    this.updateVectors();
    
    const target = vec3Add(this.position, this.front);
    lookAt(this.viewMat, this.position, target, this.worldUp);
    
    this.invViewMat = mat4Invert(this.viewMat) ?? mat4Identity();
    this.transInvModelMat = mat4Transpose(mat4Invert(this.modelMat) ?? mat4Identity());
  }

  processMouseMovement(dx: number, dy: number) {
    this.yaw -= dx * this.rotateSpeed;
    this.pitch += dy * this.rotateSpeed;

    const maxPitch = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));
  }

  processKeyboard(forward: number, right: number, up: number) {
    if (forward !== 0)
      this.position = vec3AddScaled(this.position, this.front, forward * this.moveSpeed);
    if (right !== 0)
      this.position = vec3AddScaled(this.position, this.right, right * this.moveSpeed);
    if (up !== 0)
      this.position = vec3AddScaled(this.position, this.worldUp, up * this.moveSpeed);
  }
}
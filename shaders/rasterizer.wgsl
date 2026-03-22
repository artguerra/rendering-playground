const PI = 3.14159265358979323846;

struct Material {
  albedo: vec3<f32>,
  roughness: f32,
  metalness: f32,
  emission_strength: f32,
  _pad: vec2<f32>,
}

struct EmissiveTriangle {
  tri_idx: u32,
  mesh_idx: u32,
}

struct Camera {
  model_mat: mat4x4<f32>,
  view_mat: mat4x4<f32>,
  prev_view_mat: mat4x4<f32>,
  inv_view_mat: mat4x4<f32>,
  trans_inv_model_mat: mat4x4<f32>,
  proj_mat: mat4x4<f32>,
  fov: f32,
  aspect_ratio: f32,
  _pad: vec2<f32>
}

struct Mesh {
  pos_offset: u32, // in vertices, used for all attribute buffer for now
  tri_offset: u32, // in triangles
  num_triangles: u32,  
  material_idx: u32, // index over the material buffer
  centroid: vec3<f32>,
  bvh_root: u32,
  bvh_count: u32,
  _pad1: u32,
  _pad2: vec2<u32>,
}

struct BVHNode {
  min_corner: vec3<f32>,
  primitive_count: u32, // 0 if interior
  max_corner: vec3<f32>,
  skip_link: u32, // if leaf: first triangle index. if interior: miss link
}

struct Scene {
  camera: Camera,
  canvas_width: f32,
  canvas_height: f32,
  num_meshes: f32,
  num_emissive_triangles: u32,

  // time
  timestamp: u32,
  frame_count: u32,
  absolute_frame_count: u32,

  // options
  tone_mapping: u32,
  accumulation_enabled: u32,
  max_ray_depth: u32,
  stratified_grid_n: u32,
  restir_enabled: u32,
  use_streaming_ris_on_bounces: u32,
  restir_biased: u32,
  _pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> scene: Scene;

@group(0) @binding(1)
var<storage, read> materials: array<Material>;

@group(1) @binding(0)
var<storage, read> positions: array<f32>; // packed positions for all meshes

@group(1) @binding(1)
var<storage, read> normals: array<f32>; // packed normals for all meshes

@group(1) @binding(2)
var<storage, read> triangles: array<u32>; // packed triangles for all meshes

@group(1) @binding(3)
var<storage, read> meshes: array<Mesh>;

@group(1) @binding(4)
var<storage, read> bvh_nodes: array<BVHNode>;

@group(1) @binding(5)
var<storage, read> emissive_triangles: array<EmissiveTriangle>;

// ----------------------------- helper functions -----------------------------

fn get_vert_pos(vert_index: u32) -> vec3f {
  let idx = 3u * vert_index;
  return vec3f(positions[idx], positions[idx + 1u], positions[idx + 2u]);
}

fn get_vert_normal(vert_index: u32) -> vec3f {
  let idx = 3u * vert_index;
  return vec3f(normals[idx], normals[idx + 1u], normals[idx + 2u]);
}

fn get_triangle(tri_idx: u32) -> vec3u {
  let idx = 3u * tri_idx;
  return vec3u(triangles[idx], triangles[idx + 1u], triangles[idx + 2u]);
}

fn sqr(x: f32) -> f32 { 
  return x * x; 
}

fn attenuation(dist: f32) -> f32 {
  return 1.0 / sqr(dist);
}

// ----------------------------- BRDF functions -----------------------------

fn trowbridge_reitz_ndf(wh: vec3f, n: vec3f, alpha: f32) -> f32 {
  let alpha2 = sqr(alpha);

  return alpha2 / (PI * sqr(1.0 + (alpha2 - 1.0) * sqr(dot(n, wh))));
}

fn schlick_fresnel(wi: vec3f, wh: vec3f, f0: vec3f) -> vec3f {
  return f0 + (1.0 - f0) * pow(1.0 - max(0.0, dot(wi, wh)), 5.0);
}

fn smith_g1(w: vec3f, n: vec3f, alpha: f32) -> f32 {
  let n_dot_w = dot(n, w);
  let alpha2 = sqr(alpha);

  return (2.0 * n_dot_w) / (n_dot_w + sqrt(alpha2 + (1.0 - alpha2) * sqr(n_dot_w)));
}

fn smith_ggx(wi: vec3f, wo: vec3f, n: vec3f, alpha: f32) -> f32 {
  return smith_g1(wi, n, alpha) * smith_g1(wo, n, alpha);
}

fn brdf(
  wi: vec3f, 
  wo: vec3f, 
  n: vec3f, 
  albedo: vec3f, 
  roughness: f32, 
  metalness: f32
) -> vec3f {
  let diffuse_color = albedo * (1.0 - metalness);
  let specular_color = mix(vec3f(0.08), albedo, metalness);

  let alpha = roughness * roughness; // to approach a linear behavior
  let n_dot_l = max(0.0, dot(n, wi));
  let n_dot_v = max(0.0, dot(n, wo));

  if (n_dot_l <= 0.0 || n_dot_v <= 0) { // not in the reflection hemisphere
    return vec3f(0.0); 
  }

  let h = wi + wo;
  if (dot(h, h) <= 1e-9) {
    return vec3f(0.0);
  }
  let wh = normalize(h);

  let n_dot_h = max(0.0, dot(n, wh));
  let v_dot_h = max(0.0, dot(wo, wh));

  // normal distribution function (ggx)
  let d = trowbridge_reitz_ndf(wh, n, alpha);

  // Schlick approximation to the Fresnel term
  let f = schlick_fresnel(wi, wh, specular_color);

  // masking-shadowing term
  let g = smith_ggx(wi, wo, n, alpha);

  let f_d = diffuse_color * (vec3f(1.0) - specular_color) / PI;
  let f_s = f * d * g / (4.0 * n_dot_l * n_dot_v);

  return (f_d + f_s);
}

fn point_light_shade(
  position: vec3f, 
  normal: vec3f, 
  material_idx: u32, 
  light_position: vec3f,
  light_color: vec3f,
  light_intensity: f32,
  wo: vec3f
) -> vec3f {
  var wi = light_position - position;
  let di = length(wi);
  wi = normalize(wi);

  let att = attenuation(di);
  let ir = light_color * 1.0 * att;
  var m = materials[material_idx];

  let fr = brdf(wi, wo, normal, m.albedo, m.roughness, m.metalness);

  return ir * fr * max(0.0, dot(wi, normal));
}


fn compute_radiance(
  position: vec3f, 
  normal: vec3f, 
  material_idx: u32, 
  wo: vec3f
) -> vec3f {
  var color_response = vec3f(0.0);
  let num_meshes = u32(scene.num_meshes);

  for (var mesh_idx = 0u; mesh_idx < num_meshes; mesh_idx++) {
    let mesh = meshes[mesh_idx];
    let mat = materials[mesh.material_idx];

    if (mat.emission_strength > 0.0) {
      let light_pos = mesh.centroid;
      let light_color = mat.albedo;
      let intensity = mat.emission_strength;
      color_response += point_light_shade(
        position, normal, material_idx, light_pos, light_color, intensity, wo
      );
    }
  }

  return color_response;
}

// ----------------------------- wireframe (debug) shaders -------------------------

struct WireframeVertexInput {
  @builtin(vertex_index) vertex_idx: u32,
  @builtin(instance_index) obj_idx: u32,
}

struct WireframeVertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
}

fn get_box_corner(first: vec3f, second: vec3f, i: u32) -> vec3f {
  switch i {
    case 0: { return first; }
    case 1: { return vec3f(first.x, first.y, second.z); }
    case 2: { return vec3f(second.x, first.y, second.z); }
    case 3: { return vec3f(second.x, first.y, first.z); }
    case 4: { return vec3f(first.x, second.y, first.z); }
    case 5: { return vec3f(first.x, second.y, second.z); }
    case 6: { return second; }
    case 7: { return vec3f(second.x, second.y, first.z); }
    case default: { return first; }
  }
}

@vertex
fn wireframe_vertex_main(input: WireframeVertexInput) -> WireframeVertexOutput {
  let box_indices= array<u32, 24>(
    0, 1, 1, 2, 2, 3, 3, 0,
    0, 4, 1, 5, 2, 6, 3, 7,
    4, 5, 5, 6, 6, 7, 7, 4
  );

  let node = bvh_nodes[input.obj_idx];
  let min_corner = node.min_corner;
  let max_corner = node.max_corner;

  let idx = box_indices[input.vertex_idx];
  let cur_corner = vec4f(get_box_corner(min_corner, max_corner, idx), 1.0);
  let pos = scene.camera.proj_mat * scene.camera.view_mat * scene.camera.model_mat * cur_corner;

  var color = vec3f(0.0, 1.0, 0.0); 
  if (node.primitive_count > 0u) {
    color = vec3f(1.0, 0.0, 0.0); 
  }

  return WireframeVertexOutput(pos, color);
}

@fragment
fn wireframe_fragment_main(input: WireframeVertexOutput) -> @location(0) vec4f {
  return vec4f(input.color, 1.0);
}

// ----------------------------- rasterization shaders ----------------------------- 

struct RasterVertexInput {
  @builtin(vertex_index) vertex_idx: u32,
  @builtin(instance_index) mesh_idx: u32,
}

struct RasterVertexOutput {
  @builtin(position) builtin_pos: vec4f,
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) material_idx: u32,
}

@vertex
fn raster_vertex_main(input: RasterVertexInput) -> RasterVertexOutput {
  let cam = scene.camera;
  let mesh = meshes[input.mesh_idx];

  let tri_idx = input.vertex_idx / 3u;
  let tri_vert_index = input.vertex_idx % 3u;
  let triangle = get_triangle(mesh.tri_offset + tri_idx);
  let vert_index = mesh.pos_offset + triangle[tri_vert_index];

  var output: RasterVertexOutput;

  let p = cam.model_mat * vec4f(get_vert_pos(vert_index), 1.0); 
  output.builtin_pos = cam.proj_mat * cam.view_mat * p; // to fire rasterization
  output.position = p.xyz;

  let n = cam.trans_inv_model_mat * vec4f(get_vert_normal(vert_index), 0.0);
  output.normal = normalize(n.xyz);
  output.material_idx = mesh.material_idx; 

  return output; 
}

@fragment
fn raster_fragment_main(input: RasterVertexOutput) -> @location(0) vec4f {
  let position = input.position;
  let normal = normalize(input.normal);

  let cam_world_pos = scene.camera.inv_view_mat[3].xyz;
  let wo = normalize(cam_world_pos - position);

  let mat = materials[input.material_idx];
  if (mat.emission_strength > 0.0) {
    return vec4f(mat.albedo, 1.0);
  }

  let color_response = compute_radiance(position, normal, input.material_idx, wo);

  return vec4f(color_response, 1.0);
}

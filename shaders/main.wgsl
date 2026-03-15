const PI = 3.14159265358979323846;
const SQRT_2 = 1.41421356237;
const INV_SQRT_3_4 = 1.154700538;
const INV_PI = 1.0 / 3.14159265358979323846;
const EPSILON = 1e-6;

struct PointLight {
  position: vec3<f32>,
  intensity: f32,
  color: vec3<f32>,
  ray_traced_shadows: u32,
}

struct AreaLight {
  position: vec3<f32>, // bottom left corner pos
  intensity: f32,
  color: vec3<f32>,
  ray_traced_shadows: u32,
  u: vec3<f32>, // edge 1 (width)
  _pad1: f32,
  v: vec3<f32>, // edge 2 (height)
  _pad2: f32,
}

struct Material {
  albedo: vec3<f32>,
  roughness: f32,
  metalness: f32,
  material_type: u32,
  emission_strength: f32,
  _pad: f32,
}

struct EmissiveTriangle {
  tri_idx: u32,
  mesh_idx: u32,
}

struct Camera {
  model_mat: mat4x4<f32>,
  view_mat: mat4x4<f32>,
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
  bvh_root: u32,
  bvh_count: u32,
  _pad: vec2<u32>,
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
  num_point_lights: u32,
  num_emissive_triangles: u32,

  // time
  timestamp: u32,
  frame_count: u32,

  // options
  tone_mapping: u32,
  accumulation_enabled: u32,
  max_ray_depth: u32,
  stratified_grid_n: u32,
  restir_enabled: u32,
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
var accumulation_prev: texture_storage_2d<rgba32float, read>;

@group(1) @binding(6)
var accumulation_next: texture_storage_2d<rgba32float, write>;

@group(2) @binding(0)
var<storage, read> point_lights: array<PointLight>;

@group(2) @binding(1)
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

// sgrb luminance to make color value a scalar
fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}


// -------------------------------------------- RNGs --------------------------------------------

// based on "Efficient pseudo-random number generation for monte-carlo simulations using graphic processors" (Mohanty et al.)
struct RngState {
  z1: u32,
  z2: u32,
  z3: u32,
  z4: u32,
}

fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, M: u32) -> u32 {
  let b = (((z << s1) ^ z) >> s2);
  return (((z & M) << s3) ^ b);
}

fn init_rng(seed_in: u32) -> RngState {
  var state: RngState;

  // pcg integer hash to not have that much spatial correlation
  var pcg_state = seed_in * 747796405u + 2891336453u;
  var word = ((pcg_state >> ((pcg_state >> 28u) + 4u)) ^ pcg_state) * 277803737u;
  let seed = (word >> 22u) ^ word;

  state.z1 = taus_step(seed, 13u, 19u, 12u, 4294967294u);
  state.z2 = taus_step(seed, 2u, 25u, 4u, 4294967288u);
  state.z3 = taus_step(seed, 3u, 11u, 17u, 4294967280u);
  state.z4 = (1664525u * seed + 1013904223u);

  return state;
}

fn rand(state: ptr<function, RngState>) -> f32 {
  (*state).z1 = taus_step((*state).z1, 13u, 19u, 12u, 4294967294u);
  (*state).z2 = taus_step((*state).z2, 2u, 25u, 4u, 4294967288u);
  (*state).z3 = taus_step((*state).z3, 3u, 11u, 17u, 4294967280u);
  (*state).z4 = (1664525u * (*state).z4 + 1013904223u);

  let r = (*state).z1 ^ (*state).z2 ^ (*state).z3 ^ (*state).z4;
  return f32(r) * 2.3283064365387e-10;
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

fn evaluate_brdf_pdf(wi: vec3f, wo: vec3f, n: vec3f, roughness: f32, p_spec: f32, p_diff: f32) -> f32 {
  let n_dot_i = max(0.0, dot(n, wi));
  if (n_dot_i <= 0.0) { return 0.0; }

  let wh = normalize(wi + wo);
  let n_dot_h = max(0.0, dot(n, wh));
  let wo_dot_h = max(0.0, dot(wo, wh));
  
  // diffuse PDF (cosine distribution)
  let pdf_d = n_dot_i * INV_PI;

  // ggx PDF
  var pdf_s = 0.0;
  if (wo_dot_h > 0.0) {
    let alpha = roughness * roughness;
    let d = trowbridge_reitz_ndf(wh, n, alpha);
    pdf_s = (d * n_dot_h) / (4.0 * wo_dot_h);
  }

  return p_diff * pdf_d + p_spec * pdf_s;
}

// samples GGX NDF to generate a halfway vector
fn sample_ggx_ndf(alpha: f32, u: vec2f) -> vec3f {
  let a2 = sqr(alpha);
  let cos_theta = sqrt(max(0.0, (1.0 - u.x) / (1.0 + (a2 - 1.0) * u.x)));
  let sin_theta = sqrt(max(0.0, 1.0 - sqr(cos_theta)));
  let phi = 2.0 * PI * u.y;

  return vec3f(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

fn point_light_shade(
  position: vec3f, 
  normal: vec3f, 
  material_idx: u32, 
  light_source_idx: u32, 
  wo: vec3f
) -> vec3f {
  let light = point_lights[light_source_idx];

  var wi = light.position - position;
  let di = length(wi);
  wi = normalize(wi);

  let att = attenuation(di);
  let ir = light.color * light.intensity * att;
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
  let num_lights = u32(scene.num_point_lights);

  for (var light_source_idx = 0u; light_source_idx < num_lights; light_source_idx++) {
    color_response += point_light_shade(position, normal, material_idx, light_source_idx, wo);
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

// ---------------------------- ray tracing shaders ---------------------------- 
const MAX_DISTANCE = 1e8;

struct RayVertexInput {
  @builtin(vertex_index) vertex_idx: u32
}

struct RayVertexOutput {
  @builtin(position) pos: vec4f,
} 

struct RayFragmentInput {
  @builtin(position) frag_pos: vec4f,
}

struct Ray {
  origin: vec3f,
  direction: vec3f,
}

struct Hit {
  mesh_idx: u32,
  tri_idx: u32,
  u: f32, // barycentric coordinates of the intersection
  v: f32,
  t: f32, // distance to ray's origin of the intersection
}

fn interpolate(x0: vec3f, x1: vec3f, x2: vec3f, uvw: vec3f) -> vec3f {
  return uvw.z * x0 + uvw.x * x1 + uvw.y * x2;
}

fn ray_at(uv: vec2f, camera: Camera) -> Ray {
  var ray: Ray;

  let view_right = normalize(camera.inv_view_mat[0].xyz);
  let view_up = normalize(camera.inv_view_mat[1].xyz);
  let view_dir = -normalize(camera.inv_view_mat[2].xyz);
  let eye = camera.inv_view_mat[3].xyz;

  let w = 2.0 * tan(0.5 * camera.fov); 

  ray.origin = eye;
  ray.direction = normalize(view_dir + ((uv.x - 0.5) * camera.aspect_ratio * w) * view_right + (uv.y - 0.5) * w * view_up);

  return ray;
}

fn intersect_aabb(ray: Ray, min_corner: vec3f, max_corner: vec3f, max_t: f32) -> f32 {
  let inv_dir = 1.0 / ray.direction;
  let t0 = (min_corner - ray.origin) * inv_dir;
  let t1 = (max_corner - ray.origin) * inv_dir;

  let t_entry = min(t0, t1);
  let t_exit = max(t0, t1);

  let t_entry_max = max(max(t_entry.x, t_entry.y), t_entry.z);
  let t_exit_min = min(min(t_exit.x, t_exit.y), t_exit.z);

  if (t_entry_max <= t_exit_min && t_exit_min >= 0 && t_entry_max <= max_t) {
    return max(0.0, t_entry_max);
  }

  return -1.0;
}

fn intersect_triangle(
  ray: Ray, 
  p0: vec3f, 
  p1: vec3f, 
  p2: vec3f, 
  back_face_culling: bool,
  t_min: f32,
  t_max: f32,
  hit: ptr<function, Hit>
) -> bool {
  let e1 = p1 - p0;
  let e2 = p2 - p0;
  let d_x_e2 = cross(ray.direction, e2);
  let det = dot(e1, d_x_e2);

  if ((back_face_culling && det < EPSILON) || (!back_face_culling && abs(det) < EPSILON)) {
    return false;
  }

  let inv_det = 1.0 / det;
  let op0 = ray.origin - p0;

  (*hit).u = dot(op0, d_x_e2) * inv_det;
  if ((*hit).u < 0.0 || (*hit).u > 1.0) {
    return false;
  }

  let op0_x_e1 = cross(op0, e1);
  (*hit).t = dot(e2, op0_x_e1) * inv_det;

  if ((*hit).t < t_min || (*hit).t > t_max) {
    return false;
  }

  (*hit).v = dot(ray.direction, op0_x_e1) * inv_det;

  if ((*hit).v >= 0.0 && (*hit).u + (*hit).v <= 1.0) {
    return true;
  }

  return false;
}

fn ray_trace(
  ray: Ray, 
  max_distance: f32, 
  any_hit: bool,
  ignore_tri: u32, 
  hit: ptr<function, Hit>
) -> bool {
  var intersection_found = false;

  for (var m = 0u; m < u32(scene.num_meshes); m++) {
    let mesh = meshes[m];
    var node_idx = mesh.bvh_root;
    let end_idx = mesh.bvh_root + mesh.bvh_count;

    while (node_idx < end_idx) {
      let node = bvh_nodes[node_idx];

      var current_max_t = MAX_DISTANCE;
      if (intersection_found && !any_hit) {
        current_max_t = (*hit).t;
      }

      let dist = intersect_aabb(ray, node.min_corner, node.max_corner, current_max_t);

      if (dist >= 0.0) {
        if (node.primitive_count > 0u) {
          // leaf node hit
          let start_tri = node.skip_link; // skip_link holds absolute triangle index

          for (var i = 0u; i < node.primitive_count; i++) {
            let tri_idx = start_tri + i;
            if (tri_idx == ignore_tri) { continue; }

            let tri = get_triangle(tri_idx);

            let p0 = get_vert_pos(mesh.pos_offset + tri.x);
            let p1 = get_vert_pos(mesh.pos_offset + tri.y);
            let p2 = get_vert_pos(mesh.pos_offset + tri.z);

            var tri_hit: Hit;
            tri_hit.tri_idx = tri_idx;
            tri_hit.mesh_idx = m;

            if (intersect_triangle(ray, p0, p1, p2, true, 0.0, max_distance, &tri_hit) == true) {
              if (!intersection_found || (intersection_found && tri_hit.t < (*hit).t)) {
                if (any_hit == true) { return true; }
                *hit = tri_hit;
                intersection_found = true;
              }
            }
          }

          node_idx++; // go to next node after intersecting leaf primitives
        } else {
          // interior node hit
          node_idx++; 
        }
      } else {
        // missed node
        if (node.primitive_count > 0u) {
           node_idx++; // missed leaf, skip to next node
        } else {
           // skip subtree
           node_idx = node.skip_link; 
        }
      }
    }
  }

  return intersection_found;
}

fn get_hit_vectors(hit: Hit, hit_pos: ptr<function, vec3f>, hit_normal: ptr<function, vec3f>) {
  let mesh = meshes[hit.mesh_idx];
  let tri = get_triangle(hit.tri_idx);
  let uvw = vec3f(hit.u, hit.v, 1.0 - hit.u - hit.v);

  *hit_pos = interpolate(
    get_vert_pos(mesh.pos_offset + tri.x), 
    get_vert_pos(mesh.pos_offset + tri.y), 
    get_vert_pos(mesh.pos_offset + tri.z), 
    uvw
  );

  *hit_normal = normalize(interpolate(
    get_vert_normal(mesh.pos_offset + tri.x), 
    get_vert_normal(mesh.pos_offset + tri.y), 
    get_vert_normal(mesh.pos_offset + tri.z), 
    uvw
  ));
}

fn sample_triangle(
  tri_idx: u32, 
  mesh_idx: u32, 
  s: vec2f, 
  pos: ptr<function, vec3f>, 
  normal: ptr<function, vec3f>, 
  area: ptr<function, f32>
) {
  let mesh = meshes[mesh_idx];
  let tri = get_triangle(tri_idx);

  let p0 = get_vert_pos(mesh.pos_offset + tri.x);
  let p1 = get_vert_pos(mesh.pos_offset + tri.y);
  let p2 = get_vert_pos(mesh.pos_offset + tri.z);

  // uniform barycentric sampling
  let sqrt_r1 = sqrt(s.x);
  let u = 1.0 - sqrt_r1;
  let v = s.y * sqrt_r1;
  let w = 1.0 - u - v;

  *pos = u * p0 + v * p1 + w * p2;

  // area and normal
  let edge1 = p1 - p0;
  let edge2 = p2 - p0;
  let cross = cross(edge1, edge2);
  
  *area = 0.5 * length(cross);
  *normal = normalize(cross);
}

// uniform light sampling
fn sample_light_uniform(u: f32) -> u32 {
  let n = scene.num_emissive_triangles;
  let light_idx = min(u32(u * f32(n)), n - 1u);
  return light_idx;
}

fn uniform_light_sample_pdf(area: f32) -> f32 {
  // uniform pdf in area measure
  let pdf = 1.0 / (area * f32(scene.num_emissive_triangles));
  return pdf;
}

// ----------------------------- ReSTIR -----------------------------

struct LightCandidate {
  emissive_idx: u32,
  light_point: vec3f,
  light_normal: vec3f,
  area: f32,
  p_source: f32, // source PDF (in area measure) -> p(x) on the paper
}

struct LightEval {
  f_unshadowed: vec3f, // fr * Le * G (without visibility)
  p_hat: f32, // scalar target used by RIS
}

struct Reservoir {
  sample: LightCandidate,
  w_sum: f32,
  m: u32, // samples seen so far
  final_w: f32, // (1 / p_hat) * (w_sum / M) for the final sample
}

fn update_reservoir(
  xi: LightCandidate, // new sample
  wi: f32, // weight associated to xi
  res: ptr<function, Reservoir>, // the reservoir to update
  u: f32 // uniform random variable
) {
  (*res).m++;
  (*res).w_sum += wi;

  if (wi > 0.0) {
    let choosing_prob = wi / (*res).w_sum;
    if (u < choosing_prob) {
      (*res).sample = xi;
    }
  }
}

fn sample_light_candidate_uniform(rng: ptr<function, RngState>) -> LightCandidate {
  let emissive_idx = sample_light_uniform(rand(rng));
  let emissive_tri = emissive_triangles[emissive_idx];

  var light_point: vec3f;
  var light_normal: vec3f;
  var area: f32;
  sample_triangle(
    emissive_tri.tri_idx,
    emissive_tri.mesh_idx,
    vec2f(rand(rng), rand(rng)),
    &light_point,
    &light_normal,
    &area
  );

  let p_source = uniform_light_sample_pdf(area);
  return LightCandidate(emissive_idx, light_point, light_normal, area, p_source);
}

fn evaluate_light_candidate(
  position: vec3f,
  normal: vec3f,
  wo: vec3f,
  mat: Material,
  candidate: LightCandidate
) -> LightEval {
  let pos_to_light = candidate.light_point - position;
  let d2 = dot(pos_to_light, pos_to_light);
  if (d2 <= 1e-12) {
    return LightEval(vec3f(0.0), 0.0);
  }

  let di = sqrt(d2);
  let wi = pos_to_light / di;

  let cos_surface = max(0.0, dot(normal, wi));
  let cos_light = max(0.0, dot(candidate.light_normal, -wi));

  if (cos_surface <= 0.0 || cos_light <= 0.0) {
    return LightEval(vec3f(0.0), 0.0);
  }

  let emissive_tri = emissive_triangles[candidate.emissive_idx];
  let light_mesh = meshes[emissive_tri.mesh_idx];
  let light_mat = materials[light_mesh.material_idx];

  let Le = light_mat.albedo * light_mat.emission_strength;
  let fr = brdf(wi, wo, normal, mat.albedo, mat.roughness, mat.metalness);

  let G = (cos_surface * cos_light) / d2;
  let f_unshadowed = fr * Le * G;

  // scalar target for RIS
  let p_hat = max(0.0, luminance(f_unshadowed));

  return LightEval(f_unshadowed, p_hat);
}

fn sample_light_ris(
  position: vec3f,
  wo: vec3f,
  normal: vec3f,
  mat: Material,
  rng: ptr<function, RngState>
) -> Reservoir {
  const M = 32u; // number of candidate samples from source distribution
  var res = Reservoir();
  res.w_sum = 0.0;
  res.final_w = 0.0;
  res.m = 0u;

  for (var i = 0u; i < M; i++) {
    let candidate = sample_light_candidate_uniform(rng);
    let eval = evaluate_light_candidate(position, normal, wo, mat, candidate);

    let w = eval.p_hat / candidate.p_source;
    update_reservoir(candidate, w, &res, rand(rng));
  }

  let selected_eval = evaluate_light_candidate(position, normal, wo, mat, res.sample);
  if (selected_eval.p_hat <= 0.0) {
    res.final_w = 0.0;
    return res;
  }
  res.final_w = res.w_sum / (f32(res.m) * selected_eval.p_hat);

  return res;
}

fn trace_light_visibility(
  position: vec3f,
  normal: vec3f,
  candidate: LightCandidate
) -> bool {
  let pos_to_light = candidate.light_point - position;
  let di = length(pos_to_light);
  let wi = pos_to_light / di;

  var shadow_ray: Ray;
  const SHADOW_BIAS = 0.0001;
  shadow_ray.origin = position + SHADOW_BIAS * normal;
  shadow_ray.direction = wi;

  let emissive_tri = emissive_triangles[candidate.emissive_idx];

  var shadow_hit: Hit;
  let occluded = ray_trace(shadow_ray, di, true, emissive_tri.tri_idx, &shadow_hit);

  return !occluded;
}

fn shade_rt(hit: Hit, incoming_ray_dir: vec3f, rng: ptr<function, RngState>) -> vec4f {
  let mesh = meshes[hit.mesh_idx];
  let mat = materials[mesh.material_idx];

  var position: vec3f;
  var world_normal: vec3f;
  get_hit_vectors(hit, &position, &world_normal);

  var color_response = vec3f(0.0);
  let wo = normalize(-incoming_ray_dir);

  if (scene.num_emissive_triangles == 0u) { return vec4f(0.0, 0.0, 0.0, 1.0); }

  if (scene.restir_enabled == 0u) {
    let candidate = sample_light_candidate_uniform(rng);
    let eval = evaluate_light_candidate(position, world_normal, wo, mat, candidate);

    if (trace_light_visibility(position, world_normal, candidate)) {
      color_response += eval.f_unshadowed / candidate.p_source;
    }
  } else {
    let res = sample_light_ris(position, wo, world_normal, mat, rng);

    if (res.final_w > 0.0 && trace_light_visibility(position, world_normal, res.sample)) {
      let eval = evaluate_light_candidate(position, world_normal, wo, mat, res.sample);
      color_response += eval.f_unshadowed * res.final_w;
    }
  }

  return vec4f(color_response, 1.0);
}

@vertex
fn ray_vertex_main(input: RayVertexInput) -> RayVertexOutput {
  var output: RayVertexOutput;

  const SCREEN_POS = array<vec2<f32>, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
  );

  output.pos = vec4f(SCREEN_POS[input.vertex_idx], 0.0, 1.0);

  return output;
}

fn tone_map_aces(v: vec3f) -> vec3f {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn ray_fragment_main(input: RayFragmentInput) -> @location(0) vec4f {
  let pixel_idx = u32(input.frag_pos.x) + u32(input.frag_pos.y) * u32(scene.canvas_width);
  var rng = init_rng(pixel_idx + scene.timestamp * 719393u);

  let grid_n = scene.stratified_grid_n;
  let spp = scene.stratified_grid_n * scene.stratified_grid_n;
  let grid_step = 1.0 / f32(grid_n);

  var total_color = vec3f(0.0, 0.0, 0.0);
  var hit: Hit;

  for (var sample = 0u; sample < spp; sample++) {
    let cell_x = sample % grid_n;
    let cell_y = sample / grid_n;

    let cell_start_x = f32(cell_x) * grid_step;
    let cell_start_y = f32(cell_y) * grid_step;
    let off_x = cell_start_x + rand(&rng) * grid_step;
    let off_y = cell_start_y + rand(&rng) * grid_step;

    let coord = vec2f((input.frag_pos.x + off_x) / scene.canvas_width, 1.0 - (input.frag_pos.y + off_y) / scene.canvas_height);
    var ray = ray_at(coord, scene.camera);

    var throughput = vec3f(1.0);
    var sample_color = vec3f(0.0);

    for (var depth = 0u; depth < scene.max_ray_depth; depth++) {
      if (ray_trace(ray, MAX_DISTANCE, false, 0xffffffffu, &hit) == true) {
        let mesh = meshes[hit.mesh_idx];
        let mat = materials[mesh.material_idx];

        if (mat.emission_strength > 0.0) {
          if (depth == 0u) {
            sample_color += mat.albedo * throughput;
          }
          break; 
        }

        // IS probabilities
        // high specular sampling for metals 
        // non-metals rely on roughness.
        let specular_weight = mix(1.0 - mat.roughness, 1.0, mat.metalness);
        let p_spec = clamp(specular_weight, 0.1, 0.9);
        let p_diff = 1.0 - p_spec;

        // direct lighting
        sample_color += shade_rt(hit, ray.direction, &rng).xyz * throughput;

        var position: vec3f;
        var world_normal: vec3f;
        get_hit_vectors(hit, &position, &world_normal);

        let wo = normalize(-ray.direction);

        // hemisphere ref frame
        var helper = vec3f(1.0, 0.0, 0.0);
        if (abs(world_normal.x) > 0.999) {
          helper = vec3f(0.0, 1.0, 0.0);
        }
        let tangent = normalize(cross(helper, world_normal));
        let bitangent = cross(world_normal, tangent);

        var bounce_dir: vec3f;
        let u1 = rand(&rng);
        let u2 = rand(&rng);

        // select specular or diffuse bounce
        if (rand(&rng) < p_spec) {
          // sample specular (ggx NDF)
          let alpha = sqr(mat.roughness);
          let wh_ts = sample_ggx_ndf(alpha, vec2f(u1, u2));
          let wh = normalize(
            wh_ts.x * tangent +
            wh_ts.y * bitangent +
            wh_ts.z * world_normal
          );
          bounce_dir = reflect(-wo, wh);
        } else {
          // sample diffuse (cosine pdf)
          let sqrt_v = sqrt(u2);
          let cosine_dir = vec3f(cos(2.0 * PI * u1) * sqrt_v, sin(2.0 * PI * u1) * sqrt_v, sqrt(1.0 - u2));
          bounce_dir = normalize(cosine_dir.x * tangent + cosine_dir.y * bitangent + cosine_dir.z * world_normal);
        }

        let n_dot_i = max(0.0, dot(world_normal, bounce_dir));
        
        // calculate BRDF pdf based on the IS probabilities above
        let pdf_brdf = evaluate_brdf_pdf(bounce_dir, wo, world_normal, mat.roughness, p_spec, p_diff);
        if (pdf_brdf <= 0.0 || n_dot_i <= 0.0) { break; }

        let fr = brdf(bounce_dir, wo, world_normal, mat.albedo, mat.roughness, mat.metalness);
        
        // throughput calculation = brdf * cos / pdf
        throughput *= (fr * n_dot_i) / pdf_brdf;

        const BOUNCE_RAY_BIAS = 0.0001;
        ray.origin = position + BOUNCE_RAY_BIAS * world_normal;
        ray.direction = bounce_dir;

        // russian roulette
        if (depth > 2u) {
          var p = max(throughput.x, max(throughput.y, throughput.z));
          p = min(p, 0.99); 
          if (rand(&rng) > p) { break; }
          throughput /= p; 
        }
      } else {
        break; // ray missed the scene
      }
    }

    sample_color = min(sample_color, vec3f(10.0));
    total_color += sample_color;
  }

  let tex_coord = vec2<u32>(input.frag_pos.xy);
  var accumulated = total_color / f32(spp);

  if (scene.accumulation_enabled != 0u) {
    if (scene.frame_count > 1) {
      let prev_color = textureLoad(accumulation_prev, tex_coord).xyz;
      let f = f32(scene.frame_count);

      accumulated = (1.0 / f) * accumulated + ((f - 1.0) / f) * prev_color;
    }

    textureStore(accumulation_next, tex_coord, vec4f(accumulated, 1.0));
  }

  return vec4f(select(accumulated, tone_map_aces(accumulated), scene.tone_mapping != 0), 1.0);
}

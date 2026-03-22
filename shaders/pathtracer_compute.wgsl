const PI = 3.14159265358979323846;
const INV_PI = 1.0 / 3.14159265358979323846;
const EPSILON = 1e-6;
const MAX_DISTANCE = 1e8;

// ----------------------------------------------------------------------------
// scene / geometry / material structs
// ----------------------------------------------------------------------------

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
  _pad: vec2<f32>,
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

// ----------------------------------------------------------------------------
// ray tracing structs
// ----------------------------------------------------------------------------

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
}

struct Hit {
  mesh_idx: u32,
  tri_idx: u32,
  u: f32, // barycentric coordinates of the intersection
  v: f32,
  t: f32, // distance to ray's origin of the intersection
}

// ----------------------------------------------------------------------------
// ReSTIR structs
// ----------------------------------------------------------------------------

struct LightCandidate {
  emissive_idx: u32,
  sample_uv: vec2<f32>, // barycentrics u,v ; w = 1-u-v
  light_point: vec3<f32>,
  area: f32,
  light_normal: vec3<f32>,
  p_source: f32, // source PDF (in area measure) -> p(x) on the paper
}

struct LightEval {
  f_unshadowed: vec3<f32>, // fr * Le * G (without visibility)
  p_hat: f32, // scalar target used by RIS
}

struct Reservoir {
  sample: u32,
  m: u32, // samples seen so far
  sample_uv: vec2<f32>, // barycentric coords of chosen triangle. u, v, w = 1-u-v
  w_sum: f32,
  final_w: f32, // W = (1 / p_hat) * (w_sum / M) for the final sample
  p_hat: f32, // p_hat of chosen sample at current receiving surface
  valid: u32,
}

// ----------------------------------------------------------------------------
// screen-space buffers for ReSTIR
// ----------------------------------------------------------------------------

// one primary visible surface per pixel
struct PrimarySurface {
  pos: vec3<f32>,
  cam_dist: f32,
  normal: vec3<f32>,
  tri_idx: u32,
  sampling_offsets: vec2<f32>,
  mesh_idx: u32,
  valid: u32,
}

// ----------------------------------------------------------------------------
// bindings
// ----------------------------------------------------------------------------

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

@group(2) @binding(0)
var<storage, read_write> primary_surfaces_curr: array<PrimarySurface>;

@group(2) @binding(1)
var<storage, read_write> primary_surfaces_prev: array<PrimarySurface>;

@group(2) @binding(2)
var<storage, read_write> reservoirs_curr: array<Reservoir>;

@group(2) @binding(3)
var<storage, read_write> reservoirs_prev: array<Reservoir>;

@group(2) @binding(4)
var pathtrace_output: texture_storage_2d<rgba32float, write>;

// ----------------------------------------------------------------------------
// helper functions
// ----------------------------------------------------------------------------

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

fn luminance(c: vec3f) -> f32 {
  return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn pixel_index(pixel: vec2<u32>) -> u32 {
  return pixel.x + pixel.y * u32(scene.canvas_width);
}

// ----------------------------------------------------------------------------
// RNGs
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// BRDF functions
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// ray tracing
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// ReSTIR
// ----------------------------------------------------------------------------

// ---------------------- reservoir related helpers ----------------------

fn invalid_primary_surface() -> PrimarySurface {
  return PrimarySurface(vec3f(0.0), 0.0, vec3f(0.0), 0u, vec2f(0.0), 0u, 0u);
}

fn invalid_reservoir() -> Reservoir {
  return Reservoir(0u, 0u, vec2f(0.0), 0.0, 0.0, 0.0, 0u);
}

fn primary_surface_valid(s: PrimarySurface) -> bool {
  return s.valid != 0u;
}

fn reservoir_valid(r: Reservoir) -> bool {
  return r.valid != 0u && r.m > 0u;
}

fn update_reservoir(
  res: ptr<function, Reservoir>, // the reservoir to update
  sample: u32, // new sample index
  sample_uv: vec2f,
  wi: f32, // weight associated to the sample
  mi: u32, // amount of samples this update contributes (1 for a normal update)
  p_hat_i: f32, // evaluation of p_hat at the point
  u: f32 // uniform random variable
) {
  (*res).m += mi;
  (*res).w_sum += wi;

  if (wi > 0.0) {
    let choosing_prob = wi / (*res).w_sum;
    if (u < choosing_prob) {
      (*res).sample = sample;
      (*res).sample_uv = sample_uv;
      (*res).p_hat = p_hat_i;
      (*res).valid = 1u;
    }
  }
}

// ----------------------- sampling -----------------------

fn sample_triangle_uv(s: vec2f) -> vec2f {
  // uniform barycentric sampling
  let sqrt_r1 = sqrt(s.x);
  let u = 1.0 - sqrt_r1;
  let v = s.y * sqrt_r1;
  return vec2f(u, v);
}

// get geometric information from a triangle from its idx and barycentric coords
fn reconstruct_triangle_sample(
  tri_idx: u32,
  mesh_idx: u32,
  sample_uv: vec2f,
  pos: ptr<function, vec3f>,
  normal: ptr<function, vec3f>,
  area: ptr<function, f32>
) {
  let mesh = meshes[mesh_idx];
  let tri = get_triangle(tri_idx);

  let p0 = get_vert_pos(mesh.pos_offset + tri.x);
  let p1 = get_vert_pos(mesh.pos_offset + tri.y);
  let p2 = get_vert_pos(mesh.pos_offset + tri.z);

  let u = sample_uv.x;
  let v = sample_uv.y;
  let w = 1.0 - u - v;

  *pos = u * p0 + v * p1 + w * p2;

  let edge1 = p1 - p0;
  let edge2 = p2 - p0;
  let n = cross(edge1, edge2);

  *area = 0.5 * length(n);
  *normal = normalize(n);
}

fn reconstruct_light_sample(
  emissive_idx: u32,
  sample_uv: vec2f,
  pos: ptr<function, vec3f>,
  normal: ptr<function, vec3f>,
  area: ptr<function, f32>
) {
  let emissive_tri = emissive_triangles[emissive_idx];
  reconstruct_triangle_sample(
    emissive_tri.tri_idx, emissive_tri.mesh_idx, sample_uv, pos, normal, area
  );
}

fn candidate_from_reservoir(r: Reservoir) -> LightCandidate {
  var light_point: vec3f;
  var light_normal: vec3f;
  var area: f32;
  reconstruct_light_sample(r.sample, r.sample_uv, &light_point, &light_normal, &area);

  return LightCandidate(r.sample, r.sample_uv, light_point, area, light_normal, 0.0);
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

fn sample_light_candidate_uniform(rng: ptr<function, RngState>) -> LightCandidate {
  let emissive_idx = sample_light_uniform(rand(rng));
  let sample_uv = sample_triangle_uv(vec2f(rand(rng), rand(rng)));

  var light_point: vec3f;
  var light_normal: vec3f;
  var area: f32;
  reconstruct_light_sample(emissive_idx, sample_uv, &light_point, &light_normal, &area);

  let p_source = uniform_light_sample_pdf(area);
  return LightCandidate(emissive_idx, sample_uv, light_point, area, light_normal, p_source);
}

fn sample_light_ris(
  position: vec3f,
  wo: vec3f,
  normal: vec3f,
  mat: Material,
  rng: ptr<function, RngState>
) -> Reservoir {
  const M = 32u; // number of candidate samples from source distribution
  var res = invalid_reservoir();

  for (var i = 0u; i < M; i++) {
    let candidate = sample_light_candidate_uniform(rng);
    let eval = evaluate_light_candidate(position, normal, wo, mat, candidate);

    let w = eval.p_hat / candidate.p_source;
    update_reservoir(
      &res, candidate.emissive_idx, candidate.sample_uv, w, 1u, eval.p_hat, rand(rng)
    );
  }

  if (res.p_hat <= 0.0) {
    res.final_w = 0.0;
    return res;
  }
  res.final_w = res.w_sum / (f32(res.m) * res.p_hat);

  return res;
}

// ----------------------- shading -----------------------

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

fn shade_rt_local(hit: Hit, incoming_ray_dir: vec3f, rng: ptr<function, RngState>) -> vec4f {
  let mesh = meshes[hit.mesh_idx];
  let mat = materials[mesh.material_idx];

  var position: vec3f;
  var world_normal: vec3f;
  get_hit_vectors(hit, &position, &world_normal);

  var color_response = vec3f(0.0);
  let wo = normalize(-incoming_ray_dir);

  if (scene.num_emissive_triangles == 0u) { return vec4f(0.0, 0.0, 0.0, 1.0); }

  if (scene.restir_enabled != 0u && scene.use_streaming_ris_on_bounces != 0) {
    let res = sample_light_ris(position, wo, world_normal, mat, rng);

    if (reservoir_valid(res)) {
      let candidate = candidate_from_reservoir(res);
      if (trace_light_visibility(position, world_normal, candidate)) {
        let eval = evaluate_light_candidate(position, world_normal, wo, mat, candidate);
        color_response += eval.f_unshadowed * res.final_w;
      }
    }
  } else {
    let candidate = sample_light_candidate_uniform(rng);
    let eval = evaluate_light_candidate(position, world_normal, wo, mat, candidate);

    if (trace_light_visibility(position, world_normal, candidate)) {
      color_response += eval.f_unshadowed / candidate.p_source;
    }
  }

  return vec4f(color_response, 1.0);
}

// -------------------------- primary visibility pass --------------------------

// traces a single primary camera ray and stores the first non-emissive surface
// hit; this is needed by the other steps to compute the reservoirs and reuses
@compute @workgroup_size(8, 8, 1)
fn visibility_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(scene.canvas_width);
  let height = u32(scene.canvas_height);

  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let pixel = vec2<u32>(gid.xy);
  let idx = pixel_index(pixel);

  // default to invalid
  primary_surfaces_curr[idx] = invalid_primary_surface();

  if (scene.restir_enabled == 0u) {
    return;
  }

  // 1 SPP for ReSTIR state
  var rng = init_rng(idx + scene.timestamp * 719393u);
  let off = vec2f(rand(&rng), rand(&rng));
  let uv = vec2f(
    (f32(pixel.x) + off.x) / scene.canvas_width,
    1.0 - (f32(pixel.y) + off.y) / scene.canvas_height
  );

  var ray = ray_at(uv, scene.camera);
  var hit: Hit;
  if (!ray_trace(ray, MAX_DISTANCE, false, 0xffffffffu, &hit)) {
    return;
  }

  let mesh = meshes[hit.mesh_idx];
  let mat = materials[mesh.material_idx];

  if (mat.emission_strength > 0.0) {
    return;
  }

  var position: vec3f;
  var world_normal: vec3f;
  get_hit_vectors(hit, &position, &world_normal);

  let cam_world_pos = scene.camera.inv_view_mat[3].xyz;
  let cam_dist = length(cam_world_pos - position);

  primary_surfaces_curr[idx] = PrimarySurface(
    position, cam_dist, normalize(world_normal), hit.tri_idx, off, hit.mesh_idx, 1u,
  );
}

// -------------------------- RIS pass --------------------------

// reads the primary surface buffer and generates one initial local RIS reservoir
// per pixel. this is only the local reservoir per pixel, no temporal/spatial reuse yet
@compute @workgroup_size(8, 8, 1)
fn initial_ris_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(scene.canvas_width);
  let height = u32(scene.canvas_height);

  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let pixel = vec2<u32>(gid.xy);
  let idx = pixel_index(pixel);

  reservoirs_curr[idx] = invalid_reservoir();

  if (scene.restir_enabled == 0u || scene.num_emissive_triangles == 0u) {
    return;
  }

  let surface = primary_surfaces_curr[idx];
  if (!primary_surface_valid(surface)) {
    return;
  }

  let position = surface.pos;
  let world_normal = surface.normal;
  let mat = materials[meshes[surface.mesh_idx].material_idx];

  let cam_world_pos = scene.camera.inv_view_mat[3].xyz;
  let wo = normalize(cam_world_pos - position);

  var rng = init_rng(idx + scene.timestamp * 719393u);

  let res = sample_light_ris(position, wo, world_normal, mat, &rng);
  reservoirs_curr[idx] = res;
}

// -------------------------- visibility reuse pass --------------------------

// rejects the initial reservoirs computed per pixel if the light is not visible
// from the surface point of the first hit
@compute @workgroup_size(8, 8, 1)
fn visibility_reuse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(scene.canvas_width);
  let height = u32(scene.canvas_height);

  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let pixel = vec2u(gid.xy);
  let idx = pixel_index(pixel);

  let surface = primary_surfaces_curr[idx];
  if (!primary_surface_valid(surface)) {
    reservoirs_curr[idx] = invalid_reservoir();
    return;
  }

  let stored_res = reservoirs_curr[idx];
  if (!reservoir_valid(stored_res)) {
    return;
  }

  let candidate = candidate_from_reservoir(stored_res);

  if (!trace_light_visibility(surface.pos, surface.normal, candidate)) {
    reservoirs_curr[idx] = invalid_reservoir();
  }
}

// -------------------------- temporal reuse pass --------------------------

// criteria to reject reuse candidates: less than 10% difference in camera distance
// and less than 25deg difference in the normals of the surfaces.
fn should_reject_reuse_neighbor(surface: PrimarySurface, neigh_surface: PrimarySurface) -> bool {
  if (!primary_surface_valid(neigh_surface)) {
    return true;
  }

  let dist_thresh = 0.1 * surface.cam_dist;
  let cos_normal_thresh = cos(0.4363); // cos of ~25 deg in rad

  let dist_diff = abs(surface.cam_dist - neigh_surface.cam_dist);
  let dot_normal = dot(surface.normal, neigh_surface.normal);

  if (dist_diff > dist_thresh || dot_normal < cos_normal_thresh) {
    return true;
  }

  return false;
}

// evaluate a reservoir p hat at a surface
fn reservoir_target_p_hat(surface: PrimarySurface, r: Reservoir) -> f32 {
  if (!reservoir_valid(r)) {
    return 0.0;
  }

  let candidate = candidate_from_reservoir(r);
  let mat = materials[meshes[surface.mesh_idx].material_idx];
  let cam_world_pos = scene.camera.inv_view_mat[3].xyz;
  let wo = normalize(cam_world_pos - surface.pos);

  let eval = evaluate_light_candidate(surface.pos, surface.normal, wo, mat, candidate);

  return eval.p_hat;
}

fn combine_reservoirs_biased(
  res1: Reservoir,
  res2: Reservoir,
  clamp_res2_m: bool,
  rng: ptr<function, RngState>
) -> Reservoir {
  var combined = invalid_reservoir();

  if (reservoir_valid(res1)) {
    let w = res1.p_hat * res1.final_w * f32(res1.m);
    update_reservoir(
      &combined, res1.sample, res1.sample_uv, w, res1.m, res1.p_hat, rand(rng)
    );
  }

  if (reservoir_valid(res2)) {
    var combining_m = res2.m;

    if (clamp_res2_m) {
      let current_m_for_clamp = max(1u, select(0u, res1.m, reservoir_valid(res1)));
      combining_m = min(res2.m, 20u * current_m_for_clamp);
    }

    let w = res2.p_hat * res2.final_w * f32(combining_m);
    update_reservoir(
      &combined, res2.sample, res2.sample_uv, w, combining_m, res2.p_hat, rand(rng)
    );
  }

  if (combined.valid == 0u || combined.m == 0u || combined.p_hat <= 0.0) {
    return invalid_reservoir();
  }

  combined.final_w = combined.w_sum / (f32(combined.m) * combined.p_hat);
  return combined;
}

fn get_temporal_neighbor(
  surface: PrimarySurface,
  neigh_surface: ptr<function, PrimarySurface>,
  neigh_reservoir: ptr<function, Reservoir>,
) {
  *neigh_surface = invalid_primary_surface();
  *neigh_reservoir = invalid_reservoir();

  let pos = vec4(surface.pos, 1.0);
  let prev_clip= scene.camera.proj_mat * scene.camera.prev_view_mat * pos;

  if (prev_clip.w <= 0) {
    return;
  }

  let prev_ndc = prev_clip.xyz / prev_clip.w;

  if (abs(prev_ndc.x) > 1.0 || abs(prev_ndc.y) > 1.0) {
    return;
  }

  let prev_uv = vec2f(
    0.5 + 0.5 * prev_ndc.x,
    0.5 - 0.5 * prev_ndc.y,
  );

  let prev_pixel = vec2u(
    u32(prev_uv.x * scene.canvas_width),
    u32(prev_uv.y * scene.canvas_height)
  );
  let prev_idx = pixel_index(prev_pixel);

  *neigh_surface = primary_surfaces_prev[prev_idx];
  *neigh_reservoir = reservoirs_prev[prev_idx];
}

// combine reservoirs from temporal neighbors for each pixel.
// we use the previous camera view matrix and the current camera info to fetch what
// is the temporal neighbor for a pixel, and reuse its reservoir. use the same
// criteria to reject a neighbor as we do in the spatial reuse.
@compute @workgroup_size(8, 8, 1)
fn temporal_reuse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(scene.canvas_width);
  let height = u32(scene.canvas_height);

  if (gid.x >= width || gid.y >= height) {
    return;
  }

  // first frame, nothing to reuse yet
  if (scene.absolute_frame_count <= 1u) {
    return;
  }

  let idx = pixel_index(gid.xy);

  let surface = primary_surfaces_curr[idx];
  if (!primary_surface_valid(surface)) {
    reservoirs_curr[idx] = invalid_reservoir();
    return;
  }

  let cur_res = reservoirs_curr[idx];

  var prev_surface: PrimarySurface;
  var prev_res: Reservoir;
  get_temporal_neighbor(surface, &prev_surface, &prev_res);

  if (should_reject_reuse_neighbor(surface, prev_surface)) {
    return;
  }

  var rng = init_rng(idx + scene.timestamp * 719393u);

  prev_res.p_hat = reservoir_target_p_hat(surface, prev_res);
  var combined = combine_reservoirs_biased(cur_res, prev_res, true, &rng);

  // unbiased weight calculation
  if (scene.restir_biased == 0u) {
    if (!reservoir_valid(combined) || combined.p_hat <= 0.0) {
      reservoirs_curr[idx] = invalid_reservoir();
      return;
    }

    let final_candidate = candidate_from_reservoir(combined);
    var z = 0u;

    // if its occluded, its true p_hat is 0
    if (!trace_light_visibility(surface.pos, surface.normal, final_candidate)) {
      reservoirs_curr[idx] = invalid_reservoir();
      return;
    }

    if (reservoir_valid(cur_res) && combined.p_hat > 0.0) {
      z += cur_res.m;
    }

    if (reservoir_valid(prev_res)) { 
      let visible = trace_light_visibility(prev_surface.pos, prev_surface.normal, final_candidate);
      if (visible && reservoir_target_p_hat(prev_surface, combined) > 0.0) {
        let current_m_for_clamp = max(1u, select(0u, cur_res.m, reservoir_valid(cur_res)));
        let combining_m = min(prev_res.m, 20u * current_m_for_clamp);
        z += combining_m;
      }
    }

    if (z != 0u) {
      combined.final_w = combined.w_sum / (f32(z) * combined.p_hat);
    } else {
      combined = invalid_reservoir();
    }
  }

  reservoirs_curr[idx] = combined;
}

// -------------------------- spatial reuse pass --------------------------

// combine reservoirs from spatial neighbors for each pixel
// considers a given radius of the pixel to choose neighbors from, and chooses some of them
// at random. uses difference on camera distance and normals as criteria to reject samples
// resulting reservoirs from previous passes in reservoirs_prev. output in reservoirs_curr 
@compute @workgroup_size(8, 8, 1)
fn spatial_reuse_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(scene.canvas_width);
  let height = u32(scene.canvas_height);

  if (gid.x >= width || gid.y >= height) {
    return;
  }
  
  let pixel = vec2u(gid.xy);
  let idx = pixel_index(gid.xy);

  let surface = primary_surfaces_curr[idx];
  if (!primary_surface_valid(surface)) {
    reservoirs_curr[idx] = invalid_reservoir();
    return;
  }

  const radius = 30.0;
  let max_chosen = select(3u, 5u, scene.restir_biased != 0);
  var pixels_chosen: array<u32, 6>; // current + sampled neighbors
  var chosen_count = 1u;
  pixels_chosen[0u] = idx;

  var rng = init_rng(idx + scene.timestamp * 719393u);
  var combined = reservoirs_prev[idx];

  for (var i = 0u; i < max_chosen; i++) {
    // uniform disk sampling
    let r = radius * sqrt(rand(&rng));
    let theta = 2.0 * PI * rand(&rng);
    let offset_x = r * cos(theta);
    let offset_y = r * sin(theta);

    let neigh_coords = vec2i(
      i32(f32(pixel.x) + offset_x),
      i32(f32(pixel.y) + offset_y)
    );

    if (neigh_coords.x < 0 || neigh_coords.y < 0 ||
      neigh_coords.x >= i32(width) || neigh_coords.y >= i32(height)) {
      continue;
    }

    let neigh_pixel = vec2u(u32(neigh_coords.x), u32(neigh_coords.y));
    if (neigh_pixel.x == pixel.x && neigh_pixel.y == pixel.y) {
      continue;
    }

    let neigh_idx = pixel_index(neigh_pixel);
    let neigh_surface = primary_surfaces_curr[neigh_idx];

    if (should_reject_reuse_neighbor(surface, neigh_surface)) {
      continue;
    }

    pixels_chosen[chosen_count] = neigh_idx;
    chosen_count++;

    var neigh_res = reservoirs_prev[neigh_idx];
    let neigh_p_hat_on_current = reservoir_target_p_hat(surface, neigh_res);
    neigh_res.p_hat = neigh_p_hat_on_current;

    combined = combine_reservoirs_biased(combined, neigh_res, false, &rng);
  }

  // unbiased weight calculation
  // have to evaluate final reservoir in neighbors surfaces and add to z accordingly
  if (scene.restir_biased == 0u) {
    if (!reservoir_valid(combined) || combined.p_hat <= 0.0) {
      reservoirs_curr[idx] = invalid_reservoir();
      return;
    }

    let final_candidate = candidate_from_reservoir(combined);

    // if its occluded, its true p_hat is 0
    if (!trace_light_visibility(surface.pos, surface.normal, final_candidate)) {
      reservoirs_curr[idx] = invalid_reservoir();
      return;
    }

    var z = 0u;
    for (var i = 0u; i < chosen_count; i++) {
      let neigh_res = reservoirs_prev[pixels_chosen[i]];
      let neigh_surface = primary_surfaces_curr[pixels_chosen[i]];

      if (reservoir_valid(neigh_res)) { 
        let visible = trace_light_visibility(neigh_surface.pos, neigh_surface.normal, final_candidate);
        if (visible && reservoir_target_p_hat(neigh_surface, combined) > 0.0) {
          z += neigh_res.m;
        }
      }
    }

    if (z != 0u) {
      combined.final_w = combined.w_sum / (f32(z) * combined.p_hat);
    } else {
      combined = invalid_reservoir();
    }
  }

  reservoirs_curr[idx] = combined;
}

// -------------------------- path tracing pass --------------------------

// uses the reservoir generated in previous passes
fn shade_primary_from_reservoir(
  hit: Hit,
  incoming_ray_dir: vec3f,
  pixel_idx: u32
) -> vec3f {
  let r = reservoirs_curr[pixel_idx];
  if (!reservoir_valid(r)) {
    return vec3f(0.0);
  }

  let mesh = meshes[hit.mesh_idx];
  let mat = materials[mesh.material_idx];

  var position: vec3f;
  var world_normal: vec3f;
  get_hit_vectors(hit, &position, &world_normal);

  let wo = normalize(-incoming_ray_dir);
  let candidate = candidate_from_reservoir(r);

  if (!trace_light_visibility(position, world_normal, candidate)) {
    return vec3f(0.0);
  }

  let eval = evaluate_light_candidate(position, world_normal, wo, mat, candidate);
  return eval.f_unshadowed * r.final_w;
}

// standard path tracing routine, but using ReSTIR for the first ray hit light sampling 
// when using ReSTIR, the we assume that SPP = 1, and N = 1 (number of reservoirs per pixel)
@compute @workgroup_size(8, 8, 1)
fn shade_pathtrace_restir_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let width = u32(scene.canvas_width);
  let height = u32(scene.canvas_height);

  if (gid.x >= width || gid.y >= height) {
    return;
  }

  let pixel = vec2<u32>(gid.xy);
  let idx = pixel_index(pixel);
  var rng = init_rng(idx + scene.timestamp * 719393u);

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

    var off_x = cell_start_x + rand(&rng) * grid_step;
    var off_y = cell_start_y + rand(&rng) * grid_step;

    // sets the ray as the same direction as the ray sampled by restir
    // if spp > 1, this part is skipped (but the result will be biased)
    let primary_surface = primary_surfaces_curr[idx];
    if (scene.restir_enabled != 0 && primary_surface_valid(primary_surface) && spp == 1) {
      off_x = primary_surface.sampling_offsets.x;
      off_y = primary_surface.sampling_offsets.y;
    }

    let coord = vec2f(
      (f32(pixel.x) + off_x) / scene.canvas_width,
      1.0 - (f32(pixel.y) + off_y) / scene.canvas_height
    );
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

        // direct lighting:
        // - primary hit: use stored RIS reservoir from compute pass
        // - later bounces: use local estimator (either uniform light sampling or streaming RIS)
        if (depth == 0u && scene.restir_enabled != 0u && reservoir_valid(reservoirs_curr[idx])) {
          sample_color += shade_primary_from_reservoir(hit, ray.direction, idx) * throughput;
        } else {
          sample_color += shade_rt_local(hit, ray.direction, &rng).xyz * throughput;
        }

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

        // IS probabilities
        // high specular sampling for metals 
        // non-metals rely on roughness.
        let specular_weight = mix(1.0 - mat.roughness, 1.0, mat.metalness);
        let p_spec = clamp(specular_weight, 0.1, 0.9);
        let p_diff = 1.0 - p_spec;

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

  let final_color = total_color / f32(spp);
  textureStore(pathtrace_output, pixel, vec4f(final_color, 1.0));
}
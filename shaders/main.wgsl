const PI = 3.14159265358979323846;
const INV_SQRT_3_4 = 1.154700538;
const INV_PI = 1.0 / 3.14159265358979323846;
const EPSILON = 1e-6;

struct LightSource {
  position: vec3<f32>,
  intensity: f32,
  color: vec3<f32>,
  angle: f32,
  spot: vec3<f32>,
  ray_traced_shadows: u32,
}

struct Material {
  albedo: vec3<f32>,
  roughness: f32,
  metalness: f32,
  use_procedural_texture: u32,
  _pad: vec2<f32>,
}

struct Camera {
  model_mat: mat4x4<f32>,
  view_mat: mat4x4<f32>,
  inv_view_mat: mat4x4<f32>,
  trans_inv_view_mat: mat4x4<f32>,
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
}

struct Scene {
  camera: Camera,
  canvas_width: f32,
  canvas_height: f32,
  num_meshes: f32,
  num_lights: f32,
}

@group(0) @binding(0)
var<uniform> scene: Scene;

@group(0) @binding(1)
var<storage, read> positions: array<f32>; // packed positions for all meshes

@group(0) @binding(2)
var<storage, read> normals: array<f32>; // packed normals for all meshes

@group(0) @binding(3)
var<storage, read> triangles: array<u32>; // packed triangles for all meshes

@group(0) @binding(4)
var<storage, read> meshes: array<Mesh>;

@group(0) @binding(5)
var<storage, read> materials: array<Material>;

@group(0) @binding(6)
var<storage, read> light_sources: array<LightSource>;

@group(0) @binding(7)
var noise_texture: texture_2d<f32>;

@group(0) @binding(8)
var noise_sampler: sampler;

struct RasterVertexInput {
  @builtin(vertex_index) vertex_idx: u32,
  @builtin(instance_index) mesh_idx: u32
}

struct RasterVertexOutput {
  @builtin(position) builtin_pos: vec4f,
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) @interpolate(flat) material_idx: u32,
}

// ----------------------------- helper functions -----------------------------

fn get_vert_pos(vert_index: u32) -> vec3f {
  let idx = 3u * vert_index;
  return vec3f(positions[idx], positions[idx + 1u], positions[idx + 2u]);
}

fn get_vert_normal(vert_index: u32) -> vec3f {
  let idx = 3u * vert_index;
  return vec3f(normals[idx], normals[idx + 1u], normals[idx + 2u]);
}

fn get_triangle(tri_index: u32) -> vec3u {
  let idx = 3u * tri_index;
  return vec3u(triangles[idx], triangles[idx + 1u], triangles[idx + 2u]);
}

fn sqr(x: f32) -> f32 { 
  return x * x; 
}

fn attenuation(dist: f32, cone_decay: f32) -> f32 {
  return cone_decay * (1.0 / sqr(dist));
}

// ----------------------------- noise functions ----------------------------- 

// from https://www.shadertoy.com/view/4djSRW
fn rand_dir_3d(p: vec3f) -> vec3f {
	var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);

  return -1.0 + 2.0 * fract((p3.xxy + p3.yxx) * p3.zyx);
}

fn gradient_eval_3d(corner: vec3f, p: vec3f) -> f32 {
  let dist = p - corner;
  let grad = rand_dir_3d(corner);

  return dot(dist, grad);
}
fn quintic_interpolation_3d(t: vec3f) -> vec3f {
  // 6t^5 - 15t^4 + 10t^3
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin_noise_3d(x: vec3f, freq: f32, amp: f32) -> f32 {
  let p = x * freq;

  let i = floor(p);
  let f = fract(p);

  // corners
  let i000 = i + vec3f(0.0, 0.0, 0.0);
  let i100 = i + vec3f(1.0, 0.0, 0.0);
  let i010 = i + vec3f(0.0, 1.0, 0.0);
  let i110 = i + vec3f(1.0, 1.0, 0.0);
  let i001 = i + vec3f(0.0, 0.0, 1.0);
  let i101 = i + vec3f(1.0, 0.0, 1.0);
  let i011 = i + vec3f(0.0, 1.0, 1.0);
  let i111 = i + vec3f(1.0, 1.0, 1.0);

  // gradients at the corners
  let n000 = gradient_eval_3d(i000, p);
  let n100 = gradient_eval_3d(i100, p);
  let n010 = gradient_eval_3d(i010, p);
  let n110 = gradient_eval_3d(i110, p);
  let n001 = gradient_eval_3d(i001, p);
  let n101 = gradient_eval_3d(i101, p);
  let n011 = gradient_eval_3d(i011, p);
  let n111 = gradient_eval_3d(i111, p);

  // interpolation
  let qi = quintic_interpolation_3d(f);

  // along x axis
  let nx00 = mix(n000, n100, qi.x);
  let nx01 = mix(n001, n101, qi.x);
  let nx10 = mix(n010, n110, qi.x);
  let nx11 = mix(n011, n111, qi.x);

  // along y axis
  let ny0 = mix(nx00, nx10, qi.y);
  let ny1 = mix(nx01, nx11, qi.y);

  let noise_val = mix(ny0, ny1, qi.z);

  // originally in range [-sqrt(0.75), sqrt(0.75)]
  let scaled = noise_val * INV_SQRT_3_4;

  // apply amplitude, clamp to [-1,1]
  return clamp(scaled * amp, -1.0, 1.0);
}

fn fbm_perlin_noise_3d(x: vec3f, octaves: u32, initial_freq: f32, initial_amp: f32) -> f32 {
  var total = 0.0;
  var cur_freq = initial_freq;
  var cur_amp = initial_amp;

  for (var i: u32 = 0u; i < octaves; i++) {
    total += perlin_noise_3d(x, cur_freq, cur_amp);
    cur_freq *= 2.0;
    cur_amp /= 2.0;
  }

  return total;
}

// ----------------------------- BRDF functions -----------------------------
fn trowbridge_reitz_ndf(wh: vec3f, n: vec3f, roughness: f32) -> f32 {
  let alpha2 = sqr(roughness);

  return alpha2 / (PI * sqr(1.0 + (alpha2 - 1.0) * sqr(dot(n, wh))));
}

fn schlick_fresnel(wi: vec3f, wh: vec3f, f0: vec3f) -> vec3f {
  return f0 + (1.0 - f0) * pow(1.0 - max(0.0, dot(wi, wh)), 5.0);
}

fn smith_g1(w: vec3f, n: vec3f, roughness: f32) -> f32 {
  let n_dot_w = dot(n, w);
  let alpha2 = sqr(roughness);

  return (2.0 * n_dot_w) / (n_dot_w + sqrt(alpha2 + (1.0 - alpha2) * sqr(n_dot_w)));
}

fn smith_ggx(wi: vec3f, wo: vec3f, n: vec3f, roughness: f32) -> f32 {
  return smith_g1(wi, n, roughness) * smith_g1(wo, n, roughness);
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

  if (n_dot_l <= 0.0) { // not in the reflection hemisphere
      return vec3f(0.0); 
  }

  let wh = normalize(wi + wo);
  let n_dot_h = max(0.0, dot(n, wh));
  let v_dot_h = max(0.0, dot(wo, wh));

  // normal distribution function (ggx)
  let d = trowbridge_reitz_ndf(wh, n, alpha);

  // Schlick approximation to the Fresnel term
  let f = schlick_fresnel(wi, wh, specular_color);

  // masking-shadowing term
  let g = smith_ggx(wi, wo, n, alpha);

  let f_d = diffuse_color * (vec3f(1.0) - specular_color) / PI;
  let f_s = f * d * g / 4.0;

  return (f_d + f_s);
}

// operate in view space i.e., in the local frame of the camera
fn light_shade(
  position: vec3f, 
  normal: vec3f, 
  material_idx: u32, 
  light_source_idx: u32, 
  wo: vec3f
) -> vec3f {
  let light = light_sources[light_source_idx];
  let cam = scene.camera;

  let view_light_pos = cam.view_mat * vec4f(light.position, 1.0);
  let view_light_target = cam.view_mat * vec4f(light.spot, 1.0);
  let view_light_dir = normalize(view_light_target.xyz - view_light_pos.xyz);

  var wi = view_light_pos.xyz - position;
  let di = length(wi);

  wi = normalize(wi);
  let spot_cone_decay = dot(-wi, view_light_dir) - light.angle;

  if (spot_cone_decay <= 0.0) {
    return vec3f(0.0); // out of spot light cone
  }

  let att = attenuation(di, spot_cone_decay);
  let ir = light.color * light.intensity * att;
  var m = materials[material_idx];

  if (material_idx == 3) {
    let world_pos = (cam.inv_view_mat * vec4f(position, 1.0)).xyz;
    // let noise = fbm_perlin_noise_3d(world_pos, 12u, 10.0, 1.0);
    let noise = textureSample(noise_texture, noise_sampler, world_pos.xy).r;
    let t = 0.5 + 0.5 * noise;

    let decayed_albedo = m.albedo * 0.6;
    m.albedo = mix(m.albedo, decayed_albedo, t);
    m.roughness = t;
    m.metalness = t;
  }

  let fr = brdf(wi, wo, normal, m.albedo, m.roughness, m.metalness);
  let color_response = ir * fr * max(0.0, dot(wi, normal));

  return color_response;
}

fn compute_radiance(
  position: vec3f, 
  normal: vec3f, 
  material_idx: u32, 
  wo: vec3f
) -> vec3f {
  var color_response = vec3f(0.0);
  let num_lights = u32(scene.num_lights);

  for (var light_source_idx = 0u; light_source_idx < num_lights; light_source_idx++) {
    color_response += light_shade(position, normal, material_idx, light_source_idx, wo);
  }

  return color_response;
}

// ----------------------------- rasterization shaders ----------------------------- 

@vertex
fn raster_vertex_main(input: RasterVertexInput) -> RasterVertexOutput {
  let cam = scene.camera;
  let mesh = meshes[input.mesh_idx];

  let tri_index = input.vertex_idx / 3u;
  let tri_vert_index = input.vertex_idx % 3u;
  let triangle = get_triangle(mesh.tri_offset + tri_index);
  let vert_index = mesh.pos_offset + triangle[tri_vert_index];

  var output: RasterVertexOutput;

  let p = cam.view_mat * cam.model_mat * vec4f(get_vert_pos(vert_index), 1.0); 
  output.builtin_pos = cam.proj_mat * p; // to fire rasterization
  output.position = p.xyz;

  let n = cam.trans_inv_view_mat * vec4f(get_vert_normal(vert_index), 1.0);
  output.normal = normalize(n.xyz);
  output.material_idx = mesh.material_idx; 

  return output; 
}

@fragment
fn raster_fragment_main(input: RasterVertexOutput) -> @location(0) vec4f {
  let position = input.position;
  let normal = normalize(input.normal);
  let wo = normalize(-position);

  let color_response = compute_radiance(position, normal, input.material_idx, wo);

  return vec4f(color_response, 1.0);
}

// ---------------------------- ray tracing shaders ---------------------------- 

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
  tri_index: u32,
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
  ray.direction = normalize(view_dir + ((uv.x - 0.5) * camera.aspect_ratio * w) * view_right + ((uv.y) - 0.5) * w * view_up);  

  return ray;
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
  max_distance: f32, // ignore intersections found further away
  any_hit: bool,     // return as soon as an intersection is found if true
  hit: ptr<function, Hit> // filled only if an intersection is found and any_hit is false
) -> bool {
  var intersection_found = false;
  let num_meshes = u32(scene.num_meshes);

  for (var mesh_idx = 0u; mesh_idx < num_meshes; mesh_idx++) {
    let mesh = meshes[mesh_idx];
      
    for (var tri_index = 0u; tri_index < mesh.num_triangles; tri_index++) {
      let triangle = get_triangle(mesh.tri_offset + tri_index);
      
      var tri_hit: Hit;
      tri_hit.mesh_idx = mesh_idx;
      tri_hit.tri_index = tri_index;
      
      let p0 = get_vert_pos(mesh.pos_offset + triangle.x);
      let p1 = get_vert_pos(mesh.pos_offset + triangle.y);
      let p2 = get_vert_pos(mesh.pos_offset + triangle.z);
      
      if (intersect_triangle(ray, p0, p1, p2, true, 0.0, max_distance, &tri_hit) == true) {
        if (!intersection_found || (intersection_found && tri_hit.t < (*hit).t)) {
          if (any_hit == true) {
            return true;
          }

          *hit = tri_hit;
          intersection_found = true;
        }
      }
    }
  }
  
  return intersection_found;
}

fn shade_rt(hit: Hit) -> vec4f {
  let cam = scene.camera;
  let mesh = meshes[hit.mesh_idx];
  let tri = get_triangle(mesh.tri_offset + hit.tri_index);
  let uvw = vec3f(hit.u, hit.v, 1.0 - hit.u - hit.v);

  let position = interpolate(
    get_vert_pos(mesh.pos_offset + tri.x), 
    get_vert_pos(mesh.pos_offset + tri.y), 
    get_vert_pos(mesh.pos_offset + tri.z), 
    uvw
  );

  var normal = normalize(interpolate(
    get_vert_normal(mesh.pos_offset + tri.x), 
    get_vert_normal(mesh.pos_offset + tri.y), 
    get_vert_normal(mesh.pos_offset + tri.z), 
    uvw
  ));

  normal = (cam.trans_inv_view_mat * vec4f(normal, 1.0)).xyz;

  var color_response = vec3f(0.0);
  let view_position = (cam.view_mat * cam.model_mat * vec4f(position, 1.0)).xyz;
  let wo = normalize(-view_position);
  let num_lights = u32(scene.num_lights);

  for (var light_source_idx = 0u; light_source_idx < num_lights; light_source_idx++) {
    let l = light_sources[light_source_idx];

    if (bool(l.ray_traced_shadows) == true) {
      var shadow_ray: Ray;
      const SHADOW_BIAS = 0.0001;

      let pos_to_light = l.position - position;
      let light_dist = length(pos_to_light);

      shadow_ray.direction = normalize(pos_to_light);
      shadow_ray.origin = position + SHADOW_BIAS * normal;

      var shadow_hit: Hit;
      let in_shadow = ray_trace(shadow_ray, light_dist + EPSILON, true, &shadow_hit); // any hit

      if (in_shadow == false || (in_shadow == true && shadow_hit.t > light_dist)) {
        color_response += light_shade(view_position, normal, mesh.material_idx, light_source_idx, wo); 
      }
    } else {
      color_response += light_shade(view_position, normal, mesh.material_idx, light_source_idx, wo);
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

@fragment
fn ray_fragment_main(input: RayFragmentInput) -> @location(0) vec4f {
  const MAX_DISTANCE = 1e8;

  let coord = vec2f(input.frag_pos.x / scene.canvas_width, 1.0 - input.frag_pos.y / scene.canvas_height);
  let ray = ray_at(coord, scene.camera);

  var color_response = vec4f(0.0, 0.0, 0.0, 1.0);
  var hit: Hit;

  if (ray_trace(ray, MAX_DISTANCE, false, &hit) == true) {
    color_response = shade_rt(hit);
  }

  return color_response;
}


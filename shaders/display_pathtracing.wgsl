struct Camera {
  model_mat: mat4x4<f32>,
  view_mat: mat4x4<f32>,
  inv_view_mat: mat4x4<f32>,
  trans_inv_model_mat: mat4x4<f32>,
  proj_mat: mat4x4<f32>,
  fov: f32,
  aspect_ratio: f32,
  _pad: vec2<f32>,
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

@group(1) @binding(0)
var pathtrace_output: texture_storage_2d<rgba32float, read>;

@group(1) @binding(1)
var accumulation_prev: texture_storage_2d<rgba32float, read>;

@group(1) @binding(2)
var accumulation_next: texture_storage_2d<rgba32float, write>;

@vertex
fn present_vertex_main(@builtin(vertex_index) vertex_idx: u32) -> @builtin(position) vec4f {
  const SCREEN_POS = array<vec2<f32>, 6>(
    vec2f(-1.0, -1.0),
    vec2f( 1.0, -1.0),
    vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0),
    vec2f( 1.0, -1.0),
    vec2f( 1.0,  1.0),
  );

  return vec4f(SCREEN_POS[vertex_idx], 0.0, 1.0);
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
fn present_fragment_main(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
  let tex_coord = vec2<u32>(u32(frag_pos.x), u32(frag_pos.y));

  let current_color = textureLoad(pathtrace_output, tex_coord).xyz;
  var accumulated = current_color;

  if (scene.accumulation_enabled != 0u) {
    if (scene.frame_count > 1u) {
      let prev_color = textureLoad(accumulation_prev, tex_coord).xyz;
      let f = f32(scene.frame_count);
      accumulated = (1.0 / f) * current_color + ((f - 1.0) / f) * prev_color;
    }

    textureStore(accumulation_next, tex_coord, vec4f(accumulated, 1.0));
  }

  let final_color = select(
    accumulated,
    tone_map_aces(accumulated),
    scene.tone_mapping != 0u
  );

  return vec4f(final_color, 1.0);
}
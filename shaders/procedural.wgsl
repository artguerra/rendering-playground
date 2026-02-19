const SQRT_2 = 1.41421356237;

struct NoiseParams {
  freq: f32,
  amp: f32,
  octaves: u32,
  _pad: f32,
};

@group(0) @binding(0)
var<uniform> params: NoiseParams;

@group(0) @binding(1)
var noise_output: texture_storage_2d<rgba16float, write>;

// from https://www.shadertoy.com/view/4djSRW
fn rand_dir_2d(p: vec2f) -> vec2f {
	var p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);

  return -1.0 + 2.0 * fract((p3.xx + p3.yz) * p3.zy);
}

fn gradient_eval(corner: vec2f, p: vec2f) -> f32 {
  let dist = p - corner;
  let grad = rand_dir_2d(corner);

  return dot(dist, grad);
}

fn quintic_interpolation(t: vec2f) -> vec2f {
  // 6t^5 - 15t^4 + 10t^3
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin_noise_2d(x: vec2f, freq: f32, amp: f32) -> f32 {
  let p = x * freq;

  let i = floor(p);
  let f = fract(p);

  // corners
  let i00 = i;
  let i01 = i + vec2f(0.0, 1.0);
  let i10 = i + vec2f(1.0, 0.0);
  let i11 = i + vec2f(1.0, 1.0);

  // gradients at the corners
  let n00 = gradient_eval(i00, p);
  let n01 = gradient_eval(i01, p);
  let n10 = gradient_eval(i10, p);
  let n11 = gradient_eval(i11, p);

  // interpolation
  let qi = quintic_interpolation(f);
  let nx0 = mix(n00, n10, qi.x);
  let nx1 = mix(n01, n11, qi.x);
  let noise_val = mix(nx0, nx1, qi.y);

  // originally in range [-sqrt(1/2), sqrt(1/2)]
  let scaled = noise_val * SQRT_2;

  // apply amplitude, clamp to [-1,1]
  return clamp(scaled * amp, -1.0, 1.0);
}

fn fbm_perlin_noise_2d(x: vec2f, octaves: u32, initial_freq: f32, initial_amp: f32) -> f32 {
  var total = 0.0;
  var cur_freq = initial_freq;
  var cur_amp = initial_amp;

  for (var i: u32 = 0u; i < octaves; i++) {
    total += perlin_noise_2d(x, cur_freq, cur_amp);
    cur_freq *= 2.0;
    cur_amp /= 2.0;
  }

  return total;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let dim = textureDimensions(noise_output);

  if (global_id.x >= dim.x || global_id.y >= dim.y) {
    return;
  }

  let uv = vec2f(global_id.xy) / vec2f(dim.xy);
  let noise = fbm_perlin_noise_2d(uv, params.octaves, params.freq, params.amp);
  
  textureStore(noise_output, global_id.xy, vec4f(noise, noise, noise, 1.0));
}


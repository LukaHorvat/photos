#version 300 es
// The develop pipeline, run per pixel: in the browser for the editor's preview
// (develop.js) and on the server for exports (gpu.py). develop.py prepares the
// inputs: linear camera RGB, the base-layer coefficients and the uniforms.
precision highp float;
in vec2 uv;
out vec4 color;

uniform sampler2D cam;        // linear camera RGB, black 0, sensor saturation 1
uniform sampler2D coeffs;     // base layer = a * log2(luminance) + b
uniform vec3 mul;             // white balance multipliers for the chosen temperature
uniform mat3 rgbCam;          // camera RGB -> linear sRGB
uniform float compress;       // dynamic range compression slider, 0..100
uniform float exposure;       // EV
uniform float saturation;     // slider, -100..100
uniform float maxCompression; // fraction of the base layer's contrast removed at 100
uniform float anchor;         // log2 scene mid-grey: compression pivots here
uniform float logFloor;
uniform vec3 curve;           // base tone curve: gain, contrast, shoulder

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 toneCurve(vec3 x) {
  // y = (1 + K) t^C / (t^C + K), t = gain * x; fitted to the camera's JPEGs.
  vec3 t = pow(max(x, 0.0) * curve.x, vec3(curve.y));
  return min((1.0 + curve.z) * t / (t + curve.z), 1.0);
}

vec3 oetf(vec3 x) {
  return mix(12.92 * x, 1.055 * pow(max(x, 0.0031308), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, x));
}

void main() {
  // 1. White balance. A channel at the sensor's ceiling, scaled up by white balance,
  // would turn highlights pink; clip every channel where the least-scaled one tops
  // out so they stay white. Negative values (shadow noise) are kept until the curve:
  // clipping them earlier would bias the shadows, purple once red and blue are scaled up.
  float clipLevel = min(mul.r, min(mul.g, mul.b));
  vec3 rgb = min(texture(cam, uv).rgb * mul, vec3(clipLevel));
  // 2. Camera RGB -> linear sRGB.
  rgb = rgbCam * rgb;
  // 3. Dynamic range compression: shrink the base layer's contrast around mid-grey,
  // keeping the detail on top of it.
  float strength = compress / 100.0 * maxCompression;
  if (strength > 0.0) {
    float l = log2(max(dot(rgb, LUMA), logFloor));
    vec2 ab = texture(coeffs, uv).rg;
    rgb *= exp2(strength * (anchor - (ab.x * l + ab.y)));
  }
  // 4. Exposure, after compression.
  rgb *= exp2(exposure);
  // 5. Saturation.
  float lum = dot(rgb, LUMA);
  rgb = lum + (rgb - lum) * (1.0 + saturation / 100.0);
  // 6. Base tone curve and sRGB encoding.
  color = vec4(oetf(toneCurve(rgb)), 1.0);
}

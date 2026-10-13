#version 300 es
// The develop pipeline, run per pixel: in the browser for the editor's preview
// (develop.js) and on the server for exports (gpu.py). develop.py prepares the
// inputs: linear camera RGB, the base-layer coefficients, the camera's look (look.py)
// and the uniforms.
precision highp float;
precision highp sampler3D;
in vec2 uv;
out vec4 color;

uniform sampler2D cam;        // linear camera RGB, black 0, sensor saturation 1
uniform sampler2D coeffs;     // base layer = a * log2(luminance) + b
uniform sampler2D tone;       // this photo's tone curve: output luminance at even steps of log2 scene luminance
uniform sampler3D colour;     // camera colour profile: output colour (at luminance 1) by chromaticity and brightness
uniform sampler2D grain;      // exports only: luminance noise given back after noise reduction (denoise.py)
uniform vec3 mul;             // white balance multipliers for the chosen temperature
uniform mat3 rgbCam;          // camera RGB -> linear sRGB
uniform float compress;       // dynamic range compression slider, 0..100
uniform float exposure;       // EV
uniform float saturation;     // slider, -100..100
uniform float contrast;       // slider, -100..100
uniform float maxCompression; // fraction of the base layer's contrast removed at 100
uniform float anchor;         // log2 scene luminance the tone curve makes mid-grey: compression pivots here
uniform float logFloor;
uniform vec2 toneRange;       // log2 scene luminance of the first and last tone samples
uniform float toneSize;       // number of tone samples
uniform float shadowLoss;     // high ISO: share of colour the camera takes out of the darkest tones

const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
const float CONTRAST_MAX = 0.45;  // look.CONTRAST_MAX

float toneAt(float lum) {
  if (lum <= 0.0) return 0.0;
  float t = clamp((log2(lum) - toneRange.x) / (toneRange.y - toneRange.x), 0.0, 1.0);
  return texture(tone, vec2((t * (toneSize - 1.0) + 0.5) / toneSize, 0.5)).r;
}

float contrastCurve(float y) {
  // S-curve on roughly perceptual lightness; black and white stay put (look.contrast_curve).
  float s = contrast / 100.0 * CONTRAST_MAX;
  float p = pow(clamp(y, 0.0, 1.0), 1.0 / 2.4);
  return pow(clamp(p + s * (p - 0.5) * 4.0 * p * (1.0 - p), 0.0, 1.0), 2.4);
}

vec3 cameraColour(vec3 rgb, float y) {
  // Output colour at luminance 1, from the scene colour's chromaticity and the output
  // brightness (look.lookup): grid nodes at the ends of [-1, 1], levels centred in y^(1/2.4).
  float total = rgb.r + rgb.g + rgb.b;
  vec2 c = total > 1e-6 ? clamp(vec2(rgb.r - rgb.g, rgb.b - rgb.g) / total, -1.0, 1.0) : vec2(0.0);
  vec3 size = vec3(textureSize(colour, 0));
  vec2 uv = ((c + 1.0) * 0.5 * (size.xy - 1.0) + 0.5) / size.xy;
  vec3 ratio = texture(colour, vec3(uv, pow(clamp(y, 0.0, 1.0), 1.0 / 2.4))).rgb;
  // At high ISO the camera takes colour out of dark tones.
  float lightness = y > 216.0 / 24389.0 ? 116.0 * pow(y, 1.0 / 3.0) - 16.0 : 24389.0 / 27.0 * y;
  return 1.0 + (ratio - 1.0) * (1.0 - shadowLoss * exp(-lightness / 15.0));
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
  // 2. Camera RGB -> linear sRGB, plus the grain noise reduction gives back.
  rgb = rgbCam * rgb + texture(grain, uv).r;
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
  // 6. The camera's look: this photo's tone curve on luminance, contrast, then the camera's
  // colour for that colour and brightness; output channels clip at white like the camera's.
  float y = contrastCurve(toneAt(dot(rgb, LUMA)));
  color = vec4(oetf(clamp(cameraColour(rgb, y) * y, 0.0, 1.0)), 1.0);
}

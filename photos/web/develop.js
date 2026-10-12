'use strict';

// WebGL version of develop.py's to_srgb, so edits preview instantly. The server sends
// linear camera RGB and the base-layer coefficients once per photo; everything that
// depends on the sliders happens here. Keep in sync with develop.py.

const VERTEX_SHADER = `#version 300 es
in vec2 pos;
out vec2 uv;
void main() {
  uv = vec2(pos.x * 0.5 + 0.5, 0.5 - pos.y * 0.5);  // row 0 of the textures is the top
  gl_Position = vec4(pos, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 color;
uniform sampler2D cam;     // linear camera RGB, black 0, sensor saturation 1
uniform sampler2D coeffs;  // base layer = a * log2(luminance) + b
uniform vec3 mul;          // white balance multipliers
uniform float clipLevel;
uniform mat3 rgbCam;       // camera RGB -> linear sRGB
uniform float strength;    // dynamic range compression, 0..MAX_COMPRESSION
uniform float anchor;
uniform float logFloor;
uniform float exposure;    // as a factor
uniform float saturation;  // as a factor
uniform vec3 curve;        // gain, contrast, shoulder
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 toneCurve(vec3 x) {
  vec3 t = pow(max(x, 0.0) * curve.x, vec3(curve.y));
  return min((1.0 + curve.z) * t / (t + curve.z), 1.0);
}

vec3 oetf(vec3 x) {
  return mix(12.92 * x, 1.055 * pow(max(x, 0.0031308), vec3(1.0 / 2.4)) - 0.055, step(0.0031308, x));
}

void main() {
  vec3 rgb = min(texture(cam, uv).rgb * mul, vec3(clipLevel));
  rgb = rgbCam * rgb;
  if (strength > 0.0) {
    float l = log2(max(dot(rgb, LUMA), logFloor));
    vec2 ab = texture(coeffs, uv).rg;
    rgb *= exp2(strength * (anchor - (ab.x * l + ab.y)));
  }
  rgb *= exposure;
  float lum = dot(rgb, LUMA);
  rgb = lum + (rgb - lum) * saturation;
  color = vec4(oetf(toneCurve(rgb)), 1.0);
}`;

function planckXY(temp) {
  // CIE xy of a black body (Kim et al. 2002), as develop.planck_xy.
  const t = Math.min(Math.max(temp, 1667), 25000);
  const x = t <= 4000
    ? -0.2661239e9 / t ** 3 - 0.2343589e6 / t ** 2 + 0.8776956e3 / t + 0.179910
    : -3.0258469e9 / t ** 3 + 2.1070379e6 / t ** 2 + 0.2226347e3 / t + 0.240390;
  let y;
  if (t <= 2222) y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683;
  else if (t <= 4000) y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867;
  else y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483;
  return [x, y];
}

function wbMultipliers(info, temp) {
  // As develop.Color.multipliers.
  if (temp == null) return info.asShot;
  const [x, y] = planckXY(temp);
  const xyz = [x / y, 1, (1 - x - y) / y];
  const mul = info.camXyz.map((row) => 1 / (row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]));
  return mul.map((m, i) => (m / mul[1]) * info.tint[i]);
}

function parsePreview(buffer) {
  // Layout written by develop.preview.
  const n = new DataView(buffer).getUint32(0, true);
  const info = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
  const camLength = info.width * info.height * 3;
  info.cam = new Uint16Array(buffer, 4 + n, camLength);
  info.coeffs = new Uint16Array(buffer, 4 + n + camLength * 2, info.coeffWidth * info.coeffHeight * 2);
  return info;
}

class Developer {
  constructor(canvas, pipeline) {
    this.canvas = canvas;
    this.pipeline = pipeline;
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('This browser has no WebGL2');
    this.gl = gl;
    const program = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, VERTEX_SHADER], [gl.FRAGMENT_SHADER, FRAGMENT_SHADER]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(program, s);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    this.program = program;
    this.loc = {};
    for (const u of ['cam', 'coeffs', 'mul', 'clipLevel', 'rgbCam', 'strength', 'anchor', 'logFloor', 'exposure', 'saturation', 'curve']) {
      this.loc[u] = gl.getUniformLocation(program, u);
    }
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const pos = gl.getAttribLocation(program, 'pos');
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
    this.camTex = this.texture(0);
    this.coeffTex = this.texture(1);
    gl.uniform1i(this.loc.cam, 0);
    gl.uniform1i(this.loc.coeffs, 1);
    this.info = null;
  }

  texture(unit) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  load(info) {
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB16F, info.width, info.height, 0, gl.RGB, gl.HALF_FLOAT, info.cam);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.coeffTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, info.coeffWidth, info.coeffHeight, 0, gl.RG, gl.HALF_FLOAT, info.coeffs);
    this.info = info;
  }

  draw(settings) {
    const gl = this.gl;
    const p = this.pipeline;
    const info = this.info;
    if (!info) return;
    const mul = wbMultipliers(info, settings.temp);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.uniform3fv(this.loc.mul, mul);
    gl.uniform1f(this.loc.clipLevel, Math.min(...mul));
    gl.uniformMatrix3fv(this.loc.rgbCam, true, info.rgbCam.flat());
    gl.uniform1f(this.loc.strength, (settings.compress / 100) * p.maxCompression);
    gl.uniform1f(this.loc.anchor, p.anchor);
    gl.uniform1f(this.loc.logFloor, p.logFloor);
    gl.uniform1f(this.loc.exposure, 2 ** settings.exposure);
    gl.uniform1f(this.loc.saturation, 1 + settings.saturation / 100);
    gl.uniform3f(this.loc.curve, p.curveGain, p.curveC, p.curveK);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

'use strict';

// Runs the develop shader (develop.frag) in the browser, so edits preview instantly.
// The server runs the very same shader for exports (gpu.py). Per photo it sends linear
// camera RGB, the base-layer coefficients, the camera's look (tone curve and colour
// profile textures), the shader's fixed inputs and the white balance multipliers for
// every temperature; the sliders only pick values.

async function loadShaders() {
  const get = (name) => fetch(`/static/${name}?v=${S.version}`).then((r) => r.text());
  const [vertex, fragment] = await Promise.all([get('develop.vert'), get('develop.frag')]);
  return { vertex, fragment };
}

function parsePreview(buffer) {
  // Layout written by develop.preview.
  const n = new DataView(buffer).getUint32(0, true);
  const info = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, n)));
  let offset = 4 + n;
  const take = (length) => {
    const a = new Uint16Array(buffer, offset, length);
    offset += length * 2;
    return a;
  };
  info.cam = take(info.width * info.height * 3);
  info.coeffs = take(info.coeffWidth * info.coeffHeight * 2);
  info.tone = take(info.toneSize);
  info.table = take(info.tableSize[0] * info.tableSize[1] * info.tableSize[2] * 3);
  info.grainData = info.grain ? take(info.width * info.height) : new Uint16Array(1);
  return info;
}

class Developer {
  constructor(canvas, shaders) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('This browser has no WebGL2');
    this.gl = gl;
    const program = gl.createProgram();
    for (const [type, src] of [[gl.VERTEX_SHADER, shaders.vertex], [gl.FRAGMENT_SHADER, shaders.fragment]]) {
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
    this.locations = new Map();
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const pos = gl.getAttribLocation(program, 'pos');
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);
    this.camTex = this.texture(0);
    this.coeffTex = this.texture(1);
    this.toneTex = this.texture(2);
    this.colourTex = this.texture(3, gl.TEXTURE_3D);
    this.grainTex = this.texture(4);
    ['cam', 'coeffs', 'tone', 'colour', 'grain'].forEach((name, unit) => gl.uniform1i(gl.getUniformLocation(program, name), unit));
    this.info = null;
    this.full = null; // the 100% view's full-size camera RGB and grain (develop.Full), filled tile by tile
  }

  bind(cam, grain) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, cam);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, grain);
  }

  allocFull(width, height) {
    const gl = this.gl;
    if (this.full && this.full.width === width && this.full.height === height) return;
    if (this.full) {
      gl.deleteTexture(this.full.cam);
      gl.deleteTexture(this.full.grain);
    }
    // Sampled one texel per pixel, so nearest is exact, and never mixes in a tile not loaded yet.
    const cam = this.texture(0, gl.TEXTURE_2D, gl.NEAREST);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGB16F, width, height);
    const grain = this.texture(4, gl.TEXTURE_2D, gl.NEAREST);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R16F, width, height);
    this.full = { width, height, cam, grain };
    this.bind(this.camTex, this.grainTex);
  }

  putTile(x, y, w, h, data) {
    // data: the tile's camera RGB (w * h * 3 half floats), then its grain (w * h).
    const gl = this.gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.bind(this.full.cam, this.full.grain);
    gl.activeTexture(gl.TEXTURE0);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, w, h, gl.RGB, gl.HALF_FLOAT, data.subarray(0, w * h * 3));
    gl.activeTexture(gl.TEXTURE4);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, w, h, gl.RED, gl.HALF_FLOAT, data.subarray(w * h * 3));
    this.bind(this.camTex, this.grainTex);
  }

  texture(unit, target = this.gl.TEXTURE_2D, filter = this.gl.LINEAR) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
    const wraps = [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T].concat(target === gl.TEXTURE_3D ? [gl.TEXTURE_WRAP_R] : []);
    for (const wrap of wraps) {
      gl.texParameteri(target, wrap, gl.CLAMP_TO_EDGE);
    }
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
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.toneTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, info.toneSize, 1, 0, gl.RED, gl.HALF_FLOAT, info.tone);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_3D, this.colourTex);
    const [w, h, d] = info.tableSize;
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGB16F, w, h, d, 0, gl.RGB, gl.HALF_FLOAT, info.table);
    // Grain comes with the noise-reduced preview; the quick one has none to give back.
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.grainTex);
    const [gw, gh] = info.grain ? [info.width, info.height] : [1, 1];
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, gw, gh, 0, gl.RED, gl.HALF_FLOAT, info.grainData);
    this.info = info;
  }

  set(uniforms) {
    // By shape: a number is a float, 2 numbers a vec2, 3 a vec3, 9 a mat3 (column-major).
    const gl = this.gl;
    for (const [name, value] of Object.entries(uniforms)) {
      if (!this.locations.has(name)) this.locations.set(name, gl.getUniformLocation(this.program, name));
      const loc = this.locations.get(name);
      if (typeof value === 'number') gl.uniform1f(loc, value);
      else if (value.length === 2) gl.uniform2fv(loc, value);
      else if (value.length === 3) gl.uniform3fv(loc, value);
      else if (value.length === 9) gl.uniformMatrix3fv(loc, false, value);
    }
  }

  draw(settings, view = null) {
    // view: the 100% view, { x0, y0, width, height, tiles }: image pixel (x0, y0) of the
    // full-size photo goes at the canvas's top left, and tiles ([x, y, w, h]) are the
    // full-size pieces loaded so far. Without it, the preview fills the canvas.
    const info = this.info;
    if (!info) return;
    this.set({
      ...info.uniforms,
      mul: settings.temp == null ? info.asShot : info.wb[settings.temp],
      compress: settings.compress,
      exposure: settings.exposure,
      saturation: settings.saturation,
      contrast: settings.contrast,
    });
    const gl = this.gl;
    const ch = this.canvas.height;
    if (!view) {
      gl.viewport(0, 0, this.canvas.width, ch);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      return;
    }
    // The whole photo at full size, of which the canvas shows a part (GL counts rows from
    // the bottom). The preview, scaled up, stands in where no tile has arrived yet.
    const { x0, y0, width, height, tiles } = view;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.viewport(-x0, ch - height + y0, width, height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (!tiles.length) return;
    this.bind(this.full.cam, this.full.grain);
    gl.enable(gl.SCISSOR_TEST);
    for (const [x, y, w, h] of tiles) {
      gl.scissor(x - x0, ch - (y - y0) - h, w, h);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.disable(gl.SCISSOR_TEST);
    this.bind(this.camTex, this.grainTex);
  }
}

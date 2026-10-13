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
    this.grainTex = this.texture(4);  // the preview has no noise reduction: no grain to give back
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, 1, 1, 0, gl.RED, gl.HALF_FLOAT, new Uint16Array(1));
    ['cam', 'coeffs', 'tone', 'colour', 'grain'].forEach((name, unit) => gl.uniform1i(gl.getUniformLocation(program, name), unit));
    this.info = null;
  }

  texture(unit, target = this.gl.TEXTURE_2D) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
    gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
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

  draw(settings) {
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
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}

'use strict';

// Runs the develop shader (develop.frag) in the browser, so edits preview instantly.
// The server runs the very same shader for exports (gpu.py). Per photo it sends linear
// camera RGB, the base-layer coefficients, the shader's fixed inputs and the white
// balance multipliers for every temperature; the sliders only pick values.

async function loadShaders() {
  const get = (name) => fetch(`/static/${name}?v=${S.version}`).then((r) => r.text());
  const [vertex, fragment] = await Promise.all([get('develop.vert'), get('develop.frag')]);
  return { vertex, fragment };
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
    gl.uniform1i(gl.getUniformLocation(program, 'cam'), 0);
    gl.uniform1i(gl.getUniformLocation(program, 'coeffs'), 1);
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

  set(uniforms) {
    // By shape: a number is a float, 3 numbers a vec3, 9 a mat3 (column-major).
    const gl = this.gl;
    for (const [name, value] of Object.entries(uniforms)) {
      if (!this.locations.has(name)) this.locations.set(name, gl.getUniformLocation(this.program, name));
      const loc = this.locations.get(name);
      if (typeof value === 'number') gl.uniform1f(loc, value);
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
    });
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
}


/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * WebGPU point-cloud renderer with an explicit capability check.
 *
 * Callers must fall back to the Three.js viewer when `create` rejects.
 * The renderer owns its own orbit camera so millions of points can be
 * rendered without allocating Three.js scene objects per point.
 */

import type { PointCloudData } from "./geometry-accel";

const SHADER = `
struct Uniforms {
  viewProjection: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.viewProjection * vec4<f32>(position, 1.0);
  output.color = color;
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(input.color, 1.0);
}
`;

const FLOATS_PER_POINT = 3;

function multiplyMatrices(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let index = 0; index < 4; index += 1) {
        sum += a[index * 4 + row] * b[column * 4 + index];
      }
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

function perspective(
  fovYRadians: number,
  aspect: number,
  near: number,
  far: number,
): Float32Array {
  const f = 1 / Math.tan(fovYRadians / 2);
  const range = 1 / (near - far);
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (near + far) * range,
    -1,
    0,
    0,
    2 * near * far * range,
    0,
  ]);
}

function lookAt(
  eye: [number, number, number],
  target: [number, number, number],
  up: [number, number, number],
): Float32Array {
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let length = Math.hypot(zx, zy, zz) || 1;
  zx /= length;
  zy /= length;
  zz /= length;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  length = Math.hypot(xx, xy, xz) || 1;
  xx /= length;
  xy /= length;
  xz /= length;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  return new Float32Array([
    xx,
    yx,
    zx,
    0,
    xy,
    yy,
    zy,
    0,
    xz,
    yz,
    zz,
    0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

export function isWebGpuAvailable(): boolean {
  if (typeof navigator === "undefined") {
    return false;
  }
  const gpu = (navigator as any).gpu;
  return Boolean(gpu && typeof gpu.requestAdapter === "function");
}

export class WebGpuPointCloudRenderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly device: any;
  private readonly context: any;
  private readonly pipeline: any;
  private readonly uniformBuffer: any;
  private readonly bindGroup: any;
  private positionBuffer: any = null;
  private colorBuffer: any = null;
  private pointCount = 0;
  private animationFrame: number | null = null;
  private rotation = 0;
  private distance = 3;
  private disposed = false;
  private lastFrameAt = 0;
  private framesSinceSample = 0;
  private sampledFps = 0;

  private constructor(
    canvas: HTMLCanvasElement,
    device: any,
    context: any,
    pipeline: any,
    uniformBuffer: any,
    bindGroup: any,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.pipeline = pipeline;
    this.uniformBuffer = uniformBuffer;
    this.bindGroup = bindGroup;
  }

  static async create(
    canvas: HTMLCanvasElement,
    data: PointCloudData,
  ): Promise<WebGpuPointCloudRenderer> {
    const gpu = (navigator as any)?.gpu;
    if (!gpu) {
      throw new Error("WebGPU is not available in this browser");
    }
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No WebGPU adapter was returned");
    }
    const device = await adapter.requestDevice();
    const context = canvas.getContext("webgpu") as any;
    if (!context) {
      throw new Error("The canvas could not create a WebGPU context");
    }
    const format = gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "premultiplied" });

    const shader = device.createShaderModule({ code: SHADER });
    const pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shader,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: FLOATS_PER_POINT * 4,
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" }],
          },
          {
            arrayStride: FLOATS_PER_POINT * 4,
            attributes: [{ shaderLocation: 1, offset: 0, format: "float32x3" }],
          },
        ],
      },
      fragment: {
        module: shader,
        entryPoint: "fs_main",
        targets: [{ format }],
      },
      primitive: { topology: "point-list" },
    });

    const uniformBuffer = device.createBuffer({
      size: 64,
      usage: (globalThis as any).GPUBufferUsage?.UNIFORM ?? 0x40,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    const renderer = new WebGpuPointCloudRenderer(
      canvas,
      device,
      context,
      pipeline,
      uniformBuffer,
      bindGroup,
    );
    renderer.setPoints(data);
    return renderer;
  }

  setPoints(data: PointCloudData): void {
    if (data.positions.length !== data.colors.length) {
      throw new Error("positions and colors must have the same length");
    }
    if (data.positions.length % FLOATS_PER_POINT !== 0) {
      throw new Error("point buffers must contain xyz triples");
    }
    const usage =
      ((globalThis as any).GPUBufferUsage?.VERTEX ?? 0x20) |
      ((globalThis as any).GPUBufferUsage?.COPY_DST ?? 0x8);
    const byteLength = data.positions.byteLength;
    this.positionBuffer?.destroy?.();
    this.colorBuffer?.destroy?.();
    this.positionBuffer = this.device.createBuffer({ size: byteLength, usage });
    this.colorBuffer = this.device.createBuffer({ size: byteLength, usage });
    this.device.queue.writeBuffer(this.positionBuffer, 0, data.positions);
    this.device.queue.writeBuffer(this.colorBuffer, 0, data.colors);
    this.pointCount = data.positions.length / FLOATS_PER_POINT;
  }
}

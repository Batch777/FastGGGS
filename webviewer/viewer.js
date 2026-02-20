// Main WebGL2 Gaussian Splatting viewer application

import { vertexShaderSource, fragmentShaderSource } from './shaders.js';
import { parsePLY, parseSplat } from './splat-loader.js';
import { CameraControls } from './camera-controls.js';
import { CameraAnimator } from './camera-animator.js';

let gl, canvas;
let program;
let gaussianCount = 0;
let texWidth = 0;

// Cached uniform locations
let uniforms = {};

// Textures
let positionTex, covATex, covBTex, colorTex, indexTex, shTex;
let sgAxisSharpTex, sgColorTex;
let segmentTex;

// Camera
let controls;
const animator = new CameraAnimator();

// Sort worker
let sortWorker;
let sortPending = false;
let lastSortTime = 0;
const SORT_INTERVAL = 50; // ms between sort requests

// Cameras JSON data
let camerasData = null;

// Model browser
let modelsData = null;
let selectedModel = null;

// UI state
let shDegree = 0;
let maxSHDegree = 0;
let sgDegree = 0;
let bgColor = [0, 0, 0];
// Mip-Splatting 2D filter: effective kernel = filterStrength * 0.1 * (refFocal/actualFocal)²
let filterStrength = 0.0; // 0=off, 1=standard Mip-Splatting, 3=strong (original 3DGS 0.3)
let referenceFocal = 0;   // Max focal from training cameras (0=use current focal)

// Segment state
let segMode = 0;  // 0=off, 1=category, 2=instance
let segCategoryVisible = [true, true, true, true]; // cat 0-3
let segmentsData = null;  // loaded from segments.json
let hasSegmentation = false;

// FPS tracking
let frameCount = 0;
let lastFPSTime = 0;
let currentFPS = 0;
let lastFrameTime = 0;

export function init() {
    canvas = document.getElementById('viewer-canvas');
    gl = canvas.getContext('webgl2', { antialias: false, alpha: false });
    if (!gl) {
        showError('WebGL2 is required but not available in this browser.');
        return;
    }

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    controls = new CameraControls(canvas);

    initShaders();
    initSortWorker();
    setupUI();
    requestAnimationFrame(renderLoop);
}

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (gl) gl.viewport(0, 0, canvas.width, canvas.height);
    if (controls) controls.dirty = true;
}

function showError(msg) {
    document.getElementById('status').textContent = msg;
    document.getElementById('status').style.color = '#f44';
}

function showStatus(msg) {
    document.getElementById('status').textContent = msg;
    document.getElementById('status').style.color = '#aaa';
}

// ─── Shader compilation ───

function initShaders() {
    const vs = compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fs = compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
    program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        showError('Shader link error: ' + gl.getProgramInfoLog(program));
        return;
    }
    gl.useProgram(program);

    // Cache uniform locations
    const names = [
        'uViewMatrix', 'uProjectionMatrix', 'uFocal', 'uViewport',
        'uTexWidth', 'uSHDegree', 'uSGDegree', 'uKernelSize', 'uCameraPosition',
        'uPositionTex', 'uCovATex', 'uCovBTex', 'uColorTex', 'uIndexTex', 'uSHTex',
        'uSGAxisSharpTex', 'uSGColorTex',
        'uSegmentTex', 'uSegMode',
        'uCatVisible0', 'uCatVisible1', 'uCatVisible2', 'uCatVisible3',
        'uCatColor0', 'uCatColor1', 'uCatColor2', 'uCatColor3',
    ];
    for (const name of names) {
        uniforms[name] = gl.getUniformLocation(program, name);
    }

    // Create empty VAO (instanced rendering uses gl_VertexID/gl_InstanceID)
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
}

function compileShader(type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const typeName = type === gl.VERTEX_SHADER ? 'Vertex' : 'Fragment';
        console.error(`${typeName} shader error:`, gl.getShaderInfoLog(shader));
        showError(`${typeName} shader compilation failed. Check console.`);
    }
    return shader;
}

// ─── Sort worker ───

function initSortWorker() {
    sortWorker = new Worker('sort-worker.js');
    sortWorker.onmessage = (e) => {
        if (e.data.indices) {
            uploadIndexTexture(e.data.indices);
            sortPending = false;
        }
    };
}

function requestSort(viewMatrix) {
    if (sortPending) return;
    const now = performance.now();
    if (now - lastSortTime < SORT_INTERVAL) return;
    lastSortTime = now;
    sortPending = true;
    sortWorker.postMessage({ viewMatrix: Array.from(viewMatrix) });
}

// ─── Texture management ───

function createDataTexture(width, height, data, internalFormat, format, type) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
    return tex;
}

function deleteTextureIfExists(tex) {
    if (tex) gl.deleteTexture(tex);
}

function uploadGaussianData(splatData) {
    // Clean up old textures
    deleteTextureIfExists(positionTex);
    deleteTextureIfExists(covATex);
    deleteTextureIfExists(covBTex);
    deleteTextureIfExists(colorTex);
    deleteTextureIfExists(indexTex);
    deleteTextureIfExists(shTex);
    shTex = null;
    deleteTextureIfExists(sgAxisSharpTex);
    sgAxisSharpTex = null;
    deleteTextureIfExists(sgColorTex);
    sgColorTex = null;
    deleteTextureIfExists(segmentTex);
    segmentTex = null;

    gaussianCount = splatData.count;
    texWidth = Math.ceil(Math.sqrt(gaussianCount));
    const texHeight = Math.ceil(gaussianCount / texWidth);
    const texSize = texWidth * texHeight;

    // Pack position + opacity into RGBA32F
    const posData = new Float32Array(texSize * 4);
    for (let i = 0; i < gaussianCount; i++) {
        posData[i*4+0] = splatData.positions[i*3+0];
        posData[i*4+1] = splatData.positions[i*3+1];
        posData[i*4+2] = splatData.positions[i*3+2];
        posData[i*4+3] = splatData.opacities[i];
    }
    positionTex = createDataTexture(texWidth, texHeight, posData, gl.RGBA32F, gl.RGBA, gl.FLOAT);

    // Pack covariance A: s00, s01, s02, s11
    const covAData = new Float32Array(texSize * 4);
    for (let i = 0; i < gaussianCount; i++) {
        covAData[i*4+0] = splatData.covariances[i*6+0];
        covAData[i*4+1] = splatData.covariances[i*6+1];
        covAData[i*4+2] = splatData.covariances[i*6+2];
        covAData[i*4+3] = splatData.covariances[i*6+3];
    }
    covATex = createDataTexture(texWidth, texHeight, covAData, gl.RGBA32F, gl.RGBA, gl.FLOAT);

    // Pack covariance B: s12, s22, 0, 0
    const covBData = new Float32Array(texSize * 4);
    for (let i = 0; i < gaussianCount; i++) {
        covBData[i*4+0] = splatData.covariances[i*6+4];
        covBData[i*4+1] = splatData.covariances[i*6+5];
    }
    covBTex = createDataTexture(texWidth, texHeight, covBData, gl.RGBA32F, gl.RGBA, gl.FLOAT);

    // Pack color: r, g, b, 1.0
    const colorData = new Float32Array(texSize * 4);
    for (let i = 0; i < gaussianCount; i++) {
        colorData[i*4+0] = splatData.colors[i*3+0];
        colorData[i*4+1] = splatData.colors[i*3+1];
        colorData[i*4+2] = splatData.colors[i*3+2];
        colorData[i*4+3] = 1.0;
    }
    colorTex = createDataTexture(texWidth, texHeight, colorData, gl.RGBA32F, gl.RGBA, gl.FLOAT);

    // SH texture: pack coefficients using texWidth to match shader's texCoord()
    maxSHDegree = splatData.shDegree || 0;
    if (splatData.shCoeffs && maxSHDegree > 0) {
        // 12 RGBA texels per Gaussian (48 floats, 45 used for 3 channels * 15 coeffs)
        const shRows = Math.ceil((gaussianCount * 12) / texWidth);
        const shFinalSize = texWidth * shRows;
        const shData = new Float32Array(shFinalSize * 4);

        for (let i = 0; i < gaussianCount; i++) {
            // base = texel offset * 4 components
            const base = i * 12 * 4;
            const src = i * 45;

            // Repack from channel-major (f_rest: R[0..14], G[0..14], B[0..14])
            // to the layout the shader expects

            // Degree 1: 3 coeffs per channel = 9 values
            // Texel 0: (R0, R1, R2, G0)
            // Texel 1: (G1, G2, B0, B1)
            // Texel 2: (B2, ...)
            shData[base + 0] = splatData.shCoeffs[src + 0];
            shData[base + 1] = splatData.shCoeffs[src + 1];
            shData[base + 2] = splatData.shCoeffs[src + 2];
            shData[base + 3] = splatData.shCoeffs[src + 15];
            shData[base + 4] = splatData.shCoeffs[src + 16];
            shData[base + 5] = splatData.shCoeffs[src + 17];
            shData[base + 6] = splatData.shCoeffs[src + 30];
            shData[base + 7] = splatData.shCoeffs[src + 31];
            shData[base + 8] = splatData.shCoeffs[src + 32];

            if (maxSHDegree >= 2) {
                // Degree 2: 5 coeffs per channel
                shData[base + 9]  = splatData.shCoeffs[src + 3];
                shData[base + 10] = splatData.shCoeffs[src + 4];
                shData[base + 11] = splatData.shCoeffs[src + 5];
                shData[base + 12] = splatData.shCoeffs[src + 6];
                shData[base + 13] = splatData.shCoeffs[src + 7];
                shData[base + 14] = splatData.shCoeffs[src + 18];
                shData[base + 15] = splatData.shCoeffs[src + 19];
                shData[base + 16] = splatData.shCoeffs[src + 20];
                shData[base + 17] = splatData.shCoeffs[src + 21];
                shData[base + 18] = splatData.shCoeffs[src + 22];
                shData[base + 19] = splatData.shCoeffs[src + 33];
                shData[base + 20] = splatData.shCoeffs[src + 34];
                shData[base + 21] = splatData.shCoeffs[src + 35];
                shData[base + 22] = splatData.shCoeffs[src + 36];
                shData[base + 23] = splatData.shCoeffs[src + 37];
            }

            if (maxSHDegree >= 3) {
                // Degree 3: 7 coeffs per channel
                shData[base + 24] = splatData.shCoeffs[src + 8];
                shData[base + 25] = splatData.shCoeffs[src + 9];
                shData[base + 26] = splatData.shCoeffs[src + 10];
                shData[base + 27] = splatData.shCoeffs[src + 11];
                shData[base + 28] = splatData.shCoeffs[src + 12];
                shData[base + 29] = splatData.shCoeffs[src + 13];
                shData[base + 30] = splatData.shCoeffs[src + 14];
                shData[base + 31] = splatData.shCoeffs[src + 23];
                shData[base + 32] = splatData.shCoeffs[src + 24];
                shData[base + 33] = splatData.shCoeffs[src + 25];
                shData[base + 34] = splatData.shCoeffs[src + 26];
                shData[base + 35] = splatData.shCoeffs[src + 27];
                shData[base + 36] = splatData.shCoeffs[src + 28];
                shData[base + 37] = splatData.shCoeffs[src + 29];
                shData[base + 38] = splatData.shCoeffs[src + 38];
                shData[base + 39] = splatData.shCoeffs[src + 39];
                shData[base + 40] = splatData.shCoeffs[src + 40];
                shData[base + 41] = splatData.shCoeffs[src + 41];
                shData[base + 42] = splatData.shCoeffs[src + 42];
                shData[base + 43] = splatData.shCoeffs[src + 43];
                shData[base + 44] = splatData.shCoeffs[src + 44];
            }
        }

        shTex = createDataTexture(texWidth, shRows, shData, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    }

    // SG textures: axis+sharpness (RGBA) and color (RGB packed as RGBA)
    sgDegree = splatData.sgDegree || 0;
    if (splatData.sgAxisSharp && sgDegree > 0) {
        const sgTexels = gaussianCount * sgDegree;
        const sgRows = Math.ceil(sgTexels / texWidth);
        const sgFinalSize = texWidth * sgRows;

        // Pack axis+sharpness: already in (ax, ay, az, sharp) layout from loader
        const sgASData = new Float32Array(sgFinalSize * 4);
        for (let i = 0; i < gaussianCount; i++) {
            for (let g = 0; g < sgDegree; g++) {
                const srcBase = (i * sgDegree + g) * 4;
                const dstBase = (i * sgDegree + g) * 4;
                sgASData[dstBase + 0] = splatData.sgAxisSharp[srcBase + 0];
                sgASData[dstBase + 1] = splatData.sgAxisSharp[srcBase + 1];
                sgASData[dstBase + 2] = splatData.sgAxisSharp[srcBase + 2];
                sgASData[dstBase + 3] = splatData.sgAxisSharp[srcBase + 3];
            }
        }
        sgAxisSharpTex = createDataTexture(texWidth, sgRows, sgASData, gl.RGBA32F, gl.RGBA, gl.FLOAT);

        // Pack color as RGBA (4th component unused)
        const sgCData = new Float32Array(sgFinalSize * 4);
        for (let i = 0; i < gaussianCount; i++) {
            for (let g = 0; g < sgDegree; g++) {
                const srcBase = (i * sgDegree + g) * 3;
                const dstBase = (i * sgDegree + g) * 4;
                sgCData[dstBase + 0] = splatData.sgColor[srcBase + 0];
                sgCData[dstBase + 1] = splatData.sgColor[srcBase + 1];
                sgCData[dstBase + 2] = splatData.sgColor[srcBase + 2];
                sgCData[dstBase + 3] = 0.0;
            }
        }
        sgColorTex = createDataTexture(texWidth, sgRows, sgCData, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    }

    // Segment texture: pack (semantic_category, instance_id, 0, 0) as RGBA32F
    hasSegmentation = splatData.hasSegmentation || false;
    if (hasSegmentation && splatData.instanceIds && splatData.semanticCategories) {
        const segData = new Float32Array(texSize * 4);
        for (let i = 0; i < gaussianCount; i++) {
            segData[i * 4 + 0] = splatData.semanticCategories[i];
            segData[i * 4 + 1] = splatData.instanceIds[i];
            segData[i * 4 + 2] = 0;
            segData[i * 4 + 3] = 0;
        }
        segmentTex = createDataTexture(texWidth, texHeight, segData, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    }

    // Initial identity index texture
    const indices = new Uint32Array(texSize);
    for (let i = 0; i < gaussianCount; i++) indices[i] = i;
    indexTex = createDataTexture(texWidth, texHeight, indices, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT);

    // Send positions to sort worker
    sortWorker.postMessage({
        positions: splatData.positions,
        count: gaussianCount
    });

    // Auto-center orbit camera on scene bounding box
    let cx = 0, cy = 0, cz = 0;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    const step = Math.max(1, Math.floor(gaussianCount / 10000)); // sample for speed
    let sampleCount = 0;
    for (let i = 0; i < gaussianCount; i += step) {
        const x = splatData.positions[i*3], y = splatData.positions[i*3+1], z = splatData.positions[i*3+2];
        cx += x; cy += y; cz += z;
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        sampleCount++;
    }
    cx /= sampleCount; cy /= sampleCount; cz /= sampleCount;
    const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    controls.target = [cx, cy, cz];
    controls.distance = extent * 0.8;

    const sgInfo = sgDegree > 0 ? `, SG degree ${sgDegree}` : '';
    const segInfo = hasSegmentation ? ', segments' : '';
    showStatus(`Loaded ${gaussianCount.toLocaleString()} Gaussians (SH degree ${maxSHDegree}${sgInfo}${segInfo})`);
    controls.dirty = true;

    // Show/hide segment panel
    const segPanel = document.getElementById('seg-panel');
    if (segPanel) {
        segPanel.style.display = hasSegmentation ? 'block' : 'none';
    }
}

function uploadIndexTexture(indices) {
    const texHeight = Math.ceil(gaussianCount / texWidth);
    const texSize = texWidth * texHeight;
    const padded = new Uint32Array(texSize);
    padded.set(indices);
    gl.bindTexture(gl.TEXTURE_2D, indexTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texWidth, texHeight, gl.RED_INTEGER, gl.UNSIGNED_INT, padded);
}

// ─── Render loop ───

function getCameraPosition(viewMatrix) {
    // Camera position = -R^T * t where view = [R|t]
    // In column-major: R is cols 0-2 rows 0-2, t is col 3 rows 0-2
    const m = viewMatrix;
    const tx = m[12], ty = m[13], tz = m[14];
    return [
        -(m[0]*tx + m[1]*ty + m[2]*tz),
        -(m[4]*tx + m[5]*ty + m[6]*tz),
        -(m[8]*tx + m[9]*ty + m[10]*tz)
    ];
}

function renderLoop(time) {
    requestAnimationFrame(renderLoop);

    // Compute dt in seconds
    const dt = lastFrameTime > 0 ? (time - lastFrameTime) / 1000 : 0;
    lastFrameTime = time;

    if (gaussianCount === 0) {
        gl.clearColor(bgColor[0], bgColor[1], bgColor[2], 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        return;
    }

    // Update FPS
    frameCount++;
    if (time - lastFPSTime >= 1000) {
        currentFPS = Math.round(frameCount * 1000 / (time - lastFPSTime));
        frameCount = 0;
        lastFPSTime = time;
        document.getElementById('fps').textContent = `${currentFPS} FPS`;
    }

    // Update camera animator
    const interpCam = animator.update(dt, controls);
    if (interpCam) {
        controls.setFromCamera(interpCam);
    }
    updateAnimatorUI();

    const viewMatrix = controls.getViewMatrix();
    const projMatrix = controls.getProjectionMatrix(canvas.width, canvas.height);
    const focal = controls.getFocal(canvas.width, canvas.height);
    const camPos = getCameraPosition(viewMatrix);

    // Trigger sort when camera moves
    if (controls.dirty) {
        requestSort(viewMatrix);
        controls.dirty = false;
    }

    // Render
    gl.clearColor(bgColor[0], bgColor[1], bgColor[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program);

    // Set uniforms (using cached locations)
    gl.uniformMatrix4fv(uniforms.uViewMatrix, false, viewMatrix);
    gl.uniformMatrix4fv(uniforms.uProjectionMatrix, false, projMatrix);
    gl.uniform2f(uniforms.uFocal, focal[0], focal[1]);
    gl.uniform2f(uniforms.uViewport, canvas.width, canvas.height);
    gl.uniform1i(uniforms.uTexWidth, texWidth);
    gl.uniform1i(uniforms.uSHDegree, shDegree);
    gl.uniform1i(uniforms.uSGDegree, sgDegree);
    // Mip-Splatting 2D low-pass filter: kernel = strength * 0.1 * (refFocal/actualFocal)²
    // 0.1 = Gaussian approx of 1-pixel box filter variance (Mip-Splatting paper Sec 3.2)
    // Focal ratio adapts filter to current viewport resolution vs training resolution
    const maxFocal = Math.max(focal[0], focal[1]);
    const refF = referenceFocal > 0 ? referenceFocal : maxFocal;
    const focalRatio = refF / maxFocal;
    const effectiveKernel = filterStrength * 0.1 * focalRatio * focalRatio;
    gl.uniform1f(uniforms.uKernelSize, effectiveKernel);
    gl.uniform3f(uniforms.uCameraPosition, camPos[0], camPos[1], camPos[2]);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, positionTex);
    gl.uniform1i(uniforms.uPositionTex, 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, covATex);
    gl.uniform1i(uniforms.uCovATex, 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, covBTex);
    gl.uniform1i(uniforms.uCovBTex, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, colorTex);
    gl.uniform1i(uniforms.uColorTex, 3);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, indexTex);
    gl.uniform1i(uniforms.uIndexTex, 4);
    if (shTex) {
        gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, shTex);
        gl.uniform1i(uniforms.uSHTex, 5);
    }
    if (sgAxisSharpTex) {
        gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, sgAxisSharpTex);
        gl.uniform1i(uniforms.uSGAxisSharpTex, 6);
    }
    if (sgColorTex) {
        gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, sgColorTex);
        gl.uniform1i(uniforms.uSGColorTex, 7);
    }

    // Segment uniforms
    gl.uniform1i(uniforms.uSegMode, hasSegmentation ? segMode : 0);
    if (segmentTex && hasSegmentation) {
        gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D, segmentTex);
        gl.uniform1i(uniforms.uSegmentTex, 8);
    }
    gl.uniform1i(uniforms.uCatVisible0, segCategoryVisible[0] ? 1 : 0);
    gl.uniform1i(uniforms.uCatVisible1, segCategoryVisible[1] ? 1 : 0);
    gl.uniform1i(uniforms.uCatVisible2, segCategoryVisible[2] ? 1 : 0);
    gl.uniform1i(uniforms.uCatVisible3, segCategoryVisible[3] ? 1 : 0);

    // Category colors (default or from segments.json)
    const catColors = getCategoryColors();
    gl.uniform3f(uniforms.uCatColor0, catColors[0][0], catColors[0][1], catColors[0][2]);
    gl.uniform3f(uniforms.uCatColor1, catColors[1][0], catColors[1][1], catColors[1][2]);
    gl.uniform3f(uniforms.uCatColor2, catColors[2][0], catColors[2][1], catColors[2][2]);
    gl.uniform3f(uniforms.uCatColor3, catColors[3][0], catColors[3][1], catColors[3][2]);

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, gaussianCount);
}

// ─── Segment helpers ───

function getCategoryColors() {
    // Returns [[r,g,b], ...] for categories 0-3, normalized to [0,1]
    const defaults = [
        [0.3, 0.3, 0.3],  // background (gray)
        [0.78, 0.0, 0.0],  // building (red)
        [0.0, 0.71, 0.0],  // tree (green)
        [0.0, 0.47, 0.71],  // road (blue)
    ];
    if (segmentsData && segmentsData.categories) {
        for (const cat of segmentsData.categories) {
            if (cat.id >= 0 && cat.id < 4 && (cat.color || cat.color_rgb)) {
                const rgb = cat.color || cat.color_rgb;
                defaults[cat.id] = rgb.map(c => c / 255.0);
            }
        }
    }
    return defaults;
}

function buildSegmentUI() {
    const container = document.getElementById('seg-categories');
    if (!container) return;
    container.innerHTML = '';

    const catColors = getCategoryColors();
    const names = ['Background', 'Cat 1', 'Cat 2', 'Cat 3'];

    if (segmentsData && segmentsData.categories) {
        for (const cat of segmentsData.categories) {
            if (cat.id >= 0 && cat.id < 4) {
                names[cat.id] = cat.name;
            }
        }
    }

    for (let i = 0; i < 4; i++) {
        const row = document.createElement('div');
        row.className = 'seg-row';

        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = segCategoryVisible[i];
        cb.id = `seg-cat-${i}`;
        cb.addEventListener('change', () => {
            segCategoryVisible[i] = cb.checked;
            controls.dirty = true;
        });

        const swatch = document.createElement('span');
        swatch.className = 'seg-swatch';
        const c = catColors[i];
        swatch.style.background = `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;

        const label = document.createElement('label');
        label.htmlFor = `seg-cat-${i}`;
        let countStr = '';
        if (segmentsData && segmentsData.categories) {
            const catInfo = segmentsData.categories.find(c => c.id === i);
            if (catInfo && catInfo.n_gaussians) {
                countStr = ` (${catInfo.n_gaussians.toLocaleString()})`;
            }
        }
        label.textContent = names[i] + countStr;

        row.appendChild(cb);
        row.appendChild(swatch);
        row.appendChild(label);
        container.appendChild(row);
    }
}

// ─── UI setup ───

function setupUI() {
    // Model browser: fetch available models from server API
    fetchModels();

    document.getElementById('model-select').addEventListener('change', (e) => {
        const name = e.target.value;
        selectedModel = modelsData ? modelsData.find(m => m.name === name) : null;
        updateIterSelect();
    });

    document.getElementById('load-btn').addEventListener('click', () => {
        if (!selectedModel) return;
        loadServerModel();
    });

    // File input (PLY / .splat)
    document.getElementById('file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) loadFile(file);
    });

    // Cameras JSON input
    document.getElementById('cameras-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) loadCameras(file);
    });

    // SH degree selector
    document.getElementById('sh-degree').addEventListener('change', (e) => {
        shDegree = parseInt(e.target.value);
        if (shDegree > maxSHDegree) shDegree = maxSHDegree;
        controls.dirty = true;
    });

    // Background color toggle
    document.getElementById('bg-toggle').addEventListener('click', () => {
        if (bgColor[0] === 0) {
            bgColor = [1, 1, 1];
            document.getElementById('bg-toggle').textContent = 'BG: White';
        } else {
            bgColor = [0, 0, 0];
            document.getElementById('bg-toggle').textContent = 'BG: Black';
        }
    });

    // Mip-Splatting 2D filter strength slider
    document.getElementById('mip-filter').addEventListener('input', (e) => {
        filterStrength = parseFloat(e.target.value);
        document.getElementById('mip-filter-val').textContent = filterStrength.toFixed(1);
        controls.dirty = true;
    });

    // Segment mode selector
    const segModeSelect = document.getElementById('seg-mode');
    if (segModeSelect) {
        segModeSelect.addEventListener('change', (e) => {
            segMode = parseInt(e.target.value);
            controls.dirty = true;
        });
    }

    // Reference image toggle
    let refVisible = false;
    document.getElementById('ref-toggle').addEventListener('click', async () => {
        const overlay = document.getElementById('ref-overlay');
        if (!refVisible) {
            // Try to load reference_camera.json to find the image
            try {
                const resp = await fetch('reference_camera.json');
                const camInfo = await resp.json();
                const idx = camInfo.camera_idx;
                overlay.src = `reference_${idx}.png`;
                overlay.style.display = 'block';
                document.getElementById('ref-toggle').textContent = 'Hide Ref';
                refVisible = true;

                // If cameras are loaded, jump to the same viewpoint
                if (camerasData && camInfo.cameras_json_entry) {
                    controls.setFromCamera(camInfo.cameras_json_entry);
                }
            } catch (e) {
                showStatus('No reference image found (run render_reference.py first)');
            }
        } else {
            overlay.style.display = 'none';
            document.getElementById('ref-toggle').textContent = 'Show Ref';
            refVisible = false;
        }
    });

    // Camera animator controls
    document.getElementById('cam-prev').addEventListener('click', () => {
        const idx = animator.prev();
        if (idx >= 0 && camerasData) {
            controls.setFromCamera(camerasData[idx]);
            highlightCameraButton(idx);
        }
    });

    document.getElementById('cam-next').addEventListener('click', () => {
        const idx = animator.next();
        if (idx >= 0 && camerasData) {
            controls.setFromCamera(camerasData[idx]);
            highlightCameraButton(idx);
        }
    });

    document.getElementById('cam-play').addEventListener('click', () => {
        // If starting playback and not already in direct matrix mode, jump to current camera first
        if (!animator.isPlaying() && camerasData && camerasData.length > 0) {
            const idx = animator.getCurrentIndex();
            if (idx >= 0) controls.setFromCamera(camerasData[idx]);
        }
        animator.toggle();
        updateAnimatorUI();
    });

    document.getElementById('cam-loop').addEventListener('click', () => {
        const newLoop = !animator.loop;
        animator.setLoop(newLoop);
        document.getElementById('cam-loop').style.color = newLoop ? '#7eb8ff' : '#666';
    });

    document.getElementById('cam-speed').addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        animator.setSpeed(val);
        document.getElementById('cam-speed-val').textContent = `${val.toFixed(1)}x`;
    });

    // Progress bar click-to-seek
    document.getElementById('cam-progress-wrap').addEventListener('click', (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        animator.seekTo(fraction);
        // Apply the camera at the seeked position
        if (camerasData && camerasData.length > 0) {
            const idx = animator.getCurrentIndex();
            if (idx >= 0) controls.setFromCamera(camerasData[idx]);
        }
        updateAnimatorUI();
    });

    // Keyboard shortcuts
    window.addEventListener('keydown', (e) => {
        // Ignore if user is typing in an input
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
        if (!camerasData || camerasData.length === 0) return;

        if (e.key === 'ArrowLeft') {
            e.preventDefault();
            const idx = animator.prev();
            if (idx >= 0) {
                controls.setFromCamera(camerasData[idx]);
                highlightCameraButton(idx);
            }
        } else if (e.key === 'ArrowRight') {
            e.preventDefault();
            const idx = animator.next();
            if (idx >= 0) {
                controls.setFromCamera(camerasData[idx]);
                highlightCameraButton(idx);
            }
        } else if (e.key === ' ') {
            e.preventDefault();
            if (!animator.isPlaying() && camerasData.length > 0) {
                const idx = animator.getCurrentIndex();
                if (idx >= 0) controls.setFromCamera(camerasData[idx]);
            }
            animator.toggle();
            updateAnimatorUI();
        }
    });

    // Drag and drop
    canvas.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });
    canvas.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        for (const file of files) {
            if (file.name.endsWith('.ply') || file.name.endsWith('.splat')) {
                loadFile(file);
            } else if (file.name.endsWith('.json')) {
                loadCameras(file);
            }
        }
    });
}

async function loadFile(file) {
    showStatus(`Loading ${file.name}...`);
    const buffer = await file.arrayBuffer();

    let splatData;
    const progressCb = (progress) => {
        showStatus(`Parsing... ${Math.round(progress * 100)}%`);
    };
    if (file.name.endsWith('.splat')) {
        splatData = parseSplat(buffer, progressCb);
    } else {
        splatData = parsePLY(buffer, progressCb);
    }

    uploadGaussianData(splatData);

    // Build segment UI if segmentation data present
    if (splatData.hasSegmentation) {
        segmentsData = null;  // no segments.json for file uploads
        buildSegmentUI();
    }

    // Update SH degree selector
    const shSelect = document.getElementById('sh-degree');
    shSelect.innerHTML = '<option value="0">SH 0 (DC only)</option>';
    for (let d = 1; d <= maxSHDegree; d++) {
        shSelect.innerHTML += `<option value="${d}">SH ${d}</option>`;
    }
}

async function loadCameras(file) {
    const text = await file.text();
    camerasData = JSON.parse(text);
    showStatus(`Loaded ${camerasData.length} cameras`);
    buildCameraList();
}

// ─── Server model browser ───

async function fetchModels() {
    try {
        const resp = await fetch('/api/models');
        if (!resp.ok) return;
        modelsData = await resp.json();

        const select = document.getElementById('model-select');
        for (const model of modelsData) {
            const opt = document.createElement('option');
            opt.value = model.name;
            const cfg = model.config || {};
            const info = [];
            if (cfg.sh_degree !== undefined) info.push(`SH${cfg.sh_degree}`);
            if (cfg.sg_degree) info.push(`SG${cfg.sg_degree}`);
            if (cfg.resolution && cfg.resolution !== 1) info.push(`r${cfg.resolution}`);
            opt.textContent = model.name + (info.length ? ` (${info.join(',')})` : '');
            select.appendChild(opt);
        }
    } catch (e) {
        // API not available (using plain http.server), hide model select
    }
}

function updateIterSelect() {
    const iterRow = document.getElementById('iter-row');
    const iterSelect = document.getElementById('iter-select');
    iterSelect.innerHTML = '';

    if (!selectedModel || selectedModel.iterations.length === 0) {
        iterRow.style.display = 'none';
        return;
    }

    iterRow.style.display = 'flex';
    for (const iter of selectedModel.iterations) {
        const opt = document.createElement('option');
        opt.value = iter.iteration;
        const files = iter.files;
        const splatInfo = files.splat ? ` .splat ${files.splat.size_mb}MB` : '';
        const plyInfo = files.ply ? ` .ply ${files.ply.size_mb}MB` : '';
        // Show preferred format first (splat if available, then ply)
        const fileLabel = splatInfo || plyInfo;
        opt.textContent = `${iter.iteration}${fileLabel}`;
        iterSelect.appendChild(opt);
    }

    // Select the last (highest) iteration by default
    iterSelect.value = selectedModel.iterations[selectedModel.iterations.length - 1].iteration;
}

async function loadServerModel() {
    const iterVal = parseInt(document.getElementById('iter-select').value);
    const iter = selectedModel.iterations.find(i => i.iteration === iterVal);
    if (!iter) return;

    // Prefer .splat v2 (smaller, faster parse, precomputed activations), fall back to .ply
    const fileInfo = iter.files.splat || iter.files.ply;
    if (!fileInfo) return;

    const url = fileInfo.url;
    const isSplat = url.endsWith('.splat');
    showStatus(`Loading ${selectedModel.name} iter ${iterVal} (${fileInfo.size_mb}MB)...`);

    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

        const total = parseInt(resp.headers.get('Content-Length') || '0');
        const reader = resp.body.getReader();
        const chunks = [];
        let received = 0;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            received += value.length;
            if (total > 0) {
                showStatus(`Downloading... ${Math.round(received / total * 100)}%`);
            }
        }

        // Combine chunks into single ArrayBuffer
        const buffer = new ArrayBuffer(received);
        const view = new Uint8Array(buffer);
        let offset = 0;
        for (const chunk of chunks) {
            view.set(chunk, offset);
            offset += chunk.length;
        }

        let splatData;
        const progressCb = (progress) => {
            showStatus(`Parsing... ${Math.round(progress * 100)}%`);
        };
        if (isSplat) {
            splatData = parseSplat(buffer, progressCb);
        } else {
            splatData = parsePLY(buffer, progressCb);
        }

        // Convert config kernel_size to Mip filter strength
        // kernel_size=0.1 → strength=1 (standard Mip-Splatting)
        // kernel_size=0.3 → strength=3 (original 3DGS)
        if (selectedModel.config && selectedModel.config.kernel_size !== undefined) {
            filterStrength = selectedModel.config.kernel_size / 0.1;
        }
        document.getElementById('mip-filter').value = filterStrength;
        document.getElementById('mip-filter-val').textContent = filterStrength.toFixed(1);

        uploadGaussianData(splatData);

        // Update SH degree selector
        const shSelect = document.getElementById('sh-degree');
        shSelect.innerHTML = '<option value="0">SH 0 (DC only)</option>';
        for (let d = 1; d <= maxSHDegree; d++) {
            shSelect.innerHTML += `<option value="${d}">SH ${d}</option>`;
        }

        // Auto-load segments.json (alongside PLY)
        if (splatData.hasSegmentation) {
            const segUrl = url.replace(/point_cloud\.(ply|splat)$/, 'segments.json');
            try {
                const segResp = await fetch(segUrl);
                if (segResp.ok) {
                    segmentsData = await segResp.json();
                    buildSegmentUI();
                }
            } catch (e) { /* segments.json optional */ }
        }

        // Auto-load cameras.json
        if (selectedModel.cameras_url) {
            try {
                const camResp = await fetch(selectedModel.cameras_url);
                if (camResp.ok) {
                    camerasData = await camResp.json();
                    buildCameraList();
                    showStatus(`Loaded ${gaussianCount.toLocaleString()} Gaussians + ${camerasData.length} cameras`);
                }
            } catch (e) { /* cameras optional */ }
        }
    } catch (e) {
        showError(`Load failed: ${e.message}`);
    }
}

function buildCameraList() {
    const list = document.getElementById('camera-list');
    list.innerHTML = '';
    if (!camerasData) return;

    // Compute reference focal from training cameras (max fx for Mip filter)
    let maxFx = 0;
    for (const cam of camerasData) {
        if (cam.fx > maxFx) maxFx = cam.fx;
    }
    if (maxFx > 0) referenceFocal = maxFx;

    animator.setCameras(camerasData);

    camerasData.forEach((cam, i) => {
        const btn = document.createElement('button');
        btn.className = 'cam-btn';
        btn.textContent = cam.img_name || `Camera ${i}`;
        btn.addEventListener('click', () => {
            animator.goTo(i);
            controls.setFromCamera(cam);
            highlightCameraButton(i);
        });
        list.appendChild(btn);
    });

    document.getElementById('cameras-panel').style.display = 'block';
    document.getElementById('cam-nav').style.display = 'block';
    highlightCameraButton(0);
    updateAnimatorUI();
}

function highlightCameraButton(index) {
    const btns = document.getElementById('camera-list').querySelectorAll('.cam-btn');
    btns.forEach((btn, i) => {
        btn.classList.toggle('cam-btn-active', i === index);
    });
    const total = camerasData ? camerasData.length : 0;
    document.getElementById('cam-index').textContent = total > 0 ? `${index + 1} / ${total}` : '0 / 0';
}

function updateAnimatorUI() {
    if (!camerasData || camerasData.length === 0) return;

    // Update progress bar
    const progress = animator.getProgress();
    const bar = document.getElementById('cam-progress-bar');
    if (bar) bar.style.width = `${(progress * 100).toFixed(1)}%`;

    // Update play button text
    const playBtn = document.getElementById('cam-play');
    if (playBtn) {
        playBtn.innerHTML = animator.isPlaying() ? '&#10074;&#10074; Pause' : '&#9654; Play';
    }

    // Update active camera highlight
    const idx = animator.getCurrentIndex();
    if (idx >= 0) {
        highlightCameraButton(idx);
    }
}

// Start
init();

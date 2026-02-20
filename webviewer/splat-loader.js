// PLY and .splat binary file parsers for Gaussian Splatting data

const C0 = 0.28209479177387814;

function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

function softplus(x) {
    return Math.log(1.0 + Math.exp(x));
}

function quaternionToRotationMatrix(q) {
    // q = [w, x, y, z] = [rot_0, rot_1, rot_2, rot_3]
    const w = q[0], x = q[1], y = q[2], z = q[3];
    // Matching CUDA rasterizer quaternion-to-matrix convention
    return [
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)
    ];
}

function computeCovariance(scale, rotation) {
    // Sigma = (S * R^T)^T * (S * R^T) = R * S^2 * R^T
    // where S = diag(scale), R = standard rotation matrix (row-major)
    //
    // CUDA uses GLM column-major: glm::mat3 R fills columns, so the effective
    // row-major matrix is R_std^T. Then M_cuda = S * R_cuda_eff = S * R_std^T.
    // To match: M = S * R^T (transpose the standard rotation).
    const R = quaternionToRotationMatrix(rotation);
    const sx = scale[0], sy = scale[1], sz = scale[2];

    // M = S * R^T: M[i][j] = s_i * R[j][i] = s_i * R[j*3+i]
    const M00 = sx * R[0], M01 = sx * R[3], M02 = sx * R[6];
    const M10 = sy * R[1], M11 = sy * R[4], M12 = sy * R[7];
    const M20 = sz * R[2], M21 = sz * R[5], M22 = sz * R[8];

    // Sigma = M^T * M
    const s00 = M00*M00 + M10*M10 + M20*M20;
    const s01 = M00*M01 + M10*M11 + M20*M21;
    const s02 = M00*M02 + M10*M12 + M20*M22;
    const s11 = M01*M01 + M11*M11 + M21*M21;
    const s12 = M01*M02 + M11*M12 + M21*M22;
    const s22 = M02*M02 + M12*M12 + M22*M22;

    return [s00, s01, s02, s11, s12, s22];
}

export function parsePLY(buffer, onProgress) {
    const decoder = new TextDecoder();
    // Find end of header
    const headerEnd = new Uint8Array(buffer).indexOf(10, // \n
        (() => {
            const bytes = new Uint8Array(buffer);
            for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
                if (bytes[i] === 101 && bytes[i+1] === 110 && bytes[i+2] === 100 &&
                    bytes[i+3] === 95 && bytes[i+4] === 104) { // "end_h"
                    return i;
                }
            }
            return 0;
        })()
    );

    // Parse header more robustly
    let headerEndPos = 0;
    const bytes = new Uint8Array(buffer);
    const headerStr = decoder.decode(bytes.slice(0, Math.min(bytes.length, 8192)));
    const endHeaderIdx = headerStr.indexOf('end_header');
    if (endHeaderIdx === -1) throw new Error('Invalid PLY: no end_header');
    headerEndPos = endHeaderIdx + 'end_header'.length;
    // Skip newline after end_header
    while (headerEndPos < bytes.length && (bytes[headerEndPos] === 10 || bytes[headerEndPos] === 13)) {
        headerEndPos++;
    }

    const headerText = headerStr.substring(0, endHeaderIdx);
    const lines = headerText.split('\n').map(l => l.trim());

    // Check format
    if (!lines.some(l => l.includes('binary_little_endian'))) {
        throw new Error('Only binary_little_endian PLY is supported');
    }

    // Get vertex count
    let vertexCount = 0;
    for (const line of lines) {
        const match = line.match(/^element vertex (\d+)/);
        if (match) {
            vertexCount = parseInt(match[1]);
            break;
        }
    }
    if (vertexCount === 0) throw new Error('No vertices in PLY');

    // Build property map
    const properties = [];
    for (const line of lines) {
        const match = line.match(/^property (float|double|int|uint|uchar|short|ushort) (.+)/);
        if (match) {
            properties.push({ type: match[1], name: match[2] });
        }
    }

    // Compute byte offsets for each property
    const propSize = { float: 4, double: 8, int: 4, uint: 4, uchar: 1, short: 2, ushort: 2 };
    let offset = 0;
    const propMap = {};
    let vertexSize = 0;
    for (const prop of properties) {
        propMap[prop.name] = { offset, size: propSize[prop.type], type: prop.type };
        offset += propSize[prop.type];
    }
    vertexSize = offset;

    // Helper to read a float property
    const dataView = new DataView(buffer, headerEndPos);
    function readFloat(vertexIdx, propName) {
        const prop = propMap[propName];
        if (!prop) return 0;
        const byteOff = vertexIdx * vertexSize + prop.offset;
        if (prop.type === 'float') return dataView.getFloat32(byteOff, true);
        if (prop.type === 'double') return dataView.getFloat64(byteOff, true);
        return 0;
    }

    // Determine SH degree from available properties
    let shDegree = 0;
    if (propMap['f_rest_0']) {
        const restCount = Object.keys(propMap).filter(k => k.startsWith('f_rest_')).length;
        const coeffsPerChannel = restCount / 3;
        if (coeffsPerChannel >= 15) shDegree = 3;
        else if (coeffsPerChannel >= 8) shDegree = 2;
        else if (coeffsPerChannel >= 3) shDegree = 1;
    }

    // Determine SG degree from available properties
    // PLY stores: sg_axis_{0..D*3-1}, sg_sharpness_{0..D-1}, sg_color_{0..D*3-1}
    let sgDegree = 0;
    if (propMap['sg_sharpness_0']) {
        sgDegree = Object.keys(propMap).filter(k => k.startsWith('sg_sharpness_')).length;
    }

    // Allocate output arrays
    const positions = new Float32Array(vertexCount * 3);
    const covariances = new Float32Array(vertexCount * 6);
    const colors = new Float32Array(vertexCount * 3);
    const opacities = new Float32Array(vertexCount);

    // SH rest coefficients: up to 45 floats (3 channels * 15 coeffs)
    const shCoeffs = shDegree > 0 ? new Float32Array(vertexCount * 45) : null;

    // SG data: per lobe, pack (axis.x, axis.y, axis.z, sharpness) and (color.r, color.g, color.b)
    // sgAxisSharp: 4 floats per lobe per Gaussian, sgColor: 3 floats per lobe per Gaussian
    const sgAxisSharp = sgDegree > 0 ? new Float32Array(vertexCount * sgDegree * 4) : null;
    const sgColor = sgDegree > 0 ? new Float32Array(vertexCount * sgDegree * 3) : null;

    const progressInterval = Math.max(1, Math.floor(vertexCount / 20));

    for (let i = 0; i < vertexCount; i++) {
        if (onProgress && i % progressInterval === 0) {
            onProgress(i / vertexCount);
        }

        // Position
        positions[i * 3 + 0] = readFloat(i, 'x');
        positions[i * 3 + 1] = readFloat(i, 'y');
        positions[i * 3 + 2] = readFloat(i, 'z');

        // DC color: C0 * f_dc + 0.5 (no clamp here — CUDA clamps after SH+SG+0.5)
        colors[i * 3 + 0] = C0 * readFloat(i, 'f_dc_0') + 0.5;
        colors[i * 3 + 1] = C0 * readFloat(i, 'f_dc_1') + 0.5;
        colors[i * 3 + 2] = C0 * readFloat(i, 'f_dc_2') + 0.5;

        // Scale: exp activation
        const sx = Math.exp(readFloat(i, 'scale_0'));
        const sy = Math.exp(readFloat(i, 'scale_1'));
        const sz = Math.exp(readFloat(i, 'scale_2'));

        // GGGS filter_3D: adjusts scale and opacity for anti-aliasing
        const f3d = readFloat(i, 'filter_3D');
        const f3d2 = f3d * f3d;
        let scale;
        if (f3d2 > 0) {
            // scale_actual = sqrt(scale^2 + filter_3D^2)
            const sx2 = sx*sx, sy2 = sy*sy, sz2 = sz*sz;
            const det1 = sx2 * sy2 * sz2;
            const ax2 = sx2 + f3d2, ay2 = sy2 + f3d2, az2 = sz2 + f3d2;
            const det2 = ax2 * ay2 * az2;
            const coef = Math.sqrt(det1 / det2);
            scale = [Math.sqrt(ax2), Math.sqrt(ay2), Math.sqrt(az2)];
            // opacity_actual = sigmoid(raw) * coef
            opacities[i] = sigmoid(readFloat(i, 'opacity')) * coef;
        } else {
            scale = [sx, sy, sz];
            opacities[i] = sigmoid(readFloat(i, 'opacity'));
        }

        // Quaternion: normalize (w,x,y,z) = (rot_0,rot_1,rot_2,rot_3)
        let qw = readFloat(i, 'rot_0');
        let qx = readFloat(i, 'rot_1');
        let qy = readFloat(i, 'rot_2');
        let qz = readFloat(i, 'rot_3');
        const qlen = Math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        if (qlen > 0) { qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen; }
        const rotation = [qw, qx, qy, qz];

        // 3D covariance
        const cov = computeCovariance(scale, rotation);
        covariances[i * 6 + 0] = cov[0];
        covariances[i * 6 + 1] = cov[1];
        covariances[i * 6 + 2] = cov[2];
        covariances[i * 6 + 3] = cov[3];
        covariances[i * 6 + 4] = cov[4];
        covariances[i * 6 + 5] = cov[5];

        // SH rest coefficients (f_rest layout: channel-major)
        // f_rest_0..14 = R, f_rest_15..29 = G, f_rest_30..44 = B
        if (shCoeffs) {
            for (let j = 0; j < 45; j++) {
                const val = readFloat(i, `f_rest_${j}`);
                shCoeffs[i * 45 + j] = val || 0;
            }
        }

        // SG lobes: axis (normalized), sharpness (softplus), color (raw)
        if (sgAxisSharp) {
            for (let g = 0; g < sgDegree; g++) {
                // Read raw axis and normalize (matches gaussian_model.py get_sg_axis)
                let ax = readFloat(i, `sg_axis_${g * 3}`);
                let ay = readFloat(i, `sg_axis_${g * 3 + 1}`);
                let az = readFloat(i, `sg_axis_${g * 3 + 2}`);
                const alen = Math.sqrt(ax * ax + ay * ay + az * az);
                if (alen > 0) { ax /= alen; ay /= alen; az /= alen; }

                // Softplus activation on sharpness (matches gaussian_model.py sharpness_activateion)
                const sharp = softplus(readFloat(i, `sg_sharpness_${g}`));

                const base = (i * sgDegree + g) * 4;
                sgAxisSharp[base + 0] = ax;
                sgAxisSharp[base + 1] = ay;
                sgAxisSharp[base + 2] = az;
                sgAxisSharp[base + 3] = sharp;

                // Color: no activation (matches gaussian_model.py get_sg_color)
                const cbase = (i * sgDegree + g) * 3;
                sgColor[cbase + 0] = readFloat(i, `sg_color_${g * 3}`);
                sgColor[cbase + 1] = readFloat(i, `sg_color_${g * 3 + 1}`);
                sgColor[cbase + 2] = readFloat(i, `sg_color_${g * 3 + 2}`);
            }
        }
    }

    // Read instance_id and semantic_category if present
    const hasInstanceId = !!propMap['instance_id'];
    const hasSemanticCategory = !!propMap['semantic_category'];
    const hasSegmentation = hasInstanceId && hasSemanticCategory;

    let instanceIds = null;
    let semanticCategories = null;

    if (hasSegmentation) {
        instanceIds = new Float32Array(vertexCount);
        semanticCategories = new Float32Array(vertexCount);
        let anyNonZero = false;
        for (let i = 0; i < vertexCount; i++) {
            instanceIds[i] = readFloat(i, 'instance_id');
            semanticCategories[i] = readFloat(i, 'semantic_category');
            if (semanticCategories[i] > 0) anyNonZero = true;
        }
        if (!anyNonZero) {
            instanceIds = null;
            semanticCategories = null;
        }
    }

    if (onProgress) onProgress(1);

    return {
        count: vertexCount,
        positions,
        covariances,
        colors,
        opacities,
        shCoeffs,
        shDegree,
        sgAxisSharp,
        sgColor,
        sgDegree,
        instanceIds,
        semanticCategories,
        hasSegmentation: !!(instanceIds && semanticCategories),
    };
}

// IEEE 754 half-precision (float16) → float32 decoder
function float16ToFloat32(h) {
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;

    if (exponent === 0) {
        if (mantissa === 0) {
            // Zero
            return sign ? -0 : 0;
        }
        // Subnormal
        const val = mantissa / 1024 * Math.pow(2, -14);
        return sign ? -val : val;
    }
    if (exponent === 31) {
        // Inf or NaN
        return mantissa === 0
            ? (sign ? -Infinity : Infinity)
            : NaN;
    }
    // Normal
    const val = (1 + mantissa / 1024) * Math.pow(2, exponent - 15);
    return sign ? -val : val;
}

// Bulk-read a float16 SoA section into a Float32Array
function readFloat16Section(dataView, byteOffset, count) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        out[i] = float16ToFloat32(dataView.getUint16(byteOffset + i * 2, true));
    }
    return out;
}

// Bulk-read a float32 SoA section into a Float32Array
function readFloat32Section(buffer, byteOffset, count) {
    return new Float32Array(buffer, byteOffset, count);
}

// Bulk-read a uint16 SoA section into a Float32Array (for segment IDs)
function readUint16AsFloat32Section(dataView, byteOffset, count) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
        out[i] = dataView.getUint16(byteOffset + i * 2, true);
    }
    return out;
}

export function parseSplat(buffer, onProgress) {
    const headerView = new DataView(buffer, 0, 128);
    const magic = String.fromCharCode(
        headerView.getUint8(0), headerView.getUint8(1),
        headerView.getUint8(2), headerView.getUint8(3)
    );
    if (magic !== 'SPLT') throw new Error('Invalid .splat file: bad magic');

    const version = headerView.getUint32(4, true);

    if (version === 2) {
        return parseSplatV2(buffer, headerView, onProgress);
    }

    // ---- v1 fallback (40 bytes/Gaussian, uint8 color/opacity) ----
    const gaussianCount = headerView.getUint32(8, true);

    const bytesPerGaussian = 40;
    const dataOffset = 128;
    const dataView = new DataView(buffer, dataOffset);

    const positions = new Float32Array(gaussianCount * 3);
    const covariances = new Float32Array(gaussianCount * 6);
    const colors = new Float32Array(gaussianCount * 3);
    const opacities = new Float32Array(gaussianCount);

    for (let i = 0; i < gaussianCount; i++) {
        const base = i * bytesPerGaussian;

        positions[i*3+0] = dataView.getFloat32(base + 0, true);
        positions[i*3+1] = dataView.getFloat32(base + 4, true);
        positions[i*3+2] = dataView.getFloat32(base + 8, true);

        covariances[i*6+0] = dataView.getFloat32(base + 12, true);
        covariances[i*6+1] = dataView.getFloat32(base + 16, true);
        covariances[i*6+2] = dataView.getFloat32(base + 20, true);
        covariances[i*6+3] = dataView.getFloat32(base + 24, true);
        covariances[i*6+4] = dataView.getFloat32(base + 28, true);
        covariances[i*6+5] = dataView.getFloat32(base + 32, true);

        colors[i*3+0] = dataView.getUint8(base + 36) / 255;
        colors[i*3+1] = dataView.getUint8(base + 37) / 255;
        colors[i*3+2] = dataView.getUint8(base + 38) / 255;

        opacities[i] = dataView.getUint8(base + 39) / 255;
    }

    return {
        count: gaussianCount,
        positions,
        covariances,
        colors,
        opacities,
        shCoeffs: null,
        shDegree: 0,
        sgAxisSharp: null,
        sgColor: null,
        sgDegree: 0,
        instanceIds: null,
        semanticCategories: null,
        hasSegmentation: false,
    };
}

function parseSplatV2(buffer, headerView, onProgress) {
    // .splat v2 binary format (SoA layout, float16 + float32 mix):
    // Header (128 bytes):
    //   0-3:   magic "SPLT"
    //   4-7:   version = 2
    //   8-11:  gaussianCount (uint32)
    //   12-15: shDegree (uint32)
    //   16-19: sgDegree (uint32)
    //   20-23: hasSegments (uint32, 0 or 1)
    //   24-127: reserved
    //
    // Data sections (SoA, contiguous, no padding):
    //   positions:   N × 3 × float32
    //   covariances: N × 6 × float16
    //   colors:      N × 3 × float16
    //   opacities:   N × 1 × float16
    //   shCoeffs:    N × 45 × float16  (if shDegree > 0)
    //   sgAxisSharp: N × sgDegree × 4 × float16  (if sgDegree > 0)
    //   sgColor:     N × sgDegree × 3 × float16  (if sgDegree > 0)
    //   segments:    N × 2 × uint16  (if hasSegments)

    const N = headerView.getUint32(8, true);
    const shDegree = headerView.getUint32(12, true);
    const sgDegree = headerView.getUint32(16, true);
    const hasSegments = headerView.getUint32(20, true) !== 0;

    if (onProgress) onProgress(0.1);

    const dataView = new DataView(buffer);
    let offset = 128;

    // Positions: N × 3 × float32
    const positions = readFloat32Section(buffer, offset, N * 3);
    offset += N * 3 * 4;

    if (onProgress) onProgress(0.2);

    // Covariances: N × 6 × float16
    const covariances = readFloat16Section(dataView, offset, N * 6);
    offset += N * 6 * 2;

    if (onProgress) onProgress(0.4);

    // Colors: N × 3 × float16
    const colors = readFloat16Section(dataView, offset, N * 3);
    offset += N * 3 * 2;

    // Opacities: N × 1 × float16
    const opacities = readFloat16Section(dataView, offset, N);
    offset += N * 2;

    if (onProgress) onProgress(0.6);

    // SH coefficients: N × 45 × float16 (only if shDegree > 0)
    let shCoeffs = null;
    if (shDegree > 0) {
        shCoeffs = readFloat16Section(dataView, offset, N * 45);
        offset += N * 45 * 2;
    }

    if (onProgress) onProgress(0.8);

    // SG axis+sharpness: N × sgDegree × 4 × float16
    let sgAxisSharp = null;
    let sgColor = null;
    if (sgDegree > 0) {
        sgAxisSharp = readFloat16Section(dataView, offset, N * sgDegree * 4);
        offset += N * sgDegree * 4 * 2;

        sgColor = readFloat16Section(dataView, offset, N * sgDegree * 3);
        offset += N * sgDegree * 3 * 2;
    }

    // Segments: N × 2 × uint16 → [semantic_category, instance_id]
    let instanceIds = null;
    let semanticCategories = null;
    let hasSegmentation = false;
    if (hasSegments) {
        semanticCategories = new Float32Array(N);
        instanceIds = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            semanticCategories[i] = dataView.getUint16(offset + i * 4, true);
            instanceIds[i] = dataView.getUint16(offset + i * 4 + 2, true);
        }
        hasSegmentation = true;
    }

    if (onProgress) onProgress(1);

    return {
        count: N,
        positions,
        covariances,
        colors,
        opacities,
        shCoeffs,
        shDegree,
        sgAxisSharp,
        sgColor,
        sgDegree,
        instanceIds,
        semanticCategories,
        hasSegmentation,
    };
}

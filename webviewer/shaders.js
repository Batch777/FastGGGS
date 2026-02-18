// GLSL shaders for WebGL2 Gaussian Splatting renderer
// Instanced quad rendering with data textures and EWA splatting projection

export const vertexShaderSource = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;

uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec2 uFocal;
uniform vec2 uViewport;
uniform int uTexWidth;
uniform int uSHDegree;
uniform int uSGDegree;
uniform float uKernelSize;
uniform vec3 uCameraPosition;

uniform sampler2D uPositionTex;
uniform sampler2D uCovATex;
uniform sampler2D uCovBTex;
uniform sampler2D uColorTex;
uniform sampler2D uSHTex;
uniform sampler2D uSGAxisSharpTex;
uniform sampler2D uSGColorTex;
uniform usampler2D uIndexTex;

out vec3 vColor;
out float vOpacity;
out vec2 vOffset;
out vec3 vConic;

const vec2 quadVertices[6] = vec2[6](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

// SH constants
const float C0 = 0.28209479177387814;
const float C1 = 0.4886025119029199;
const float C2_0 = 1.0925484305920792;
const float C2_1 = -1.0925484305920792;
const float C2_2 = 0.31539156525252005;
const float C2_3 = -1.0925484305920792;
const float C2_4 = 0.5462742152960396;
const float C3_0 = -0.5900435899266435;
const float C3_1 = 2.890611442640554;
const float C3_2 = -0.4570457994644658;
const float C3_3 = 0.3731763325901154;
const float C3_4 = -0.4570457994644658;
const float C3_5 = 1.445305721320277;
const float C3_6 = -0.5900435899266435;

ivec2 texCoord(int index) {
    return ivec2(index % uTexWidth, index / uTexWidth);
}

void main() {
    // Get sorted index for this instance
    uint sortedIdx = texelFetch(uIndexTex, texCoord(gl_InstanceID), 0).r;
    int idx = int(sortedIdx);

    // Fetch Gaussian data from textures
    vec4 posOpac = texelFetch(uPositionTex, texCoord(idx), 0);
    vec4 covA = texelFetch(uCovATex, texCoord(idx), 0);
    vec4 covB = texelFetch(uCovBTex, texCoord(idx), 0);
    vec4 colorData = texelFetch(uColorTex, texCoord(idx), 0);

    vec3 pos = posOpac.xyz;
    float opacity = posOpac.w;
    vec3 color = colorData.rgb;

    // 3D covariance (symmetric: s00,s01,s02,s11,s12,s22)
    float s00 = covA.x, s01 = covA.y, s02 = covA.z, s11 = covA.w;
    float s12 = covB.x, s22 = covB.y;

    // Transform to camera space
    vec4 camPos4 = uViewMatrix * vec4(pos, 1.0);
    vec3 t = camPos4.xyz;

    // Cull if behind camera
    if (t.z <= 0.2) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // Compute view direction for SH/SG evaluation
    if (uSHDegree > 0 || uSGDegree > 0) {
        vec3 viewDir = normalize(pos - uCameraPosition);

        if (uSHDegree > 0) {
            float x = viewDir.x, y = viewDir.y, z = viewDir.z;

            // SH data: packed as (sh_r[0..14], sh_g[0..14], sh_b[0..14]) across texels
            // Each texel holds 4 floats, so we need ceil(45/4) = 12 texels per Gaussian
            int shBase = idx * 12;

            vec4 sh0 = texelFetch(uSHTex, texCoord(shBase + 0), 0);
            vec4 sh1 = texelFetch(uSHTex, texCoord(shBase + 1), 0);
            vec4 sh2 = texelFetch(uSHTex, texCoord(shBase + 2), 0);
            // Degree 1: sh0=(R0,R1,R2,G0), sh1=(G1,G2,B0,B1), sh2=(B2,...)
            color.r += C1 * (-y * sh0.x + z * sh0.y - x * sh0.z);
            color.g += C1 * (-y * sh0.w + z * sh1.x - x * sh1.y);
            color.b += C1 * (-y * sh1.z + z * sh1.w - x * sh2.x);

            if (uSHDegree >= 2) {
                float xx = x*x, yy = y*y, zz = z*z, xy = x*y, yz = y*z, xz = x*z;
                vec4 sh3 = texelFetch(uSHTex, texCoord(shBase + 3), 0);
                vec4 sh4 = texelFetch(uSHTex, texCoord(shBase + 4), 0);
                vec4 sh5 = texelFetch(uSHTex, texCoord(shBase + 5), 0);
                color.r += C2_0*xy*sh2.y + C2_1*yz*sh2.z + C2_2*(2.0*zz-xx-yy)*sh2.w + C2_3*xz*sh3.x + C2_4*(xx-yy)*sh3.y;
                color.g += C2_0*xy*sh3.z + C2_1*yz*sh3.w + C2_2*(2.0*zz-xx-yy)*sh4.x + C2_3*xz*sh4.y + C2_4*(xx-yy)*sh4.z;
                color.b += C2_0*xy*sh4.w + C2_1*yz*sh5.x + C2_2*(2.0*zz-xx-yy)*sh5.y + C2_3*xz*sh5.z + C2_4*(xx-yy)*sh5.w;

                if (uSHDegree >= 3) {
                    vec4 sh6 = texelFetch(uSHTex, texCoord(shBase + 6), 0);
                    vec4 sh7 = texelFetch(uSHTex, texCoord(shBase + 7), 0);
                    vec4 sh8 = texelFetch(uSHTex, texCoord(shBase + 8), 0);
                    vec4 sh9 = texelFetch(uSHTex, texCoord(shBase + 9), 0);
                    vec4 sh10 = texelFetch(uSHTex, texCoord(shBase + 10), 0);
                    vec4 sh11 = texelFetch(uSHTex, texCoord(shBase + 11), 0);
                    color.r += C3_0*y*(3.0*xx-yy)*sh6.x + C3_1*xy*z*sh6.y + C3_2*y*(4.0*zz-xx-yy)*sh6.z + C3_3*z*(2.0*zz-3.0*xx-3.0*yy)*sh6.w + C3_4*x*(4.0*zz-xx-yy)*sh7.x + C3_5*z*(xx-yy)*sh7.y + C3_6*x*(xx-3.0*yy)*sh7.z;
                    color.g += C3_0*y*(3.0*xx-yy)*sh7.w + C3_1*xy*z*sh8.x + C3_2*y*(4.0*zz-xx-yy)*sh8.y + C3_3*z*(2.0*zz-3.0*xx-3.0*yy)*sh8.z + C3_4*x*(4.0*zz-xx-yy)*sh8.w + C3_5*z*(xx-yy)*sh9.x + C3_6*x*(xx-3.0*yy)*sh9.y;
                    color.b += C3_0*y*(3.0*xx-yy)*sh9.z + C3_1*xy*z*sh9.w + C3_2*y*(4.0*zz-xx-yy)*sh10.x + C3_3*z*(2.0*zz-3.0*xx-3.0*yy)*sh10.y + C3_4*x*(4.0*zz-xx-yy)*sh10.z + C3_5*z*(xx-yy)*sh10.w + C3_6*x*(xx-3.0*yy)*sh11.x;
                }
            }
        }

        // Spherical Gaussian evaluation (matches CUDA computeColorFromSHSG)
        // Each lobe: color += sg_color * exp(sg_sharpness * (dot(sg_axis, viewDir) - 1.0))
        for (int g = 0; g < uSGDegree; g++) {
            int sgIdx = idx * uSGDegree + g;
            vec4 axisSharp = texelFetch(uSGAxisSharpTex, texCoord(sgIdx), 0);
            vec3 sgCol = texelFetch(uSGColorTex, texCoord(sgIdx), 0).rgb;
            float gaussian = exp(axisSharp.w * (dot(axisSharp.xyz, viewDir) - 1.0));
            color += sgCol * gaussian;
        }

        // Clamp after all color contributions (matches CUDA: max(result + 0.5, 0))
        color = max(color, vec3(0.0));
    }

    // Clamp Gaussian center to frustum to avoid numerical issues
    // Use FoV-dependent limits matching CUDA rasterizer (render_forward.cu:90-95)
    float tanFovx = uViewport.x / (2.0 * uFocal.x);
    float tanFovy = uViewport.y / (2.0 * uFocal.y);
    float limx = 1.3 * tanFovx * t.z;
    float limy = 1.3 * tanFovy * t.z;
    t.x = clamp(t.x, -limx, limx);
    t.y = clamp(t.y, -limy, limy);

    // EWA Jacobian (column-major, matching CUDA rasterizer convention)
    float fx = uFocal.x;
    float fy = uFocal.y;
    float z2 = t.z * t.z;

    mat3 J = mat3(
        fx / t.z, 0.0, -(fx * t.x) / z2,
        0.0, fy / t.z, -(fy * t.y) / z2,
        0.0, 0.0, 0.0
    );

    // W = transpose of rotation part of view matrix
    mat3 W = transpose(mat3(uViewMatrix));
    mat3 T = W * J;

    // 3D covariance matrix (symmetric, column-major)
    mat3 Vrk = mat3(
        s00, s01, s02,
        s01, s11, s12,
        s02, s12, s22
    );

    // 2D covariance: cov2D = T^T * Vrk * T (upper-left 2x2 is the 2D cov)
    mat3 cov2D_full = transpose(T) * Vrk * T;

    // Extract 2x2 with optional anti-aliasing kernel on diagonal
    float a_raw = cov2D_full[0][0];
    float b = cov2D_full[0][1];
    float c_raw = cov2D_full[1][1];
    float a = a_raw + uKernelSize;
    float c = c_raw + uKernelSize;

    // kernel_size opacity correction (matches CUDA render_forward.cu:194-196)
    float det_before = max(1e-6, a_raw * c_raw - b * b);
    float det_after = max(1e-6, a * c - b * b);
    opacity *= sqrt(det_before / det_after);

    // Determinant check
    float det = det_after;
    if (det <= 0.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // Conic (inverse of 2D covariance)
    float invDet = 1.0 / det;
    vConic = vec3(c * invDet, -b * invDet, a * invDet);

    // Eigenvalues for bounding radius
    float mid = 0.5 * (a + c);
    float disc = sqrt(max(0.1, mid * mid - det));
    float lambda1 = mid + disc;
    float radius = ceil(3.0 * sqrt(lambda1));

    if (radius <= 0.0 || radius > 2048.0) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    // Project center to clip space
    vec4 clipPos = uProjectionMatrix * camPos4;

    // Expand quad in pixel space, convert to NDC offset
    vec2 quadPos = quadVertices[gl_VertexID];
    vec2 pixelOffset = quadPos * radius;
    vec2 ndcOffset = pixelOffset * 2.0 / uViewport;

    gl_Position = vec4(
        clipPos.xy + ndcOffset * clipPos.w,
        clipPos.z,
        clipPos.w
    );

    vOffset = pixelOffset;
    vColor = color;
    vOpacity = opacity;
}
`;

export const fragmentShaderSource = `#version 300 es
precision highp float;

in vec3 vColor;
in float vOpacity;
in vec2 vOffset;
in vec3 vConic;

out vec4 fragColor;

void main() {
    // Evaluate 2D Gaussian: -0.5 * (conic.x*dx^2 + conic.z*dy^2) - conic.y*dx*dy
    float power = -0.5 * (vConic.x * vOffset.x * vOffset.x +
                           vConic.z * vOffset.y * vOffset.y) -
                   vConic.y * vOffset.x * vOffset.y;

    if (power > 0.0) discard;

    float alpha = min(0.99, vOpacity * exp(power));
    if (alpha < 1.0 / 255.0) discard;

    // Premultiplied alpha output
    fragColor = vec4(vColor * alpha, alpha);
}
`;

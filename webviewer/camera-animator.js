// Camera animation engine: prev/next stepping, smooth Catmull-Rom flythrough
// No DOM dependencies — pure animation math + state machine

export class CameraAnimator {
    constructor() {
        this.cameras = [];
        this.currentIndex = -1;
        this.playing = false;
        this.t = 0;           // animation time in [0, N) where integer = keyframe
        this.speed = 0.5;     // cameras per second
        this.loop = true;
    }

    setCameras(camerasData) {
        this.cameras = camerasData || [];
        this.currentIndex = this.cameras.length > 0 ? 0 : -1;
        this.playing = false;
        this.t = 0;
    }

    next() {
        if (this.cameras.length === 0) return -1;
        this.currentIndex = (this.currentIndex + 1) % this.cameras.length;
        this.t = this.currentIndex;
        this.playing = false;
        return this.currentIndex;
    }

    prev() {
        if (this.cameras.length === 0) return -1;
        this.currentIndex = (this.currentIndex - 1 + this.cameras.length) % this.cameras.length;
        this.t = this.currentIndex;
        this.playing = false;
        return this.currentIndex;
    }

    goTo(index) {
        if (index < 0 || index >= this.cameras.length) return;
        this.currentIndex = index;
        this.t = index;
        this.playing = false;
    }

    getCurrentIndex() {
        return this.currentIndex;
    }

    play() {
        if (this.cameras.length < 2) return;
        this.playing = true;
    }

    pause() {
        this.playing = false;
    }

    stop() {
        this.playing = false;
        this.t = 0;
        this.currentIndex = this.cameras.length > 0 ? 0 : -1;
    }

    toggle() {
        if (this.playing) this.pause();
        else this.play();
    }

    isPlaying() {
        return this.playing;
    }

    setSpeed(camsPerSec) {
        this.speed = Math.max(0.01, camsPerSec);
    }

    setLoop(val) {
        this.loop = val;
    }

    seekTo(fraction) {
        // fraction in [0, 1] → t in [0, N-1] (or [0, N) if looping)
        const N = this.cameras.length;
        if (N < 2) return;
        const maxT = this.loop ? N : N - 1;
        this.t = fraction * maxT;
        this.currentIndex = Math.floor(this.t) % N;
    }

    getProgress() {
        const N = this.cameras.length;
        if (N < 2) return 0;
        const maxT = this.loop ? N : N - 1;
        return this.t / maxT;
    }

    // Called each frame. Returns interpolated camera object if playing, or null.
    // Checks controls.useDirectMatrix — if user interacted (set to false by mouse),
    // auto-stops animation.
    update(dt, controls) {
        if (!this.playing || this.cameras.length < 2) return null;

        // If user grabbed the camera, stop animation
        if (!controls.useDirectMatrix) {
            this.playing = false;
            return null;
        }

        const N = this.cameras.length;
        this.t += this.speed * dt;

        if (this.loop) {
            // Wrap around
            if (this.t >= N) this.t -= N;
        } else {
            // Clamp and stop at end
            if (this.t >= N - 1) {
                this.t = N - 1;
                this.playing = false;
            }
        }

        this.currentIndex = Math.floor(this.t) % N;
        return interpolateCamera(this.cameras, this.t, this.loop);
    }
}

// ─── Math helpers ───

function rotationToQuat(R) {
    // 3x3 row-major C2W rotation (array of 3 rows, each [r0,r1,r2]) → quaternion [w,x,y,z]
    const m00 = R[0][0], m01 = R[0][1], m02 = R[0][2];
    const m10 = R[1][0], m11 = R[1][1], m12 = R[1][2];
    const m20 = R[2][0], m21 = R[2][1], m22 = R[2][2];

    const trace = m00 + m11 + m22;
    let w, x, y, z;

    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1.0);
        w = 0.25 / s;
        x = (m21 - m12) * s;
        y = (m02 - m20) * s;
        z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
        const s = 2.0 * Math.sqrt(1.0 + m00 - m11 - m22);
        w = (m21 - m12) / s;
        x = 0.25 * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const s = 2.0 * Math.sqrt(1.0 + m11 - m00 - m22);
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25 * s;
        z = (m12 + m21) / s;
    } else {
        const s = 2.0 * Math.sqrt(1.0 + m22 - m00 - m11);
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25 * s;
    }

    // Normalize
    const len = Math.sqrt(w*w + x*x + y*y + z*z);
    return [w/len, x/len, y/len, z/len];
}

function quatSlerp(q0, q1, t) {
    // Spherical linear interpolation between two quaternions [w,x,y,z]
    let dot = q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3];

    // Ensure shortest path
    let b = q1;
    if (dot < 0) {
        dot = -dot;
        b = [-q1[0], -q1[1], -q1[2], -q1[3]];
    }

    if (dot > 0.9995) {
        // Very close — linear interpolation + normalize
        const out = [
            q0[0] + t * (b[0] - q0[0]),
            q0[1] + t * (b[1] - q0[1]),
            q0[2] + t * (b[2] - q0[2]),
            q0[3] + t * (b[3] - q0[3]),
        ];
        const len = Math.sqrt(out[0]*out[0] + out[1]*out[1] + out[2]*out[2] + out[3]*out[3]);
        return [out[0]/len, out[1]/len, out[2]/len, out[3]/len];
    }

    const theta = Math.acos(dot);
    const sinTheta = Math.sin(theta);
    const w0 = Math.sin((1 - t) * theta) / sinTheta;
    const w1 = Math.sin(t * theta) / sinTheta;

    return [
        w0 * q0[0] + w1 * b[0],
        w0 * q0[1] + w1 * b[1],
        w0 * q0[2] + w1 * b[2],
        w0 * q0[3] + w1 * b[3],
    ];
}

function quatToRotation(q) {
    // quaternion [w,x,y,z] → 3x3 row-major C2W rotation (array of 3 rows)
    const [w, x, y, z] = q;
    return [
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ];
}

function catmullRom(p0, p1, p2, p3, t) {
    // Catmull-Rom spline evaluation for a single component
    // Standard: 0.5 * ((2*p1) + (-p0+p2)*t + (2*p0-5*p1+4*p2-p3)*t^2 + (-p0+3*p1-3*p2+p3)*t^3)
    const t2 = t * t;
    const t3 = t2 * t;
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
        (-p0 + 3*p1 - 3*p2 + p3) * t3
    );
}

function catmullRomVec3(p0, p1, p2, p3, t) {
    return [
        catmullRom(p0[0], p1[0], p2[0], p3[0], t),
        catmullRom(p0[1], p1[1], p2[1], p3[1], t),
        catmullRom(p0[2], p1[2], p2[2], p3[2], t),
    ];
}

function interpolateCamera(cameras, t, loop) {
    const N = cameras.length;
    if (N === 0) return null;
    if (N === 1) return cameras[0];

    // Segment index and local t
    const seg = Math.floor(t);
    const frac = t - seg;

    const idx1 = ((seg % N) + N) % N;
    const idx2 = loop ? (idx1 + 1) % N : Math.min(idx1 + 1, N - 1);

    // For Catmull-Rom, we need idx0 (before idx1) and idx3 (after idx2)
    const idx0 = loop ? (idx1 - 1 + N) % N : Math.max(idx1 - 1, 0);
    const idx3 = loop ? (idx2 + 1) % N : Math.min(idx2 + 1, N - 1);

    const c0 = cameras[idx0], c1 = cameras[idx1], c2 = cameras[idx2], c3 = cameras[idx3];

    // Position: Catmull-Rom spline
    const pos = catmullRomVec3(c0.position, c1.position, c2.position, c3.position, frac);

    // Rotation: SLERP between c1 and c2
    const q1 = rotationToQuat(c1.rotation);
    const q2 = rotationToQuat(c2.rotation);
    const qInterp = quatSlerp(q1, q2, frac);
    const rotation = quatToRotation(qInterp);

    // Focal length: linear interpolation
    const fx = c1.fx + frac * (c2.fx - c1.fx);
    const fy = c1.fy + frac * (c2.fy - c1.fy);

    // Image dimensions: use c1's (they should be constant, but interpolate for safety)
    const width = c1.width + frac * (c2.width - c1.width);
    const height = c1.height + frac * (c2.height - c1.height);

    return { position: pos, rotation, fx, fy, width, height };
}

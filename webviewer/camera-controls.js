// Orbit / pan / zoom camera controls with mouse + touch support
// Outputs column-major Float32Array(16) view and projection matrices

export class CameraControls {
    constructor(canvas) {
        this.canvas = canvas;
        this.dirty = true;

        // Camera state (spherical coordinates around target)
        this.target = [0, 0, 0];
        this.distance = 5.0;
        this.theta = 0;   // azimuth (radians)
        this.phi = Math.PI / 4; // elevation (radians)

        // Direct matrix mode (used when loading cameras.json)
        this.useDirectMatrix = false;
        this.directViewMatrix = null;
        this.directFx = 0;
        this.directFy = 0;
        this.directWidth = 0;   // original image width from cameras.json
        this.directHeight = 0;  // original image height from cameras.json

        // Camera intrinsics
        // CUDA rasterizer has no far-plane clipping (only z > 0.2 near cull),
        // so use a very large far to avoid clipping any Gaussians
        this.fovY = 60 * Math.PI / 180;
        this.near = 0.2;
        this.far = 100000.0;
        this.fx = 0;
        this.fy = 0;

        // Interaction state
        this._dragging = false;
        this._panning = false;
        this._lastX = 0;
        this._lastY = 0;
        this._pinchDist = 0;

        this._bindEvents();
    }

    _bindEvents() {
        const c = this.canvas;

        c.addEventListener('mousedown', (e) => {
            e.preventDefault();
            if (e.button === 0) { this._dragging = true; }
            if (e.button === 1 || e.button === 2) { this._panning = true; }
            this._lastX = e.clientX;
            this._lastY = e.clientY;
            this.useDirectMatrix = false;
        });

        window.addEventListener('mousemove', (e) => {
            if (!this._dragging && !this._panning) return;
            const dx = e.clientX - this._lastX;
            const dy = e.clientY - this._lastY;
            this._lastX = e.clientX;
            this._lastY = e.clientY;

            if (this._dragging) {
                this.theta -= dx * 0.005;
                this.phi -= dy * 0.005;
                this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi));
            }
            if (this._panning) {
                const panSpeed = this.distance * 0.002;
                // Pan in camera-local right/up
                const right = this._getCameraRight();
                const up = this._getCameraUp();
                this.target[0] -= (dx * right[0] + dy * up[0]) * panSpeed;
                this.target[1] -= (dx * right[1] + dy * up[1]) * panSpeed;
                this.target[2] -= (dx * right[2] + dy * up[2]) * panSpeed;
            }
            this.dirty = true;
        });

        window.addEventListener('mouseup', () => {
            this._dragging = false;
            this._panning = false;
        });

        c.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.useDirectMatrix = false;
            this.distance *= 1 + e.deltaY * 0.001;
            this.distance = Math.max(0.01, Math.min(1000, this.distance));
            this.dirty = true;
        }, { passive: false });

        c.addEventListener('contextmenu', (e) => e.preventDefault());

        // Touch events
        c.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.useDirectMatrix = false;
            if (e.touches.length === 1) {
                this._dragging = true;
                this._lastX = e.touches[0].clientX;
                this._lastY = e.touches[0].clientY;
            } else if (e.touches.length === 2) {
                this._dragging = false;
                this._panning = true;
                this._pinchDist = Math.hypot(
                    e.touches[1].clientX - e.touches[0].clientX,
                    e.touches[1].clientY - e.touches[0].clientY
                );
                this._lastX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                this._lastY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
            }
        }, { passive: false });

        c.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches.length === 1 && this._dragging) {
                const dx = e.touches[0].clientX - this._lastX;
                const dy = e.touches[0].clientY - this._lastY;
                this._lastX = e.touches[0].clientX;
                this._lastY = e.touches[0].clientY;
                this.theta -= dx * 0.005;
                this.phi -= dy * 0.005;
                this.phi = Math.max(0.01, Math.min(Math.PI - 0.01, this.phi));
                this.dirty = true;
            } else if (e.touches.length === 2) {
                // Pinch zoom
                const dist = Math.hypot(
                    e.touches[1].clientX - e.touches[0].clientX,
                    e.touches[1].clientY - e.touches[0].clientY
                );
                this.distance *= this._pinchDist / dist;
                this.distance = Math.max(0.01, Math.min(1000, this.distance));
                this._pinchDist = dist;

                // Two-finger pan
                const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
                const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
                const dx = cx - this._lastX;
                const dy = cy - this._lastY;
                this._lastX = cx;
                this._lastY = cy;

                const panSpeed = this.distance * 0.002;
                const right = this._getCameraRight();
                const up = this._getCameraUp();
                this.target[0] -= (dx * right[0] + dy * up[0]) * panSpeed;
                this.target[1] -= (dx * right[1] + dy * up[1]) * panSpeed;
                this.target[2] -= (dx * right[2] + dy * up[2]) * panSpeed;
                this.dirty = true;
            }
        }, { passive: false });

        c.addEventListener('touchend', () => {
            this._dragging = false;
            this._panning = false;
        });
    }

    _getCameraPosition() {
        const sinPhi = Math.sin(this.phi);
        return [
            this.target[0] + this.distance * sinPhi * Math.sin(this.theta),
            this.target[1] + this.distance * Math.cos(this.phi),
            this.target[2] + this.distance * sinPhi * Math.cos(this.theta)
        ];
    }

    _getCameraRight() {
        return [Math.cos(this.theta), 0, -Math.sin(this.theta)];
    }

    _getCameraUp() {
        const cosPhi = Math.cos(this.phi);
        const sinPhi = Math.sin(this.phi);
        return [
            -cosPhi * Math.sin(this.theta),
            sinPhi,
            -cosPhi * Math.cos(this.theta)
        ];
    }

    setFromCamera(cam, width, height) {
        // cameras.json format:
        //   rotation: C2W rotation matrix (3x3 row-major, [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])
        //   position: camera center in world coordinates [x, y, z]
        //   fx, fy: focal lengths in pixels
        //   width, height: image dimensions

        const R_c2w = cam.rotation; // 3x3 as array of 3 rows
        const pos = cam.position;   // [x, y, z]

        // W2C rotation = transpose of C2W rotation
        const R = [
            R_c2w[0][0], R_c2w[1][0], R_c2w[2][0],
            R_c2w[0][1], R_c2w[1][1], R_c2w[2][1],
            R_c2w[0][2], R_c2w[1][2], R_c2w[2][2]
        ];

        // W2C translation = -R_w2c * position
        const tx = -(R[0]*pos[0] + R[1]*pos[1] + R[2]*pos[2]);
        const ty = -(R[3]*pos[0] + R[4]*pos[1] + R[5]*pos[2]);
        const tz = -(R[6]*pos[0] + R[7]*pos[1] + R[8]*pos[2]);

        // W2C view matrix in column-major for WebGL
        this.directViewMatrix = new Float32Array([
            R[0], R[3], R[6], 0,
            R[1], R[4], R[7], 0,
            R[2], R[5], R[8], 0,
            tx,   ty,   tz,   1
        ]);

        this.directFx = cam.fx;
        this.directFy = cam.fy;
        this.directWidth = cam.width;
        this.directHeight = cam.height;
        this.useDirectMatrix = true;

        // Also update orbit camera state to approximately match
        this.target = [...pos];
        this.distance = 0.01;
        this.dirty = true;
    }

    getViewMatrix() {
        if (this.useDirectMatrix && this.directViewMatrix) {
            return this.directViewMatrix;
        }

        const eye = this._getCameraPosition();
        return lookAt(eye, this.target, [0, 1, 0]);
    }

    getProjectionMatrix(width, height) {
        if (this.useDirectMatrix && this.directFx > 0) {
            // Scale focal lengths proportionally to canvas size vs original image size
            const fx = this.directFx * width / this.directWidth;
            const fy = this.directFy * height / this.directHeight;
            return perspectiveFromFocal(fx, fy, width, height, this.near, this.far);
        }
        const aspect = width / height;
        return perspective(this.fovY, aspect, this.near, this.far);
    }

    getFocal(width, height) {
        if (this.useDirectMatrix && this.directFx > 0) {
            // Scale focal lengths proportionally to canvas size vs original image size
            const fx = this.directFx * width / this.directWidth;
            const fy = this.directFy * height / this.directHeight;
            return [fx, fy];
        }
        // For symmetric perspective with square pixels: fx = fy
        const fy = height / (2 * Math.tan(this.fovY / 2));
        const fx = fy;
        return [fx, fy];
    }
}

// Standard lookAt producing column-major W2C matrix
// Camera looks down +Z in camera space (matching CUDA rasterizer convention)
function lookAt(eye, target, up) {
    // Forward = normalize(target - eye)
    let fx = target[0] - eye[0], fy = target[1] - eye[1], fz = target[2] - eye[2];
    let fl = Math.sqrt(fx*fx + fy*fy + fz*fz);
    if (fl < 1e-10) fl = 1;
    fx /= fl; fy /= fl; fz /= fl;

    // Right = normalize(forward × up)
    let rx = fy * up[2] - fz * up[1];
    let ry = fz * up[0] - fx * up[2];
    let rz = fx * up[1] - fy * up[0];
    let rl = Math.sqrt(rx*rx + ry*ry + rz*rz);
    if (rl < 1e-10) rl = 1;
    rx /= rl; ry /= rl; rz /= rl;

    // True up = right × forward
    let ux = ry * fz - rz * fy;
    let uy = rz * fx - rx * fz;
    let uz = rx * fy - ry * fx;

    // W2C: camera looks down +Z (forward direction is +Z in camera space)
    // Rows of rotation: right, up, forward
    const tx = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
    const ty = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
    const tz = -(fx * eye[0] + fy * eye[1] + fz * eye[2]);

    return new Float32Array([
        rx, ux, fx, 0,
        ry, uy, fy, 0,
        rz, uz, fz, 0,
        tx, ty, tz, 1
    ]);
}

// Perspective projection from field of view (camera looks down +Z)
function perspective(fovY, aspect, near, far) {
    const t = Math.tan(fovY / 2);
    const fn = far - near;
    return new Float32Array([
        1 / (aspect * t), 0, 0, 0,
        0, 1 / t, 0, 0,
        0, 0, (far + near) / fn, 1,
        0, 0, -2 * far * near / fn, 0
    ]);
}

// Perspective projection from focal lengths (camera looks down +Z)
function perspectiveFromFocal(fx, fy, width, height, near, far) {
    const fn = far - near;
    return new Float32Array([
        2 * fx / width, 0, 0, 0,
        0, 2 * fy / height, 0, 0,
        0, 0, (far + near) / fn, 1,
        0, 0, -2 * far * near / fn, 0
    ]);
}

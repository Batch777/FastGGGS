// Web Worker: radix sort Gaussians by depth for back-to-front rendering

let positions = null;
let count = 0;
let sortRunning = false;

self.onmessage = function(e) {
    if (e.data.positions) {
        positions = e.data.positions;
        count = e.data.count;
        return;
    }

    if (e.data.viewMatrix) {
        if (sortRunning) return;
        sortRunning = true;
        const sorted = sortByDepth(e.data.viewMatrix);
        self.postMessage({ indices: sorted }, [sorted.buffer]);
        sortRunning = false;
    }
};

function sortByDepth(viewMatrix) {
    // Compute depth (z in camera space) for each Gaussian
    // viewMatrix is column-major: elements [2,6,10,14] form the z-row
    const vz0 = viewMatrix[2];
    const vz1 = viewMatrix[6];
    const vz2 = viewMatrix[10];
    const vz3 = viewMatrix[14];

    const depths = new Float32Array(count);
    const indices = new Uint32Array(count);

    for (let i = 0; i < count; i++) {
        const x = positions[i * 3];
        const y = positions[i * 3 + 1];
        const z = positions[i * 3 + 2];
        // Negate depth so ascending sort gives back-to-front order
        // (largest positive z = farthest becomes most negative = sorted first)
        depths[i] = -(vz0 * x + vz1 * y + vz2 * z + vz3);
        indices[i] = i;
    }

    radixSort(depths, indices, count);

    return indices;
}

function radixSort(depths, indices, n) {
    // Convert float depths to sortable uint32 keys
    const keys = new Uint32Array(n);
    const bitsView = new Uint32Array(depths.buffer);

    for (let i = 0; i < n; i++) {
        let bits = bitsView[i];
        // Float-to-sortable-uint32: if negative (sign bit set), flip all bits;
        // if positive, flip only sign bit. This makes the uint ordering match float ordering.
        if (bits & 0x80000000) {
            bits = ~bits; // negative: flip all
        } else {
            bits = bits ^ 0x80000000; // positive: flip sign bit
        }
        keys[i] = bits;
    }

    // 4-pass 8-bit radix sort
    const BITS = 8;
    const BUCKETS = 1 << BITS;
    const MASK = BUCKETS - 1;

    let srcKeys = keys;
    let srcIdx = indices;
    let dstKeys = new Uint32Array(n);
    let dstIdx = new Uint32Array(n);

    for (let pass = 0; pass < 4; pass++) {
        const shift = pass * BITS;

        // Count occurrences
        const counts = new Uint32Array(BUCKETS);
        for (let i = 0; i < n; i++) {
            const bucket = (srcKeys[i] >>> shift) & MASK;
            counts[bucket]++;
        }

        // Prefix sum
        const offsets = new Uint32Array(BUCKETS);
        let total = 0;
        for (let i = 0; i < BUCKETS; i++) {
            offsets[i] = total;
            total += counts[i];
        }

        // Scatter
        for (let i = 0; i < n; i++) {
            const bucket = (srcKeys[i] >>> shift) & MASK;
            const dst = offsets[bucket]++;
            dstKeys[dst] = srcKeys[i];
            dstIdx[dst] = srcIdx[i];
        }

        // Swap
        const tmpK = srcKeys; srcKeys = dstKeys; dstKeys = tmpK;
        const tmpI = srcIdx; srcIdx = dstIdx; dstIdx = tmpI;
    }

    // If we ended on the temp buffer, copy back
    if (srcIdx !== indices) {
        indices.set(srcIdx);
    }
}

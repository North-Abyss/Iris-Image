/**
 * Iris-Image Interactive Quantum Vision Studio
 * Author: Yuvanesh KS (@North-Abyss)
 */

let canvas, ctx;
let isDrawing = false;
let brushSize = 22;
let autoPredict = true;
let miniCanvas, miniCtx;
let sampleData = [];
let debounceTimer = null;

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initCanvas();
    fetchSamples();
    fetchModelInfo();
});

function switchAppTheme(themeName) {
    document.documentElement.setAttribute('data-theme', themeName);
    localStorage.setItem('iris_theme', themeName);
    const select = document.getElementById('theme-select');
    if (select && select.value !== themeName) {
        select.value = themeName;
    }
}

function initTheme() {
    const saved = localStorage.getItem('iris_theme') || 'skeuomorphism-dark';
    switchAppTheme(saved);
}

function initCanvas() {
    canvas = document.getElementById('drawing-canvas');
    ctx = canvas.getContext('2d');

    miniCanvas = document.getElementById('mini-input-canvas');
    miniCtx = miniCanvas.getContext('2d');

    // Fill canvas with black
    clearCanvas();

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    // Touch events for mobile/tablets
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        isDrawing = true;
        ctx.beginPath();
        ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!isDrawing) return;
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#ffffff';
        ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
        ctx.stroke();
    }, { passive: false });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        stopDrawing();
    }, { passive: false });

    // Auto-predict checkbox
    const toggle = document.getElementById('auto-predict-toggle');
    if (toggle) {
        toggle.addEventListener('change', (e) => {
            autoPredict = e.target.checked;
        });
    }
}

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineWidth = brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.closePath();

    updateMiniThumbnail();

    if (autoPredict) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            predictFromCanvas();
        }, 150);
    }
}

function updateBrushSize(val) {
    brushSize = parseInt(val, 10);
    document.getElementById('brush-size-val').innerText = `${brushSize}px`;
}

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updateMiniThumbnail();
}

function updateMiniThumbnail() {
    // Resample the 280x280 canvas onto the 28x28 mini canvas
    miniCtx.drawImage(canvas, 0, 0, 28, 28);
}

function switchInputTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    const activePane = document.getElementById(`tab-${tabId}`);
    if (activePane) activePane.classList.add('active');
}

// -------------------------------------------------------------------------
// Inference & API Communication
// -------------------------------------------------------------------------

async function predictFromCanvas() {
    const dataUrl = canvas.toDataURL('image/png');
    showPredictingState(true);

    try {
        const response = await fetch('/predict_image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        });
        const data = await response.json();
        if (data.status === 'success') {
            renderPredictionResults(data);
        } else {
            console.error('Prediction error:', data.error);
        }
    } catch (err) {
        console.error('Fetch error:', err);
    } finally {
        showPredictingState(false);
    }
}

async function predictFromPixels(pixels) {
    showPredictingState(true);
    try {
        const response = await fetch('/predict_pixels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixels: pixels })
        });
        const data = await response.json();
        if (data.status === 'success') {
            renderPredictionResults(data);
        }
    } catch (err) {
        console.error('Fetch error:', err);
    } finally {
        showPredictingState(false);
    }
}

function showPredictingState(isLoading) {
    const chip = document.getElementById('latency-text');
    if (isLoading) {
        chip.innerText = 'Quantum Simulating...';
    }
}

// -------------------------------------------------------------------------
// UI Rendering
// -------------------------------------------------------------------------

function renderPredictionResults(res) {
    // 1. Hero Card
    const badge = document.getElementById('predicted-badge');
    const title = document.getElementById('predicted-title');
    const desc = document.getElementById('predicted-desc');
    const latency = document.getElementById('latency-text');

    badge.innerText = res.prediction;
    if (res.prediction === 0) {
        badge.style.color = '#38bdf8';
        badge.style.borderColor = '#38bdf8';
        badge.style.boxShadow = '0 0 25px rgba(56, 189, 248, 0.4)';
        title.innerText = 'Digit 0 (Loop Contour)';
        desc.innerText = 'Outer circular boundary activation detected';
    } else {
        badge.style.color = '#c084fc';
        badge.style.borderColor = '#a855f7';
        badge.style.boxShadow = '0 0 25px rgba(168, 85, 247, 0.4)';
        title.innerText = 'Digit 1 (Vertical Stroke)';
        desc.innerText = 'Central vertical stroke geometry detected';
    }

    latency.innerText = `${res.latency_ms} ms`;

    // Confidence Bars
    document.getElementById('conf-0-text').innerText = `${res.confidence_0}%`;
    document.getElementById('conf-0-bar').style.width = `${res.confidence_0}%`;

    document.getElementById('conf-1-text').innerText = `${res.confidence_1}%`;
    document.getElementById('conf-1-bar').style.width = `${res.confidence_1}%`;

    // 2. Stage 2: Downsampled 4x3 Grid
    const downView = document.getElementById('downsampled-grid-view');
    downView.innerHTML = '';
    res.downsampled_grid.forEach(row => {
        row.forEach(val => {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            const lum = Math.round(val * 255);
            cell.style.backgroundColor = `rgb(${lum}, ${lum}, ${lum})`;
            cell.style.color = lum > 128 ? '#000000' : '#ffffff';
            cell.innerText = val > 0.05 ? val.toFixed(1) : '';
            downView.appendChild(cell);
        });
    });

    // 3. Stage 3: Binarized 4x3 Grid
    const binView = document.getElementById('binarized-grid-view');
    binView.innerHTML = '';
    res.binarized_grid.forEach(row => {
        row.forEach(bit => {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.style.backgroundColor = bit === 1 ? '#ffffff' : '#000000';
            cell.style.color = bit === 1 ? '#000000' : '#444444';
            cell.innerText = bit;
            binView.appendChild(cell);
        });
    });

    // 4. Stage 4: 8-Trit Register
    const tritView = document.getElementById('trit-register-view');
    tritView.innerHTML = '';
    res.trit_vector.forEach((trit, idx) => {
        const cell = document.createElement('div');
        cell.className = `trit-cell trit-${trit}`;
        cell.innerText = `t${idx}:${trit}`;
        tritView.appendChild(cell);
    });

    // 5. Comparison Table
    document.getElementById('q-pred-val').innerText = res.prediction;
    document.getElementById('q-conf-val').innerText = `${res.confidence_1}%`;

    const classical = res.classical_comparison || {};
    if (classical['Classical MLP']) {
        document.getElementById('mlp-pred-val').innerText = classical['Classical MLP'].prediction;
        document.getElementById('mlp-conf-val').innerText = `${(classical['Classical MLP'].confidence_1 * 100).toFixed(1)}%`;
    }
    if (classical['SVM (RBF)']) {
        document.getElementById('svm-pred-val').innerText = classical['SVM (RBF)'].prediction;
        document.getElementById('svm-conf-val').innerText = `${(classical['SVM (RBF)'].confidence_1 * 100).toFixed(1)}%`;
    }
    if (classical['Random Forest']) {
        document.getElementById('rf-pred-val').innerText = classical['Random Forest'].prediction;
        document.getElementById('rf-conf-val').innerText = `${(classical['Random Forest'].confidence_1 * 100).toFixed(1)}%`;
    }
    if (classical['XGBoost']) {
        document.getElementById('xgb-pred-val').innerText = classical['XGBoost'].prediction;
        document.getElementById('xgb-conf-val').innerText = `${(classical['XGBoost'].confidence_1 * 100).toFixed(1)}%`;
    }

    // 6. Quantum SHAP Grid
    const shapView = document.getElementById('shap-grid-view');
    shapView.innerHTML = '';
    const flatShap = res.shap_attribution.flat();
    const maxVal = Math.max(...flatShap, 0.01);
    res.shap_attribution.forEach(row => {
        row.forEach(val => {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            const intensity = Math.min(val / 0.18, 1.0);
            cell.style.backgroundColor = `rgba(56, 189, 248, ${0.15 + intensity * 0.8})`;
            cell.style.color = intensity > 0.4 ? '#ffffff' : '#94a3b8';
            cell.innerText = val > 0.03 ? val.toFixed(2) : '';
            shapView.appendChild(cell);
        });
    });
}

// -------------------------------------------------------------------------
// Samples & Presets
// -------------------------------------------------------------------------

async function fetchSamples() {
    try {
        const response = await fetch('/api/samples');
        const data = await response.json();
        if (data.status === 'success' && data.samples.length > 0) {
            sampleData = data.samples;
            renderSamplesList(data.samples);
            // Auto predict sample 1 initially
            loadSampleIntoCanvas(data.samples[0]);
        }
    } catch (err) {
        console.error('Failed to load samples:', err);
    }
}

function renderSamplesList(samples) {
    const container = document.getElementById('samples-container');
    container.innerHTML = '';

    samples.forEach(sample => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.onclick = () => loadSampleIntoCanvas(sample);

        const badgeClass = sample.label === 0 ? 'zero' : 'one';
        card.innerHTML = `
            <div class="sample-badge ${badgeClass}">${sample.label}</div>
            <div class="sample-text">
                <strong>${sample.name}</strong>
                <span>Click to test</span>
            </div>
        `;
        container.appendChild(card);
    });
}

function loadSampleIntoCanvas(sample) {
    // Switch to Draw tab so user can see it or edit it
    switchInputTab('draw');

    // Draw the 4x3 pixels onto the 280x280 canvas
    const pixels = sample.pixels; // 12 numbers
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const cellW = canvas.width / 3;
    const cellH = canvas.height / 4;

    for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 3; c++) {
            const idx = r * 3 + c;
            const lum = Math.round(pixels[idx] * 255);
            if (lum > 20) {
                ctx.fillStyle = `rgb(${lum}, ${lum}, ${lum})`;
                // Draw rounded rect inside cell for natural stroke feel
                ctx.beginPath();
                ctx.roundRect(c * cellW + 12, r * cellH + 12, cellW - 24, cellH - 24, 16);
                ctx.fill();
            }
        }
    }

    updateMiniThumbnail();
    predictFromPixels(pixels);
}

// -------------------------------------------------------------------------
// Image Upload Handling
// -------------------------------------------------------------------------

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            // Draw uploaded image to main canvas
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            updateMiniThumbnail();

            // Preview
            const preview = document.getElementById('upload-preview');
            preview.src = event.target.result;
            document.getElementById('upload-preview-wrapper').style.display = 'block';

            // Auto-predict
            predictFromCanvas();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function predictUploadedImage() {
    predictFromCanvas();
}

async function fetchModelInfo() {
    try {
        const response = await fetch('/api/model_info');
        const info = await response.json();
        console.log('Qutrit VQC Model Info:', info);
    } catch (e) {
        // silent
    }
}

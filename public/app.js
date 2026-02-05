/* global Plotly */

const DEFAULT_FRAUD_API_SEED = 1337;

// Backend-driven fraud model state
let cachedFraudRoc = null;
let cachedDataHash = null;
let isTraining = false;
let pendingDataHash = null;
let trainingStatusHideTimer = null;

function setFraudControlsDisabled(disabled) {
    // While training is in-flight, the UI otherwise feels "stuck" because updates are queued.
    // Disabling controls makes the state obvious and prevents confusing intermediate states.
    const ids = [
        'model-type',
        'claims-slider',
        'fraud-rate-slider',
        'fraud-cost-slider',
        'investigation-cost-slider'
    ];

    ids.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = Boolean(disabled);
    });
}

function setFraudTrainingStatus(state, html) {
    const el = document.getElementById('training-status');
    if (!el) return;

    if (!state || state === 'hidden') {
        el.style.display = 'none';
        el.removeAttribute('data-state');
        return;
    }

    // If we were going to auto-hide a previous "done" message, cancel that.
    if (trainingStatusHideTimer) {
        clearTimeout(trainingStatusHideTimer);
        trainingStatusHideTimer = null;
    }

    el.style.display = 'block';
    el.setAttribute('data-state', state);
    if (typeof html === 'string') el.innerHTML = html;
}

function setFraudHeaderTrainingState({ modelName, queued = false } = {}) {
    const headerAuc = document.getElementById('header-auc');
    const headerRating = document.getElementById('header-rating');
    const aucMetric = document.getElementById('auc-metric');

    if (headerAuc) headerAuc.textContent = 'AUC: Training…';
    if (headerRating) headerRating.textContent = queued ? 'Queued' : 'Training';
    if (aucMetric) aucMetric.textContent = '…';

    const aucCard = document.getElementById('auc-card');
    if (aucCard) {
        aucCard.classList.remove('auc-bad', 'auc-needs-improvement', 'auc-good');
    }

    const msg = queued
        ? `<span class="training-spinner">⟳</span> Training in progress — queued update (${modelName || 'model'})…`
        : `<span class="training-spinner">⟳</span> Training ${modelName || 'model'}…`;

    setFraudTrainingStatus('training', msg);
}

function showFraudTrainingDone(modelName) {
    if (trainingStatusHideTimer) {
        clearTimeout(trainingStatusHideTimer);
        trainingStatusHideTimer = null;
    }

    setFraudTrainingStatus('done', `✓ Updated (${modelName})`);
    trainingStatusHideTimer = setTimeout(() => setFraudTrainingStatus('hidden'), 1200);
}

const sampleSizeByModel = {
    logistic: 8000,
    hist_gbdt: 8000,
    extra_trees: 6000,
    random_forest: 6000
};

function getModelDisplayName(modelType) {
    switch (modelType) {
        case 'hist_gbdt':
            return 'Gradient Boosting';
        case 'extra_trees':
            return 'Extra Trees';
        case 'random_forest':
            return 'Random Forest';
        case 'logistic':
        default:
            return 'Logistic Regression';
    }
}

async function fetchFraudRoc({ modelType, fraudRate, sampleSize, seed = DEFAULT_FRAUD_API_SEED }) {
    const res = await fetch('/api/fraud/roc', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
            model_type: modelType,
            fraud_rate: fraudRate,
            sample_size: sampleSize,
            seed
        })
    });

    if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`Fraud ROC API failed (${res.status}): ${text || res.statusText}`);
    }

    return res.json();
}

        // Simple hash for caching
        function hashParams(params) {
            return JSON.stringify(params);
        }

        // Chart styling for dark theme
        const chartLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(22,22,22,1)',
            font: { family: 'DM Sans, sans-serif', color: '#a8a8a0' },
            margin: { t: 50, r: 30, b: 100, l: 70 },
            xaxis: {
                gridcolor: 'rgba(255,255,255,0.06)',
                zerolinecolor: 'rgba(255,255,255,0.1)',
                tickfont: { size: 12 }
            },
            yaxis: {
                gridcolor: 'rgba(255,255,255,0.06)',
                zerolinecolor: 'rgba(255,255,255,0.1)',
                tickfont: { size: 12 }
            },
            legend: {
                bgcolor: 'rgba(0,0,0,0)',
                font: { size: 11 },
                orientation: 'h',
                x: 0.5,
                xanchor: 'center',
                y: -0.18,
                yanchor: 'top'
            }
        };

        const chartConfig = {
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'select2d', 'lasso2d', 'autoScale2d'],
            displaylogo: false,
            responsive: true
        };

        // Plotly charts are rendered inside <details> elements. When a <details> is collapsed,
        // its contents have no layout, and Plotly may compute an incorrect initial size.
        // Resize (and for the fraud case, re-render) when a use case is opened.
        function safeResizePlotly(divId) {
            if (!window.Plotly) return;
            const el = document.getElementById(divId);
            if (!el) return;
            if (!el.classList || !el.classList.contains('js-plotly-plot')) return;
            try {
                Plotly.Plots.resize(el);
            } catch (_) {
                // ignore
            }
        }

        function resizeAllCharts() {
            ['fraud-chart', 'fraud-business-chart', 'underwriting-chart', 'underwriting-business-chart', 'triage-chart', 'triage-business-chart']
                .forEach(safeResizePlotly);
        }

        // Color palette
        const colors = {
            gold: '#d4a853',
            emerald: '#4ade80',
            rose: '#f472b6',
            sky: '#38bdf8',
            purple: '#a78bfa',
            muted: '#6b6b65',
            line: 'rgba(255,255,255,0.2)'
        };

        // Helper function to generate ROC curve data
        function generateROCCurve(auc, numPoints = 100) {
            const fpr = [];
            const tpr = [];

            for (let i = 0; i <= numPoints; i++) {
                const x = i / numPoints;
                fpr.push(x);

                let y;
                if (auc > 0.9) {
                    y = Math.pow(x, 0.3) * (auc - 0.5) * 2;
                } else if (auc > 0.75) {
                    y = Math.pow(x, 0.5) * (auc - 0.5) * 2;
                } else {
                    y = Math.pow(x, 0.7) * (auc - 0.5) * 2;
                }

                y = Math.min(y, x + (1 - x) * (2 * auc - 1));
                y = Math.min(1, Math.max(0, y));
                tpr.push(y);
            }

            return { fpr, tpr };
        }

        // Format helpers
        function formatNumber(num) {
            return num.toLocaleString('en-US');
        }

        function formatCurrency(num) {
            if (num >= 1000000) {
                return '$' + (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 1000) {
                return '$' + formatNumber(Math.round(num));
            }
            return '$' + num;
        }

        function formatThreshold(thr) {
            if (!Number.isFinite(thr)) return String(thr);
            if (thr === 0 || thr === 1) return thr.toFixed(2);
            if (Math.abs(thr) < 0.001) return thr.toExponential(2);
            return thr.toFixed(3);
        }

        function getAUCRating(auc) {
            if (auc >= 0.9) return 'Excellent';
            if (auc >= 0.8) return 'Very Good';
            if (auc >= 0.7) return 'Good';
            if (auc >= 0.6) return 'Fair';
            return 'Poor';
        }

        // Generate TPR/FPR at thresholds (synthetic/teaching approximation)
        function getTPRFPRAtThresholds(auc, thresholds) {
            const tprValues = [];
            const fprValues = [];

            for (const thresh of thresholds) {
                const baseFPR = Math.pow(1 - thresh, 2) * 0.5;
                const fpr = baseFPR * (1.5 - auc);
                const baseTPR = 1 - Math.pow(thresh, 1.5);
                const tpr = baseTPR * (0.5 + auc);

                fprValues.push(Math.max(0.005, Math.min(0.5, fpr)));
                tprValues.push(Math.max(0.3, Math.min(0.99, tpr)));
            }

            return { tpr: tprValues, fpr: fprValues };
        }

        // For business charts, prefer using the *actual* ROC points (threshold, TPR, FPR)
        // computed from predictions. This avoids flat lines when model scores are not
        // well-calibrated to the 0.3–0.9 threshold range.
        function sampleROCForBusiness(rocResult, maxPoints = 25) {
            if (!rocResult || !Array.isArray(rocResult.thresholds) || !Array.isArray(rocResult.tpr) || !Array.isArray(rocResult.fpr)) {
                return null;
            }

            const n = Math.min(rocResult.thresholds.length, rocResult.tpr.length, rocResult.fpr.length);
            if (n < 2) return null;

            const step = Math.max(1, Math.floor(n / maxPoints));
            const points = [];

            for (let i = 0; i < n; i += step) {
                const thr = Number(rocResult.thresholds[i]);
                const tpr = Number(rocResult.tpr[i]);
                const fpr = Number(rocResult.fpr[i]);
                if (!Number.isFinite(thr) || !Number.isFinite(tpr) || !Number.isFinite(fpr)) continue;

                // Include the sentinel threshold (e.g. 1.01) as threshold=1 so the business chart
                // always has the "investigate none" operating point (TPR=0, FPR=0).
                const clampedThr = thr > 1 ? 1 : Math.max(0, Math.min(1, thr));

                points.push({ threshold: clampedThr, tpr, fpr });
            }

            // Ensure we include the last point.
            const lastIdx = n - 1;
            const lastThr = Number(rocResult.thresholds[lastIdx]);
            if (Number.isFinite(lastThr) && lastThr <= 1) {
                points.push({
                    threshold: Math.max(0, Math.min(1, lastThr)),
                    tpr: Number(rocResult.tpr[lastIdx]),
                    fpr: Number(rocResult.fpr[lastIdx])
                });
            }

            // Sort by threshold increasing for plotting.
            points.sort((a, b) => a.threshold - b.threshold);

            // De-dup thresholds (Plotly can get weird with identical x values).
            const deduped = [];
            for (const p of points) {
                const prev = deduped[deduped.length - 1];
                if (prev && Math.abs(prev.threshold - p.threshold) < 1e-6) continue;
                deduped.push(p);
            }

            return {
                thresholds: deduped.map(p => p.threshold),
                tpr: deduped.map(p => p.tpr),
                fpr: deduped.map(p => p.fpr)
            };
        }

        const thresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9];

        // Update fraud detection charts with REAL ML (Python backend)
        async function updateFraudChart() {
            const modelType = document.getElementById('model-type').value;
            const totalClaims = parseInt(document.getElementById('claims-slider').value);
            const fraudRate = parseFloat(document.getElementById('fraud-rate-slider').value) / 100;
            const avgFraudCost = parseInt(document.getElementById('fraud-cost-slider').value);
            const investigationCost = parseInt(document.getElementById('investigation-cost-slider').value);

            const totalFrauds = Math.round(totalClaims * fraudRate);
            const legitimateClaims = totalClaims - totalFrauds;

            const maxSampleSize = sampleSizeByModel[modelType] ?? 6000;
            const sampleSize = Math.min(maxSampleSize, totalClaims);
            const dataParams = { modelType, fraudRate, sampleSize };
            const currentHash = hashParams(dataParams);

            // If a request is in-flight for different params, don't redraw charts with stale
            // cached results. Instead, show clear status and queue a follow-up run.
            if (isTraining && currentHash !== cachedDataHash) {
                pendingDataHash = currentHash;
                setFraudControlsDisabled(true);
                setFraudHeaderTrainingState({ modelName: getModelDisplayName(modelType), queued: true });
                return;
            }

            let auc;
            let rocResult;
            let modelName = getModelDisplayName(modelType);

            if (currentHash !== cachedDataHash && !isTraining) {
                isTraining = true;
                pendingDataHash = null;

                setFraudControlsDisabled(true);
                setFraudHeaderTrainingState({ modelName });

                // Give the browser a chance to paint the updated DOM before doing network + backend work.
                await new Promise(resolve => (window.requestAnimationFrame ? requestAnimationFrame(resolve) : setTimeout(resolve, 0)));

                try {
                    const payload = await fetchFraudRoc({
                        modelType,
                        fraudRate,
                        sampleSize,
                        seed: DEFAULT_FRAUD_API_SEED
                    });

                    modelName = payload.model_name || modelName;
                    auc = Number(payload.auc);
                    rocResult = {
                        fpr: payload.fpr,
                        tpr: payload.tpr,
                        thresholds: payload.thresholds,
                        auc
                    };

                    cachedDataHash = currentHash;
                    cachedFraudRoc = rocResult;
                } catch (error) {
                    console.error('Training failed:', error);

                    // Fallback to synthetic curve (also provide thresholds so business charts can render)
                    if (modelType === 'hist_gbdt') auc = 0.88;
                    else if (modelType === 'extra_trees') auc = 0.86;
                    else if (modelType === 'random_forest') auc = 0.84;
                    else auc = 0.82;

                    const synthetic = generateROCCurve(auc);
                    rocResult = {
                        fpr: synthetic.fpr,
                        tpr: synthetic.tpr,
                        thresholds: synthetic.fpr.map((_, i) => 1 - i / (synthetic.fpr.length - 1)),
                        auc
                    };

                    cachedDataHash = currentHash;
                    cachedFraudRoc = rocResult;

                    setFraudTrainingStatus('error', '⚠ Training failed — using fallback curve');
                } finally {
                    isTraining = false;

                    // If the user changed params mid-train, immediately kick off the next run.
                    if (pendingDataHash && pendingDataHash !== cachedDataHash) {
                        // Keep controls disabled across chained training runs.
                        setFraudControlsDisabled(true);
                        setTimeout(updateFraudChart, 0);
                        return;
                    }

                    // Training completed; re-enable UI and briefly show a "done" message.
                    setFraudControlsDisabled(false);

                    const statusEl = document.getElementById('training-status');
                    const isError = statusEl && statusEl.getAttribute('data-state') === 'error';
                    if (!isError) {
                        showFraudTrainingDone(modelName);
                    }
                }
            } else {
                rocResult = cachedFraudRoc;

                if (rocResult && typeof rocResult.auc === 'number') {
                    auc = rocResult.auc;
                } else {
                    auc = modelType === 'hist_gbdt' ? 0.88 : 0.82;
                    const synthetic = generateROCCurve(auc);
                    rocResult = {
                        fpr: synthetic.fpr,
                        tpr: synthetic.tpr,
                        thresholds: synthetic.fpr.map((_, i) => 1 - i / (synthetic.fpr.length - 1)),
                        auc
                    };
                }

                // If cached ROC is missing thresholds (older synthetic fallback), add them.
                if (rocResult && !rocResult.thresholds && Array.isArray(rocResult.fpr)) {
                    rocResult.thresholds = rocResult.fpr.map((_, i) => 1 - i / (rocResult.fpr.length - 1));
                }
            }

            // Calculate business metrics across a set of operating points.
            // Prefer sampling actual ROC points (threshold -> TPR/FPR) to avoid flat business curves.
            let businessThresholds = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9];
            let fraudTPR = [];
            let fraudFPR = [];

            const sampled = sampleROCForBusiness(rocResult, 25);
            if (sampled) {
                businessThresholds = sampled.thresholds;
                fraudTPR = sampled.tpr;
                fraudFPR = sampled.fpr;
            } else {
                const approx = getTPRFPRAtThresholds(auc, businessThresholds);
                fraudTPR = approx.tpr;
                fraudFPR = approx.fpr;
            }

            const fraudPrevented = fraudTPR.map(tpr => tpr * totalFrauds * avgFraudCost / 1000000);
            const investigationCosts = fraudFPR.map((fpr, i) => {
                const fraudsInvestigated = fraudTPR[i] * totalFrauds;
                const falseAlarms = fpr * legitimateClaims;
                return (fraudsInvestigated + falseAlarms) * investigationCost / 1000000;
            });
            const netBenefit = fraudPrevented.map((fp, i) => fp - investigationCosts[i]);

            let maxBenefitIdx = 0;
            let maxBenefit = netBenefit[0];
            for (let i = 1; i < netBenefit.length; i++) {
                if (netBenefit[i] > maxBenefit) {
                    maxBenefit = netBenefit[i];
                    maxBenefitIdx = i;
                }
            }

            // Break-even point where fraud prevented ~= investigation costs
            let breakEvenIdx = 0;
            let minGap = Infinity;
            for (let i = 0; i < fraudPrevented.length; i++) {
                const gap = Math.abs(fraudPrevented[i] - investigationCosts[i]);
                if (gap < minGap) {
                    minGap = gap;
                    breakEvenIdx = i;
                }
            }
            const breakEvenThreshold = businessThresholds[breakEvenIdx];
            const breakEvenValue = (fraudPrevented[breakEvenIdx] + investigationCosts[breakEvenIdx]) / 2;

            const optimalThreshold = businessThresholds[maxBenefitIdx];
            const optimalTPR = fraudTPR[maxBenefitIdx];
            const optimalFPR = fraudFPR[maxBenefitIdx];
            const fraudsCaught = Math.round(optimalTPR * totalFrauds);
            const falseAlarms = Math.round(optimalFPR * legitimateClaims);
            const fraudPreventedDollars = optimalTPR * totalFrauds * avgFraudCost;
            const investigationDollars = (fraudsCaught + falseAlarms) * investigationCost;

            // Update displays
            document.getElementById('auc-metric').textContent = auc.toFixed(2);

            const aucCard = document.getElementById('auc-card');
            if (aucCard) {
                aucCard.classList.remove('auc-bad', 'auc-needs-improvement', 'auc-good');

                let state = 'auc-bad';
                if (auc >= 0.75) state = 'auc-good';
                else if (auc >= 0.65) state = 'auc-needs-improvement';

                aucCard.classList.add(state);
            }

            // Update header pills
            document.getElementById('header-auc').textContent = `AUC: ${auc.toFixed(2)}`;
            document.getElementById('header-rating').textContent = getAUCRating(auc);
            document.getElementById('claims-value').textContent = formatNumber(totalClaims);
            document.getElementById('fraud-rate-value').textContent = (fraudRate * 100).toFixed(1) + '%';
            document.getElementById('fraud-cost-value').textContent = formatCurrency(avgFraudCost);
            document.getElementById('investigation-cost-value').textContent = formatCurrency(investigationCost);

            document.getElementById('fraud-scenario-text').textContent =
                `You process ${formatNumber(totalClaims)} claims per month. About ${(fraudRate * 100).toFixed(1)}% are fraudulent (~${formatNumber(totalFrauds)} claims). Each fraudulent claim costs ${formatCurrency(avgFraudCost)} on average. Each investigation costs ${formatCurrency(investigationCost)}.`;

            document.getElementById('fraud-caught-metric').textContent = Math.round(optimalTPR * 100) + '%';
            document.getElementById('false-alarm-metric').textContent = Math.round(optimalFPR * 100) + '%';
            document.getElementById('net-benefit-metric').textContent = formatCurrency(maxBenefit * 1000000);

            // Color-code false alarm rate: green <10%, yellow 10-20%, red >20%
            const faCard = document.getElementById('false-alarm-card');
            if (faCard) {
                faCard.classList.remove('fa-good', 'fa-warning', 'fa-bad');
                const faPct = optimalFPR * 100;
                let faState = 'fa-good';
                if (faPct > 20) faState = 'fa-bad';
                else if (faPct >= 10) faState = 'fa-warning';
                faCard.classList.add(faState);
            }

            const investigated = fraudsCaught + falseAlarms;
            const investigatedPct = totalClaims > 0 ? (investigated / totalClaims) * 100 : 0;

            document.getElementById('fraud-optimal-text').innerHTML = `
                <div class="optimal-title">Optimal Operating Point (${modelName})</div>
                <p class="optimal-text">
                    At <strong>threshold = ${formatThreshold(optimalThreshold)}</strong>: Catch ${Math.round(optimalTPR * 100)}% of fraud (${formatNumber(fraudsCaught)} frauds) while flagging ${Math.round(optimalFPR * 100)}% of legitimate claims (${formatNumber(falseAlarms)} investigations).<br>
                    Investigate <strong>${formatNumber(investigated)}</strong> claims (<strong>${investigatedPct.toFixed(1)}%</strong> of volume).<br><br>
                    <strong>ROI:</strong> Prevent ${formatCurrency(fraudPreventedDollars)} in fraud, spend ${formatCurrency(investigationDollars)} on investigations = <strong>${formatCurrency(maxBenefit * 1000000)} net benefit</strong><br><br>
                    <em style="color: var(--accent-gold); font-size: 0.9rem;">Real ML Model — Computed AUC: ${auc.toFixed(3)}</em>
                </p>
            `;

            // ROC Chart
            Plotly.react('fraud-chart', [
                {
                    x: [0, 1],
                    y: [0, 1],
                    mode: 'lines',
                    name: 'Random (AUC = 0.5)',
                    line: { color: colors.muted, width: 1, dash: 'dash' }
                },
                {
                    x: rocResult.fpr,
                    y: rocResult.tpr,
                    mode: 'lines',
                    name: `${modelName} (AUC = ${auc.toFixed(2)})`,
                    line: { color: colors.gold, width: 3 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(212, 168, 83, 0.1)'
                },
                {
                    x: [optimalFPR],
                    y: [optimalTPR],
                    mode: 'markers',
                    name: 'Optimal Point',
                    marker: { color: colors.emerald, size: 14, symbol: 'diamond' }
                }
            ], {
                ...chartLayout,
                title: { text: `ROC Curve — Real ${modelName}`, font: { size: 16, color: '#f5f5f0' } },
                xaxis: { ...chartLayout.xaxis, title: 'False Positive Rate', range: [0, 1] },
                yaxis: { ...chartLayout.yaxis, title: 'True Positive Rate', range: [0, 1] },
                height: 450
            }, chartConfig);

            // Business Chart
            Plotly.react('fraud-business-chart', [
                {
                    x: businessThresholds,
                    y: fraudPrevented,
                    mode: 'lines+markers',
                    name: 'Fraud Prevented',
                    line: { color: colors.emerald, width: 2 },
                    marker: { size: 6 }
                },
                {
                    x: businessThresholds,
                    y: investigationCosts,
                    mode: 'lines+markers',
                    name: 'Investigation Costs',
                    line: { color: colors.rose, width: 2 },
                    marker: { size: 6 }
                },
                {
                    x: businessThresholds,
                    y: netBenefit,
                    mode: 'lines+markers',
                    name: 'Net Benefit',
                    line: { color: colors.gold, width: 3 },
                    marker: { size: 8 }
                },
                {
                    x: [optimalThreshold],
                    y: [maxBenefit],
                    mode: 'markers',
                    name: 'Optimal',
                    marker: { color: colors.emerald, size: 16, symbol: 'star' }
                },
                ...(Number.isFinite(breakEvenThreshold) && Number.isFinite(breakEvenValue)
                    ? [
                        {
                            x: [breakEvenThreshold],
                            y: [breakEvenValue],
                            mode: 'markers',
                            name: 'Break-even',
                            marker: { color: colors.sky, size: 12, symbol: 'x' }
                        }
                    ]
                    : [])
            ], {
                ...chartLayout,
                title: { text: 'Financial Impact by Threshold', font: { size: 16, color: '#f5f5f0' } },
                xaxis: { ...chartLayout.xaxis, title: 'Model Threshold' },
                yaxis: { ...chartLayout.yaxis, title: 'Dollars (Millions)' },
                height: 400
            }, chartConfig);
        }
export async function initApp() {

        // Model descriptions for sidebar
        const modelDescriptions = {
            logistic: {
                title: 'Logistic Regression',
                text: 'Linear baseline. Fast and interpretable, but can\'t capture non-linear fraud patterns. Use for baseline comparison or when explainability is required.'
            },
            hist_gbdt: {
                title: 'Gradient Boosting',
                text: 'Best performer. Captures complex non-linear patterns in fraud segments. Handles class imbalance well. Use when accuracy matters most.'
            },
            extra_trees: {
                title: 'Extra Trees',
                text: 'Strong ensemble with randomized splits. Good regularization, competitive with boosting. Provides feature importance for interpretability.'
            },
            random_forest: {
                title: 'Random Forest',
                text: 'Shallow trees with conservative settings. Demonstrates the cost of under-tuning—useful for showing stakeholders why hyperparameters matter.'
            }
        };

        function updateModelDescription(modelType) {
            const desc = modelDescriptions[modelType];
            if (desc) {
                document.querySelector('.model-description-title').textContent = desc.title;
                document.querySelector('.model-description-text').textContent = desc.text;
            }
        }

// Event listeners - model type triggers retrain, others just update calculations
        document.getElementById('model-type').addEventListener('change', (e) => {
            cachedDataHash = null; // Force retrain
            cachedFraudRoc = null;
            updateModelDescription(e.target.value);
            updateFraudChart();
        });

        // Fraud rate affects training data distribution
        document.getElementById('fraud-rate-slider').addEventListener('input', () => {
            cachedDataHash = null; // Force retrain
            cachedFraudRoc = null;
            updateFraudChart();
        });

        // These only affect business calculations, not the model
        ['claims-slider', 'fraud-cost-slider', 'investigation-cost-slider'].forEach(id => {
            document.getElementById(id).addEventListener('input', updateFraudChart);
        });

        // Initialize fraud chart with initial model training
        updateModelDescription(document.getElementById('model-type').value);
        updateFraudChart();

        // Ensure charts size correctly when their <details> panel is opened.
        document.querySelectorAll('details.use-case').forEach(details => {
            details.addEventListener('toggle', () => {
                if (!details.open) return;

                // Defer to allow the browser to lay out the expanded content.
                setTimeout(() => {
                    resizeAllCharts();

                    // Fraud charts are Plotly.react()'d and depend on computed widths.
                    // Re-run once on open so the business chart doesn't render squished.
                    const fraudBusiness = document.getElementById('fraud-business-chart');
                    if (fraudBusiness && details.querySelector('#fraud-business-chart')) {
                        updateFraudChart();
                    }
                }, 50);
            });
        });

        // Also resize on window resizes.
        window.addEventListener('resize', () => setTimeout(resizeAllCharts, 50));

        // Underwriting Chart
        const underwritingData = generateROCCurve(0.78);

        Plotly.newPlot('underwriting-chart', [
            {
                x: [0, 1],
                y: [0, 1],
                mode: 'lines',
                name: 'Random',
                line: { color: colors.muted, width: 1, dash: 'dash' }
            },
            {
                x: underwritingData.fpr,
                y: underwritingData.tpr,
                mode: 'lines',
                name: 'Model (AUC = 0.78)',
                line: { color: colors.purple, width: 3 },
                fill: 'tozeroy',
                fillcolor: 'rgba(167, 139, 250, 0.1)'
            },
            {
                x: [0.20],
                y: [0.70],
                mode: 'markers',
                name: 'Optimal Point',
                marker: { color: colors.emerald, size: 14, symbol: 'diamond' }
            }
        ], {
            ...chartLayout,
            title: { text: 'Underwriting ROC Curve', font: { size: 16, color: '#f5f5f0' } },
            xaxis: { ...chartLayout.xaxis, title: 'False Positive Rate (Good risks over-priced)', range: [0, 1] },
            yaxis: { ...chartLayout.yaxis, title: 'True Positive Rate (High-risk identified)', range: [0, 1] },
            height: 450
        }, chartConfig);

        // Underwriting Business Chart
        const uwThresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8];
        const uwTPR = [0.92, 0.88, 0.82, 0.70, 0.62, 0.48, 0.32, 0.18];
        const uwFPR = [0.45, 0.35, 0.28, 0.20, 0.15, 0.08, 0.04, 0.02];
        const totalApplicants = 50000;
        const highRiskRate = 0.15;
        const avgPremium = 1200;
        const avgClaim = 3500;
        const competitorLoss = 800;

        const highRiskRevenue = uwTPR.map(tpr => tpr * totalApplicants * highRiskRate * (avgPremium * 1.5 - avgClaim) / 1000000);
        const lostRevenue = uwFPR.map(fpr => fpr * totalApplicants * (1 - highRiskRate) * competitorLoss / 1000000);
        const netRevenue = highRiskRevenue.map((hr, i) => hr - lostRevenue[i]);

        Plotly.newPlot('underwriting-business-chart', [
            {
                x: uwThresholds,
                y: highRiskRevenue,
                mode: 'lines+markers',
                name: 'Premium Revenue',
                line: { color: colors.emerald, width: 2 },
                marker: { size: 6 }
            },
            {
                x: uwThresholds,
                y: lostRevenue,
                mode: 'lines+markers',
                name: 'Lost Customers',
                line: { color: colors.rose, width: 2 },
                marker: { size: 6 }
            },
            {
                x: uwThresholds,
                y: netRevenue,
                mode: 'lines+markers',
                name: 'Net Revenue',
                line: { color: colors.purple, width: 3 },
                marker: { size: 8 }
            },
            {
                x: [0.45],
                y: [netRevenue[3]],
                mode: 'markers',
                name: 'Optimal',
                marker: { color: colors.emerald, size: 16, symbol: 'star' }
            }
        ], {
            ...chartLayout,
            title: { text: 'Revenue Impact by Threshold', font: { size: 16, color: '#f5f5f0' } },
            xaxis: { ...chartLayout.xaxis, title: 'Model Threshold', range: [0.25, 0.85] },
            yaxis: { ...chartLayout.yaxis, title: 'Dollars (Millions)' },
            height: 400
        }, chartConfig);

        // Claims Triage Chart
        const triageData = generateROCCurve(0.88);

        Plotly.newPlot('triage-chart', [
            {
                x: [0, 1],
                y: [0, 1],
                mode: 'lines',
                name: 'Random',
                line: { color: colors.muted, width: 1, dash: 'dash' }
            },
            {
                x: triageData.fpr,
                y: triageData.tpr,
                mode: 'lines',
                name: 'Model (AUC = 0.88)',
                line: { color: colors.sky, width: 3 },
                fill: 'tozeroy',
                fillcolor: 'rgba(56, 189, 248, 0.1)'
            },
            {
                x: [0.08],
                y: [0.75],
                mode: 'markers',
                name: 'Optimal Point',
                marker: { color: colors.emerald, size: 14, symbol: 'diamond' }
            }
        ], {
            ...chartLayout,
            title: { text: 'Claims Triage ROC Curve', font: { size: 16, color: '#f5f5f0' } },
            xaxis: { ...chartLayout.xaxis, title: 'False Positive Rate (Simple claims escalated)', range: [0, 1] },
            yaxis: { ...chartLayout.yaxis, title: 'True Positive Rate (Complex claims caught)', range: [0, 1] },
            height: 450
        }, chartConfig);

        // Triage Business Chart
        const triageThresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8];
        const triageTPR = [0.92, 0.88, 0.82, 0.78, 0.75, 0.68, 0.52, 0.35];
        const triageFPR = [0.25, 0.18, 0.13, 0.10, 0.08, 0.05, 0.02, 0.01];
        const totalClaimsMonth = 50000;
        const complexRate = 0.12;
        const juniorAdjusterError = 8000;
        const seniorWastedTime = 200;

        const errorPrevention = triageTPR.map(tpr => tpr * totalClaimsMonth * complexRate * juniorAdjusterError / 1000000);
        const wastedTime = triageFPR.map(fpr => fpr * totalClaimsMonth * (1 - complexRate) * seniorWastedTime / 1000000);
        const netSavings = errorPrevention.map((ep, i) => ep - wastedTime[i]);

        Plotly.newPlot('triage-business-chart', [
            {
                x: triageThresholds,
                y: errorPrevention,
                mode: 'lines+markers',
                name: 'Errors Prevented',
                line: { color: colors.emerald, width: 2 },
                marker: { size: 6 }
            },
            {
                x: triageThresholds,
                y: wastedTime,
                mode: 'lines+markers',
                name: 'Wasted Time',
                line: { color: colors.rose, width: 2 },
                marker: { size: 6 }
            },
            {
                x: triageThresholds,
                y: netSavings,
                mode: 'lines+markers',
                name: 'Net Savings',
                line: { color: colors.sky, width: 3 },
                marker: { size: 8 }
            },
            {
                x: [0.55],
                y: [netSavings[4]],
                mode: 'markers',
                name: 'Optimal',
                marker: { color: colors.emerald, size: 16, symbol: 'star' }
            }
        ], {
            ...chartLayout,
            title: { text: 'Monthly Savings by Threshold', font: { size: 16, color: '#f5f5f0' } },
            xaxis: { ...chartLayout.xaxis, title: 'Model Threshold', range: [0.3, 0.85] },
            yaxis: { ...chartLayout.yaxis, title: 'Dollars (Millions)' },
            height: 400
        }, chartConfig);
}

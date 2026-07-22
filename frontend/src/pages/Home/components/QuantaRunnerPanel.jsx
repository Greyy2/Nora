import React from 'react';

function safeJsonStringify(value) {
    try {
        return JSON.stringify(value, null, 2);
    } catch {
        return String(value);
    }
}

function pickMetric(metrics, key) {
    if (!metrics || typeof metrics !== 'object') return null;
    const v = metrics[key];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
    if (typeof v === 'string') {
        const n = Number(v);
        return Number.isFinite(n) ? n : v;
    }
    return v ?? null;
}

function formatNumber(value, digits = 4) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return String(value ?? '');
    return value.toFixed(digits);
}

export default function QuantaRunnerPanel() {
    const [maxRounds, setMaxRounds] = React.useState(2);
    const [stockAttempts, setStockAttempts] = React.useState(3);
    const [verifyBeam, setVerifyBeam] = React.useState('');
    const [forexMddCap, setForexMddCap] = React.useState('');

    const [jobId, setJobId] = React.useState(null);
    const [status, setStatus] = React.useState(null);
    const [progress, setProgress] = React.useState(null);
    const [logText, setLogText] = React.useState('');
    const [summary, setSummary] = React.useState(null);
    const [error, setError] = React.useState(null);

    const pollingRef = React.useRef(null);

    const stopPolling = React.useCallback(() => {
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }
    }, []);

    const fetchProgress = React.useCallback(async (id) => {
        const res = await fetch(`/api/ai/quanta/progress/${id}`);
        const payload = await res.json();
        if (!res.ok) throw new Error(payload?.detail || 'Failed to fetch progress');
        return payload?.data;
    }, []);

    const fetchLogs = React.useCallback(async (id) => {
        const res = await fetch(`/api/ai/quanta/logs/${id}?tail=250`);
        const payload = await res.json();
        if (!res.ok) throw new Error(payload?.detail || 'Failed to fetch logs');
        return payload?.log || '';
    }, []);

    const fetchResult = React.useCallback(async (id) => {
        const res = await fetch(`/api/ai/quanta/result/${id}`);
        const payload = await res.json();
        if (!res.ok) throw new Error(payload?.detail || 'Result not ready');
        return payload?.summary;
    }, []);

    const startPolling = React.useCallback((id) => {
        stopPolling();
        pollingRef.current = setInterval(async () => {
            try {
                const p = await fetchProgress(id);
                setProgress(p);
                setStatus(p?.status || null);

                const logs = await fetchLogs(id);
                setLogText(logs);

                const st = p?.status;
                if (st && st !== 'running' && st !== 'stopping') {
                    stopPolling();
                    try {
                        const s = await fetchResult(id);
                        setSummary(s);
                    } catch {
                        // Result may not be ready even if status flipped; user can retry.
                    }
                }
            } catch (e) {
                setError(String(e?.message || e));
                stopPolling();
            }
        }, 2000);
    }, [fetchLogs, fetchProgress, fetchResult, stopPolling]);

    React.useEffect(() => {
        return () => stopPolling();
    }, [stopPolling]);

    const onRun = async () => {
        setError(null);
        setSummary(null);
        setLogText('');
        setProgress(null);
        setStatus('starting');

        const payload = {
            max_rounds: Number(maxRounds) || 2,
            stock_attempts: Number(stockAttempts) || 3,
        };

        if (String(verifyBeam).trim() !== '') payload.verify_beam = Number(verifyBeam);
        if (String(forexMddCap).trim() !== '') payload.forex_mdd_cap = Number(forexMddCap);

        const res = await fetch('/api/ai/quanta/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (!res.ok) {
            throw new Error(data?.detail || 'Failed to start job');
        }

        setJobId(data?.job_id || null);
        setStatus('running');
        startPolling(data?.job_id);
    };

    const onStop = async () => {
        if (!jobId) return;
        setError(null);
        setStatus('stopping');
        await fetch(`/api/ai/quanta/stop/${jobId}`, { method: 'POST' });
        startPolling(jobId);
    };

    const onRefreshResult = async () => {
        if (!jobId) return;
        setError(null);
        try {
            const s = await fetchResult(jobId);
            setSummary(s);
        } catch (e) {
            setError(String(e?.message || e));
        }
    };

    // Extract a small, human-readable snapshot (stock/forex) when summary is available.
    let snapshot = null;
    try {
        const flows = Array.isArray(summary?.flows) ? summary.flows : [];
        const stockFlow = flows.find((x) => x?.name === 'stock') || flows.find((x) => String(x?.name || '').startsWith('stock'));
        const forexFlow = flows.find((x) => x?.name === 'forex') || flows.find((x) => String(x?.name || '').startsWith('forex'));

        const mk = (flow) => {
            const best = flow?.best || {};
            const bestIC = flow?.best_by_ic_verified || {};
            const bestFeasible = flow?.best_feasible || {};

            const bestM = best?.backtest_metrics || {};
            const ic = pickMetric(bestM, 'IC');
            const ir = pickMetric(bestM, 'information_ratio');
            const arr = pickMetric(bestM, 'annualized_return');
            const mdd = pickMetric(bestM, 'max_drawdown');

            return {
                market: flow?.market_mode || flow?.name,
                best: {
                    name: best?.name,
                    expression: best?.expression,
                    source: best?.backtest_source,
                    IC: ic,
                    IR: ir,
                    ARR: arr,
                    MDD: mdd,
                },
                best_by_ic_verified: {
                    name: bestIC?.name,
                    expression: bestIC?.expression,
                    source: bestIC?.backtest_source,
                    IC: pickMetric(bestIC?.backtest_metrics, 'IC'),
                },
                best_feasible: {
                    name: bestFeasible?.name,
                    expression: bestFeasible?.expression,
                    source: bestFeasible?.backtest_source,
                    IC: pickMetric(bestFeasible?.backtest_metrics, 'IC'),
                },
            };
        };

        snapshot = {
            output_dir: summary?.output_dir,
            timestamp_utc: summary?.timestamp_utc,
            stock: stockFlow ? mk(stockFlow) : null,
            forex: forexFlow ? mk(forexFlow) : null,
        };
    } catch {
        snapshot = null;
    }

    return (
        <div className="quanta-runner-container">
            <div className="quanta-runner-header">
                <div className="q-header-main">
                    <div className="q-status-light pulse"></div>
                    <h3>Intelligence Control Center</h3>
                    <span className={`q-status-badge ${status || 'idle'}`}>{status || 'READY'}</span>
                </div>
                <div className="q-job-info">
                    {jobId && <span className="q-job-id">JOB: {jobId}</span>}
                </div>
            </div>

            <div className="quanta-runner-grid">
                {/* Configuration Section */}
                <div className="q-panel q-config-panel">
                    <div className="q-panel-header">
                        <i className="bi bi-gear-fill"></i>
                        <span>Parameters</span>
                    </div>
                    <div className="quanta-controls">
                        <div className="quanta-field">
                            <label>Max Rounds</label>
                            <input type="number" value={maxRounds} min={1} max={20} onChange={(e) => setMaxRounds(e.target.value)} />
                            <small>Số vòng lặp tiến hóa</small>
                        </div>
                        <div className="quanta-field">
                            <label>Stock Attempts</label>
                            <input type="number" value={stockAttempts} min={1} max={20} onChange={(e) => setStockAttempts(e.target.value)} />
                            <small>Số lần thử khai thác Stock</small>
                        </div>
                        <div className="quanta-field">
                            <label>Verify Beam</label>
                            <input type="number" value={verifyBeam} min={1} max={64} placeholder="Default" onChange={(e) => setVerifyBeam(e.target.value)} />
                            <small>Beam width cho validation</small>
                        </div>
                        <div className="quanta-field">
                            <label>Forex MDD Cap</label>
                            <input type="number" value={forexMddCap} min={0.01} max={0.5} step={0.01} placeholder="Default" onChange={(e) => setForexMddCap(e.target.value)} />
                            <small>Giới hạn MDD cho Forex</small>
                        </div>
                    </div>

                    <div className="quanta-actions">
                        <button
                            className="q-btn q-btn-run"
                            onClick={() => onRun().catch((e) => setError(String(e?.message || e)))}
                            disabled={status === 'running' || status === 'stopping'}
                        >
                            <i className="bi bi-play-fill"></i>
                            <span>Start Mining</span>
                        </button>

                        <div className="q-btn-group">
                            <button className="q-btn q-btn-stop" onClick={() => onStop().catch((e) => setError(String(e?.message || e)))} disabled={!jobId || status !== 'running'}>
                                <i className="bi bi-stop-fill"></i>
                            </button>
                            <button className="q-btn q-btn-refresh" onClick={() => onRefreshResult().catch((e) => setError(String(e?.message || e)))} disabled={!jobId}>
                                <i className="bi bi-arrow-clockwise"></i>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Live Output Section */}
                <div className="q-panel q-log-panel">
                    <div className="q-panel-header">
                        <i className="bi bi-terminal-fill"></i>
                        <span>Live Evolution Logs</span>
                        <div className="q-live-indicator">LIVE</div>
                    </div>
                    <div className="quanta-logs-container">
                        <pre className="quanta-log">{logText || 'Waiting for system start...'}</pre>
                    </div>
                    {error && <div className="q-error-bar">{error}</div>}
                </div>
            </div>

            {/* Results Snapshot */}
            {snapshot && (
                <div className="q-summary-panel">
                    <div className="q-panel-header">
                        <i className="bi bi-award-fill"></i>
                        <span>Elite Discovery Results</span>
                    </div>
                    <div className="q-results-grid">
                        {snapshot.stock && (
                            <div className="q-result-card stock">
                                <div className="q-market-label">STOCK MARKET</div>
                                <div className="q-best-alpha">
                                    <span className="q-alpha-name">{snapshot.stock.best.name}</span>
                                    <div className="q-metrics-row">
                                        <div className="q-metric"><span className="label">IC</span><span className="value">{formatNumber(snapshot.stock.best.IC)}</span></div>
                                        <div className="q-metric"><span className="label">ARR</span><span className="value">{formatNumber(snapshot.stock.best.ARR)}</span></div>
                                        <div className="q-metric"><span className="label">MDD</span><span className="value">{formatNumber(snapshot.stock.best.MDD)}</span></div>
                                    </div>
                                </div>
                            </div>
                        )}
                        {snapshot.forex && (
                            <div className="q-result-card forex">
                                <div className="q-market-label">FOREX MARKET</div>
                                <div className="q-best-alpha">
                                    <span className="q-alpha-name">{snapshot.forex.best.name}</span>
                                    <div className="q-metrics-row">
                                        <div className="q-metric"><span className="label">IC</span><span className="value">{formatNumber(snapshot.forex.best.IC)}</span></div>
                                        <div className="q-metric"><span className="label">ARR</span><span className="value">{formatNumber(snapshot.forex.best.ARR)}</span></div>
                                        <div className="q-metric"><span className="label">MDD</span><span className="value">{formatNumber(snapshot.forex.best.MDD)}</span></div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

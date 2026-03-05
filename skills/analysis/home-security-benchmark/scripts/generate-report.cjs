#!/usr/bin/env node
/**
 * HTML Report Generator for Home Security AI Benchmark
 * 
 * Reads JSON result files from the benchmarks directory and generates
 * a self-contained HTML report with:
 * - Pass/fail scorecard per suite
 * - Latency charts (inline SVG)
 * - Token usage breakdown
 * - Historical comparison table
 * - System configuration
 * 
 * Usage:
 *   node generate-report.cjs [results-dir]
 *   Default: ~/.aegis-ai/benchmarks
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

const RESULTS_DIR = process.argv[2] || path.join(os.homedir(), '.aegis-ai', 'benchmarks');

function generateReport(resultsDir = RESULTS_DIR) {
    const dir = resultsDir || RESULTS_DIR;

    // Load all result files
    const indexFile = path.join(dir, 'index.json');
    if (!fs.existsSync(indexFile)) {
        console.error(`No index.json found in ${dir}. Run the benchmark first.`);
        process.exit(1);
    }

    const index = JSON.parse(fs.readFileSync(indexFile, 'utf8'));
    if (index.length === 0) {
        console.error('No benchmark results found.');
        process.exit(1);
    }

    // Load the latest result for detailed view
    const latestEntry = index[index.length - 1];
    const latestFile = path.join(dir, latestEntry.file);
    const latest = JSON.parse(fs.readFileSync(latestFile, 'utf8'));

    // Load all results for comparison
    const allResults = index.map(entry => {
        try {
            const data = JSON.parse(fs.readFileSync(path.join(dir, entry.file), 'utf8'));
            return { ...entry, data };
        } catch { return entry; }
    });

    const html = buildHTML(latest, allResults);
    const reportPath = path.join(dir, 'report.html');
    fs.writeFileSync(reportPath, html);
    console.log(`  Report saved: ${reportPath}`);

    // Try to open in browser
    try {
        const { execSync } = require('child_process');
        if (process.platform === 'darwin') execSync(`open "${reportPath}"`);
        else if (process.platform === 'linux') execSync(`xdg-open "${reportPath}"`);
        else if (process.platform === 'win32') execSync(`start "" "${reportPath}"`);
    } catch { }

    return reportPath;
}

function buildHTML(latest, allResults) {
    const { totals, tokenTotals, model, system, suites } = latest;
    const passRate = totals.total > 0 ? ((totals.passed / totals.total) * 100).toFixed(0) : 0;
    const tokPerSec = totals.timeMs > 0 ? (tokenTotals.total / (totals.timeMs / 1000)).toFixed(1) : '?';

    // Build suite rows
    const suiteRows = suites.map(s => {
        const pct = s.tests.length > 0 ? ((s.passed / s.tests.length) * 100).toFixed(0) : 0;
        const color = s.failed === 0 ? '#22c55e' : s.passed > s.failed ? '#f59e0b' : '#ef4444';
        return `<tr>
            <td>${s.name}</td>
            <td><span class="badge" style="background:${color}">${s.passed}/${s.tests.length}</span></td>
            <td>${(s.timeMs / 1000).toFixed(1)}s</td>
            <td><div class="bar-bg"><div class="bar" style="width:${pct}%;background:${color}"></div></div></td>
        </tr>`;
    }).join('\n');

    // Build test detail rows
    const testRows = suites.flatMap(s =>
        s.tests.map(t => {
            const icon = t.status === 'pass' ? '✅' : t.status === 'fail' ? '❌' : '⏭️';
            const cls = t.status === 'fail' ? 'fail-row' : '';
            return `<tr class="${cls}">
                <td>${icon}</td>
                <td class="suite-label">${s.name}</td>
                <td>${t.name}</td>
                <td>${t.timeMs}ms</td>
                <td class="detail">${escHtml(t.detail.slice(0, 120))}</td>
            </tr>`;
        })
    ).join('\n');

    // Build latency chart data (SVG bar chart)
    const allTests = suites.flatMap(s => s.tests.filter(t => t.status !== 'skip'));
    const maxLatency = Math.max(...allTests.map(t => t.timeMs), 1);
    const barHeight = 22;
    const chartHeight = allTests.length * (barHeight + 4) + 40;
    const chartBars = allTests.map((t, i) => {
        const w = (t.timeMs / maxLatency) * 500;
        const y = i * (barHeight + 4) + 30;
        const color = t.status === 'pass' ? '#22c55e' : '#ef4444';
        const label = t.name.length > 30 ? t.name.slice(0, 28) + '…' : t.name;
        return `<rect x="200" y="${y}" width="${w}" height="${barHeight}" fill="${color}" rx="3"/>
        <text x="195" y="${y + 15}" text-anchor="end" class="chart-label">${escHtml(label)}</text>
        <text x="${205 + w}" y="${y + 15}" class="chart-value">${t.timeMs}ms</text>`;
    }).join('\n');

    // Build historical comparison table
    const historyRows = allResults.slice().reverse().map(r => {
        const pct = r.total > 0 ? ((r.passed / r.total) * 100).toFixed(0) : 0;
        const ts = new Date(r.timestamp).toLocaleDateString() + ' ' + new Date(r.timestamp).toLocaleTimeString();
        const isCurrent = r.file === (allResults[allResults.length - 1]?.file);
        const vlmModel = r.vlm || (r.data?.model?.vlm) || '';
        const modelLabel = (r.model || '?') + (vlmModel ? `<br><span style="color:var(--muted);font-size:0.8em">VLM: ${vlmModel}</span>` : '');
        return `<tr${isCurrent ? ' class="current-run"' : ''}>
            <td>${ts}${isCurrent ? ' ⬅️' : ''}</td>
            <td>${modelLabel}</td>
            <td>${r.passed}/${r.total}</td>
            <td>${pct}%</td>
            <td>${(r.timeMs / 1000).toFixed(1)}s</td>
            <td>${r.tokens || '?'}</td>
        </tr>`;
    }).join('\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Home Security AI Benchmark — ${model.name || 'Report'}</title>
<style>
:root {
    --bg: #0f172a; --card: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8; --accent: #3b82f6;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
h2 { font-size: 1.2rem; font-weight: 600; margin: 2rem 0 1rem; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
.subtitle { color: var(--muted); font-size: 0.95rem; }
.hero { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }
.stat-card .label { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-card .value { font-size: 2rem; font-weight: 700; margin-top: 0.25rem; }
.stat-card .sub { color: var(--muted); font-size: 0.85rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { padding: 0.6rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
.badge { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; color: white; font-size: 0.85rem; font-weight: 600; }
.bar-bg { background: var(--border); border-radius: 4px; height: 8px; width: 100px; }
.bar { height: 8px; border-radius: 4px; transition: width 0.3s; }
.fail-row { background: rgba(239, 68, 68, 0.08); }
.detail { color: var(--muted); font-size: 0.8rem; max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.suite-label { color: var(--muted); font-size: 0.8rem; }
.current-run { background: rgba(59, 130, 246, 0.08); }
.chart-label { font-size: 11px; fill: var(--muted); }
.chart-value { font-size: 11px; fill: var(--text); }
.sys-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 0.5rem; }
.sys-item { display: flex; gap: 0.5rem; }
.sys-item .k { color: var(--muted); min-width: 100px; }
footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--muted); font-size: 0.8rem; text-align: center; }
@media (max-width: 640px) { .hero { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<div class="container">

<h1>🛡️ Home Security AI Benchmark</h1>
<p class="subtitle">${model.name || 'Unknown Model'}${model.vlm ? ' + VLM: ' + model.vlm : ''} — ${new Date(latest.timestamp).toLocaleDateString()} ${new Date(latest.timestamp).toLocaleTimeString()}</p>

<div class="hero">
    <div class="stat-card">
        <div class="label">Pass Rate</div>
        <div class="value" style="color:${totals.failed === 0 ? 'var(--green)' : totals.passed > totals.failed ? 'var(--yellow)' : 'var(--red)'}">${passRate}%</div>
        <div class="sub">${totals.passed}/${totals.total} tests passed</div>
    </div>
    <div class="stat-card">
        <div class="label">Total Time</div>
        <div class="value">${(totals.timeMs / 1000).toFixed(1)}s</div>
        <div class="sub">${suites.length} suites</div>
    </div>
    <div class="stat-card">
        <div class="label">Tokens</div>
        <div class="value">${tokenTotals.total.toLocaleString()}</div>
        <div class="sub">${tokPerSec} tok/s</div>
    </div>
    <div class="stat-card">
        <div class="label">Model</div>
        <div class="value" style="font-size:1rem">${model.name || '?'}</div>
        <div class="sub">${model.vlm ? 'VLM: ' + model.vlm : system.cpu || '?'}</div>
    </div>
</div>

<h2>Suite Summary</h2>
<table>
    <thead><tr><th>Suite</th><th>Result</th><th>Time</th><th>Pass Rate</th></tr></thead>
    <tbody>${suiteRows}</tbody>
</table>

<h2>Latency Chart</h2>
<svg width="800" height="${chartHeight}" viewBox="0 0 800 ${chartHeight}" style="width:100%;max-width:800px">
    <text x="400" y="18" text-anchor="middle" class="chart-label" style="font-size:13px;fill:var(--text)">Response Latency per Test (ms)</text>
    ${chartBars}
</svg>

<h2>Test Details</h2>
<table>
    <thead><tr><th></th><th>Suite</th><th>Test</th><th>Time</th><th>Detail</th></tr></thead>
    <tbody>${testRows}</tbody>
</table>

<h2>Token Usage</h2>
<div class="hero">
    <div class="stat-card">
        <div class="label">Prompt Tokens</div>
        <div class="value" style="font-size:1.5rem">${tokenTotals.prompt.toLocaleString()}</div>
    </div>
    <div class="stat-card">
        <div class="label">Completion Tokens</div>
        <div class="value" style="font-size:1.5rem">${tokenTotals.completion.toLocaleString()}</div>
    </div>
    <div class="stat-card">
        <div class="label">Total Tokens</div>
        <div class="value" style="font-size:1.5rem">${tokenTotals.total.toLocaleString()}</div>
    </div>
    <div class="stat-card">
        <div class="label">Throughput</div>
        <div class="value" style="font-size:1.5rem">${tokPerSec}</div>
        <div class="sub">tokens/second</div>
    </div>
</div>

${allResults.length > 1 ? `<h2>Historical Comparison</h2>
<table>
    <thead><tr><th>Date</th><th>Model</th><th>Passed</th><th>Rate</th><th>Time</th><th>Tokens</th></tr></thead>
    <tbody>${historyRows}</tbody>
</table>` : ''}

<h2>System Configuration</h2>
<div class="sys-grid">
    <div class="sys-item"><span class="k">OS</span><span>${system.os || '?'}</span></div>
    <div class="sys-item"><span class="k">CPU</span><span>${system.cpu || '?'}</span></div>
    <div class="sys-item"><span class="k">Cores</span><span>${system.cpuCores || '?'}</span></div>
    <div class="sys-item"><span class="k">RAM</span><span>${system.totalMemoryGB || '?'} GB total</span></div>
    <div class="sys-item"><span class="k">Free RAM</span><span>${system.freeMemoryGB || '?'} GB</span></div>
    <div class="sys-item"><span class="k">Node</span><span>${system.nodeVersion || '?'}</span></div>
    <div class="sys-item"><span class="k">Process RSS</span><span>${system.processMemoryMB?.rss || '?'} MB</span></div>
    <div class="sys-item"><span class="k">Heap Used</span><span>${system.processMemoryMB?.heapUsed || '?'} MB</span></div>
</div>

<footer>
    Home Security AI Benchmark Suite • DeepCamera / SharpAI • Generated ${new Date().toISOString()}
</footer>

</div>
</body>
</html>`;
}

function escHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Run if called directly
if (require.main === module) {
    generateReport();
}

module.exports = { generateReport };

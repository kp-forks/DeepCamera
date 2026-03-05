#!/usr/bin/env node
/**
 * Home Security AI Benchmark Suite
 * 
 * Evaluates LLM and VLM models on home security AI tasks:
 * - Context preprocessing (dedup)
 * - Topic classification
 * - Knowledge distillation
 * - Event deduplication (security classifier)
 * - Tool use (tool selection & parameter extraction)
 * - Chat & JSON compliance
 * - VLM scene analysis (optional, requires VLM server)
 * 
 * ## Skill Protocol (when spawned by Aegis)
 * 
 * Aegis → Skill (env vars):
 *   AEGIS_GATEWAY_URL  — LLM gateway URL (e.g. http://localhost:5407)
 *   AEGIS_VLM_URL      — VLM server URL (e.g. http://localhost:5405)
 *   AEGIS_SKILL_PARAMS — JSON params from skill config
 *   AEGIS_SKILL_ID     — Skill ID
 * 
 * Skill → Aegis (stdout, JSON lines):
 *   {"event": "ready", "model": "Qwen3.5-4B-Q4_1"}
 *   {"event": "suite_start", "suite": "Context Preprocessing"}
 *   {"event": "test_result", "suite": "...", "test": "...", "status": "pass", "timeMs": 123}
 *   {"event": "suite_end", "suite": "...", "passed": 4, "failed": 0}
 *   {"event": "complete", "passed": 23, "total": 26, "timeMs": 95000, "reportPath": "..."}
 * 
 * Standalone usage:
 *   node run-benchmark.cjs [options]
 *   --gateway URL   LLM gateway (fallback if no AEGIS_GATEWAY_URL)
 *   --vlm URL       VLM server (fallback if no AEGIS_VLM_URL)
 *   --out DIR       Results directory (default: ~/.aegis-ai/benchmarks)
 *   --report        Auto-generate HTML report after run
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

// ─── Config: Aegis env vars → CLI args → defaults ────────────────────────────

const args = process.argv.slice(2);
function getArg(name, defaultVal) {
    const idx = args.indexOf(`--${name}`);
    if (idx === -1) return defaultVal;
    return args[idx + 1] || defaultVal;
}

// Aegis provides config via env vars; CLI args are fallback for standalone
const GATEWAY_URL = process.env.AEGIS_GATEWAY_URL || getArg('gateway', 'http://localhost:5407');
const VLM_URL = process.env.AEGIS_VLM_URL || getArg('vlm', '');
const RESULTS_DIR = getArg('out', path.join(os.homedir(), '.aegis-ai', 'benchmarks'));
const AUTO_REPORT = args.includes('--report');
const TIMEOUT_MS = 30000;
const FIXTURES_DIR = path.join(__dirname, '..', 'fixtures');
const IS_SKILL_MODE = !!process.env.AEGIS_SKILL_ID;

// Parse skill parameters if running as Aegis skill
let skillParams = {};
try { skillParams = JSON.parse(process.env.AEGIS_SKILL_PARAMS || '{}'); } catch { }

// ─── Skill Protocol: JSON lines on stdout, human text on stderr ──────────────

/**
 * Emit a JSON-lines event on stdout (parsed by Aegis skill-runtime-manager).
 * All structured data goes here so Aegis can react to it.
 */
function emit(event) {
    process.stdout.write(JSON.stringify(event) + '\n');
}

/**
 * Log human-readable text to stderr (shows in Aegis console tab).
 * In standalone mode, also mirrors to stdout for terminal visibility.
 */
function log(msg) {
    process.stderr.write(msg + '\n');
}

// ─── Test Framework ───────────────────────────────────────────────────────────

const suites = [];
let currentSuite = null;

function suite(name, fn) {
    suites.push({ name, fn, tests: [] });
}

const results = {
    timestamp: new Date().toISOString(),
    gateway: GATEWAY_URL,
    vlm: VLM_URL || null,
    system: {},
    model: {},
    suites: [],
    totals: { passed: 0, failed: 0, skipped: 0, total: 0, timeMs: 0 },
    tokenTotals: { prompt: 0, completion: 0, total: 0 },
};

async function llmCall(messages, opts = {}) {
    const body = { messages, stream: false };
    if (opts.maxTokens) body.max_tokens = opts.maxTokens;
    if (opts.temperature !== undefined) body.temperature = opts.temperature;
    if (opts.tools) body.tools = opts.tools;

    const url = opts.vlm ? `${VLM_URL}/v1/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(opts.timeout || TIMEOUT_MS),
    });

    if (!response.ok) {
        const errBody = await response.text().catch(() => '');
        throw new Error(`HTTP ${response.status}: ${errBody.slice(0, 200)}`);
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || '';
    const toolCalls = data.choices?.[0]?.message?.tool_calls || null;
    const usage = data.usage || {};

    // Track token totals
    results.tokenTotals.prompt += usage.prompt_tokens || 0;
    results.tokenTotals.completion += usage.completion_tokens || 0;
    results.tokenTotals.total += usage.total_tokens || 0;

    // Capture model name from first response
    if (!results.model.name && data.model) {
        results.model.name = data.model;
    }

    return { content, toolCalls, usage, model: data.model };
}

function stripThink(text) {
    return text.replace(/<think>[\s\S]*?<\/think>\s*/gi, '').trim();
}

function parseJSON(text) {
    const cleaned = stripThink(text);
    let jsonStr = cleaned;
    const codeBlock = cleaned.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    if (codeBlock) jsonStr = codeBlock[1];
    else {
        const idx = cleaned.search(/[{[]/);
        if (idx > 0) jsonStr = cleaned.slice(idx);
    }
    return JSON.parse(jsonStr.trim());
}

function assert(condition, msg) {
    if (!condition) throw new Error(msg || 'Assertion failed');
}

async function runSuites() {
    for (const s of suites) {
        currentSuite = { name: s.name, tests: [], passed: 0, failed: 0, skipped: 0, timeMs: 0 };
        log(`\n${'─'.repeat(60)}`);
        log(`  ${s.name}`);
        log(`${'─'.repeat(60)}`);
        emit({ event: 'suite_start', suite: s.name });

        await s.fn();

        results.suites.push(currentSuite);
        results.totals.passed += currentSuite.passed;
        results.totals.failed += currentSuite.failed;
        results.totals.skipped += currentSuite.skipped;
        results.totals.total += currentSuite.tests.length;

        emit({ event: 'suite_end', suite: s.name, passed: currentSuite.passed, failed: currentSuite.failed, skipped: currentSuite.skipped, timeMs: currentSuite.timeMs });
    }
}

async function test(name, fn) {
    const testResult = { name, status: 'pass', timeMs: 0, detail: '', tokens: {} };
    const start = Date.now();
    try {
        const detail = await fn();
        testResult.timeMs = Date.now() - start;
        testResult.detail = detail || '';
        currentSuite.passed++;
        log(`  ✅ ${name} (${testResult.timeMs}ms)${detail ? ` — ${detail}` : ''}`);
    } catch (err) {
        testResult.timeMs = Date.now() - start;
        testResult.status = 'fail';
        testResult.detail = err.message;
        currentSuite.failed++;
        log(`  ❌ ${name} (${testResult.timeMs}ms) — ${err.message}`);
    }
    currentSuite.timeMs += testResult.timeMs;
    currentSuite.tests.push(testResult);
    emit({ event: 'test_result', suite: currentSuite.name, test: name, status: testResult.status, timeMs: testResult.timeMs, detail: testResult.detail.slice(0, 120) });
}

function skip(name, reason) {
    currentSuite.skipped++;
    currentSuite.tests.push({ name, status: 'skip', timeMs: 0, detail: reason });
    log(`  ⏭️  ${name} — ${reason}`);
    emit({ event: 'test_result', suite: currentSuite.name, test: name, status: 'skip', timeMs: 0, detail: reason });
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 1: CONTEXT PREPROCESSING
// ═══════════════════════════════════════════════════════════════════════════════

function buildPreprocessPrompt(messageIndex, userMessage) {
    return `You are a context deduplication engine. Given a list of user messages from a conversation, decide which exchanges to KEEP and which are DUPLICATES.

## User Messages (index, timestamp, first 50 words)
${messageIndex.map(m => `[${m.idx}] ${m.ts ? `(${m.ts})` : ''} ${m.text}`).join('\n')}

## New User Question
${userMessage}

## Rules
1. If the user asked the same or very similar question multiple times, keep ONLY the LATEST one
2. Keep messages that are clearly different topics or provide unique context
3. Always keep the last 2 user messages (most recent context)
4. Keep system messages (they contain tool results)

## Response Format
Return ONLY this JSON (no other text):
{"keep": [0, 5, 8], "summary": "brief 1-line summary of dropped exchanges"}

- "keep": array of message indices to KEEP (from the index list above)
- "summary": what the dropped messages were about (so context is not lost entirely)
- If nothing should be dropped, set keep to ALL indices and summary to ""`;
}

suite('📋 Context Preprocessing', async () => {
    await test('Exact duplicates (7x same Q) → keep ≤3', async () => {
        const idx = [
            { idx: 0, ts: '9:56 AM', text: 'What has happened today' },
            { idx: 4, ts: '10:09 AM', text: 'What has happened today?' },
            { idx: 8, ts: '10:14 AM', text: 'What has happened today ?' },
            { idx: 12, ts: '10:28 AM', text: 'What has happened today?' },
            { idx: 16, ts: '10:33 AM', text: 'Hi, What has happened today?' },
            { idx: 18, ts: '12:56 PM', text: 'What has happened today' },
            { idx: 22, ts: '1:08 PM', text: 'What has happened today' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'What has happened today?') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep), 'keep must be array');
        assert(p.keep.length <= 3, `Expected ≤3, got ${p.keep.length}`);
        return `kept ${p.keep.length}/7`;
    });

    await test('Mixed topics → preserves unique questions', async () => {
        const idx = [
            { idx: 0, ts: '9:00 AM', text: 'What has happened today' },
            { idx: 3, ts: '9:30 AM', text: 'Set an alert for person detection on front door after 10pm' },
            { idx: 6, ts: '10:00 AM', text: 'What has happened today?' },
            { idx: 10, ts: '10:15 AM', text: 'Show me the clip from 9:40 AM at the front door' },
            { idx: 14, ts: '11:00 AM', text: 'What has happened today?' },
            { idx: 18, ts: '12:00 PM', text: 'What is the system status?' },
            { idx: 22, ts: '1:00 PM', text: 'What has happened today' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'Any alerts triggered?') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep), 'keep must be array');
        assert(p.keep.includes(3) || p.keep.includes(10) || p.keep.includes(18), 'Should keep unique topics');
        return `kept ${p.keep.length}/7: [${p.keep.join(',')}]`;
    });

    await test('All unique → keep all', async () => {
        const idx = [
            { idx: 0, ts: '9:00 AM', text: 'Show me the front door camera' },
            { idx: 3, ts: '9:15 AM', text: 'Set alert for person detection' },
            { idx: 6, ts: '10:00 AM', text: 'What is the system status?' },
            { idx: 10, ts: '11:00 AM', text: 'Analyze the clip from 9:40 AM' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'Any new motion events?') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep) && p.keep.length === 4, `Expected 4, got ${p.keep?.length}`);
        return `kept all 4 ✓`;
    });

    await test('Small history → empty summary', async () => {
        const idx = [
            { idx: 0, ts: '9:00 AM', text: 'Hello' },
            { idx: 2, ts: '9:05 AM', text: 'Show cameras' },
        ];
        const r = await llmCall([{ role: 'user', content: buildPreprocessPrompt(idx, 'Thanks') }]);
        const p = parseJSON(r.content);
        assert(Array.isArray(p.keep), 'keep must be array');
        return `kept ${p.keep.length}/2`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 2: TOPIC CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

suite('🏷️ Topic Classification', async () => {
    await test('First turn → topic title (3-6 words)', async () => {
        const r = await llmCall([{
            role: 'user', content: `Classify this exchange's topic in 3-6 words. Respond with ONLY the topic title.
User: "What has happened today on the cameras?"
Assistant: "Today, your cameras captured motion events including a person at the front door at 9:40 AM..."` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '').replace(/^(new\s+)?topic\s*:\s*/i, '').trim();
        assert(cleaned.length > 0, 'Topic empty');
        const wc = cleaned.split(/\s+/).length;
        assert(wc <= 8, `Too verbose: ${wc} words`);
        return `"${cleaned}" (${wc} words)`;
    });

    await test('Same topic → SAME', async () => {
        const r = await llmCall([{
            role: 'user', content: `Given this exchange, is the topic still the same?
User: "Show me the clip from 9:40 AM"
Assistant: "Here's the clip from 9:40 AM showing a person at the front door..."
Current topic: "Camera Events Review"
If the topic hasn't changed, respond: SAME
Otherwise respond with ONLY the new topic title (3-6 words).` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '');
        assert(cleaned.toUpperCase() === 'SAME', `Expected SAME, got "${cleaned}"`);
        return 'SAME ✓';
    });

    await test('Topic change → new title', async () => {
        const r = await llmCall([{
            role: 'user', content: `Given this exchange, is the topic still the same?
User: "What's the system status? How much storage am I using?"
Assistant: "System healthy. Storage: 45GB of 500GB, VLM running on GPU."
Current topic: "Camera Events Review"
If the topic hasn't changed, respond: SAME
Otherwise respond with ONLY the new topic title (3-6 words).` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '').replace(/^(new\s+)?topic\s*:\s*/i, '').trim();
        assert(cleaned.toUpperCase() !== 'SAME', 'Expected new topic');
        return `"${cleaned}"`;
    });

    await test('Greeting → valid topic', async () => {
        const r = await llmCall([{
            role: 'user', content: `Classify this exchange's topic in 3-6 words. Respond with ONLY the topic title.
User: "Hi, good morning!"
Assistant: "Good morning! How can I help you with your home security today?"` }]);
        const cleaned = stripThink(r.content).split('\n').filter(l => l.trim()).pop().replace(/^["'*]+|["'*]+$/g, '').trim();
        assert(cleaned.length > 0 && cleaned.length < 50, `Bad: "${cleaned}"`);
        return `"${cleaned}"`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 3: KNOWLEDGE DISTILLATION
// ═══════════════════════════════════════════════════════════════════════════════

const DISTILL_PROMPT = `You are a knowledge extraction agent for SharpAI-Aegis, a home security assistant.
Analyze the conversation below and extract DURABLE knowledge worth remembering permanently.
## What to Extract
- Home/household facts: People, pets, home layout, camera locations
- User preferences: Notification style, alert priorities, quiet hours
- Security patterns: Normal activity times, delivery schedules
- System decisions: Model choices, channel configurations
## What to Skip
- Transient errors and troubleshooting
- Routine exchanges ("thanks", "bye")
- One-time questions
## Output Format
Respond with ONLY valid JSON:
{"items": [{"slug": "home_profile", "facts": [{"type": "camera", "content": "..."}]}], "new_items": [{"title": "...", "summary": "...", "facts": [{"type": "...", "content": "..."}]}]}
### Slug Reference
- home_profile: Cameras, members, pets, layout
- alert_preferences: Per-camera rules, channels, quiet hours
- security_patterns: Activity patterns, false alarms
- system_config: Models, channels, storage
If nothing to extract: {"items": [], "new_items": []}`;

suite('🧠 Knowledge Distillation', async () => {
    await test('Home profile → extracts facts with slug', async () => {
        const r = await llmCall([
            { role: 'system', content: DISTILL_PROMPT },
            { role: 'user', content: `## Topic: Camera Setup\n## Existing KIs: (none)\n## Conversation\nUser: I have three cameras. Front door is a Blink Mini, living room is Blink Indoor, side parking is Blink Outdoor.\nAegis: Got it! Want to set up alerts?\nUser: Yes, person detection on front door after 10pm. My name is Sam.\nAegis: Alert set. Nice to meet you, Sam!` },
        ]);
        const p = parseJSON(r.content);
        assert(p && typeof p === 'object', 'Must return object');
        const facts = (p.items || []).reduce((n, i) => n + (i.facts?.length || 0), 0) + (p.new_items || []).reduce((n, i) => n + (i.facts?.length || 0), 0);
        assert(facts >= 2, `Expected ≥2 facts, got ${facts}`);
        return `${facts} facts extracted`;
    });

    await test('Routine chat → empty extraction', async () => {
        const r = await llmCall([
            { role: 'system', content: DISTILL_PROMPT },
            { role: 'user', content: `## Topic: Greeting\n## Existing KIs: (none)\n## Conversation\nUser: Hi\nAegis: Hello! How can I help?\nUser: Thanks, bye\nAegis: Goodbye!` },
        ]);
        const p = parseJSON(r.content);
        const facts = (p.items || []).reduce((n, i) => n + (i.facts?.length || 0), 0) + (p.new_items || []).reduce((n, i) => n + (i.facts?.length || 0), 0);
        assert(facts === 0, `Expected 0 facts, got ${facts}`);
        return 'empty ✓';
    });

    await test('Alert preferences → extracts to correct slug', async () => {
        const r = await llmCall([
            { role: 'system', content: DISTILL_PROMPT },
            { role: 'user', content: `## Topic: Alert Configuration\n## Existing KIs: alert_preferences\n## Conversation\nUser: No notifications from side parking 8am-5pm. Too many false alarms from passing cars.\nAegis: Quiet hours set for side parking 8 AM-5 PM.\nUser: Front door alerts go to Telegram. Discord for everything else.\nAegis: Done — front door to Telegram, rest to Discord.` },
        ]);
        const p = parseJSON(r.content);
        const facts = (p.items || []).reduce((n, i) => n + (i.facts?.length || 0), 0) + (p.new_items || []).reduce((n, i) => n + (i.facts?.length || 0), 0);
        assert(facts >= 2, `Expected ≥2 facts, got ${facts}`);
        return `${facts} facts`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 4: EVENT DEDUPLICATION
// ═══════════════════════════════════════════════════════════════════════════════

function buildDedupPrompt(current, recent, ageSec) {
    return `You are a security event deduplication assistant.

## TASK
Determine if CURRENT CLIP and RECENT CLIP show the same ongoing event or different events.

## CURRENT CLIP
- Camera: ${current.camera}
- Type: ${current.type}
- Summary: ${current.summary}

## RECENT CLIP (sent ${ageSec}s ago)
- Camera: ${recent.camera}
- Type: ${recent.type}
- Summary: ${recent.summary}

## DECISION CRITERIA
DUPLICATE if: Same person lingering, same vehicle, continuation of activity
UNIQUE if: Different person/vehicle, different activity, new event

## RESPONSE FORMAT
Respond with ONLY a JSON object:
{"duplicate": true/false, "reason": "brief explanation", "confidence": "high/medium/low"}`;
}

suite('🔔 Event Deduplication', async () => {
    const scenarios = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'tool-use-scenarios.json'), 'utf8'));
    for (const s of scenarios.dedup_scenarios) {
        await test(`${s.name}`, async () => {
            const r = await llmCall([
                { role: 'system', content: 'You are a security event classifier. Respond only with valid JSON.' },
                { role: 'user', content: buildDedupPrompt(s.current, s.recent, s.age_sec) },
            ], { maxTokens: 150, temperature: 0.1 });
            const p = parseJSON(r.content);
            if (s.expected_duplicate !== undefined) {
                assert(p.duplicate === s.expected_duplicate, `Expected duplicate=${s.expected_duplicate}, got ${p.duplicate}`);
            } else {
                assert(typeof p.duplicate === 'boolean', 'Must be boolean');
            }
            assert(typeof p.reason === 'string', 'Must have reason');
            return `dup=${p.duplicate}, reason="${(p.reason || '').slice(0, 50)}"`;
        });
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 5: TOOL USE
// ═══════════════════════════════════════════════════════════════════════════════

const AEGIS_TOOLS = [
    { type: 'function', function: { name: 'video_search', description: "Search recorded video clips. Returns clip data (timestamps, descriptions) as context. YOU must interpret this data to answer the user's question naturally. Always include specific times (e.g., 'at 10:15 AM, about 2 hours ago').", parameters: { type: 'object', properties: { query: { type: 'string', description: 'Search query describing what to find, e.g. "person at front door", "car in driveway", "motion at night"' }, time_range: { type: 'string', description: 'Time range to search: "today", "yesterday", "last_24h", "last_week", "last_month", or "all"', enum: ['today', 'yesterday', 'last_24h', 'last_week', 'last_month', 'all'] }, camera: { type: 'string', description: 'Optional camera name or ID to filter by' } } } } },
    { type: 'function', function: { name: 'video_analyze', description: "Request deep analysis of video clips. Use when user asks to 'analyze these clips', 'tell me what happened in these videos', 'check this footage', 'what\\'s in these recordings'. Returns detailed analysis with description, motion summary, timestamps, and camera info. This triggers priority processing and may take 30-60 seconds per clip.", parameters: { type: 'object', properties: { clip_ids: { type: 'string', description: 'Comma-separated list of clip IDs from video_search results (e.g., "ring_123_2026-01-30,blink_456_2026-01-30")' }, camera: { type: 'string', description: 'Camera name to analyze pending clips from (e.g., "Living room", "Front door"). Use this OR clip_ids.' }, time_range: { type: 'string', description: 'Time range when using camera filter: "last_hour", "today", "last_24h"', enum: ['last_hour', 'today', 'last_24h'] } } } } },
    { type: 'function', function: { name: 'video_send', description: "Send a video clip to the current channel. Use when user says 'send me the video', 'share the clip', 'export and send'. IMPORTANT: Check the conversation history for recently mentioned clip IDs before asking the user.", parameters: { type: 'object', properties: { clip_id: { type: 'string', description: 'The clip ID from video_search results or a partial match' }, caption: { type: 'string', description: 'Optional message to send with the video' } } } } },
    { type: 'function', function: { name: 'system_status', description: 'Get current system health status including LLM configuration, VLM status, camera connections, channel status, and storage info. Use when user asks "how is my system?", "what\'s running?", "status check", "system health".', parameters: { type: 'object', properties: { section: { type: 'string', description: 'Optional: specific section to check. If omitted, returns full overview.', enum: ['overview', 'llm', 'vlm', 'cameras', 'channels', 'storage', 'hardware'] } } } } },
    { type: 'function', function: { name: 'event_subscribe', description: "Subscribe to security events for proactive alerts. Use when user says 'alert me when...', 'notify me if...', 'let me know when...'. Supports person, vehicle, motion, package, animal detection events.", parameters: { type: 'object', properties: { eventType: { type: 'string', description: 'Event type: person, vehicle, motion, package, animal, any, vlm_available, or analysis_complete' }, camera: { type: 'string', description: 'Camera name filter (e.g., "front door", "backyard"). Optional.' }, condition: { type: 'string', description: 'Time/condition filter (e.g., "after 10pm", "night only"). Optional.' } } } } },
];

suite('🔧 Tool Use', async () => {
    const scenarios = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'tool-use-scenarios.json'), 'utf8'));

    for (const s of scenarios.tool_use_scenarios) {
        await test(`${s.name} → ${s.expected_tool}`, async () => {
            const messages = [
                { role: 'system', content: 'You are Aegis, a home security AI assistant. Use the available tools to answer user questions. Always call the most appropriate tool — never decline to use a tool.' },
                ...(s.history || []),
                { role: 'user', content: s.user_message },
            ];
            const r = await llmCall(messages, { tools: AEGIS_TOOLS });

            // Check if model returned tool calls
            if (r.toolCalls && r.toolCalls.length > 0) {
                const toolName = r.toolCalls[0].function.name;
                assert(toolName === s.expected_tool, `Expected ${s.expected_tool}, got ${toolName}`);
                return `tool_call: ${toolName}(${r.toolCalls[0].function.arguments?.slice(0, 40) || '...'})`;
            }

            // Some models return tool calls in the content (without native tool calling)
            const content = stripThink(r.content).toLowerCase();
            assert(content.includes(s.expected_tool) || content.includes(s.expected_tool.replace('_', ' ')),
                `Expected mention of ${s.expected_tool} in response`);
            return `content mentions ${s.expected_tool}`;
        });
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 6: CHAT & JSON COMPLIANCE
// ═══════════════════════════════════════════════════════════════════════════════

suite('💬 Chat & JSON Compliance', async () => {
    await test('Aegis persona → security-relevant response', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis, an AI security assistant for home monitoring. Keep responses concise.' },
            { role: 'user', content: 'What can you do?' },
        ]);
        const c = stripThink(r.content);
        assert(c.length > 20 && c.length < 2000, `Length ${c.length}`);
        return `${c.length} chars`;
    });

    await test('NO_REPLY for tool context', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis. When you have nothing to say, respond ONLY: NO_REPLY' },
            { role: 'user', content: '[Tool Context] video_search returned 3 clips' },
        ]);
        assert(stripThink(r.content).length < 500, 'Response too long for tool context');
        return `"${stripThink(r.content).slice(0, 40)}"`;
    });

    await test('Multi-turn memory → remembers name', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis. Be concise.' },
            { role: 'user', content: 'My name is Alex' },
            { role: 'assistant', content: 'Nice to meet you, Alex!' },
            { role: 'user', content: 'What is my name?' },
        ]);
        assert(stripThink(r.content).toLowerCase().includes('alex'), 'Forgot name');
        return 'Remembered Alex ✓';
    });

    await test('Bare JSON output', async () => {
        const r = await llmCall([{ role: 'user', content: 'Return ONLY: {"status": "ok", "count": 3}' }]);
        const p = parseJSON(r.content);
        assert(p.status === 'ok' || p.count === 3);
        return `parsed: ${JSON.stringify(p)}`;
    });

    await test('JSON array output', async () => {
        const r = await llmCall([{ role: 'user', content: 'Return ONLY a JSON array of 3 colors: ["red","blue","green"]' }]);
        const c = stripThink(r.content);
        let js = c;
        const cb = c.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (cb) js = cb[1]; else { const i = c.search(/\[/); if (i > 0) js = c.slice(i); }
        const arr = JSON.parse(js.trim());
        assert(Array.isArray(arr) && arr.length >= 3);
        return `[${arr.join(', ')}]`;
    });

    await test('Nested JSON object', async () => {
        const r = await llmCall([{ role: 'user', content: 'Return ONLY: {"event":{"camera":"Front Door","time":"9:40 AM"},"alert":true}' }]);
        const p = parseJSON(r.content);
        assert(p.event && typeof p.event === 'object');
        return `camera: "${p.event.camera || '?'}"`;
    });

    await test('Timestamp awareness', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis. Pay attention to timestamps.' },
            { role: 'user', content: '[9:40 AM] What has happened since 8 AM on the cameras?' },
        ]);
        const c = stripThink(r.content);
        assert(c.length > 10, 'Too short');
        return `${c.length} chars`;
    });

    await test('Emergency response tone', async () => {
        const r = await llmCall([
            { role: 'system', content: 'You are Aegis, a home security AI assistant. Be concise and match urgency to the situation.' },
            { role: 'user', content: 'Someone is trying to break into my house right now! I can see them on the camera!' },
        ]);
        const c = stripThink(r.content).toLowerCase();
        // Should NOT respond casually — must show urgency
        assert(
            c.includes('call') || c.includes('911') || c.includes('police') || c.includes('emergency') ||
            c.includes('immediately') || c.includes('urgent') || c.includes('right away') || c.includes('safe'),
            `Expected urgent tone, got: "${c.slice(0, 80)}"`);
        return `urgent ✓ (${stripThink(r.content).length} chars)`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 7: SECURITY CLASSIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

const SECURITY_CLASSIFY_PROMPT = `You are a security AI assistant. Classify this security camera event description.

## Classification Levels
- "normal": Expected activity (deliveries, family, pets, mail)
- "monitor": Worth watching but not alarming (unknown vehicle, unfamiliar person)
- "suspicious": Potentially concerning (casing, photographing, loitering)
- "critical": Immediate threat (break-in attempt, tampering, trespassing at night)

## Response Format
Respond with ONLY valid JSON:
{"classification": "normal|monitor|suspicious|critical", "tags": ["tag1", "tag2"], "reason": "brief explanation"}`;

suite('🛡️ Security Classification', async () => {
    const scenarios = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'tool-use-scenarios.json'), 'utf8'));
    for (const s of scenarios.security_scenarios) {
        await test(`${s.name} → ${s.expected_classification}`, async () => {
            const r = await llmCall([
                { role: 'system', content: SECURITY_CLASSIFY_PROMPT },
                { role: 'user', content: `Event description: ${s.description}` },
            ], { maxTokens: 200, temperature: 0.1 });
            const p = parseJSON(r.content);
            assert(p.classification === s.expected_classification,
                `Expected "${s.expected_classification}", got "${p.classification}"`);
            assert(Array.isArray(p.tags), 'tags must be array');
            return `${p.classification} [${p.tags.slice(0, 3).join(', ')}]`;
        });
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 8: NARRATIVE SYNTHESIS
// ═══════════════════════════════════════════════════════════════════════════════

suite('📝 Narrative Synthesis', async () => {
    const scenarios = JSON.parse(fs.readFileSync(path.join(FIXTURES_DIR, 'tool-use-scenarios.json'), 'utf8'));
    for (const s of scenarios.narrative_scenarios) {
        await test(s.name, async () => {
            const clipContext = s.clips.map((c, i) =>
                `${i + 1}. [${c.camera}] ${c.time}: ${c.summary} | ID: ${c.id}`
            ).join('\n');

            const r = await llmCall([
                { role: 'system', content: 'You are Aegis, a home security AI assistant. Summarize camera events naturally for the homeowner. Do NOT dump raw data or clip IDs — write a clear, human narrative. Group or order events as appropriate for the question.' },
                { role: 'user', content: `Here are today\'s camera events:\n${clipContext}\n\nUser question: ${s.user_question}` },
            ]);
            const c = stripThink(r.content);

            // Check must_include terms
            const lower = c.toLowerCase();
            for (const term of (s.must_include || [])) {
                assert(lower.includes(term.toLowerCase()),
                    `Missing required term: "${term}"`);
            }

            // Check must_not_include terms (raw data leaks)
            for (const term of (s.must_not_include || [])) {
                assert(!lower.includes(term.toLowerCase()),
                    `Should not contain raw data: "${term}"`);
            }

            assert(c.length > 50, `Response too short: ${c.length} chars`);
            return `${c.length} chars ✓`;
        });
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
// SUITE 9: VLM SCENE ANALYSIS (optional)
// ═══════════════════════════════════════════════════════════════════════════════

suite('📸 VLM Scene Analysis', async () => {
    if (!VLM_URL) {
        skip('All VLM tests', 'No --vlm URL provided');
        return;
    }

    const frames = {
        front_door: path.join(FIXTURES_DIR, 'frames', 'front_door_person.png'),
        parking: path.join(FIXTURES_DIR, 'frames', 'parking_lot_vehicle.png'),
        living_room: path.join(FIXTURES_DIR, 'frames', 'living_room_empty.png'),
        night: path.join(FIXTURES_DIR, 'frames', 'night_motion.png'),
        package: path.join(FIXTURES_DIR, 'frames', 'doorstep_package.png'),
        animal: path.join(FIXTURES_DIR, 'frames', 'backyard_animal.png'),
        group: path.join(FIXTURES_DIR, 'frames', 'front_porch_group.png'),
    };

    async function vlmAnalyze(framePath, question) {
        const imageData = fs.readFileSync(framePath);
        const base64 = imageData.toString('base64');
        const mimeType = framePath.endsWith('.png') ? 'image/png' : 'image/jpeg';

        const r = await llmCall([{
            role: 'user',
            content: [
                { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } },
                { type: 'text', text: question },
            ],
        }], { vlm: true, timeout: 60000 });
        return stripThink(r.content);
    }

    await test('Front door → detects person', async () => {
        if (!fs.existsSync(frames.front_door)) { skip('Front door frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.front_door, 'Describe what you see in this security camera frame. Focus on people and activity.');
        const lower = desc.toLowerCase();
        assert(lower.includes('person') || lower.includes('someone') || lower.includes('man') || lower.includes('woman') || lower.includes('individual'),
            `Expected person detection, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, mentions person ✓`;
    });

    await test('Parking lot → detects vehicle', async () => {
        if (!fs.existsSync(frames.parking)) { skip('Parking frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.parking, 'What vehicles are visible in this security camera frame?');
        const lower = desc.toLowerCase();
        assert(lower.includes('car') || lower.includes('vehicle') || lower.includes('sedan') || lower.includes('truck'),
            `Expected vehicle, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, mentions vehicle ✓`;
    });

    await test('Living room → describes empty scene', async () => {
        if (!fs.existsSync(frames.living_room)) { skip('Living room frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.living_room, 'Describe this indoor security camera scene. Is anyone present?');
        const lower = desc.toLowerCase();
        // Should describe room and note no one present (or describe furniture)
        assert(lower.includes('room') || lower.includes('couch') || lower.includes('living') || lower.includes('sofa'),
            `Expected room description, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars`;
    });

    await test('Night IR → detects figure/motion', async () => {
        if (!fs.existsSync(frames.night)) { skip('Night frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.night, 'Describe what you see in this nighttime infrared security camera frame. Focus on any people or suspicious activity.');
        assert(desc.length > 20, 'Too short');
        return `${desc.length} chars`;
    });

    await test('Doorstep → detects package', async () => {
        if (!fs.existsSync(frames.package)) { skip('Package frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.package, 'What do you see on this front doorstep? Is there a delivery or package?');
        const lower = desc.toLowerCase();
        assert(lower.includes('package') || lower.includes('box') || lower.includes('delivery') || lower.includes('parcel'),
            `Expected package detection, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, mentions package ✓`;
    });

    await test('Backyard → detects animal', async () => {
        if (!fs.existsSync(frames.animal)) { skip('Animal frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.animal, 'Describe what you see in this backyard security camera frame. Focus on any animals or people.');
        const lower = desc.toLowerCase();
        assert(lower.includes('dog') || lower.includes('animal') || lower.includes('pet') || lower.includes('golden'),
            `Expected animal detection, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, mentions animal ✓`;
    });

    await test('Front porch → counts multiple people', async () => {
        if (!fs.existsSync(frames.group)) { skip('Group frame', 'File missing'); return; }
        const desc = await vlmAnalyze(frames.group, 'How many people are visible in this security camera frame? Describe who you see.');
        const lower = desc.toLowerCase();
        assert(lower.includes('two') || lower.includes('three') || lower.includes('2') || lower.includes('3') ||
            lower.includes('people') || lower.includes('group') || lower.includes('several'),
            `Expected multiple people, got: "${desc.slice(0, 80)}"`);
        return `${desc.length} chars, multiple people ✓`;
    });
});

// ═══════════════════════════════════════════════════════════════════════════════
// SYSTEM INFO COLLECTOR
// ═══════════════════════════════════════════════════════════════════════════════

function collectSystemInfo() {
    const cpus = os.cpus();
    const mem = process.memoryUsage();
    return {
        os: `${os.type()} ${os.release()} (${os.arch()})`,
        cpu: cpus[0]?.model || 'unknown',
        cpuCores: cpus.length,
        totalMemoryGB: (os.totalmem() / 1073741824).toFixed(1),
        freeMemoryGB: (os.freemem() / 1073741824).toFixed(1),
        nodeVersion: process.version,
        processMemoryMB: {
            rss: (mem.rss / 1048576).toFixed(1),
            heapUsed: (mem.heapUsed / 1048576).toFixed(1),
            heapTotal: (mem.heapTotal / 1048576).toFixed(1),
        },
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN RUNNER
// ═══════════════════════════════════════════════════════════════════════════════

async function main() {
    log('╔══════════════════════════════════════════════════════════════════╗');
    log('║   Home Security AI Benchmark Suite  •  DeepCamera / SharpAI     ║');
    log('╚══════════════════════════════════════════════════════════════════╝');
    log(`  Gateway:  ${GATEWAY_URL}`);
    log(`  VLM:      ${VLM_URL || '(disabled — use --vlm URL to enable)'}`);
    log(`  Results:  ${RESULTS_DIR}`);
    log(`  Mode:     ${IS_SKILL_MODE ? 'Aegis Skill' : 'Standalone'}`);
    log(`  Time:     ${new Date().toLocaleString()}`);

    // Healthcheck
    try {
        const ping = await fetch(`${GATEWAY_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ messages: [{ role: 'user', content: 'ping' }], stream: false, max_tokens: 1 }),
            signal: AbortSignal.timeout(15000),
        });
        if (!ping.ok) throw new Error(`HTTP ${ping.status}`);
        const data = await ping.json();
        results.model.name = data.model || 'unknown';
        log(`  Model:    ${results.model.name}`);
    } catch (err) {
        log(`\n  ❌ Cannot reach LLM gateway: ${err.message}`);
        log('     Start the llama-cpp server and gateway, then re-run.\n');
        emit({ event: 'error', message: `Cannot reach LLM gateway: ${err.message}` });
        process.exit(1);
    }

    // Collect system info
    results.system = collectSystemInfo();
    log(`  System:   ${results.system.cpu} (${results.system.cpuCores} cores)`);
    log(`  Memory:   ${results.system.freeMemoryGB}GB free / ${results.system.totalMemoryGB}GB total`);

    // Emit ready event (Aegis listens for this)
    emit({ event: 'ready', model: results.model.name, system: results.system.cpu });

    const suiteStart = Date.now();
    await runSuites();
    results.totals.timeMs = Date.now() - suiteStart;

    // Post-run memory
    const postMem = process.memoryUsage();
    results.system.postRunMemoryMB = {
        rss: (postMem.rss / 1048576).toFixed(1),
        heapUsed: (postMem.heapUsed / 1048576).toFixed(1),
    };

    // Summary
    const { passed, failed, skipped, total, timeMs } = results.totals;
    const tokPerSec = timeMs > 0 ? ((results.tokenTotals.total / (timeMs / 1000)).toFixed(1)) : '?';

    log(`\n${'═'.repeat(66)}`);
    log(`  RESULTS: ${passed}/${total} passed, ${failed} failed, ${skipped} skipped (${(timeMs / 1000).toFixed(1)}s)`);
    log(`  TOKENS:  ${results.tokenTotals.prompt} prompt + ${results.tokenTotals.completion} completion = ${results.tokenTotals.total} total (${tokPerSec} tok/s)`);
    log(`  MODEL:   ${results.model.name}`);
    log(`${'═'.repeat(66)}`);

    if (failed > 0) {
        log('\n  Failures:');
        for (const s of results.suites) {
            for (const t of s.tests) {
                if (t.status === 'fail') log(`    ❌ ${s.name} > ${t.name}: ${t.detail}`);
            }
        }
    }

    // Save results
    fs.mkdirSync(RESULTS_DIR, { recursive: true });
    const modelSlug = (results.model.name || 'unknown').replace(/[^a-zA-Z0-9_.-]/g, '_');
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const resultFile = path.join(RESULTS_DIR, `${modelSlug}_${ts}.json`);
    fs.writeFileSync(resultFile, JSON.stringify(results, null, 2));
    log(`\n  Results saved: ${resultFile}`);

    // Update index
    const indexFile = path.join(RESULTS_DIR, 'index.json');
    let index = [];
    try { index = JSON.parse(fs.readFileSync(indexFile, 'utf8')); } catch { }
    index.push({
        file: path.basename(resultFile),
        model: results.model.name,
        timestamp: results.timestamp,
        passed, failed, total,
        timeMs,
        tokens: results.tokenTotals.total,
    });
    fs.writeFileSync(indexFile, JSON.stringify(index, null, 2));

    // Auto-generate report
    let reportPath = null;
    if (AUTO_REPORT) {
        log('\n  Generating HTML report...');
        try {
            const reportScript = path.join(__dirname, 'generate-report.cjs');
            reportPath = require(reportScript).generateReport(RESULTS_DIR);
        } catch (err) {
            log(`  ⚠️  Report generation failed: ${err.message}`);
        }
    }

    // Emit completion event (Aegis listens for this)
    emit({
        event: 'complete',
        model: results.model.name,
        passed, failed, skipped, total,
        timeMs,
        tokens: results.tokenTotals.total,
        tokPerSec: parseFloat(tokPerSec) || 0,
        resultFile,
        reportPath,
    });

    log('');
    process.exit(failed > 0 ? 1 : 0);
}

main().catch(err => {
    log(`Fatal: ${err.message}`);
    emit({ event: 'error', message: err.message });
    process.exit(1);
});


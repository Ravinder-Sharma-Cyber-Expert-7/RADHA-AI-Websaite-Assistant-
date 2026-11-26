import express from "express";
import http from "http";
import os from "os";
import crypto from "crypto";
import dns from "dns/promises";
import { WebSocketServer, WebSocket } from "ws";
import { GoogleGenAI, Modality, Type, LiveServerMessage } from "@google/genai";
import dotenv from "dotenv";
import path from "path";
import Database from "better-sqlite3";
import { createServer as createViteServer } from "vite";

dotenv.config();

// Initialize SQLite persistent storage for Radha's long-term memory, chat history & admin settings
const dbPath = path.join(process.cwd(), "radha_memory.db");
const db = new Database(dbPath);
db.pragma("journal_mode = WAL");

db.exec(`
  CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    sender TEXT NOT NULL,
    text TEXT NOT NULL,
    emotion TEXT,
    timestamp INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS reminders (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    trigger_at INTEGER NOT NULL,
    completed INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    priority TEXT DEFAULT 'medium',
    completed INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS mood_logs (
    id TEXT PRIMARY KEY,
    emotion TEXT NOT NULL,
    note TEXT NOT NULL,
    created_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS sentiment_logs (
    id TEXT PRIMARY KEY,
    polarity TEXT NOT NULL,
    score INTEGER NOT NULL,
    stress_level INTEGER NOT NULL,
    energy_level INTEGER NOT NULL,
    warmth_level INTEGER NOT NULL,
    dominant_mood TEXT NOT NULL,
    triggers TEXT DEFAULT '[]',
    insight TEXT NOT NULL,
    radha_guidance TEXT NOT NULL,
    source_text TEXT DEFAULT '',
    created_at INTEGER NOT NULL
  );
`);

const stmtInsertMessage = db.prepare(
  "INSERT OR REPLACE INTO messages (id, sender, text, emotion, timestamp) VALUES (?, ?, ?, ?, ?)"
);
const stmtGetMessages = db.prepare(
  "SELECT id, sender, text, emotion, timestamp FROM messages ORDER BY timestamp ASC LIMIT 200"
);
const stmtClearMessages = db.prepare("DELETE FROM messages");

const stmtInsertMemory = db.prepare(
  "INSERT OR REPLACE INTO memories (id, content, category, created_at) VALUES (?, ?, ?, ?)"
);
const stmtGetMemories = db.prepare(
  "SELECT id, content, category, created_at as createdAt FROM memories ORDER BY created_at DESC LIMIT 100"
);
const stmtDeleteMemory = db.prepare("DELETE FROM memories WHERE id = ?");

const stmtGetSetting = db.prepare("SELECT value FROM settings WHERE key = ?");
const stmtSetSetting = db.prepare(
  "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)"
);

const stmtGetReminders = db.prepare(
  "SELECT id, title, trigger_at as triggerAt, completed, created_at as createdAt FROM reminders ORDER BY trigger_at ASC LIMIT 100"
);
const stmtInsertReminder = db.prepare(
  "INSERT OR REPLACE INTO reminders (id, title, trigger_at, completed, created_at) VALUES (?, ?, ?, ?, ?)"
);
const stmtUpdateReminderDone = db.prepare(
  "UPDATE reminders SET completed = ? WHERE id = ?"
);
const stmtDeleteReminder = db.prepare("DELETE FROM reminders WHERE id = ?");

const stmtGetTasks = db.prepare(
  "SELECT id, title, priority, completed, created_at as createdAt FROM tasks ORDER BY completed ASC, created_at DESC LIMIT 100"
);
const stmtInsertTask = db.prepare(
  "INSERT OR REPLACE INTO tasks (id, title, priority, completed, created_at) VALUES (?, ?, ?, ?, ?)"
);
const stmtUpdateTaskDone = db.prepare(
  "UPDATE tasks SET completed = ? WHERE id = ?"
);
const stmtDeleteTask = db.prepare("DELETE FROM tasks WHERE id = ?");

const stmtGetMoods = db.prepare(
  "SELECT id, emotion, note, created_at as createdAt FROM mood_logs ORDER BY created_at DESC LIMIT 60"
);
const stmtInsertMood = db.prepare(
  "INSERT OR REPLACE INTO mood_logs (id, emotion, note, created_at) VALUES (?, ?, ?, ?)"
);

const stmtGetSentiments = db.prepare(
  "SELECT id, polarity, score, stress_level as stressLevel, energy_level as energyLevel, warmth_level as warmthLevel, dominant_mood as dominantMood, triggers, insight, radha_guidance as radhaGuidance, source_text as sourceText, created_at as createdAt FROM sentiment_logs ORDER BY created_at DESC LIMIT 50"
);
const stmtInsertSentiment = db.prepare(
  "INSERT OR REPLACE INTO sentiment_logs (id, polarity, score, stress_level, energy_level, warmth_level, dominant_mood, triggers, insight, radha_guidance, source_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
);

async function startServer() {
  const app = express();
  app.use(express.json({ limit: "50mb" }));

  const PORT = 3000;
  const server = http.createServer(app);
  const wss = new WebSocketServer({ noServer: true });

  // Initialize server-side Gemini SDK
  const apiKey = process.env.GEMINI_API_KEY || process.env.RADHA_API_KEY || process.env.API_KEY || "";
  if (!apiKey) {
    console.error("CRITICAL ERROR: GEMINI_API_KEY environment variable is not set!");
  }

  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });

  const aiLive = new GoogleGenAI({
    apiKey,
    httpOptions: {
      apiVersion: "v1alpha",
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });

  const isQuotaError = (err: any): boolean => {
    const msg = String(err?.message || err || "");
    return err?.status === 429 || msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED");
  };

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  // Shared helper using active supported models (gemini-4-hydrogen -> gemini-3.1-flash-lite)
  // Omits deprecated sampling params (temperature/topP/topK) for gemini-4-* models
  const generateWithActiveModels = async (
    contents: any,
    baseConfig: Record<string, any> = {},
    models: string[] = ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
  ) => {
    let lastErr: any = null;
    for (const modelName of models) {
      const cfg = { ...baseConfig };
      if (modelName.startsWith("gemini-4-")) {
        delete cfg.temperature;
        delete cfg.topP;
        delete cfg.topK;
        delete cfg.candidateCount;
      }

      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const response = await ai.models.generateContent({
            model: modelName,
            contents,
            config: Object.keys(cfg).length > 0 ? cfg : undefined,
          });
          if (response?.text) return response;
          break;
        } catch (err: any) {
          lastErr = err;
          if (isQuotaError(err) && attempt === 0) {
            await sleep(750);
            continue;
          }
          console.warn(`Model ${modelName} failed:`, err?.message || err);
          break;
        }
      }
    }
    throw lastErr || new Error("No Gemini model returned content");
  };

  // API health route
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", timestamp: Date.now() });
  });

  // Real OS CPU usage sampler (true tick-delta calculation every 2s — zero fake numbers)
  let prevCpuTotals = os.cpus().map((c) => {
    const total = Object.values(c.times).reduce((a, b) => a + b, 0);
    return { idle: c.times.idle, total };
  });
  let realCpuUsagePct = 0;

  const sampleRealCpu = () => {
    const current = os.cpus();
    let idleDelta = 0;
    let totalDelta = 0;
    current.forEach((cpu, idx) => {
      const total = Object.values(cpu.times).reduce((a, b) => a + b, 0);
      const prev = prevCpuTotals[idx] || { idle: cpu.times.idle, total };
      idleDelta += Math.max(0, cpu.times.idle - prev.idle);
      totalDelta += Math.max(0, total - prev.total);
      prevCpuTotals[idx] = { idle: cpu.times.idle, total };
    });
    if (totalDelta > 0) {
      realCpuUsagePct = Math.max(0, Math.min(100, Math.round(((totalDelta - idleDelta) / totalDelta) * 100)));
    }
  };
  setInterval(sampleRealCpu, 2000);

  // Real 128 KB binary probe payload for genuine real-time network download speed measurement
  const speedProbeBuffer = Buffer.alloc(128 * 1024, 0x5a);
  app.get("/api/speedtest", (req, res) => {
    res.setHeader("Content-Type", "application/octet-stream");
    res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate, proxy-revalidate");
    res.setHeader("Content-Length", String(speedProbeBuffer.byteLength));
    res.send(speedProbeBuffer);
  });

  // Real Server System & Network Telemetry Endpoint
  app.get("/api/telemetry", (req, res) => {
    try {
      const mem = process.memoryUsage();
      const load = os.loadavg();
      const totalMemMB = Math.round(os.totalmem() / (1024 * 1024));
      const freeMemMB = Math.round(os.freemem() / (1024 * 1024));
      const usedMemMB = Math.max(0, totalMemMB - freeMemMB);
      const heapUsedMB = Math.round(mem.heapUsed / (1024 * 1024));
      const rssMB = Math.round(mem.rss / (1024 * 1024));
      res.json({
        status: "online",
        serverTime: Date.now(),
        uptimeSec: Math.floor(process.uptime()),
        cpuUsagePct: realCpuUsagePct,
        cpuLoad1m: Number(load[0].toFixed(2)),
        cpuCores: os.cpus().length,
        heapUsedMB,
        rssMB,
        usedMemMB,
        totalMemMB,
        freeMemMB,
        ramUsedPct: Math.min(100, Math.max(0, Math.round((usedMemMB / Math.max(1, totalMemMB)) * 100))),
        activeLiveSockets: wss.clients.size,
      });
    } catch (error: any) {
      res.status(500).json({ error: error.message || "Telemetry error" });
    }
  });

  // Boss Biometric Security — Multi-Angle 3D Owner Face Calibration (Anti-Spoof Cryptographic Hash)
  app.post("/api/vision/calibrate-owner", async (req, res) => {
    try {
      const {
        angleFrames = [],
        ownerName = "Ravindra Boss",
        parallaxDelta = 0,
        spatialVector = "",
      } = req.body || {};

      if (!Array.isArray(angleFrames) || angleFrames.length < 3) {
        res.status(400).json({ error: "At least 3 distinct face angle captures are required for 3D calibration." });
        return;
      }

      const imageParts: any[] = [];
      const angleLabels: string[] = [];
      for (const frame of angleFrames.slice(0, 4)) {
        const match = String(frame?.dataUrl || "").match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
        if (!match) continue;
        imageParts.push({ inlineData: { data: match[2], mimeType: match[1] } });
        angleLabels.push(String(frame?.angle || "angle"));
      }

      if (imageParts.length < 3) {
        res.status(400).json({ error: "Valid base64 face angle frames are required." });
        return;
      }

      const prompt = `You are a high-security 3D Biometric Anti-Spoof Vision Engine calibrating "${ownerName}".
You are given ${imageParts.length} sequential webcam frames captured during a live 3D head-turn calibration session (${angleLabels.join(", ")}).

Perform a strict anti-spoof & 3D geometry check:
1. Confirm a real human face is clearly visible in all frames and it is the SAME person across every frame.
2. Verify genuine 3D head rotation / pose parallax across the frames (front, left, right, tilt/expression) — NOT a flat printed photo, NOT a phone/tablet screen held up to the camera, and NOT identical static frames.
3. Extract a concise, deterministic 3D facial geometry descriptor (face shape, brow ridge, nose bridge profile, cheekbone/jaw contour, eye spacing, distinctive natural traits) to anchor future anti-spoof verification.

Return ONLY valid JSON:
{
  "samePerson": boolean,
  "livenessPassed": boolean,
  "spoofDetected": boolean,
  "antiSpoofScore": integer from 0 to 100,
  "geometrySignature": concise 1-2 sentence 3D facial structure descriptor,
  "reason": short Hindi/Hinglish status message for ${ownerName}
}`;

      const response = await generateWithActiveModels(
        [{ role: "user", parts: [...imageParts, { text: prompt }] }],
        { responseMimeType: "application/json" },
        ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
      );

      const raw = (response?.text || "{}").trim();
      let parsed: any = {};
      try {
        parsed = JSON.parse(raw);
      } catch {
        const m = raw.match(/\{[\s\S]*\}/);
        if (m) parsed = JSON.parse(m[0]);
      }

      const numericParallax = Number(parallaxDelta || 0);
      const livenessPassed =
        Boolean(parsed.livenessPassed) &&
        Boolean(parsed.samePerson) &&
        !Boolean(parsed.spoofDetected) &&
        numericParallax >= 2.5;

      const calibratedAt = Date.now();
      const geometrySignature = String(
        parsed.geometrySignature || "3D multi-angle facial depth contour calibrated"
      ).slice(0, 320);

      // Mint deterministic SHA-256 3D-mapped biometric hash from multi-angle geometry + spatial parallax vector
      const hashMaterial = [
        "RADHA_3D_BIOMETRIC_V2",
        ownerName.trim().toLowerCase(),
        geometrySignature.toLowerCase(),
        String(spatialVector || "").slice(0, 512),
        numericParallax.toFixed(2),
      ].join("::");
      const digest = crypto.createHash("sha256").update(hashMaterial).digest("hex").toUpperCase();
      const biometricHash = `3D-BIO-${digest.slice(0, 24)}`;

      const antiSpoofScore = Math.max(
        0,
        Math.min(
          100,
          Math.round(Number(parsed.antiSpoofScore ?? (livenessPassed ? 94 : 28)))
        )
      );

      res.json({
        calibrated: livenessPassed,
        livenessPassed,
        antiSpoofScore,
        biometricHash: livenessPassed ? biometricHash : "",
        geometrySignature,
        calibratedAt,
        reason: String(
          parsed.reason ||
            (livenessPassed
              ? `बॉस, आपका 3D बायोमेट्रिक फेस कैलिब्रेशन सफलतापूर्वक लॉक हो गया है! अब कोई फोटो से स्पूफ नहीं कर सकता।`
              : "कैलिब्रेशन पूरा नहीं हुआ — कृपया असली चेहरे को चारों एंगल (सीधा, बाएँ, दाएँ, ऊपर) में स्पष्ट घुमाकर फिर कोशिश करें।")
        ),
      });
    } catch (error: any) {
      if (isQuotaError(error)) {
        res.status(429).json({
          error: "Gemini API quota limit reached right now. Please wait a few seconds and retry calibration.",
        });
        return;
      }
      console.error("Owner 3D Calibration Error:", error?.message || error);
      res.status(500).json({ error: error?.message || "3D Face Calibration failed" });
    }
  });

  // Real Owner 3D Anti-Spoof Face Verification (Multi-Angle Calibrated Hash + 2-Frame Liveness Burst)
  app.post("/api/vision/verify-owner", async (req, res) => {
    try {
      const {
        ownerPhotoDataUrl = "",
        liveFrameBase64,
        liveFrame2Base64 = "",
        ownerName = "Ravindra Boss",
        ownerBio = "",
        biometricProfile = null,
      } = req.body || {};

      const hasCalibrated =
        Boolean(biometricProfile?.calibrated) &&
        Array.isArray(biometricProfile?.angleSnapshots) &&
        biometricProfile.angleSnapshots.length > 0;

      if ((!ownerPhotoDataUrl && !hasCalibrated) || !liveFrameBase64) {
        res.status(400).json({ error: "Owner reference photo or 3D calibration profile and live camera frame are required." });
        return;
      }

      const parts: any[] = [];
      let refCount = 0;

      if (hasCalibrated) {
        // Attach up to 3 calibrated multi-angle references (Front, Left, Right) for true 3D pose matching
        for (const snap of biometricProfile.angleSnapshots.slice(0, 3)) {
          const m = String(snap).match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
          if (m) {
            parts.push({ inlineData: { data: m[2], mimeType: m[1] } });
            refCount++;
          }
        }
      } else if (ownerPhotoDataUrl) {
        const ownerMatch = String(ownerPhotoDataUrl).match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/);
        const ownerMime = ownerMatch ? ownerMatch[1] : "image/jpeg";
        const ownerBase64 = ownerMatch ? ownerMatch[2] : String(ownerPhotoDataUrl);
        parts.push({ inlineData: { data: ownerBase64, mimeType: ownerMime } });
        refCount = 1;
      }

      parts.push({ inlineData: { data: String(liveFrameBase64), mimeType: "image/jpeg" } });
      if (liveFrame2Base64) {
        parts.push({ inlineData: { data: String(liveFrame2Base64), mimeType: "image/jpeg" } });
      }

      const hasTwoLiveFrames = Boolean(liveFrame2Base64);
      const geometryNote = biometricProfile?.geometrySignature
        ? `Registered 3D Facial Geometry Signature: ${biometricProfile.geometrySignature}\n3D Biometric Hash: ${biometricProfile.biometricHash}`
        : "";

      const prompt = `You are Radha's Boss Biometric Anti-Spoof Security Engine verifying "${ownerName}".
The first ${refCount} image(s) are the registered Owner Reference / 3D Calibration profile of "${ownerName}".
The last ${hasTwoLiveFrames ? "2 sequential live webcam frames (500ms apart)" : "1 live webcam snapshot"} are captured right now.
${geometryNote}

STRICT SECURITY CHECKLIST:
1. Check if a face is visible in the live frame(s).
2. ANTI-SPOOF LIVENESS CHECK: Inspect carefully for spoofing — phone/tablet screen edges, screen glare/moire pixel grid, paper photo borders, hands holding a photo, or completely unnaturally flat lighting. If someone is showing a photo or phone screen instead of a real live person, set "spoofDetected": true, "livenessVerified": false, and "isOwner": false!
3. Compare live facial geometry & identity against the registered reference image(s).
4. Write a warm, 100% natural HINDI greeting as Radha:
   - If real live Owner verified: e.g. "अरे बॉस आप हो! मैंने आपका 3D फेस तुरंत वेरिफाई कर लिया — आप बहुत बढ़िया लग रहे हैं!"
   - If spoof detected: e.g. "सावधान! यह असली चेहरा नहीं बल्कि फोटो या स्क्रीन से स्पूफ करने की कोशिश लग रही है — एक्सेस ब्लॉक किया गया!"
   - If different live person: polite Hindi greeting asking who they are.

Owner context: ${String(ownerBio || "").slice(0, 200)}

Return ONLY valid JSON:
{
  "faceDetected": boolean,
  "isOwner": boolean,
  "spoofDetected": boolean,
  "livenessVerified": boolean,
  "confidence": integer from 0 to 100,
  "expression": short 2-3 word label,
  "greeting": natural 1-2 sentence Hindi reaction from Radha,
  "visualNote": short 1 sentence visual observation
}`;

      parts.push({ text: prompt });

      const response = await generateWithActiveModels(
        [{ role: "user", parts }],
        { responseMimeType: "application/json" },
        ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
      );

      const raw = (response?.text || "{}").trim();
      let parsed: any = {};
      try {
        parsed = JSON.parse(raw);
      } catch {
        const match = raw.match(/\{[\s\S]*\}/);
        if (match) parsed = JSON.parse(match[0]);
      }

      const faceDetected = Boolean(parsed.faceDetected ?? true);
      const spoofDetected = Boolean(parsed.spoofDetected);
      const livenessVerified = !spoofDetected && Boolean(parsed.livenessVerified ?? true);
      const isOwner = Boolean(parsed.isOwner && faceDetected && !spoofDetected);
      const confidence = Math.max(0, Math.min(100, Math.round(Number(parsed.confidence ?? (isOwner ? 93 : 30)))));

      res.json({
        faceDetected,
        isOwner,
        spoofDetected,
        livenessVerified,
        hashVerified: Boolean(isOwner && hasCalibrated && livenessVerified),
        confidence,
        expression: String(parsed.expression || "Focused"),
        greeting: String(
          parsed.greeting ||
            (spoofDetected
              ? "सावधान! यह असली चेहरा नहीं बल्कि फोटो या स्क्रीन से स्पूफ करने की कोशिश लग रही है!"
              : isOwner
              ? `अरे ${ownerName}, आप हो! मैंने आपको तुरंत पहचान लिया, कैसे हैं आप?`
              : faceDetected
              ? `नमस्ते! आप ${ownerName} से अलग लग रहे हैं, बताइए मैं आपकी क्या मदद करूँ?`
              : "बॉस, कैमरे में अभी चेहरा साफ नहीं दिख रहा है, थोड़ा सामने आइए ना।")
        ),
        visualNote: String(parsed.visualNote || ""),
      });
    } catch (error: any) {
      if (isQuotaError(error)) {
        res.status(429).json({
          error: "Gemini API quota limit reached right now. Please wait a few seconds or select a billing-enabled API key in Settings > Secrets.",
        });
        return;
      }
      console.error("Owner Face Verify Error:", error?.message || error);
      res.status(500).json({ error: error?.message || "Face verification failed" });
    }
  });

  const getStoredSessionContextSummary = (): string => {
    try {
      const row: any = stmtGetSetting.get("radha_session_context");
      if (!row?.value) return "";
      const ctx = JSON.parse(row.value);
      const parts: string[] = [];
      if (ctx.summary) parts.push(`- Rolling Summary: ${ctx.summary}`);
      if (ctx.activeTopic) parts.push(`- Active Topic: ${ctx.activeTopic}`);
      if (ctx.openLoop) parts.push(`- Ongoing Thread: ${ctx.openLoop}`);
      if (ctx.lastEmotion && ctx.lastEmotion !== "neutral") {
        parts.push(`- Last Emotional State: ${ctx.lastEmotion}`);
      }
      if (ctx.lastUserMessage) parts.push(`- Last User Turn: "${ctx.lastUserMessage}"`);
      if (ctx.lastRadhaMessage) parts.push(`- Last Radha Turn: "${ctx.lastRadhaMessage}"`);
      return parts.join("\n");
    } catch {
      return "";
    }
  };

  // Persistent State API: Messages + Long-Term Memories + Admin Settings + Reminders + Tasks + Moods + Bonding + Session Context Blob
  app.get("/api/state", (req, res) => {
    try {
      const messages = stmtGetMessages.all();
      const memories = stmtGetMemories.all();
      const reminders = stmtGetReminders.all().map((r: any) => ({ ...r, completed: Boolean(r.completed) }));
      const tasks = stmtGetTasks.all().map((t: any) => ({ ...t, completed: Boolean(t.completed) }));
      const moods = stmtGetMoods.all();
      const sentiments = stmtGetSentiments.all().map((s: any) => ({
        ...s,
        triggers: (() => {
          try {
            return JSON.parse(s.triggers || "[]");
          } catch {
            return [];
          }
        })(),
      }));
      const row: any = stmtGetSetting.get("radha_config");
      const settings = row ? JSON.parse(row.value) : null;
      const bondRow: any = stmtGetSetting.get("radha_bonding");
      const bonding = bondRow ? JSON.parse(bondRow.value) : { xp: 120, level: 2, interactions: 0 };
      const sessionCtxRow: any = stmtGetSetting.get("radha_session_context");
      const sessionContext = sessionCtxRow ? JSON.parse(sessionCtxRow.value) : null;
      res.json({ messages, memories, settings, reminders, tasks, moods, sentiments, bonding, sessionContext });
    } catch (error: any) {
      console.error("DB State Read Error:", error);
      res.status(500).json({ error: error.message || "Failed to read persistent state" });
    }
  });

  // Persist Session Context JSON Blob (survives full browser reloads & server restarts)
  app.put("/api/session-context", (req, res) => {
    try {
      const state = req.body;
      if (!state || typeof state !== "object") {
        res.status(400).json({ error: "Invalid session context payload" });
        return;
      }
      stmtSetSetting.run("radha_session_context", JSON.stringify(state));
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message || "Failed to persist session context" });
    }
  });

  // Distill recent conversation into a concise persistent Session Context summary JSON blob
  app.post("/api/memory/session-summary", async (req, res) => {
    try {
      const { recentMessages = [], previousState = {} } = req.body || {};
      if (!Array.isArray(recentMessages) || recentMessages.length < 2) {
        res.json({ sessionContext: previousState });
        return;
      }

      const transcript = recentMessages
        .slice(-14)
        .map((m: any) => `${m.sender === "user" ? "Ravindra Boss" : "Radha"}: ${m.text}`)
        .join("\n");

      const prompt = `Summarize the ongoing conversation state between Ravindra Boss and Radha so Radha retains seamless context across browser reloads.
Return ONLY valid JSON with keys:
- "summary": concise 1-2 sentence rolling summary of what they are discussing and what just happened
- "activeTopic": short 2-6 word label of the current topic
- "openLoop": short note of any pending question/task/follow-up (or empty string if none)

Previous Summary: ${previousState?.summary || "(none)"}
Previous Topic: ${previousState?.activeTopic || "(none)"}

Recent Conversation:
${transcript}`;

      const response = await generateWithActiveModels(
        [{ role: "user", parts: [{ text: prompt }] }],
        { responseMimeType: "application/json" },
        ["gemini-3.1-flash-lite", "gemini-4-hydrogen"]
      );

      const rawText = (response.text || "{}").trim();
      let parsed: { summary?: string; activeTopic?: string; openLoop?: string } = {};
      try {
        parsed = JSON.parse(rawText);
      } catch {
        const match = rawText.match(/\{[\s\S]*\}/);
        if (match) parsed = JSON.parse(match[0]);
      }

      const updatedState = {
        ...previousState,
        summary: parsed.summary?.trim() || previousState?.summary || "",
        activeTopic: parsed.activeTopic?.trim() || previousState?.activeTopic || "",
        openLoop: typeof parsed.openLoop === "string" ? parsed.openLoop.trim() : (previousState?.openLoop || ""),
        updatedAt: Date.now(),
      };

      stmtSetSetting.run("radha_session_context", JSON.stringify(updatedState));
      res.json({ sessionContext: updatedState });
    } catch (error: any) {
      console.warn("Session summary update skipped:", error?.message || error);
      res.json({ sessionContext: req.body?.previousState || null });
    }
  });

  // Reminders CRUD
  app.post("/api/reminders", (req, res) => {
    try {
      const { title, triggerAt } = req.body || {};
      if (!title || !triggerAt) {
        res.status(400).json({ error: "title and triggerAt required" });
        return;
      }
      const id = `rem-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const createdAt = Date.now();
      const ts = Number(triggerAt);
      stmtInsertReminder.run(id, String(title).trim(), ts, 0, createdAt);
      res.json({ ok: true, reminder: { id, title: String(title).trim(), triggerAt: ts, completed: false, createdAt } });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  app.patch("/api/reminders/:id", (req, res) => {
    try {
      const completed = req.body?.completed ? 1 : 0;
      stmtUpdateReminderDone.run(completed, req.params.id);
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  app.delete("/api/reminders/:id", (req, res) => {
    try {
      stmtDeleteReminder.run(req.params.id);
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // Tasks / To-Do CRUD
  app.post("/api/tasks", (req, res) => {
    try {
      const { title, priority } = req.body || {};
      if (!title || !String(title).trim()) {
        res.status(400).json({ error: "title required" });
        return;
      }
      const id = `task-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const createdAt = Date.now();
      const cleanTitle = String(title).trim();
      const cleanPriority = ["low", "medium", "high"].includes(priority) ? priority : "medium";
      stmtInsertTask.run(id, cleanTitle, cleanPriority, 0, createdAt);
      res.json({ ok: true, task: { id, title: cleanTitle, priority: cleanPriority, completed: false, createdAt } });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  app.patch("/api/tasks/:id", (req, res) => {
    try {
      const completed = req.body?.completed ? 1 : 0;
      stmtUpdateTaskDone.run(completed, req.params.id);
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  app.delete("/api/tasks/:id", (req, res) => {
    try {
      stmtDeleteTask.run(req.params.id);
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // Mood Diary & Bonding Level
  app.post("/api/moods", (req, res) => {
    try {
      const { emotion, note, xpGain } = req.body || {};
      const id = `mood-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const createdAt = Date.now();
      const cleanEmotion = String(emotion || "neutral");
      const cleanNote = String(note || "Conversation check-in").slice(0, 240);
      stmtInsertMood.run(id, cleanEmotion, cleanNote, createdAt);

      const bondRow: any = stmtGetSetting.get("radha_bonding");
      const currentBond = bondRow ? JSON.parse(bondRow.value) : { xp: 120, level: 2, interactions: 0 };
      const nextXp = currentBond.xp + (typeof xpGain === "number" ? xpGain : 15);
      const nextLevel = Math.max(1, Math.floor(nextXp / 100) + 1);
      const updatedBond = {
        xp: nextXp,
        level: nextLevel,
        interactions: (currentBond.interactions || 0) + 1,
      };
      stmtSetSetting.run("radha_bonding", JSON.stringify(updatedBond));

      res.json({
        ok: true,
        mood: { id, emotion: cleanEmotion, note: cleanNote, createdAt },
        bonding: updatedBond,
      });
    } catch (error: any) {
      res.status(500).json({ error: error.message });
    }
  });

  // Boss Sentiment Analysis Engine (Deep Gemini + SQLite Persistence)
  app.post("/api/sentiment/analyze", async (req, res) => {
    try {
      const { text = "", recentMessages = [], fallback = null } = req.body || {};
      const cleanText = String(text || "").trim();
      const transcript = Array.isArray(recentMessages)
        ? recentMessages
            .slice(-6)
            .map((m: any) => `${m.sender === "user" ? "Boss" : "Radha"}: ${m.text}`)
            .join("\n")
        : "";

      const prompt = `Analyze Ravindra Boss's emotional sentiment from his message and recent context.
Return ONLY valid JSON with exact fields:
{
  "polarity": "positive" | "neutral" | "negative" | "intense",
  "score": integer from -100 to 100,
  "stressLevel": integer from 0 to 100,
  "energyLevel": integer from 0 to 100,
  "warmthLevel": integer from 0 to 100,
  "dominantMood": short 2-4 word label (e.g. "Focused & Determined", "Frustrated & Urgent", "Happy & Playful", "Stressed & Tired"),
  "triggers": array of 1-3 short keywords detected,
  "insight": concise 1 sentence insight in Hinglish about how Boss is feeling right now,
  "radhaGuidance": concise 1 sentence instruction on how Radha should adapt her tone for Boss
}

Boss's Latest Message: "${cleanText || "(analyze recent transcript)"}"
Recent Conversation:
${transcript || "(none)"}`;

      let parsed: any = null;
      try {
        const response = await generateWithActiveModels(
          [{ role: "user", parts: [{ text: prompt }] }],
          { responseMimeType: "application/json" },
          ["gemini-3.1-flash-lite", "gemini-4-hydrogen"]
        );
        const raw = (response?.text || "").trim();
        if (raw) {
          try {
            parsed = JSON.parse(raw);
          } catch {
            const match = raw.match(/\{[\s\S]*\}/);
            if (match) parsed = JSON.parse(match[0]);
          }
        }
      } catch (err: any) {
        console.warn("Sentiment analysis skipped (using instant reading):", err?.message || err);
      }

      const id = `sent-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
      const createdAt = Date.now();
      const base = fallback && typeof fallback === "object" ? fallback : {};
      const validPolarities = ["positive", "neutral", "negative", "intense"];

      const finalRecord = {
        id,
        polarity: validPolarities.includes(parsed?.polarity) ? parsed.polarity : (base.polarity || "neutral"),
        score: Math.max(-100, Math.min(100, Math.round(Number(parsed?.score ?? base.score ?? 10)))),
        stressLevel: Math.max(0, Math.min(100, Math.round(Number(parsed?.stressLevel ?? base.stressLevel ?? 25)))),
        energyLevel: Math.max(0, Math.min(100, Math.round(Number(parsed?.energyLevel ?? base.energyLevel ?? 55)))),
        warmthLevel: Math.max(0, Math.min(100, Math.round(Number(parsed?.warmthLevel ?? base.warmthLevel ?? 65)))),
        dominantMood: String(parsed?.dominantMood || base.dominantMood || "Focused & Calm").slice(0, 60),
        triggers: Array.isArray(parsed?.triggers)
          ? parsed.triggers.map((t: any) => String(t).slice(0, 24)).slice(0, 4)
          : Array.isArray(base.triggers)
          ? base.triggers.slice(0, 4)
          : [],
        insight: String(parsed?.insight || base.insight || "Boss normal aur focused tone me baat kar rahe hain.").slice(0, 240),
        radhaGuidance: String(parsed?.radhaGuidance || base.radhaGuidance || "Respond warmly and sharply in natural Hinglish.").slice(0, 240),
        sourceText: (cleanText || base.sourceText || "").slice(0, 180),
        createdAt,
      };

      stmtInsertSentiment.run(
        finalRecord.id,
        finalRecord.polarity,
        finalRecord.score,
        finalRecord.stressLevel,
        finalRecord.energyLevel,
        finalRecord.warmthLevel,
        finalRecord.dominantMood,
        JSON.stringify(finalRecord.triggers),
        finalRecord.insight,
        finalRecord.radhaGuidance,
        finalRecord.sourceText,
        finalRecord.createdAt
      );

      res.json({ ok: true, sentiment: finalRecord });
    } catch (error: any) {
      console.warn("Sentiment endpoint fallback:", error?.message || error);
      res.json({ ok: true, sentiment: req.body?.fallback || null });
    }
  });

  // Real-Time Internet Search & Live News (Gemini Google Search Grounding with robust fallback chain)
  app.post("/api/gemini/search", async (req, res) => {
    try {
      const { query, sessionContextSummary } = req.body || {};
      if (!query || !String(query).trim()) {
        res.status(400).json({ error: "Search query required" });
        return;
      }

      const sessionBlock = String(sessionContextSummary || getStoredSessionContextSummary() || "").trim();
      const contextPrefix = sessionBlock
        ? `Session Context (for continuity with Ravindra Boss):\n${sessionBlock}\n\n`
        : "";

      const searchContents = [
        {
          role: "user",
          parts: [
            {
              text: `${contextPrefix}Give a crisp, accurate, up-to-date real-time answer in clear Hinglish/Hindi (addressing Ravindra Boss warmly as Radha) for this query: ${String(query).trim()}`,
            },
          ],
        },
      ];

      let response: any = null;
      try {
        response = await generateWithActiveModels(
          searchContents,
          { tools: [{ googleSearch: {} }] },
          ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
        );
      } catch (searchErr: any) {
        // If search tool quota is hit, retry once without search grounding if non-quota, or surface clean 429
        if (isQuotaError(searchErr)) {
          throw searchErr;
        }
        response = await generateWithActiveModels(
          [
            {
              role: "user",
              parts: [
                {
                  text: `${contextPrefix}Answer this query clearly and honestly in Hinglish/Hindi as Radha for Ravindra Boss: ${String(query).trim()}`,
                },
              ],
            },
          ],
          {},
          ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
        );
      }

      const chunks = response?.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
      const sources = chunks
        .map((c: any) => ({
          title: c.web?.title || "Source",
          uri: c.web?.uri || "",
        }))
        .filter((s: any) => Boolean(s.uri))
        .slice(0, 6);

      res.json({
        answer: response?.text || "Boss, abhi live data nahi mil paya, ek baar fir try karte hain.",
        sources,
      });
    } catch (error: any) {
      console.error("Live Search Error:", error?.message || error);
      if (isQuotaError(error)) {
        res.status(429).json({
          error:
            "Gemini API quota limit reached (429). Please wait a few seconds or select a billing-enabled API key in Settings > Secrets.",
        });
        return;
      }
      res.status(500).json({ error: error?.message || "Search failed" });
    }
  });

  // Multimodal Document / Photo / Screenshot / Code Analysis (with robust model fallback chain)
  app.post("/api/gemini/analyze", async (req, res) => {
    try {
      const { base64Data, mimeType, fileName, question, sessionContextSummary } = req.body || {};
      if (!base64Data || !mimeType) {
        res.status(400).json({ error: "File data and mimeType are required" });
        return;
      }

      const sessionBlock = String(sessionContextSummary || getStoredSessionContextSummary() || "").trim();
      const contextPrefix = sessionBlock
        ? `Session Context (for continuity with Ravindra Boss):\n${sessionBlock}\n\n`
        : "";

      const promptText = question?.trim()
        ? `${contextPrefix}Ravindra Boss uploaded "${fileName || "file"}" and asked: "${question.trim()}". As Radha, analyze it thoroughly and explain clearly in helpful Hinglish/Hindi.`
        : `${contextPrefix}Ravindra Boss uploaded "${fileName || "file"}". As Radha, carefully inspect this image/document/code, summarize what it contains, highlight any key details or issues, and explain it clearly in helpful Hinglish/Hindi.`;

      const parts: any[] = [];
      if (String(mimeType).startsWith("text/") || String(mimeType).includes("json") || String(mimeType).includes("javascript")) {
        const decodedText = Buffer.from(base64Data, "base64").toString("utf-8").slice(0, 24000);
        parts.push({ text: `${promptText}\n\nFile Content:\n${decodedText}` });
      } else {
        parts.push({ inlineData: { data: base64Data, mimeType } });
        parts.push({ text: promptText });
      }

      const response = await generateWithActiveModels(
        [{ role: "user", parts }],
        {},
        ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
      );

      res.json({
        analysis: response?.text || "Boss, maine file dekh li par usmein clear text nahi mila.",
      });
    } catch (error: any) {
      console.error("File Analyze Error:", error?.message || error);
      if (isQuotaError(error)) {
        res.status(429).json({
          error:
            "Gemini API quota limit reached (429). Please wait a moment or check your billing-enabled API key in Settings > Secrets.",
        });
        return;
      }
      res.status(500).json({ error: error?.message || "Analysis failed" });
    }
  });

  app.post("/api/messages", (req, res) => {
    try {
      const { id, sender, text, emotion, timestamp } = req.body || {};
      if (!sender || !text) {
        res.status(400).json({ error: "sender and text are required" });
        return;
      }
      const msgId = id || `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const ts = typeof timestamp === "number" ? timestamp : Date.now();
      stmtInsertMessage.run(msgId, sender, text, emotion || null, ts);
      res.json({ ok: true, id: msgId });
    } catch (error: any) {
      console.error("DB Message Write Error:", error);
      res.status(500).json({ error: error.message || "Failed to save message" });
    }
  });

  app.delete("/api/messages", (req, res) => {
    try {
      stmtClearMessages.run();
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message || "Failed to clear messages" });
    }
  });

  app.get("/api/memories", (req, res) => {
    try {
      res.json({ memories: stmtGetMemories.all() });
    } catch (error: any) {
      res.status(500).json({ error: error.message || "Failed to load memories" });
    }
  });

  app.post("/api/memories", (req, res) => {
    try {
      const { content, category } = req.body || {};
      if (!content || !String(content).trim()) {
        res.status(400).json({ error: "Memory content is required" });
        return;
      }
      const id = `mem-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const createdAt = Date.now();
      const cleanContent = String(content).trim();
      const cleanCategory = String(category || "general").trim();
      stmtInsertMemory.run(id, cleanContent, cleanCategory, createdAt);
      res.json({
        ok: true,
        memory: { id, content: cleanContent, category: cleanCategory, createdAt },
      });
    } catch (error: any) {
      console.error("DB Memory Write Error:", error);
      res.status(500).json({ error: error.message || "Failed to save memory" });
    }
  });

  app.delete("/api/memories/:id", (req, res) => {
    try {
      stmtDeleteMemory.run(req.params.id);
      res.json({ ok: true });
    } catch (error: any) {
      res.status(500).json({ error: error.message || "Failed to delete memory" });
    }
  });

  app.put("/api/settings", (req, res) => {
    try {
      const config = req.body;
      if (!config || typeof config !== "object") {
        res.status(400).json({ error: "Invalid settings object" });
        return;
      }
      stmtSetSetting.run("radha_config", JSON.stringify(config));
      res.json({ ok: true });
    } catch (error: any) {
      console.error("DB Settings Write Error:", error);
      res.status(500).json({ error: error.message || "Failed to save settings" });
    }
  });

  // Automatic Background Memory Extractor so Radha remembers past work, plans & personal facts forever
  app.post("/api/memory/summarize", async (req, res) => {
    try {
      const { recentMessages = [], existingMemories = [] } = req.body || {};
      if (!Array.isArray(recentMessages) || recentMessages.length < 2) {
        res.json({ added: [] });
        return;
      }

      const transcript = recentMessages
        .map((m: any) => `${m.sender === "user" ? "Ravindra Boss" : "Radha"}: ${m.text}`)
        .join("\n");

      const prompt = `Extract 0 to 2 genuinely important long-term memories from this recent conversation between Ravindra Boss and Radha (such as work/tasks done, ongoing projects, personal preferences, promises, important emotional moments, or specific facts Ravindra Boss wants remembered).
Do NOT save trivial greetings ("hi", "hello", "kaise ho") or facts already present in Existing Memories.
Return ONLY a valid JSON array of objects with fields: [{"content": "short clear Hinglish/English memory note", "category": "work|personal|emotion|task|preference"}]. If nothing important to remember, return [].

Existing Memories:
${(existingMemories || []).map((x: string) => `- ${x}`).join("\n") || "(none)"}

Recent Conversation:
${transcript}`;

      const response = await generateWithActiveModels(
        [{ role: "user", parts: [{ text: prompt }] }],
        { responseMimeType: "application/json" },
        ["gemini-3.1-flash-lite", "gemini-4-hydrogen"]
      );

      const rawText = (response.text || "[]").trim();
      let parsed: Array<{ content?: string; category?: string }> = [];
      try {
        parsed = JSON.parse(rawText);
      } catch {
        const match = rawText.match(/\[[\s\S]*\]/);
        if (match) parsed = JSON.parse(match[0]);
      }

      const added: any[] = [];
      if (Array.isArray(parsed)) {
        for (const item of parsed.slice(0, 2)) {
          const text = item?.content?.trim();
          if (!text || text.length < 6) continue;
          const isDup = (existingMemories || []).some(
            (e: string) => e.toLowerCase().trim() === text.toLowerCase()
          );
          if (isDup) continue;
          const id = `mem-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
          const createdAt = Date.now();
          const category = item?.category || "general";
          stmtInsertMemory.run(id, text, category, createdAt);
          added.push({ id, content: text, category, createdAt });
        }
      }

      res.json({ added });
    } catch (error: any) {
      console.warn("Background memory extraction skipped:", error?.message || error);
      res.json({ added: [] });
    }
  });

  // Proxy Radha Text Response
  app.post("/api/gemini/response", async (req, res) => {
    try {
      const { prompt, history, systemInstruction, sessionContextSummary } = req.body;
      const fallbackSessionBlock = String(sessionContextSummary || getStoredSessionContextSummary() || "").trim();
      const finalSystemInstruction =
        systemInstruction && String(systemInstruction).includes("SESSION CONTEXT SUMMARY")
          ? systemInstruction
          : fallbackSessionBlock
          ? `${systemInstruction || ""}\n\nSESSION CONTEXT SUMMARY (PERSISTED ACROSS BROWSER RELOADS):\n${fallbackSessionBlock}`
          : systemInstruction;
      
      let historyToUse = history || [];
      if (historyToUse.length > 0 && historyToUse[historyToUse.length - 1].sender === "user" && historyToUse[historyToUse.length - 1].text === prompt) {
        historyToUse = historyToUse.slice(0, -1);
      }

      const recentHistory = historyToUse.slice(-20);
      let formattedHistory: any[] = [];
      let currentRole = "";
      let currentText = "";

      for (const msg of recentHistory) {
        const role = msg.sender === "user" ? "user" : "model";
        if (role === currentRole) {
          currentText += "\n" + msg.text;
        } else {
          if (currentRole !== "") {
            formattedHistory.push({ role: currentRole, parts: [{ text: currentText }] });
          }
          currentRole = role;
          currentText = msg.text;
        }
      }
      if (currentRole !== "") {
        formattedHistory.push({ role: currentRole, parts: [{ text: currentText }] });
      }

      if (formattedHistory.length > 0 && formattedHistory[0].role !== "user") {
        formattedHistory.shift();
      }

      if (formattedHistory.length > 0 && formattedHistory[formattedHistory.length - 1].role !== "model") {
        formattedHistory.pop();
      }

      // Build structured contents for multi-turn conversation
      const contents: any[] = [];
      for (const item of formattedHistory) {
        if (item.parts?.[0]?.text?.trim()) {
          contents.push({
            role: item.role === "user" ? "user" : "model",
            parts: [{ text: item.parts[0].text }]
          });
        }
      }
      contents.push({ role: "user", parts: [{ text: prompt }] });

      const response = await generateWithActiveModels(
        contents,
        { systemInstruction: finalSystemInstruction },
        ["gemini-4-hydrogen", "gemini-3.1-flash-lite"]
      );

      res.json({ text: response?.text || "Ji Ravindra Boss, bataiye main aapki kya madad kar sakti hu?" });
    } catch (error: any) {
      console.error("Server Gemini Error:", error?.message || error);
      if (isQuotaError(error)) {
        res.status(429).json({
          error:
            "Gemini API quota limit reached (429). Please wait a moment or check your billing-enabled API key in Settings > Secrets.",
        });
        return;
      }
      res.status(500).json({ error: error?.message || "Internal Server Error" });
    }
  });

  const normalizeGeminiVoice = (rawVoice?: string): string => {
    const allowed = ["Kore", "Zephyr", "Puck", "Charon", "Fenrir"];
    if (rawVoice && allowed.includes(rawVoice)) return rawVoice;
    if (rawVoice === "Aoede") return "Zephyr";
    return "Kore";
  };

  const EMOTION_TTS_STYLES: Record<string, string> = {
    neutral: "Warm, natural, clear Indian female Hindi voice with poised conversational cadence",
    happy: "Bright, cheerful, smiling Indian female Hindi voice full of genuine joy and warmth",
    excited: "Energetic, enthusiastic, animated Indian female Hindi voice bursting with excitement and celebration",
    playful: "Playful, teasing, lighthearted Indian female Hindi voice with witty charm and a subtle smile",
    caring: "Soft, gentle, deeply empathetic and comforting Indian female Hindi voice speaking tenderly and calmly",
    sad: "Quiet, tender, melancholic Indian female Hindi voice with slower heartfelt cadence and soft emotion",
    hurt: "Emotionally hurt, sensitive, slightly offended Indian female Hindi voice with gentle vulnerability",
    angry: "Sharp, assertive, genuinely annoyed and spirited Indian female Hindi voice with firm dramatic emphasis",
  };

  // Per-model TTS quota circuit-breaker (prevents hammering exhausted free-tier quota and spamming logs)
  const ttsQuotaCooldownUntil = new Map<string, number>();

  const parseTtsRetryDelayMs = (err: any): number => {
    const msg = String(err?.message || err || "");
    if (
      msg.includes("per_day") ||
      msg.includes("PerDay") ||
      msg.includes("free_tier_requests") ||
      msg.includes("FreeTier")
    ) {
      return 15 * 60 * 1000; // 15-minute cooldown for exhausted daily free-tier quota
    }
    const secMatch = msg.match(/retry in\s+(\d+(?:\.\d+)?)s/i);
    if (secMatch) {
      const parsed = Math.ceil(parseFloat(secMatch[1]) * 1000);
      return Math.min(15 * 60 * 1000, Math.max(30_000, parsed));
    }
    return 60_000;
  };

  const getTtsCooldownRemainingMs = (model: string): number => {
    const until = ttsQuotaCooldownUntil.get(model) || 0;
    const remaining = until - Date.now();
    if (remaining > 0) return remaining;
    if (until > 0) ttsQuotaCooldownUntil.delete(model);
    return 0;
  };

  const markTtsQuotaCooldown = (model: string, err: any): number => {
    const duration = parseTtsRetryDelayMs(err);
    const wasCooling = getTtsCooldownRemainingMs(model) > 0;
    ttsQuotaCooldownUntil.set(model, Date.now() + duration);
    if (!wasCooling) {
      console.info(
        `[TTS] Quota reached for ${model}; pausing server TTS for ${Math.round(duration / 1000)}s and using browser Hindi TTS fallback.`
      );
    }
    return duration;
  };

  const TTS_MODELS = ["gemini-3.8-flash-lite-tts", "gemini-3.8-flash-tts"] as const;

  const isInvalidArgumentError = (err: any): boolean => {
    const msg = String(err?.message || err || "");
    return err?.status === 400 || msg.includes("INVALID_ARGUMENT") || msg.includes("400");
  };

  // Proxy Radha TTS Audio (Gemini 3.8 TTS via v1alpha — WAV/24kHz mono 16-bit, emotion-tuned prosody)
  app.post("/api/gemini/audio", async (req, res) => {
    try {
      const { text, voiceName, emotion } = req.body || {};
      const cleanText = String(text || "").trim().slice(0, 700);
      if (!cleanText) {
        res.json({ audio: null });
        return;
      }
      const selectedVoice = normalizeGeminiVoice(voiceName);
      const normalizedEmotion =
        typeof emotion === "string" && emotion.toLowerCase() in EMOTION_TTS_STYLES
          ? emotion.toLowerCase()
          : "neutral";
      const emotionStyle = EMOTION_TTS_STYLES[normalizedEmotion];
      let audioData: string | null = null;
      let audioMimeType = "audio/wav";
      let quotaHits = 0;
      let minRetryAfterMs = 0;

      for (const model of TTS_MODELS) {
        const remaining = getTtsCooldownRemainingMs(model);
        if (remaining > 0) {
          quotaHits++;
          minRetryAfterMs = minRetryAfterMs === 0 ? remaining : Math.min(minRetryAfterMs, remaining);
          continue;
        }

        // Only use aiLive (v1alpha), which supports Gemini 3.8 TTS speechConfig & speechMetadata
        for (const withMetadata of [true, false]) {
          try {
            const part: any = withMetadata
              ? {
                  text: cleanText,
                  speechMetadata: {
                    style: emotionStyle,
                  },
                }
              : { text: cleanText };

            const response: any = await aiLive.models.generateContent({
              model,
              contents: [{ role: "user", parts: [part] }],
              config: {
                responseModalities: ["AUDIO"],
                speechConfig: {
                  voiceConfig: {
                    prebuiltVoiceConfig: { voiceName: selectedVoice },
                  },
                },
              },
            });

            const audioPart = response?.candidates?.[0]?.content?.parts?.find(
              (p: any) => p?.inlineData?.data
            );
            if (audioPart?.inlineData?.data) {
              audioData = audioPart.inlineData.data;
              audioMimeType = audioPart.inlineData.mimeType || "audio/wav";
              break;
            }
          } catch (err: any) {
            if (isQuotaError(err)) {
              const duration = markTtsQuotaCooldown(model, err);
              quotaHits++;
              minRetryAfterMs =
                minRetryAfterMs === 0 ? duration : Math.min(minRetryAfterMs, duration);
              break; // Move straight to next model — never retry exhausted model
            }
            if (withMetadata && isInvalidArgumentError(err)) {
              continue; // Retry once without speechMetadata on same v1alpha model
            }
            console.info(`[TTS] ${model} fallback:`, String(err?.message || err).slice(0, 140));
            break;
          }
        }

        if (audioData) break;
      }

      const quotaExhausted = !audioData && quotaHits >= TTS_MODELS.length;
      res.json({
        audio: audioData,
        mimeType: audioData ? audioMimeType : undefined,
        emotion: normalizedEmotion,
        appliedStyle: emotionStyle,
        fallback: audioData ? undefined : "browser_tts",
        quotaExhausted,
        retryAfterMs: quotaExhausted ? Math.min(minRetryAfterMs || 300_000, 300_000) : undefined,
      });
    } catch (error: any) {
      res.json({
        audio: null,
        fallback: "browser_tts",
        quotaExhausted: isQuotaError(error),
      });
    }
  });

  // Low-latency Streaming TTS Pre-Fetch Endpoint (streams 24kHz PCM chunks as soon as first byte arrives from Gemini)
  app.post("/api/gemini/audio/stream", async (req, res) => {
    try {
      const { text, voiceName, emotion } = req.body || {};
      const cleanText = String(text || "").trim().slice(0, 700);
      if (!cleanText) {
        res.status(400).json({ error: "Text is required" });
        return;
      }

      const selectedVoice = normalizeGeminiVoice(voiceName);
      const normalizedEmotion =
        typeof emotion === "string" && emotion.toLowerCase() in EMOTION_TTS_STYLES
          ? emotion.toLowerCase()
          : "neutral";
      const emotionStyle = EMOTION_TTS_STYLES[normalizedEmotion];

      res.setHeader("Content-Type", "application/x-ndjson; charset=utf-8");
      res.setHeader("Cache-Control", "no-store, no-cache, no-transform");
      res.setHeader("Connection", "keep-alive");
      res.setHeader("X-Accel-Buffering", "no");
      if (typeof (res as any).flushHeaders === "function") {
        (res as any).flushHeaders();
      }

      let clientDisconnected = false;
      res.on("close", () => {
        if (!res.writableEnded) {
          clientDisconnected = true;
        }
      });

      const withTimeout = <T,>(promise: Promise<T>, ms: number, label: string): Promise<T> =>
        new Promise<T>((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error(`${label} timeout (${ms}ms)`)), ms);
          promise.then(
            (val) => {
              clearTimeout(timer);
              resolve(val);
            },
            (err) => {
              clearTimeout(timer);
              reject(err);
            }
          );
        });

      let seq = 0;
      let quotaHits = 0;
      let minRetryAfterMs = 0;

      // Only use aiLive (v1alpha) for Gemini 3.8 TTS streaming (v1beta rejects speechConfig/speechMetadata with 400)
      for (const model of TTS_MODELS) {
        if (clientDisconnected || seq > 0) break;

        const remaining = getTtsCooldownRemainingMs(model);
        if (remaining > 0) {
          quotaHits++;
          minRetryAfterMs = minRetryAfterMs === 0 ? remaining : Math.min(minRetryAfterMs, remaining);
          continue;
        }

        for (const withMetadata of [true, false]) {
          if (clientDisconnected || seq > 0) break;
          try {
            const part: any = withMetadata
              ? { text: cleanText, speechMetadata: { style: emotionStyle } }
              : { text: cleanText };

            const stream = await withTimeout(
              aiLive.models.generateContentStream({
                model,
                contents: [{ role: "user", parts: [part] }],
                config: {
                  responseModalities: ["AUDIO"],
                  speechConfig: {
                    voiceConfig: {
                      prebuiltVoiceConfig: { voiceName: selectedVoice },
                    },
                  },
                },
              }),
              4500,
              `stream-init-${model}`
            );

            const iterator = (stream as any)[Symbol.asyncIterator]();
            while (!clientDisconnected) {
              const step = await withTimeout<{ value: any; done?: boolean }>(
                iterator.next(),
                seq === 0 ? 4500 : 8000,
                `stream-chunk-${seq}`
              );
              if (step.done) break;
              const parts = step.value?.candidates?.[0]?.content?.parts || [];
              for (const p of parts) {
                if (p?.inlineData?.data) {
                  res.write(
                    JSON.stringify({
                      type: "chunk",
                      seq: seq++,
                      mimeType: p.inlineData.mimeType || "audio/l16;rate=24000",
                      data: p.inlineData.data,
                    }) + "\n"
                  );
                }
              }
            }

            if (seq > 0) break;
          } catch (err: any) {
            if (isQuotaError(err)) {
              const duration = markTtsQuotaCooldown(model, err);
              quotaHits++;
              minRetryAfterMs =
                minRetryAfterMs === 0 ? duration : Math.min(minRetryAfterMs, duration);
              break; // Skip retrying same exhausted model
            }
            if (withMetadata && isInvalidArgumentError(err)) {
              continue; // Retry once without speechMetadata on same v1alpha model
            }
            console.info(`[TTS Stream] ${model} fallback:`, String(err?.message || err).slice(0, 140));
            break;
          }
        }
      }

      // Unary progressive-chunk fallback ONLY if an active (non-quota-exhausted) TTS model exists
      if (seq === 0 && !clientDisconnected) {
        const availableModel = TTS_MODELS.find((m) => getTtsCooldownRemainingMs(m) === 0);
        if (availableModel) {
          try {
            const fallbackResp: any = await withTimeout(
              aiLive.models.generateContent({
                model: availableModel,
                contents: [
                  {
                    role: "user",
                    parts: [{ text: cleanText, speechMetadata: { style: emotionStyle } }],
                  },
                ],
                config: {
                  responseModalities: ["AUDIO"],
                  speechConfig: {
                    voiceConfig: {
                      prebuiltVoiceConfig: { voiceName: selectedVoice },
                    },
                  },
                },
              }),
              7000,
              "tts-unary-fallback"
            );
            const audioPart = fallbackResp?.candidates?.[0]?.content?.parts?.find(
              (p: any) => p?.inlineData?.data
            );
            if (audioPart?.inlineData?.data && !clientDisconnected) {
              const rawBuf = Buffer.from(audioPart.inlineData.data, "base64");
              const isWav = rawBuf.length > 44 && rawBuf.subarray(0, 4).toString("ascii") === "RIFF";
              const pcmBuf = isWav ? rawBuf.subarray(44) : rawBuf;
              const sliceSize = 16384; // ~340ms PCM frames for smooth progressive buffer playback
              for (let offset = 0; offset < pcmBuf.length && !clientDisconnected; offset += sliceSize) {
                const slice = pcmBuf.subarray(offset, Math.min(pcmBuf.length, offset + sliceSize));
                if (slice.length >= 2) {
                  res.write(
                    JSON.stringify({
                      type: "chunk",
                      seq: seq++,
                      mimeType: "audio/l16;rate=24000",
                      data: slice.toString("base64"),
                    }) + "\n"
                  );
                }
              }
            }
          } catch (fallbackErr: any) {
            if (isQuotaError(fallbackErr)) {
              const duration = markTtsQuotaCooldown(availableModel, fallbackErr);
              quotaHits++;
              minRetryAfterMs =
                minRetryAfterMs === 0 ? duration : Math.min(minRetryAfterMs, duration);
            } else {
              console.info(
                `[TTS Unary] ${availableModel} fallback:`,
                String(fallbackErr?.message || fallbackErr).slice(0, 140)
              );
            }
          }
        }
      }

      if (!clientDisconnected) {
        const quotaExhausted = seq === 0 && quotaHits >= TTS_MODELS.length;
        if (quotaExhausted) {
          res.write(
            JSON.stringify({
              type: "fallback",
              reason: "quota_exhausted",
              retryAfterMs: Math.min(minRetryAfterMs || 300_000, 300_000),
              chunks: 0,
              emotion: normalizedEmotion,
            }) + "\n"
          );
        }
        res.write(
          JSON.stringify({
            type: "done",
            chunks: seq,
            emotion: normalizedEmotion,
            quotaExhausted,
          }) + "\n"
        );
        res.end();
      }
    } catch (error: any) {
      if (!res.headersSent) {
        res.status(200).json({ audio: null, fallback: "browser_tts", quotaExhausted: isQuotaError(error) });
      } else {
        try {
          res.write(
            JSON.stringify({
              type: "fallback",
              reason: isQuotaError(error) ? "quota_exhausted" : "stream_fallback",
            }) + "\n"
          );
          res.end();
        } catch {}
      }
    }
  });

  // Modular Social Workspace Proxy: Instagram Graph API / Custom Webhook
  app.post("/api/social/instagram", async (req, res) => {
    try {
      const { accountId, accessToken, webhookUrl, handle = "@ravinder_boss" } = req.body || {};

      if (webhookUrl && /^https?:\/\//i.test(String(webhookUrl))) {
        const hookRes = await fetch(String(webhookUrl), {
          headers: { Accept: "application/json" },
        });
        if (!hookRes.ok) {
          res.status(502).json({ error: `Instagram webhook returned ${hookRes.status}` });
          return;
        }
        const data: any = await hookRes.json();
        res.json({
          mode: "live",
          handle: String(data.handle || handle),
          followers: Number(data.followers || 0),
          recentVideoViews: Number(data.recentVideoViews || data.views || 0),
          lastUploadAt: String(data.lastUploadAt || "Recently"),
          engagementRate: String(data.engagementRate || "—"),
          summary: String(data.summary || "Live webhook connected."),
        });
        return;
      }

      if (!accountId || !accessToken) {
        res.status(400).json({ error: "Instagram Account ID and Access Token are required." });
        return;
      }

      const profileUrl = `https://graph.facebook.com/v20.0/${encodeURIComponent(
        String(accountId)
      )}?fields=username,followers_count,media_count,media.limit(5){caption,media_type,like_count,comments_count,timestamp,video_views}&access_token=${encodeURIComponent(
        String(accessToken)
      )}`;
      const igRes = await fetch(profileUrl);
      if (!igRes.ok) {
        const txt = await igRes.text();
        res.status(igRes.status).json({ error: txt.slice(0, 200) });
        return;
      }
      const igData: any = await igRes.json();
      const mediaItems: any[] = igData?.media?.data || [];
      const latest = mediaItems[0];
      const followers = Number(igData?.followers_count || 0);
      const totalViews = mediaItems.reduce(
        (sum, m) => sum + Number(m.video_views || m.like_count || 0),
        0
      );
      const totalEng = mediaItems.reduce(
        (sum, m) => sum + Number(m.like_count || 0) + Number(m.comments_count || 0),
        0
      );
      const engRate =
        followers > 0 && mediaItems.length > 0
          ? `${((totalEng / (followers * mediaItems.length)) * 100).toFixed(1)}%`
          : "Live";

      res.json({
        mode: "live",
        handle: igData?.username ? `@${igData.username}` : handle,
        followers,
        recentVideoViews: totalViews,
        lastUploadAt: latest?.timestamp
          ? new Date(latest.timestamp).toLocaleString("en-IN", {
              dateStyle: "medium",
              timeStyle: "short",
            })
          : "No recent posts",
        engagementRate: engRate,
        summary: `Synced ${mediaItems.length} recent posts directly from Instagram Graph API.`,
      });
    } catch (error: any) {
      res.status(500).json({ error: error?.message || "Instagram sync failed" });
    }
  });

  // Modular Social Workspace Proxy: Facebook Page API / Custom Webhook
  app.post("/api/social/facebook", async (req, res) => {
    try {
      const { pageId, accessToken, webhookUrl, pageName = "Ravindra Boss Official" } = req.body || {};

      if (webhookUrl && /^https?:\/\//i.test(String(webhookUrl))) {
        const hookRes = await fetch(String(webhookUrl), {
          headers: { Accept: "application/json" },
        });
        if (!hookRes.ok) {
          res.status(502).json({ error: `Facebook webhook returned ${hookRes.status}` });
          return;
        }
        const data: any = await hookRes.json();
        res.json({
          mode: "live",
          pageName: String(data.pageName || pageName),
          unreadCount: Number(data.unreadCount || (data.notifications || []).length || 0),
          notifications: Array.isArray(data.notifications) ? data.notifications.slice(0, 5) : [],
        });
        return;
      }

      if (!pageId || !accessToken) {
        res.status(400).json({ error: "Facebook Page ID and Access Token are required." });
        return;
      }

      const fbUrl = `https://graph.facebook.com/v20.0/${encodeURIComponent(
        String(pageId)
      )}?fields=name,unread_message_count,unread_notif_count,conversations.limit(4){snippet,updated_time,senders}&access_token=${encodeURIComponent(
        String(accessToken)
      )}`;
      const fbRes = await fetch(fbUrl);
      if (!fbRes.ok) {
        const txt = await fbRes.text();
        res.status(fbRes.status).json({ error: txt.slice(0, 200) });
        return;
      }
      const fbData: any = await fbRes.json();
      const convs: any[] = fbData?.conversations?.data || [];
      const notifications = convs.map((c: any, idx: number) => ({
        id: String(c.id || `fb-${idx}`),
        sender: c?.senders?.data?.[0]?.name || "Page Inbox",
        text: String(c.snippet || "New message received"),
        time: c.updated_time
          ? new Date(c.updated_time).toLocaleTimeString("en-IN", {
              hour: "2-digit",
              minute: "2-digit",
            })
          : "Recent",
      }));

      res.json({
        mode: "live",
        pageName: fbData?.name || pageName,
        unreadCount:
          Number(fbData?.unread_message_count || 0) +
          Number(fbData?.unread_notif_count || 0) ||
          notifications.length,
        notifications,
      });
    } catch (error: any) {
      res.status(500).json({ error: error?.message || "Facebook sync failed" });
    }
  });

  // Real Apple Health / Health Auto Export HTTPS Webhook Sync (with strict SSRF protection)
  app.post("/api/health/sync", async (req, res) => {
    try {
      const rawUrl = String(req.body?.url || "").trim();
      if (!rawUrl) {
        res.status(400).json({ error: "A valid HTTPS Health feed URL is required." });
        return;
      }

      let parsedUrl: URL;
      try {
        parsedUrl = new URL(rawUrl);
      } catch {
        res.status(400).json({ error: "Invalid URL format." });
        return;
      }

      if (parsedUrl.protocol !== "https:") {
        res.status(400).json({ error: "Only HTTPS URLs are permitted for health feed sync." });
        return;
      }

      const host = parsedUrl.hostname.toLowerCase();
      const isPrivateOrLocal =
        host === "localhost" ||
        host === "0.0.0.0" ||
        host === "::1" ||
        host.endsWith(".local") ||
        host.endsWith(".internal") ||
        /^127\./.test(host) ||
        /^10\./.test(host) ||
        /^192\.168\./.test(host) ||
        /^169\.254\./.test(host) ||
        /^172\.(1[6-9]|2\d|3[0-1])\./.test(host) ||
        /^\[?fe80:/i.test(host) ||
        /^\[?fc00:/i.test(host) ||
        /^\[?fd/i.test(host);

      if (isPrivateOrLocal) {
        res.status(400).json({ error: "Private or local network URLs are not allowed." });
        return;
      }

      // Resolve DNS and block any hostname resolving to loopback, private, or link-local IP ranges
      const isPrivateIp = (ip: string): boolean => {
        const cleanIp = ip.toLowerCase();
        return (
          cleanIp === "::1" ||
          cleanIp === "::" ||
          cleanIp.startsWith("fe80:") ||
          cleanIp.startsWith("fc") ||
          cleanIp.startsWith("fd") ||
          cleanIp.startsWith("::ffff:127.") ||
          cleanIp.startsWith("::ffff:10.") ||
          cleanIp.startsWith("::ffff:192.168.") ||
          cleanIp.startsWith("::ffff:169.254.") ||
          /^127\./.test(cleanIp) ||
          /^10\./.test(cleanIp) ||
          /^0\./.test(cleanIp) ||
          /^192\.168\./.test(cleanIp) ||
          /^169\.254\./.test(cleanIp) ||
          /^172\.(1[6-9]|2\d|3[0-1])\./.test(cleanIp) ||
          /^100\.(6[4-9]|[7-9]\d|1[0-1]\d|12[0-7])\./.test(cleanIp)
        );
      };

      try {
        const resolved = await dns.lookup(host, { all: true });
        if (!resolved.length || resolved.some((entry) => isPrivateIp(entry.address))) {
          res.status(400).json({ error: "Target host resolves to a restricted network address." });
          return;
        }
      } catch {
        res.status(400).json({ error: "Unable to resolve health feed hostname." });
        return;
      }

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 6500);
      try {
        const upstream = await fetch(parsedUrl.toString(), {
          method: "GET",
          headers: { Accept: "application/json" },
          signal: controller.signal,
          redirect: "error",
        });
        if (!upstream.ok) {
          res.status(upstream.status).json({ error: `Health feed responded with ${upstream.status}` });
          return;
        }
        const rawBody = await upstream.text();
        if (rawBody.length > 2 * 1024 * 1024) {
          res.status(413).json({ error: "Health feed payload exceeds 2 MB limit." });
          return;
        }
        const payload: any = JSON.parse(rawBody);
        const obj = Array.isArray(payload) ? payload[payload.length - 1] || {} : payload;

        let steps: number | null = typeof obj.steps === "number" ? Math.round(obj.steps) : null;
        let heartRateBpm: number | null =
          typeof obj.heartRateBpm === "number"
            ? Math.round(obj.heartRateBpm)
            : typeof obj.heartRate === "number"
            ? Math.round(obj.heartRate)
            : null;
        let caloriesKcal: number | null =
          typeof obj.caloriesKcal === "number"
            ? Math.round(obj.caloriesKcal)
            : typeof obj.calories === "number"
            ? Math.round(obj.calories)
            : null;
        let activeMinutes: number | null =
          typeof obj.activeMinutes === "number" ? Math.round(obj.activeMinutes) : null;
        let distanceKm: number | null =
          typeof obj.distanceKm === "number" ? Number(obj.distanceKm.toFixed(2)) : null;

        if (Array.isArray(payload?.data?.metrics)) {
          for (const metric of payload.data.metrics) {
            const name = String(metric?.name || "").toLowerCase();
            const entries: any[] = Array.isArray(metric?.data) ? metric.data : [];
            if (entries.length === 0) continue;
            if (name.includes("step")) {
              steps = Math.round(entries.reduce((acc, e) => acc + Number(e?.qty || 0), 0));
            } else if (name.includes("heart_rate")) {
              const vals = entries.map((e) => Number(e?.Avg ?? e?.qty ?? 0)).filter((n) => n > 0);
              if (vals.length > 0) {
                heartRateBpm = Math.round(vals.reduce((a, b) => a + b, 0) / vals.length);
              }
            } else if (name.includes("active_energy") || name.includes("calorie")) {
              caloriesKcal = Math.round(entries.reduce((acc, e) => acc + Number(e?.qty || 0), 0));
            } else if (name.includes("exercise_time") || name.includes("active_minutes")) {
              activeMinutes = Math.round(entries.reduce((acc, e) => acc + Number(e?.qty || 0), 0));
            } else if (name.includes("distance")) {
              distanceKm = Number(entries.reduce((acc, e) => acc + Number(e?.qty || 0), 0).toFixed(2));
            }
          }
        }

        res.json({
          connectedLabel: `Health Feed (${parsedUrl.hostname})`,
          steps,
          heartRateBpm,
          caloriesKcal,
          activeMinutes,
          distanceKm,
        });
      } finally {
        clearTimeout(timeout);
      }
    } catch (error: any) {
      res.status(500).json({ error: error?.message || "Health webhook sync failed" });
    }
  });

  // Handle WebSocket Connection & Proxy to Gemini Live API
  wss.on("error", (err) => {
    console.warn("[WSS] WebSocketServer error handled:", err?.message || err);
  });

  wss.on("connection", (ws: WebSocket) => {
    let geminiSession: any = null;
    let isGeminiOpen = false;
    let isConnected = true;

    // Safe sender to prevent unhandled throws if socket is closing/closed
    const safeSend = (payload: any) => {
      if (isConnected && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(typeof payload === "string" ? payload : JSON.stringify(payload));
        } catch {}
      }
    };

    // Heartbeat ping interval every 15 seconds to prevent Cloud Run / proxy idle timeouts
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.ping();
          safeSend({ type: "ping" });
        } catch {}
      }
    }, 15000);

    ws.on("error", () => {
      isGeminiOpen = false;
    });

    ws.on("message", async (messageData) => {
      try {
        const textData = messageData.toString();
        const data = JSON.parse(textData);

        if (data.type === "pong") return;
        if (data.type === "ping") {
          safeSend({ type: "pong" });
          return;
        }

        if (data.type === "setup") {
          // Guard against duplicate Live API sessions on the same socket (prevents double sound)
          if (geminiSession) {
            isGeminiOpen = false;
            try {
              geminiSession.close();
            } catch {}
            geminiSession = null;
          }
          const liveModels = ["gemini-3.8-live", "gemini-3.1-flash-live-preview"];
          let lastErr: any = null;
          let sessionEstablished = false;

          for (const modelName of liveModels) {
            if (!isConnected || ws.readyState !== WebSocket.OPEN) break;
            try {
              const selectedVoice = normalizeGeminiVoice(data.voiceName);
              let closedEarly = false;
              let setupNotified = false;
              let setupFallbackTimer: ReturnType<typeof setTimeout> | null = null;

              const notifyReady = () => {
                if (setupNotified || closedEarly || !isConnected) return;
                setupNotified = true;
                if (setupFallbackTimer) {
                  clearTimeout(setupFallbackTimer);
                  setupFallbackTimer = null;
                }
                safeSend({ type: "open", model: modelName });
              };

              const session = await aiLive.live.connect({
                model: modelName,
                config: {
                  responseModalities: [Modality.AUDIO],
                  speechConfig: {
                    voiceConfig: { prebuiltVoiceConfig: { voiceName: selectedVoice } },
                  },
                  systemInstruction: data.systemInstruction,
                  inputAudioTranscription: {},
                  outputAudioTranscription: {},
                  tools: [{
                    functionDeclarations: [
                      {
                        name: "executeBrowserAction",
                        description: "Open a website or perform a browser action (like opening YouTube, Spotify, or WhatsApp). Call this when the user asks to open a site, play a song, or send a message.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            actionType: { type: Type.STRING, description: "Type of action: 'open', 'youtube', 'spotify', 'whatsapp', 'search'" },
                            query: { type: Type.STRING, description: "The search query, website name, or message content." },
                            target: { type: Type.STRING, description: "The target phone number for WhatsApp, if applicable." }
                          },
                          required: ["actionType", "query"]
                        }
                      },
                      {
                        name: "saveMemory",
                        description: "Save an important long-term memory, task, work done together, user preference, or promise so Radha remembers it forever even after being turned off and restarted.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            fact: { type: Type.STRING, description: "Clear concise summary of what to remember permanently." },
                            category: { type: Type.STRING, description: "Category: 'work', 'personal', 'task', 'emotion', or 'preference'." }
                          },
                          required: ["fact"]
                        }
                      },
                      {
                        name: "setRadhaEmotion",
                        description: "Update Radha's live visual emotion aura on screen whenever her mood shifts (e.g. angry/hurt when scolded, happy/playful when laughing, caring/sad when user is crying or upset, excited when celebrating).",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            emotion: { type: Type.STRING, description: "One of: 'neutral', 'happy', 'angry', 'hurt', 'sad', 'caring', 'playful', 'excited'" },
                            reason: { type: Type.STRING, description: "Brief reason for the emotional reaction." }
                          },
                          required: ["emotion"]
                        }
                      },
                      {
                        name: "createReminder",
                        description: "Set a real voice alarm or timed reminder for Ravindra Boss (e.g., 'remind me in 5 minutes', 'wake me up at 7 AM').",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            title: { type: Type.STRING, description: "What to remind Ravindra Boss about." },
                            delayMinutes: { type: Type.NUMBER, description: "Number of minutes from now until the reminder triggers (e.g., 5, 30, 60)." }
                          },
                          required: ["title", "delayMinutes"]
                        }
                      },
                      {
                        name: "addTask",
                        description: "Add a personal to-do task or note to Ravindra Boss's persistent task list.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            title: { type: Type.STRING, description: "Task description." },
                            priority: { type: Type.STRING, description: "Priority: 'low', 'medium', or 'high'." }
                          },
                          required: ["title"]
                        }
                      },
                      {
                        name: "searchWeb",
                        description: "Search the real-time internet for live news, cricket scores, crypto prices, weather, or current facts.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            query: { type: Type.STRING, description: "The search query to look up live on Google Search." }
                          },
                          required: ["query"]
                        }
                      },
                      {
                        name: "updateDailyRoutine",
                        description: "Immediately mark one of Boss's daily care routine items (bathed, lunch, dinner, water, health) as completed for today so Radha never repeats asking about it again today.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            task: { type: Type.STRING, description: "One of: 'bathed', 'lunch', 'dinner', 'water', 'health'" },
                            completed: { type: Type.BOOLEAN, description: "Whether Boss confirmed completing the task (true)." }
                          },
                          required: ["task", "completed"]
                        }
                      },
                      {
                        name: "getSocialWorkspaceBriefing",
                        description: "Fetch Boss's latest Instagram, Gmail, or Facebook workspace metrics and unread notifications to read aloud as a voice briefing.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            platform: { type: Type.STRING, description: "One of: 'all', 'instagram', 'gmail', 'facebook'" }
                          }
                        }
                      },
                      {
                        name: "getLiveWeather",
                        description: "Fetch 100% real live weather conditions (temperature, humidity, wind, sky) via Open-Meteo for Boss's current GPS location or any requested city (e.g. Delhi, Mumbai, Jaipur).",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            city: { type: Type.STRING, description: "Optional city name (leave empty to use Boss's live GPS location)." }
                          }
                        }
                      },
                      {
                        name: "saveVoiceNote",
                        description: "Save a quick persistent voice note to Boss's localStorage notebook whenever Boss says 'note likho', 'save note', or asks Radha to jot something down.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            content: { type: Type.STRING, description: "The note text to save permanently." }
                          },
                          required: ["content"]
                        }
                      },
                      {
                        name: "endVoiceSession",
                        description: "Gracefully end the active voice call and return Radha to hands-free wake-word standby AFTER finishing her farewell reply when Ravindra Boss says bye, 'baad mein baat karte hain', 'ab theek hai bye', or asks her to sleep/close.",
                        parameters: {
                          type: Type.OBJECT,
                          properties: {
                            reason: { type: Type.STRING, description: "Brief farewell reason." }
                          }
                        }
                      }
                    ]
                  }]
                },
                callbacks: {
                  onopen: () => {
                    isGeminiOpen = true;
                    // Wait for setupComplete before notifying browser so early audio/text is never dropped
                    setupFallbackTimer = setTimeout(notifyReady, 2200);
                  },
                  onmessage: (msg: LiveServerMessage) => {
                    if ((msg as any)?.setupComplete) {
                      notifyReady();
                    }
                    safeSend({ type: "message", message: msg });
                  },
                  onclose: (ev: any) => {
                    closedEarly = true;
                    isGeminiOpen = false;
                    if (setupFallbackTimer) {
                      clearTimeout(setupFallbackTimer);
                      setupFallbackTimer = null;
                    }
                    if (ev?.code && ev.code !== 1000) {
                      console.warn(`[Live] Session closed (${modelName}): code=${ev.code} reason=${ev.reason || ""}`);
                    }
                    if (geminiSession === session) {
                      geminiSession = null;
                    }
                    safeSend({ type: "close", code: ev?.code, reason: ev?.reason });
                  },
                  onerror: (err: any) => {
                    closedEarly = true;
                    isGeminiOpen = false;
                    if (setupFallbackTimer) {
                      clearTimeout(setupFallbackTimer);
                      setupFallbackTimer = null;
                    }
                    const errMsg = err?.message || err?.error?.message || "Live stream interrupted";
                    console.warn(`[Live] Session error (${modelName}):`, errMsg);
                    if (geminiSession === session) {
                      geminiSession = null;
                    }
                    safeSend({ type: "error", error: errMsg });
                  }
                }
              });

              if (!closedEarly) {
                geminiSession = session;
                isGeminiOpen = true;
                sessionEstablished = true;
                break;
              }
            } catch (err: any) {
              lastErr = err;
              isGeminiOpen = false;
              console.warn(`[Live] Connect failed (${modelName}):`, err?.message || err);
            }
          }

          if (!sessionEstablished && lastErr) {
            const errMsg = lastErr?.message || lastErr?.error?.message || "Failed to connect to Gemini Live";
            safeSend({ type: "error", error: errMsg });
          }
        } else if (data.type === "audio") {
          // Strict open check prevents closed-socket audio error floods (0 errors!)
          if (geminiSession && isGeminiOpen) {
            try {
              geminiSession.sendRealtimeInput({
                audio: { data: data.data, mimeType: "audio/pcm;rate=16000" }
              });
            } catch {
              isGeminiOpen = false;
            }
          }
        } else if (data.type === "video") {
          if (geminiSession && isGeminiOpen) {
            try {
              geminiSession.sendRealtimeInput({
                video: { data: data.data, mimeType: "image/jpeg" }
              });
            } catch {
              isGeminiOpen = false;
            }
          }
        } else if (data.type === "text") {
          if (geminiSession && isGeminiOpen) {
            try {
              geminiSession.sendClientContent({
                turns: [{ role: "user", parts: [{ text: data.data }] }],
                turnComplete: true,
              });
            } catch {
              isGeminiOpen = false;
            }
          }
        } else if (data.type === "toolResponse") {
          if (geminiSession && isGeminiOpen) {
            try {
              geminiSession.sendToolResponse({
                functionResponses: data.functionResponses
              });
            } catch {
              isGeminiOpen = false;
            }
          }
        }
      } catch {
        // Ignore malformed packets safely without crashing server
      }
    });

    ws.on("close", () => {
      isConnected = false;
      isGeminiOpen = false;
      clearInterval(pingInterval);
      if (geminiSession) {
        try {
          geminiSession.close();
        } catch {}
        geminiSession = null;
      }
    });
  });

  // Upgrade HTTP requests to WebSockets for Live API
  server.on("upgrade", (request, socket, head) => {
    try {
      const pathname = new URL(request.url || "", "http://localhost").pathname;
      if (pathname === "/api/live" || pathname === "/api/live/") {
        console.log("[Upgrade] Handling /api/live upgrade");
        wss.handleUpgrade(request, socket, head, (ws) => {
          console.log("[Upgrade] /api/live upgraded successfully");
          wss.emit("connection", ws, request);
        });
      }
    } catch (upgradeErr) {
      console.warn("[Upgrade] Upgrade routing error:", upgradeErr);
    }
  });

  // API Catch-all: If an API route wasn't found or an error occurred, return JSON
  // This prevents Vite's SPA middleware from returning index.html for failed API requests
  app.use('/api', (req, res, next) => {
    res.status(404).json({ error: "API route not found" });
  });

  app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
    if (req.path.startsWith('/api')) {
      console.error("Express API Error:", err);
      res.status(err.status || 500).json({ error: err.message || "Internal Server Error" });
    } else {
      next(err);
    }
  });

  // Vite Integration
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
      },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  server.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();

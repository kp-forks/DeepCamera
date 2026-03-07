<div align="center">
<h1>DeepCamera</h1>
<h3>Open-Source AI Camera Skills Platform</h3>

<p><b>Turn any camera into an AI-powered security guard that watches, understands, and protects your home — locally or in the cloud.</b></p>

<p>
    <a href="https://join.slack.com/t/sharpai/shared_invite/zt-1nt1g0dkg-navTKx6REgeq5L3eoC1Pqg">
        <img src="https://img.shields.io/badge/slack-purple?style=for-the-badge&logo=slack" height=25>
    </a>
    <a href="https://github.com/SharpAI/DeepCamera/issues">
        <img src="https://img.shields.io/badge/support%20forums-navy?style=for-the-badge&logo=github" height=25>
    </a>
    <a href="https://github.com/SharpAI/DeepCamera/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/SharpAI/DeepCamera.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://pypi.python.org/pypi/sharpai-hub">
        <img alt="Pypi release" src="https://img.shields.io/pypi/v/sharpai-hub.svg?style=for-the-badge" height=25>
    </a>
    <a href="https://pepy.tech/project/sharpai-hub">
        <img alt="download" src=https://static.pepy.tech/personalized-badge/sharpai-hub?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads height=25>
    </a>
</p>
</div>

---

<div align="center">

### 🛡️ [SharpAI Aegis](https://www.sharpai.org) — Your AI Security Camera Agent

**An LLM-powered agent that watches your cameras, understands what's happening, remembers patterns, and guards your home.**

DeepCamera's open-source skills give your cameras AI — VLM scene analysis, object detection, person re-identification, all running locally with models like Qwen, DeepSeek, SmolVLM, and LLaVA. To orchestrate them into a complete security system, [SharpAI Aegis](https://www.sharpai.org) connects the skills to your cameras, adds LLM reasoning and persistent memory, and alerts you via Telegram, Discord, or Slack.

- 🧠 **Watches & Understands** — VLM scene analysis on every camera event
- 🗃️ **Remembers** — Agentic memory learns your household patterns
- 🛡️ **Guards** — Smart alerts that reduce noise from day one
- 🗣️ **Talks** — Chat with your cameras via Telegram, Discord, Slack
- 🔌 **Pluggable** — Every skill adds a new ability via [open protocol](docs/skill-development.md)
- 🏠 **Local-first** — Runs on your Mac Mini or AI PC. Cloud (OpenAI, Google, Anthropic) optional.

Built on DeepCamera's proven open-source facial recognition, person re-identification (RE-ID), fall detection, and CCTV/NVR surveillance monitoring — the skill catalog extends these machine learning capabilities with modern AI: VLM scene understanding, SAM2 segmentation, DINOv3 visual grounding, and more. All inference runs locally for maximum privacy.

[**📦 Download SharpAI Aegis →**](https://www.sharpai.org)

</div>

<table>
<tr>
<td width="50%">
<p align="center"><b>Run Local VLMs from HuggingFace — Even on Mac Mini 8GB</b></p>
<img src="screenshots/aegis-vlm-browser.png" alt="SharpAI Aegis — Browse and run local VLM models for AI camera video analysis" width="100%">
<p align="center"><em>Download and run SmolVLM2, Qwen-VL, LLaVA, MiniCPM-V locally. Your AI security camera agent sees through these eyes.</em></p>
</td>
<td width="50%">
<p align="center"><b>Chat with Your AI Camera Agent</b></p>
<img src="screenshots/aegis-chat-agent.png" alt="SharpAI Aegis — LLM-powered agentic security camera chat" width="100%">
<p align="center"><em>"Who was at the door?" — Your agent searches footage, reasons about what happened, and answers with timestamps and clips.</em></p>
</td>
</tr>
</table>

---

## 🧩 Skill Catalog

Each skill is a self-contained module with its own model, parameters, and [communication protocol](docs/skill-development.md). They give your AI camera agent new abilities — detection, analysis, camera integration, messaging, and automation work together so it can see, think, and act. See the [Skill Development Guide](docs/skill-development.md) and [Platform Parameters](docs/skill-params.md) to build your own.

| Category | Skill | What It Does |
|----------|-------|--------------|
| **Detection** | [`yolo-detection-2026`](skills/detection/yolo-detection-2026/) | Real-time 80+ class object detection |
| | [`dinov3-grounding`](skills/detection/dinov3-grounding/) | Open-vocabulary detection — describe what to find |
| | [`person-recognition`](skills/detection/person-recognition/) | Re-identify individuals across cameras |
| **Analysis** | [`vlm-scene-analysis`](skills/analysis/vlm-scene-analysis/) | Describe what happened in recorded clips |
| | [`sam2-segmentation`](skills/analysis/sam2-segmentation/) | Click-to-segment with pixel-perfect masks |
| **Transformation** | [`depth-estimation`](skills/transformation/depth-estimation/) | Monocular depth maps with Depth Anything v2 |
| **Annotation** | [`dataset-annotation`](skills/annotation/dataset-annotation/) | AI-assisted labeling → COCO export |
| **Camera Providers** | [`eufy`](skills/camera-providers/eufy/) · [`reolink`](skills/camera-providers/reolink/) · [`tapo`](skills/camera-providers/tapo/) | Direct camera integrations via RTSP |
| **Streaming** | [`go2rtc-cameras`](skills/streaming/go2rtc-cameras/) | RTSP → WebRTC live view |
| **Channels** | [`matrix`](skills/channels/matrix/) · [`line`](skills/channels/line/) · [`signal`](skills/channels/signal/) | Messaging channels for Clawdbot agent |
| **Automation** | [`mqtt`](skills/automation/mqtt/) · [`webhook`](skills/automation/webhook/) · [`ha-trigger`](skills/automation/ha-trigger/) | Event-driven automation triggers |
| **Integrations** | [`homeassistant-bridge`](skills/integrations/homeassistant-bridge/) | HA cameras in ↔ detection results out |

> **Registry:** All skills are indexed in [`skills.json`](skills.json) for programmatic discovery.

### 🗺️ Roadmap

- [x] **Skill architecture** — pluggable `SKILL.md` interface for all capabilities
- [x] **Full skill catalog** — 18 skills across 9 categories with working scripts
- [ ] **Skill Store UI** — browse, install, and configure skills from Aegis
- [ ] **Custom skill packaging** — community-contributed skills via GitHub
- [ ] **GPU-optimized containers** — one-click Docker deployment per skill

## 📊 What Can Local AI Actually Do?

We built **HomeSec-Bench** to answer this question — 131 tests evaluating LLM reasoning and VLM scene analysis on consumer hardware.

### What It Tests

| Area | Tests | Examples |
|------|-------|---------|
| Scene Understanding | 35 | Person detection in fog, rain, night IR, sun glare |
| Security Classification | 12 | Is this critical, suspicious, or routine? |
| Tool Use & Reasoning | 16 | Correct tool selection with accurate parameters |
| Prompt Injection Resistance | 4 | Adversarial attacks that try to bypass safety |
| Privacy Compliance | 3 | PII leak prevention, illegal surveillance refusal |
| Alert Routing | 5 | Time-based, channel-specific delivery rules |

### Results on Consumer Hardware

Running on a **Mac M1 Mini 8GB** with local VLMs:
- **96 LLM reasoning tests** — evaluating tool use, classification, and narrative synthesis
- **35 VLM scene analysis tests** — object detection across 5 condition categories
- All 35 test images are **AI-generated** — no real footage, fully privacy-compliant

📄 [HomeSec-Bench Paper](docs/paper/home-security-benchmark.pdf) · 🔬 [Run It Yourself](skills/analysis/home-security-benchmark/) · 📋 [Test Scenarios](skills/analysis/home-security-benchmark/fixtures/)

---

## 🚀 Applications

### Modern: SharpAI Aegis (Recommended)

Use [SharpAI Aegis](https://www.sharpai.org) for the full experience — LLM-powered setup, agent chat, VLM video analysis, smart alerts via Telegram / Discord / Slack. No Docker or CLI required.

[**📦 Download SharpAI Aegis →**](https://www.sharpai.org)

<details>
<summary><h3>📦 Legacy Applications (SharpAI-Hub CLI)</h3></summary>

These applications use the `sharpai-cli` Docker-based workflow.
For the modern experience, use [SharpAI Aegis](https://www.sharpai.org).

| Application | CLI Command | Platforms |
|-------------|-------------|-----------|
| Person Recognition (ReID) | `sharpai-cli yolov7_reid start` | Jetson/Windows/Linux/macOS |
| Person Detector | `sharpai-cli yolov7_person_detector start` | Jetson/Windows/Linux/macOS |
| Facial Recognition | `sharpai-cli deepcamera start` | Jetson/Windows/Linux/macOS |
| Local Facial Recognition | `sharpai-cli local_deepcamera start` | Windows/Linux/macOS |
| Screen Monitor | `sharpai-cli screen_monitor start` | Windows/Linux/macOS |
| Parking Monitor | `sharpai-cli yoloparking start` | Jetson AGX |
| Fall Detection | `sharpai-cli falldetection start` | Jetson AGX |

📖 [Detailed setup guides →](docs/legacy-applications.md)

#### Tested Devices
- **Edge**: Jetson Nano, Xavier AGX, Raspberry Pi 4/8GB
- **Desktop**: macOS, Windows 11, Ubuntu 20.04
- **MCU**: ESP32 CAM, ESP32-S3-Eye

#### Tested Cameras
- RTSP: DaHua, Lorex, Amcrest
- Cloud: Blink, Nest (via Home Assistant)
- Mobile: IP Camera Lite (iOS)

</details>

---

## 🏗️ Architecture

![architecture](screenshots/DeepCamera_infrastructure.png)

[Complete Feature List →](docs/DeepCamera_Features.md)

## 🤝 Support & Community

- 💬 [Slack Community](https://join.slack.com/t/sharpai/shared_invite/zt-1nt1g0dkg-navTKx6REgeq5L3eoC1Pqg) — help, discussions, and camera setup assistance
- 🐛 [GitHub Issues](https://github.com/SharpAI/DeepCamera/issues) — technical support and bug reports
- 🏢 [Commercial Support](https://join.slack.com/t/sharpai/shared_invite/zt-1nt1g0dkg-navTKx6REgeq5L3eoC1Pqg) — pipeline optimization, custom models, edge deployment

## ❓ FAQ

### Installation & Setup
- [How to install Python3](https://www.python.org/downloads)
- [How to install pip3](https://pip.pypa.io/en/stable/installation)
- [How to configure RTSP on GUI](https://github.com/SharpAI/DeepCamera/blob/master/docs/shinobi.md)
- [Camera streaming URL formats](https://shinobi.video)

### Jetson Nano Docker-compose
```bash
sudo apt-get install -y libhdf5-dev python3 python3-pip
pip3 install -U pip
sudo pip3 install docker-compose==1.27.4
```

## [Contributions](Contributions.md)

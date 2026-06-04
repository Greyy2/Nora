<div align="center">
  <img src="docs/images/grey-home.png" alt="Grey / NoraQuantEngine dashboard" width="92%" />
</div>

<div align="center">

# Grey / NoraQuantEngine

**Nền tảng nghiên cứu định lượng, backtest, trading dashboard và AI factor discovery.**

Grey gom backend FastAPI, frontend React/Vite, engine backtest, Walk-Forward Analysis, Monte Carlo, SOVA AI và QuantaAlpha vào một workspace thống nhất cho nghiên cứu chiến lược giao dịch.

<p>
  <a href="https://github.com/Greyy2/Nora"><img src="https://img.shields.io/badge/GitHub-Greyy2%2FNora-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub repository" /></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI backend" />
  <img src="https://img.shields.io/badge/React%20%2B%20Vite-Frontend-646CFF?style=flat-square&logo=vite&logoColor=white" alt="React Vite frontend" />
  <img src="https://img.shields.io/badge/MongoDB-Storage-47A248?style=flat-square&logo=mongodb&logoColor=white" alt="MongoDB storage" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker ready" />
</p>

<p>
  <a href="#tong-quan">Tổng quan</a> ·
  <a href="#module-chinh">Module chính</a> ·
  <a href="#demo-giao-dien">Demo giao diện</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#an-toan-public-repo">An toàn repo</a>
</p>

</div>

---

## Tổng Quan

Grey là một hệ thống full-stack phục vụ quy trình nghiên cứu định lượng từ dữ liệu thị trường đến kiểm định chiến lược, stress test, AI analysis và export báo cáo. Repo public này chỉ chứa code logic, cấu hình mẫu và tài liệu minh họa; dữ liệu chạy thật, kết quả, token và secret được loại khỏi Git.

```text
Market Data -> Backtest / WFA / Monte Carlo -> Trading Dashboard -> SOVA / QuantaAlpha -> Reports / Export
```

| Layer | Thành phần | Vai trò |
| :--- | :--- | :--- |
| Frontend | React, Vite, Bootstrap, Plotly, Chart.js, lightweight-charts | Dashboard cho Home, Backtest, Trading, Client AI, Settings |
| Backend | FastAPI, Uvicorn, Pydantic | API orchestration, health checks, artifact routing |
| Storage | MongoDB, local runtime mounts | Lưu campaign, result document và lịch sử thực thi khi chạy local |
| Research Core | Backtest, WFA, Monte Carlo | Kiểm định chiến lược, robustness và stress test |
| AI Core | Grey AI, SOVA, QuantaAlpha | Phân tích thị trường, factor mining và factor library |
| Export | Google Sheets / Excel endpoints | Xuất kết quả sang workflow báo cáo |

---

## Module Chính

| Module | Màn hình | API / Engine | Giá trị chính |
| :--- | :--- | :--- | :--- |
| Nora Trading | `/trading` | `/api/single-core`, `/api/chart-data` | Kiểm tra chiến lược trên chart, data window và regime overlay |
| Nora Backtest | `/backtest` | `/api/backtest`, `/api/wfa`, `/api/carlo`, `/api/campaigns` | Chạy campaign, đọc kết quả, WFA, Monte Carlo và top strategies |
| Nora Client AI | `/client` | `/api/ai/quanta`, `/api/ai/quanta/v2` | Factor mining, factor library, backtest factor và live logs |
| SOVA AI | `/api/sova/*` | SOVA engine | AI reasoning layer cho phân tích và điều phối workflow |
| Sheets Export | `/api/sheets/*` | Google Sheets service | Export backtest/trade report khi cấu hình OAuth local |

---

## Demo Giao Diện

<div align="center">
  <img src="docs/images/grey-client-mining.png" alt="Nora Client AI factor mining workspace" width="90%" />
  <p><em>Nora Client AI workspace cho factor mining và live evolution logs.</em></p>
</div>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/grey-client-factor-library.png" alt="Nora Client factor library" width="100%" />
      <br />
      <strong>Factor Library</strong>
    </td>
    <td align="center" width="50%">
      <img src="docs/images/grey-backtest-dashboard.png" alt="Nora Backtest dashboard" width="100%" />
      <br />
      <strong>Backtest Dashboard</strong>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/grey-backtest-results.png" alt="Nora Backtest results" width="100%" />
      <br />
      <strong>Backtest Results</strong>
    </td>
    <td align="center" width="50%">
      <img src="docs/images/grey-trading-chart.png" alt="Nora Trading chart" width="100%" />
      <br />
      <strong>Nora Trading</strong>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/images/grey-sova-architecture.png" alt="SOVA AI architecture" width="100%" />
      <br />
      <strong>SOVA AI Architecture</strong>
    </td>
    <td align="center" width="50%">
      <img src="docs/images/grey-sova-training.png" alt="SOVA AI training workflow" width="100%" />
      <br />
      <strong>SOVA Training Workflow</strong>
    </td>
  </tr>
</table>

---

## Quick Start

### 1. Clone repo

```bash
git clone git@github.com:Greyy2/Nora.git
cd Nora
```

### 2. Cấu hình môi trường local

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

Nếu dùng Google Sheets export, tạo thêm `sheet/.env` từ `sheet/.env.example` và điền OAuth secret ở máy local.

### 3. Chạy bằng Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

```text
Frontend: http://localhost:5720/backtest
Backend:  http://localhost:8000
MongoDB:  mongodb://localhost:27020
```

### 4. Chạy local để phát triển

Backend:

```bash
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Mở ứng dụng tại:

```text
http://localhost:5721/backtest
```

---

## Cấu Trúc Repo

```text
Grey/
├── backend/          # FastAPI app, routes, services, engines, optimize modules
├── frontend/         # React/Vite UI source
├── docker/           # Dockerfiles and docker-compose
├── docs/images/      # README screenshots
├── nginx/            # Deployment nginx config
├── .env.example      # Local config template without secrets
└── start.sh          # Helper script for local/server startup
```

Các thư mục runtime như `data/`, `results/`, `log/`, `tmp/`, `frontend/node_modules/`, `frontend/dist/`, `sheet/` và experiment artifacts không được commit.

---

## An Toàn Public Repo

Repo này được chuẩn bị theo nguyên tắc public-safe:

| Không commit | Lý do |
| :--- | :--- |
| `.env`, `.env.local`, `sheet/*.json`, `sheet/.env` | Chứa API key, OAuth secret, token hoặc service account |
| `data/`, `results/`, `backend/engines/*/data` | Dữ liệu thị trường và artifact chạy thật |
| `frontend/node_modules/`, `frontend/dist/` | Dependency/build output có thể tái tạo |
| `log/`, `tmp/`, `mlruns/`, `cache/`, `*.pkl`, `*.csv` | Runtime logs, model/cache/result files |

Grey là hệ thống nghiên cứu và kiểm định chiến lược. Kết quả backtest, AI analysis và factor mining không phải lời khuyên đầu tư; mọi claim hiệu năng cần benchmark có thể tái lập trước khi dùng cho production.

<div align="center">

**Grey / NoraQuantEngine**  
Research -> Backtest -> Trading -> AI -> Export

</div>

# 部署指南（GitHub + Hugging Face Spaces）

本项目是一个 Flask 应用，推荐部署方式：

```
GitHub（公开代码，README 放演示链接）
        │  git push
        ▼
Hugging Face Space（Docker，免费 CPU，16 GB 内存）
        │  自动构建
        ▼
https://<user>-<space>.hf.space
```

> 为什么不用 GitHub Pages：Pages 只能托管静态文件，不能运行 Flask/JAX。
> 为什么推荐 HF Spaces：免费 CPU Space 有 16 GB 内存；JAX 反演编译需要数 GB，
> Render 免费版 512 MB 会 OOM。

---

## 1. 创建 Hugging Face Space

1. 登录 <https://huggingface.co> → **New Space**。
2. 填写：
   - Owner / Space name：例如 `yourname/srt-workbench`
   - License：MIT（或与你仓库一致）
   - SDK：**Docker** → **Blank**
   - Hardware：**CPU basic · 2 vCPU · 16 GB · FREE**
   - Visibility：Public
3. 创建后会得到一个 git 地址，形如：
   `https://huggingface.co/spaces/yourname/srt-workbench`

## 2. 推送代码

Space 根目录的 `README.md` 必须带 YAML 元数据（Docker SDK / 端口）：

```yaml
---
title: SRT Workbench
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

模板见 [`deploy/hf/README.md`](../deploy/hf/README.md)。两种推送方式：

**A. 从本仓库直接推（推荐）**

```bash
git remote add hf https://huggingface.co/spaces/yourname/srt-workbench
# 把 deploy/hf/README.md 复制为 Space 根 README（仅这次提交，不要影响 GitHub 的 README）
cp deploy/hf/README.md README.md
git add README.md && git commit -m "docs: HF Space metadata"
git push hf main
```

**B. 在 HF 网页上手动上传**：把仓库打包上传，确保根目录有 `Dockerfile`、
`requirements.txt`、`app/`、`diff_srt/`、`launcher.py`，以及带元数据的 `README.md`。

推送后 HF 会自动构建镜像（3–10 分钟，取决于 wheel 下载）。

## 3. 运行时配置

`Dockerfile` 已内置演示模式的环境变量：

| 变量 | 值 | 作用 |
|---|---|---|
| `SRT_DEMO` | `1` | 临时运行目录、单并发 |
| `SRT_ALLOW_FORTRAN` | `0` | 关闭 Windows Fortran 引擎 |
| `SRT_MAX_NZ` | `20` | 限制离散层数 |
| `SRT_MAX_ITER` | `300` | 限制反演迭代数 |
| `SRT_MAX_WORKERS` | `1` | 同时只跑一个任务 |
| `SRT_RATE_LIMIT` / `SRT_RATE_WINDOW` | `40` / `600` | 每 IP 10 分钟 40 次提交 |
| `SRT_RUNS_DIR` / `SRT_JAX_CACHE` | `/tmp/...` | 任务与编译缓存 |
| `HOST` / `PORT` | `0.0.0.0` / `7860` | HF 要求的端口 |

如需调整，在 Space 的 **Settings → Variables and secrets** 里覆盖即可，不必改代码。

## 4. 验证清单

部署完成后依次检查：

```bash
BASE=https://yourname-srt-workbench.hf.space

curl -s $BASE/api/health
# {"ok":true,"engine":"jax","fortran":false,...}

# JAX 正演（提交任务 → 轮询）
curl -s -X POST $BASE/api/jax/forward -H 'Content-Type: application/json' \
  -d '{"H":5,"cover":0.494,"d_L":1,"tree_diam":5,"tree_type":4,"lad":6,
       "tau_L":0.4843,"ro_L":0.4746,"ro_soil":0.3379,"hot_spot":8,
       "sza":45,"saa":270,"slope":0,"aspect":90,"alpha_slope":0,
       "beta_aspect":90,"i0":1,"nz":20,"maxiter":10,
       "vza":[0,15,30,45,60],"vaa":[0,90,180,270]}'

curl -s $BASE/api/jobs/<job_id>          # 直到 status=done
curl -s $BASE/api/jobs/<job_id>/result | head -c 400
```

浏览器端检查：正演曲线、反演损失/观测拟合、敏感性热图。

## 5. 更新 GitHub README 的演示链接

把 `README.md` 顶部的占位链接替换为：

```
**在线演示：** https://yourname-srt-workbench.hf.space
```

## 6. 可选

- **保持唤醒**：HF 免费 Space 空闲一段时间会休眠，首次请求需冷启动
  （约 30–60 s + JAX 预热）。可用外部定时 ping，但请遵守 HF 的使用政策。
- **持久编译缓存**：给 Space 挂载持久存储，把 `SRT_JAX_CACHE` 指到该目录，
  重启后无需重新编译。
- **自定义域名**：HF 支持在 Space 设置里绑定自定义域名（需要 CNAME）。
- **访问控制**：目前是公开无鉴权；如需限制，可在反向代理层加 Basic Auth，
  或给 `SRT_RATE_LIMIT` 设更小的值。

## 7. 关于 Fortran 引擎

公开演示默认 **不启用** Fortran：

- 原 Fortran 参考代码的版权/许可需要确认后才能公开；
- Windows 的 `exe_SRT_topo_static.exe` 无法在 Linux 容器中运行。

如果确认可以公开源码，可以在 Dockerfile 里加 gfortran 编译阶段：

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends gfortran \
 && gfortran -static -std=legacy -fallow-argument-mismatch -O2 \
      -o SRT_topo/exe_SRT_topo_static SRT_topo/*.f
ENV SRT_ALLOW_FORTRAN=1
```

同时把 `SRT_topo/` 纳入仓库（去掉 `.gitignore` 里的对应行），并把
`SRT_ALLOW_FORTRAN` 从 `0` 改为 `1`。若不确定许可，请保持 JAX-only。

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| 构建失败在 `pip install jax[cpu]` | 确认 Python 3.12；必要时放宽 `jax` 版本 |
| 首次访问很慢 | Space 冷启动 + JAX 编译，属正常 |
| 反演 OOM | 降低 `SRT_MAX_NZ` / `SRT_MAX_ITER`，或升级硬件 |
| `/api/forward` 返回 503 | 演示模式未启用 Fortran（预期行为），使用 `/api/jax/forward` |
| 429 | 触发限流，等待窗口结束或调大 `SRT_RATE_LIMIT` |

# Posture Analyzer

一个适合桌面和手机浏览器使用的本地姿态分析网页。

## 项目简介

本项目基于 MediaPipe Pose Landmarker，在浏览器中实时分析上半身姿态，用于观察坐姿稳定度、头部偏移、肩线平衡和躯干倾斜情况。所有计算都在本地完成，不上传摄像头画面。

## 功能特性

- 实时摄像头姿态识别
- 姿态总分与状态标签显示
- 四项核心指标可视化
  - 头部偏移
  - 肩线平衡
  - 躯干倾斜
  - 稳定度
- 趋势曲线与会话统计
- 根据当前姿态输出中文纠正建议
- 针对手机浏览器优化的布局、按钮尺寸和安全区适配

## 技术栈

- 原生 HTML / CSS / JavaScript
- `@mediapipe/tasks-vision` 通过 CDN 加载
- 无构建步骤，可直接部署为静态站点

## 本地运行

摄像头权限通常要求 `HTTPS` 或 `localhost` 安全上下文。

如果你的系统已安装 Python，可以在项目目录运行：

```powershell
cd C:\Users\Administrator\Desktop\github\posture-analyzer
py -m http.server 5500
```

然后打开：

- `http://localhost:5500`

## 部署到 GitHub Pages

1. 将仓库推送到 GitHub 默认分支。
2. 打开仓库 `Settings > Pages`。
3. 在 `Build and deployment` 中选择 `Deploy from a branch`。
4. 选择默认分支和 `/ (root)`。
5. 保存后等待 GitHub Pages 发布。

## 使用建议

- 尽量让头部、肩部和髋部都进入画面。
- 手机端建议竖屏打开，并允许摄像头权限。
- 将设备固定在较稳定的位置，避免手持晃动影响稳定度判断。
- 若浏览器提示无法调用摄像头，请确认页面运行在 `localhost` 或 `HTTPS` 环境下。

## 隐私说明

所有姿态检测均在浏览器本地执行，不会上传视频数据。

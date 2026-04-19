"""
从 PyTorch Profiler 导出的 Chrome Trace JSON 中解析 CUDA 显存采样与 record_function 区间，
计算各区间内峰值并绘图。输出默认写入项目 trace/ 目录。

用法:
  python profiler_trace_plot.py /path/to/chrome_trace.json
  # 需要把前向区间也画进浅色条时自行指定前缀，例如：
  python profiler_trace_plot.py trace.json --prefix shapelets/ model/ mix/ CL.forward_q CL.forward_k
  # 下图横轴为相对 trace 起点的绝对时刻区间，默认 [1.5, 2.0] 秒：
  python profiler_trace_plot.py trace.json --zoom-from 1.5 --zoom-to 2.0

依赖: matplotlib, numpy

关于 Profiler 里 shapelet 相关区间名称：
  - 「shapelets/L…/类名」：**前向**子块（默认不在图中标色，除非 --prefix 包含 shapelets/）。
  - 「shapelets_bw/L…/类名」：**反向**子块（默认会标色）。
  若 trace 里只有一段大反向、没有各 shapelets_bw：多半是 **gradient checkpoint** 路径导致子模块
  backward hook 不按块进 trace；跑 profile 前可设环境变量 **CSL_DETAIL_BW_IN_PROFILER=1**
  （会临时禁用 ShapeletsDistBlocks 里的 checkpoint，显存会涨）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# 默认只解析/标注各 shapelet 子块的反向 record_function（前向需用 --prefix 自行加入）
DEFAULT_SPAN_PREFIXES: Tuple[str, ...] = ("shapelets_bw/",)

# 下图局部放大：横轴为相对 trace 起点的绝对时刻 [t0, t1]（秒），默认 1.5～2.0
DEFAULT_ZOOM_FROM_S = 1.5
DEFAULT_ZOOM_TO_S = 2.0


def configure_matplotlib_chinese_font() -> None:
    """
    配置可显示中文的字体，避免标题/图例/坐标轴出现方框。
    按顺序尝试常见系统字体；并关闭 unicode 负号用 ASCII 减号，避免负号乱码。
    """
    import matplotlib.pyplot as plt

    # 顺序即回退链：macOS 常见 PingFang / Hiragino；Windows 雅黑/黑体；Linux Noto/文泉驿
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def zoom_xlim_absolute(
    rel_s: np.ndarray,
    zoom_from_s: float,
    zoom_to_s: float,
) -> Tuple[float, float]:
    """
    下图使用固定时刻区间 [zoom_from_s, zoom_to_s]（与全景图相同的相对时间轴），
    再与当前 trace 的 [rel_s[0], rel_s[-1]] 求交；若交为空则退回全区间。
    """
    if rel_s.size == 0:
        return zoom_from_s, zoom_to_s
    t_min = float(np.min(rel_s))
    t_max = float(np.max(rel_s))
    x0, x1 = float(zoom_from_s), float(zoom_to_s)
    if x0 > x1:
        x0, x1 = x1, x0
    x0 = max(x0, t_min)
    x1 = min(x1, t_max)
    if x1 <= x0:
        return t_min, t_max
    return x0, x1


@dataclass
class MemorySample:
    """单点显存采样（Chrome trace 时间戳为微秒）。"""
    ts_us: float
    allocated_bytes: float


@dataclass
class DurationSpan:
    """ph=='X' 的区间事件（record_function 等）。"""
    name: str
    ts_us: float
    dur_us: float

    @property
    def end_us(self) -> float:
        return self.ts_us + self.dur_us


@dataclass
class SpanPeakResult:
    span_name: str
    start_us: float
    end_us: float
    peak_gb: float
    peak_t_us: float


def load_chrome_trace(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_trace_events(trace: dict) -> List[dict]:
    return list(trace.get("traceEvents") or [])


def extract_cuda_memory_samples(
    events: Iterable[dict],
    require_device_cuda: bool = True,
) -> List[MemorySample]:
    """
    解析参考格式: name == "[memory]", args 含 Total Allocated;
    Device Type == 1 表示 CUDA（部分版本可能无此字段，可关 require_device_cuda）。
    """
    out: List[MemorySample] = []
    for evt in events:
        if evt.get("name") != "[memory]":
            continue
        args = evt.get("args") or {}
        if require_device_cuda and "Device Type" in args and args.get("Device Type") != 1:
            continue
        ta = args.get("Total Allocated")
        if ta is None:
            continue
        ts = float(evt.get("ts", 0))
        out.append(MemorySample(ts_us=ts, allocated_bytes=float(ta)))
    out.sort(key=lambda s: s.ts_us)
    return out


def default_span_name_filter(name: str) -> bool:
    """默认仅保留各子块反向区间（shapelets_bw/）；前向不参与浅色条与图例。"""
    if not name or name == "[memory]":
        return False
    if name.startswith("ProfilerStep"):
        return False
    if name.startswith("aten::") or name.startswith("cuda"):
        return False
    return any(name.startswith(p) for p in DEFAULT_SPAN_PREFIXES)


def extract_duration_spans(
    events: Iterable[dict],
    name_filter: Optional[Callable[[str], bool]] = None,
) -> List[DurationSpan]:
    """收集 ph=='X' 且带 dur 的区间，用于对齐 record_function 标签。"""
    filt = name_filter or default_span_name_filter
    spans: List[DurationSpan] = []
    for evt in events:
        if evt.get("ph") != "X":
            continue
        dur = evt.get("dur")
        if dur is None:
            continue
        name = evt.get("name") or ""
        if not filt(name):
            continue
        ts = float(evt.get("ts", 0))
        spans.append(DurationSpan(name=name, ts_us=ts, dur_us=float(dur)))
    spans.sort(key=lambda s: (s.ts_us, s.name))
    return spans


def samples_to_arrays_ms_gb(samples: Sequence[MemorySample]) -> Tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.array([]), np.array([])
    t = np.array([s.ts_us for s in samples], dtype=np.float64)
    v = np.array([s.allocated_bytes / (1024 ** 3) for s in samples], dtype=np.float64)
    return t, v


def peak_in_interval(
    t_us: np.ndarray,
    v_gb: np.ndarray,
    start_us: float,
    end_us: float,
) -> Tuple[float, float]:
    """区间内最大显存及对应时间（微秒）；无采样则返回 (0.0, start_us)。"""
    if t_us.size == 0:
        return 0.0, start_us
    mask = (t_us >= start_us) & (t_us <= end_us)
    if not np.any(mask):
        return 0.0, start_us
    idx = int(np.argmax(v_gb[mask]))
    sub_t = t_us[mask]
    sub_v = v_gb[mask]
    return float(sub_v[idx]), float(sub_t[idx])


def compute_span_peaks(
    samples: Sequence[MemorySample],
    spans: Sequence[DurationSpan],
) -> List[SpanPeakResult]:
    t_us, v_gb = samples_to_arrays_ms_gb(samples)
    results: List[SpanPeakResult] = []
    for sp in spans:
        peak_gb, peak_t = peak_in_interval(t_us, v_gb, sp.ts_us, sp.end_us)
        results.append(
            SpanPeakResult(
                span_name=sp.name,
                start_us=sp.ts_us,
                end_us=sp.end_us,
                peak_gb=peak_gb,
                peak_t_us=peak_t,
            )
        )
    return results


def aggregate_peaks_by_name(span_peaks: Sequence[SpanPeakResult]) -> dict:
    """同名多段区间：保留全局最大峰值。"""
    best: dict = {}
    for r in span_peaks:
        prev = best.get(r.span_name)
        if prev is None or r.peak_gb > prev["peak_gb"]:
            best[r.span_name] = {
                "peak_gb": r.peak_gb,
                "peak_t_us": r.peak_t_us,
                "start_us": r.start_us,
                "end_us": r.end_us,
            }
    return best


def _span_name_to_color(span_names: Sequence[str]) -> dict:
    """每个 record_function 名称固定一种颜色（图例与上下两图一致）。"""
    import matplotlib.pyplot as plt

    unique = sorted(set(span_names))
    if not unique:
        return {}
    cmap = plt.get_cmap("tab20")
    return {name: cmap((i % 20) / 20.0) for i, name in enumerate(unique)}


def _draw_single_panel(
    ax,
    rel_s: np.ndarray,
    v_gb: np.ndarray,
    spans: Sequence[DurationSpan],
    span_peaks: Sequence[SpanPeakResult],
    t0: float,
    name_to_color: dict,
    *,
    annotate_global: bool,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """绘制单幅显存阶梯曲线、record_function 浅色底纹、峰值散点。"""
    ax.step(rel_s, v_gb, where="post", color="#2E86AB", lw=2)

    for sp in spans:
        c = name_to_color.get(sp.name, (0.7, 0.7, 0.7, 1.0))
        s0 = (sp.ts_us - t0) / 1e6
        s1 = (sp.end_us - t0) / 1e6
        ax.axvspan(s0, s1, facecolor=c, alpha=0.22, edgecolor="none")

    rel_peak_t = [(r.peak_t_us - t0) / 1e6 for r in span_peaks]
    rel_peak_v = [r.peak_gb for r in span_peaks]
    if span_peaks:
        ax.scatter(
            rel_peak_t,
            rel_peak_v,
            c="#E94F37",
            s=42,
            zorder=5,
            edgecolors="white",
            linewidths=0.5,
        )

    g_idx = int(np.argmax(v_gb))
    g_t = float(rel_s[g_idx])
    g_v = float(v_gb[g_idx])
    ax.scatter(
        [g_t],
        [g_v],
        c="gold",
        s=160,
        zorder=6,
        marker="*",
        edgecolors="darkgoldenrod",
        linewidths=0.8,
    )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if annotate_global and rel_s.size > 0:
        xa0, xa1 = ax.get_xlim()
        dx = max((xa1 - xa0) * 0.08, 1e-9)
        ax.annotate(
            f"{g_v:.3f} GB",
            xy=(g_t, g_v),
            xytext=(g_t + dx, g_v + max(g_v * 0.04, 0.015)),
            arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.2),
            fontsize=9,
        )

    ax.set_title(title)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("已分配显存 (GB)")
    ax.grid(alpha=0.3)


def plot_memory_with_spans(
    samples: Sequence[MemorySample],
    spans: Sequence[DurationSpan],
    span_peaks: Sequence[SpanPeakResult],
    output_path: str,
    title: str = "CUDA memory vs record_function spans",
    zoom_from_s: float = DEFAULT_ZOOM_FROM_S,
    zoom_to_s: float = DEFAULT_ZOOM_TO_S,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    configure_matplotlib_chinese_font()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    t_us, v_gb = samples_to_arrays_ms_gb(samples)
    if t_us.size == 0:
        raise ValueError("无显存采样点：请确认 trace 由 profile_memory=True 导出且含 [memory] 事件")

    t0 = float(np.min(t_us))
    if spans:
        t0 = min(t0, min(s.ts_us for s in spans))
    rel_s = (t_us - t0) / 1e6

    span_names = [s.name for s in spans]
    name_to_color = _span_name_to_color(span_names)

    fig, (ax_top, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(14, 11),
        gridspec_kw={"height_ratios": [1.0, 1.2]},
    )
    fig.suptitle(
        "读图说明（默认仅反向子块）：浅色横条 = shapelets_bw/…；"
        "颜色与图例「区间:」一致；橙色圆点 = 该段时间内显存峰值；金星 = 全 trace 最大采样。"
        "缺块时请用 CSL_DETAIL_BW_IN_PROFILER=1 重采 trace；前向请加 --prefix。",
        fontsize=10,
        y=0.995,
    )

    _draw_single_panel(
        ax_top,
        rel_s,
        v_gb,
        spans,
        span_peaks,
        t0,
        name_to_color,
        annotate_global=True,
        title=f"{title} — 全景",
    )

    # 局部放大：固定相对时刻 [zoom_from_s, zoom_to_s]（默认 1.5～2.0 s），与数据范围求交
    g_idx = int(np.argmax(v_gb))
    g_t = float(rel_s[g_idx])
    xz0, xz1 = zoom_xlim_absolute(rel_s, zoom_from_s, zoom_to_s)

    zm = (rel_s >= xz0) & (rel_s <= xz1)
    if np.any(zm):
        ylo = float(np.min(v_gb[zm]))
        yhi = float(np.max(v_gb[zm]))
    else:
        ylo, yhi = float(np.min(v_gb)), float(np.max(v_gb))
    for r in span_peaks:
        tt = (r.peak_t_us - t0) / 1e6
        if xz0 <= tt <= xz1:
            ylo = min(ylo, r.peak_gb)
            yhi = max(yhi, r.peak_gb)
    if xz0 <= g_t <= xz1:
        g_v = float(v_gb[g_idx])
        ylo = min(ylo, g_v)
        yhi = max(yhi, g_v)
    pad_y = max((yhi - ylo) * 0.18, yhi * 0.03, 0.02)
    ylim_zoom = (max(0.0, ylo - pad_y), yhi + pad_y)

    _draw_single_panel(
        ax_zoom,
        rel_s,
        v_gb,
        spans,
        span_peaks,
        t0,
        name_to_color,
        annotate_global=True,
        title=f"局部放大：相对时刻 [{xz0:.4f}, {xz1:.4f}] s（请求 [{zoom_from_s:.2f}, {zoom_to_s:.2f}] s 与数据求交）",
        xlim=(xz0, xz1),
        ylim=ylim_zoom,
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#2E86AB",
            lw=2.5,
            label="显存曲线（Chrome trace 中 [memory] / Total Allocated）",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="gold",
            markeredgecolor="darkgoldenrod",
            markeredgewidth=0.8,
            markersize=14,
            label="全局峰值：整条曲线上采样点的最大值",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#E94F37",
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=9,
            label="区间内峰值：某 record_function 时间段内的最大显存",
        ),
    ]
    for name in sorted(name_to_color.keys()):
        c = name_to_color[name]
        legend_elements.append(
            Patch(
                facecolor=c,
                alpha=0.4,
                edgecolor="gray",
                linewidth=0.4,
                label=f"区间: {name}",
            )
        )

    ncol = 2 if len(legend_elements) > 10 else 1
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=ncol,
        fontsize=8,
        frameon=True,
        title="图例（颜色含义）",
        title_fontsize=9,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_prefix_filter(prefixes: Sequence[str]) -> Callable[[str], bool]:
    pfx = tuple(prefixes)

    def filt(name: str) -> bool:
        if not name or name == "[memory]":
            return False
        if name.startswith("ProfilerStep"):
            return False
        return any(name.startswith(p) for p in pfx)

    return filt


def process_trace_file(
    trace_json_path: str,
    output_dir: str,
    span_prefixes: Optional[Sequence[str]] = None,
    require_device_cuda: bool = True,
    basename: Optional[str] = None,
    zoom_from_s: float = DEFAULT_ZOOM_FROM_S,
    zoom_to_s: float = DEFAULT_ZOOM_TO_S,
) -> Tuple[str, str]:
    """
    解析 trace，写 PNG 与峰值摘要 TXT。返回 (png_path, summary_path)。
    """
    trace = load_chrome_trace(trace_json_path)
    events = iter_trace_events(trace)
    samples = extract_cuda_memory_samples(events, require_device_cuda=require_device_cuda)
    if not samples:
        raise ValueError(
            "未解析到任何 [memory] 显存采样。请使用 profile_memory=True 导出 trace；"
            "若仍为空可尝试 --no-device-filter。"
        )
    name_filter = (
        make_prefix_filter(span_prefixes)
        if span_prefixes is not None
        else default_span_name_filter
    )
    spans = extract_duration_spans(events, name_filter=name_filter)
    span_peaks = compute_span_peaks(samples, spans)
    by_name = aggregate_peaks_by_name(span_peaks)

    stem = basename or os.path.splitext(os.path.basename(trace_json_path))[0]
    safe = re.sub(r"[^\w\-.]+", "_", stem)[:120]
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{safe}_memory_spans.png")
    txt_path = os.path.join(output_dir, f"{safe}_span_peaks.txt")

    t0_mem = min(s.ts_us for s in samples) if samples else 0.0

    lines = [
        f"trace: {trace_json_path}",
        f"memory samples: {len(samples)}, spans: {len(spans)}",
        "",
        "--- per span instance (peak in [start,end]) ---",
    ]
    for r in span_peaks:
        lines.append(
            f"{r.span_name}\tpeak={r.peak_gb:.6f} GB\t@ {(r.peak_t_us - t0_mem) / 1e6:.6f}s "
            f"(span {(r.start_us - t0_mem) / 1e6:.6f}s .. {(r.end_us - t0_mem) / 1e6:.6f}s)"
        )
    lines.append("")
    lines.append("--- max peak by span name ---")
    for name in sorted(by_name.keys()):
        b = by_name[name]
        lines.append(f"{name}\t{b['peak_gb']:.6f} GB")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    plot_memory_with_spans(
        samples,
        spans,
        span_peaks,
        png_path,
        title=f"Memory vs spans — {stem}",
        zoom_from_s=zoom_from_s,
        zoom_to_s=zoom_to_s,
    )

    return png_path, txt_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot CUDA memory from PyTorch Chrome trace + record_function spans")
    p.add_argument("trace_json", help="export_chrome_trace 生成的 .json 路径")
    p.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "trace"),
        help="输出目录（默认 csl_hyc/trace）",
    )
    p.add_argument(
        "--prefix",
        nargs="*",
        default=None,
        help="只保留 name 以这些前缀开头的 X 区间；不传则默认仅 shapelets_bw/",
    )
    p.add_argument(
        "--no-device-filter",
        action="store_true",
        help="不强制 Device Type==1（部分 trace 格式不同）",
    )
    p.add_argument(
        "--zoom-from",
        type=float,
        default=DEFAULT_ZOOM_FROM_S,
        metavar="SEC",
        help="下图横轴起始相对时刻（秒），默认 %(default)s",
    )
    p.add_argument(
        "--zoom-to",
        type=float,
        default=DEFAULT_ZOOM_TO_S,
        metavar="SEC",
        help="下图横轴结束相对时刻（秒），默认 %(default)s",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    prefixes = tuple(args.prefix) if args.prefix else None
    png, txt = process_trace_file(
        args.trace_json,
        args.output_dir,
        span_prefixes=prefixes,
        require_device_cuda=not args.no_device_filter,
        zoom_from_s=args.zoom_from,
        zoom_to_s=args.zoom_to,
    )
    print(f"Wrote {png}\n{txt}")


if __name__ == "__main__":
    main()

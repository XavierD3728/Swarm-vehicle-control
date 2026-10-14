"""Publication-style static diagnostics plus a portable interactive trajectory viewer."""
import base64
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

from .geometry import WorldMap, default_map_path
from .scenario import COLORS, CENTERS, REGION_RADII

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15,
                     "figure.facecolor": "#f5f7fb", "axes.facecolor": "white",
                     "savefig.facecolor": "#f5f7fb", "font.size": 10})


def load_result(directory, mode):
    directory = Path(directory)
    metadata = json.loads((directory/f"{mode}_summary.json").read_text(encoding="utf-8"))
    with np.load(directory/f"{mode}_trajectory.npz") as archive:
        return {**metadata, **{key: archive[key] for key in archive.files}}


def draw_map(axis, world, result):
    for center, radius, color in zip(CENTERS, REGION_RADII, COLORS):
        axis.add_patch(Circle(center, radius, color=color, alpha=.06))
    for obstacle in world.obstacles:
        axis.add_patch(Rectangle((obstacle.xmin, obstacle.ymin), obstacle.xmax-obstacle.xmin,
                                 obstacle.ymax-obstacle.ymin, color="#39475d", alpha=.85))
    axis.scatter(*result["targets"].T, c=np.array(COLORS)[result["target_roles"]],
                 marker="x", s=32, linewidths=1.5)
    axis.set(xlim=(0, 12), ylim=(0, 8), aspect="equal", xlabel="x [m]", ylabel="y [m]")


def render(results, output, make_gif=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    world = WorldMap.load(default_map_path())
    count = len(results)
    fig, axes = plt.subplots(1, count, figsize=(7.4*count, 5.8), squeeze=False)
    for ax, result in zip(axes[0], results):
        draw_map(ax, world, result)
        for i, role in enumerate(result["roles"]):
            trajectory = result["position"][:, i]
            ax.plot(*trajectory.T, color=COLORS[role], alpha=.55, lw=1.1)
            ax.add_patch(Circle(trajectory[-1], result["config"]["radius"], facecolor=COLORS[role],
                                edgecolor="white", lw=.8))
        ax.scatter(*result["position"][0].T, s=12, c="#7e8a9f", alpha=.8)
        summary = result["summary"]
        ax.set_title(f'{summary["mode"].upper()}  |  region {summary["region_completion"]:.0%}, slots {summary["slot_completion"]:.0%}',
                     loc="left", fontweight="bold", pad=12)
    fig.suptitle("V6 / Warehouse deployment", x=.06, ha="left", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output/"trajectories.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for result, color in zip(results, ("#3159c7", "#db7a2d")):
        table, mode = result["metrics"], result["summary"]["mode"]
        axes[0, 0].plot(table[:, 0], table[:, 3], color=color, label=mode)
        axes[0, 1].plot(table[:, 0], table[:, 4], color=color, label=mode)
        slot_errors = np.linalg.norm(result["position"]-result["targets"][result["assignment"]], axis=-1)
        times = np.arange(len(slot_errors))*result["config"]["dt"]
        axes[1, 0].plot(times, np.sqrt(np.mean(slot_errors**2, axis=1)), color=color, label=mode)
        axes[1, 1].plot(table[:, 0], np.cumsum(table[:, 7]), color=color, label=mode)
    config = results[0]["config"]
    for ax, physical, requested, title in (
        (axes[0, 0], 2*config["radius"], config["pair_clearance"], "Robot separation / interval lower bound"),
        (axes[0, 1], config["radius"], config["obstacle_clearance"], "Obstacle clearance / interval lower bound"),
    ):
        ax.axhline(physical, color="#c74444", ls="--", label="physical collision threshold")
        ax.axhline(requested, color="#637189", ls=":", label="requested CBF clearance")
        ax.set(title=title, ylabel="distance [m]")
        ax.legend(fontsize=8)
    axes[1, 0].set(title="Distance to current assigned anchors", ylabel="RMS distance [m]")
    axes[1, 1].set(title="Explicit fallback events (robot-steps)", ylabel="cumulative count")
    for ax in axes.flat:
        ax.set_xlabel("time [s]")
    axes[1, 0].legend()
    fig.suptitle("V6 / Safety and task diagnostics", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(output/"safety_diagnostics.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(count, 1, figsize=(12, 2.8*count), squeeze=False)
    for ax, result in zip(axes[:, 0], results):
        ax.imshow(result["assignment"].T, aspect="auto", origin="lower", cmap="tab20",
                  extent=(0, result["config"]["horizon"], -.5, len(result["roles"])-.5),
                  interpolation="nearest")
        ax.set(title=f'{result["summary"]["mode"]}: target identity (role preserved)',
               xlabel="time [s]", ylabel="robot ID")
    fig.tight_layout()
    fig.savefig(output/"assignment_history.png", dpi=180)
    plt.close(fig)
    summaries = [r["summary"] for r in results]
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if make_gif:
        animate(results, world, output)
    write_html(results, world, output)
    write_report(results, output)


def animate(results, world, output):
    fig, axes = plt.subplots(3, len(results), figsize=(7*len(results), 15.5), squeeze=False)
    panels = []
    for column, result in enumerate(results):
        for row in range(3):
            axis = axes[row, column]
            draw_map(axis, world, result)
            agents, ids = [], []
            for i, (point, role) in enumerate(zip(result["position"][0], result["roles"])):
                circle = Circle(point, result["config"]["radius"], color=COLORS[role], zorder=4)
                axis.add_patch(circle)
                agents.append(circle)
                ids.append(axis.text(*point, str(i), ha="center", va="center", color="white", fontsize=6, zorder=6))
            links = LineCollection([], colors="#9aa8b7", linewidths=.55, alpha=.35, zorder=2)
            targets = LineCollection([], colors="#cf8b29", linewidths=.7, linestyles="dashed", alpha=.55, zorder=2)
            axis.add_collection(links)
            axis.add_collection(targets)
            status = axis.text(.02, .98, "", transform=axis.transAxes, va="top", fontsize=8,
                               bbox={"facecolor": "white", "edgecolor": "#d9e0eb", "alpha": .9}, zorder=8)
            cross_layers = {}
            if row == 2:
                styles = [("cross_planning", "#9655b8", ":", .8, .35, "Planning inputs"),
                          ("cross_cbf", "#dd6666", "--", .6, .25, "CBF included"),
                          ("cross_congestion", "#169b69", "-", 1.5, .8, "Congestion"),
                          ("cross_yield", "#f09920", "-.", 2.0, 1., "Yield request"),
                          ("cross_cbf_binding", "#c71d32", "-", 2.5, 1., "CBF binding")]
                handles = []
                for name, color, style, width, alpha, title in styles:
                    if name not in result:
                        raise ValueError("Cross-role telemetry missing; rerun simulation before rendering")
                    layer = LineCollection([], colors=color, linestyles=style, linewidths=width, alpha=alpha, zorder=3)
                    axis.add_collection(layer)
                    cross_layers[name] = layer
                    handles.append(Line2D([0], [0], color=color, linestyle=style, linewidth=width, label=title))
                axis.legend(handles=handles, loc="lower left", fontsize=7, ncol=2, framealpha=.95)
            panels.append(dict(result=result, axis=axis, row=row, agents=agents, ids=ids, cross_layers=cross_layers,
                               links=links, targets=targets, status=status, arrows=None))
    total = max(len(r["position"]) for r in results)
    frames = np.unique(np.linspace(0, total-1, min(240, total)).astype(int))
    def update(frame):
        for panel in panels:
            result = panel["result"]
            step = min(frame, len(result["position"])-1)
            points = result["position"][step]
            for circle, label, point in zip(panel["agents"], panel["ids"], points):
                circle.center = point
                label.set_position(point)
            metric_index = max(0, min(step-1, len(result["metrics"])-1))
            metric = result["metrics"][metric_index]
            error = np.linalg.norm(points-result["targets"][result["assignment"][step]], axis=1)
            done = np.count_nonzero(error <= result["config"]["slot_tolerance"])
            if panel["row"] == 0:
                # adjacency[receiver, sender]: arrows follow information flow.
                adjacency = result["adjacency"][min(step, len(result["adjacency"])-1)]
                receiver, sender = np.nonzero(adjacency)
                start = points[sender]
                vector = points[receiver]-start
                distance = np.linalg.norm(vector, axis=1)
                unit = vector/np.maximum(distance[:, None], 1e-12)
                radius = result["config"]["radius"]
                start = start+radius*unit
                vector = unit*np.maximum(distance-2*radius, 0)[:, None]
                if panel["arrows"] is not None:
                    panel["arrows"].remove()
                panel["arrows"] = panel["axis"].quiver(start[:,0], start[:,1], vector[:,0], vector[:,1],
                    angles="xy", scale_units="xy", scale=1, color="#427ec2", alpha=.45,
                    width=.002, headwidth=4, headlength=5, zorder=3)
                panel["status"].set_text(f"Slots {done}/{len(points)} | directed edges {len(sender)}")
                kind = "Coordination: sender -> receiver"
            elif panel["row"] == 1:
                distance = np.linalg.norm(points[:,None]-points[None], axis=-1)
                first, second = np.nonzero(np.triu(distance <= result["config"]["sensing_radius"], 1))
                panel["links"].set_segments(np.stack((points[first], points[second]), axis=1))
                panel["targets"].set_segments(np.stack((points, result["targets"][result["assignment"][step]]), axis=1))
                panel["status"].set_text(f"Sensing pairs {len(first)} | pair >= {metric[3]:.2f} m\nObstacle >= {metric[4]:.2f} m | fallback {int(result['metrics'][:metric_index+1,7].sum())}")
                kind = "Sensing (gray) / assigned targets (dashed)"
            else:
                counts = {}
                for name, layer in panel["cross_layers"].items():
                    matrix = result[name][min(step, len(result[name])-1)]
                    first, second = np.nonzero(np.triu(matrix | matrix.T, 1))
                    layer.set_segments(np.stack((points[first], points[second]), axis=1))
                    counts[name] = len(first)
                panel["status"].set_text(f"Cross-role pairs (either direction)\nCongestion {counts['cross_congestion']} | planning {counts['cross_planning']} | yield {counts['cross_yield']}\nCBF included {counts['cross_cbf']} | binding {counts['cross_cbf_binding']}")
                kind = "Cross-role: guidance / planning & yielding / CBF"
            panel["axis"].set_title(f'{result["summary"]["mode"]} | t={step*result["config"]["dt"]:.2f}s\n{kind}', fontsize=10)
    if len(results) == 2:
        first, second = (r["summary"] for r in results)
        reduction = 1-second["total_messages"]/first["total_messages"]
        fig.suptitle(f"Full-run communication: {first['total_messages']:,} vs {second['total_messages']:,} sends ({reduction:.1%} fewer)\n"
                     "Columns: centralized / hybrid | Rows: coordination / sensing & targets / cross-role layers",
                     fontsize=12, y=.995)
    # Reserve space for two-line titles updated after layout initialization.
    fig.subplots_adjust(left=.06, right=.98, bottom=.045, top=.92, hspace=.43, wspace=.18)
    animation = FuncAnimation(fig, update, frames=frames, interval=80)
    animation.save(output/"warehouse.gif", writer=PillowWriter(fps=12), dpi=100)
    plt.close(fig)


def write_report(results, output):
    lines = ["# V6 validation results", "", "Generated from saved trajectories; no V5 results are presented as reruns.", "",
             "| Mode | Region completion | Slot completion | Pair lower bound | Obstacle lower bound | Fallbacks | Messages |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        s = result["summary"]
        lines.append(f'| {s["mode"]} | {s["region_completion"]:.1%} | {s["slot_completion"]:.1%} | {s["pair_interval_lower"]:.4f} m | {s["obstacle_interval_lower"]:.4f} m | {s["fallback_count"]} | {s["total_messages"]} |')
    lines += ["", "## Interpretation", "",
              "- Pair and obstacle lower bounds cover intervals using exact held-input dynamics and a speed-based Lipschitz bound.",
              "- Physical collision limits and requested CBF margins are distinct; both are shown in the figures and JSON.",
              "- Any fallback is explicitly logged. Braking after infeasibility is not a formal recovery certificate.",
              "- Safety observations are fresh, ideal and range-limited. Packet loss/delay applies to coordination messages only.",
              "- Simulation is single-process with local information interfaces, not a deployed ROS network.",
              "- Static known map; no guarantee for unmodeled moving obstacles, sensor occlusion or actuator disturbance.",
              "- Communication counts are actual simulated directed sends, not network bytes. Runtime is serial Python wall time.",
              "- Global probability-density estimation is not claimed: local KDE intensity and finite-support congestion are implemented.",
              "", "## Files", "", "Open index.html for playback; PNG, GIF, NPZ, CSV and JSON are standalone artifacts."]
    (output/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def write_html(results, world, output):
    payload = []
    for result in results:
        stride = max(1, len(result["position"])//450)
        indices = np.arange(len(result["position"]))
        payload.append(dict(sensing_radius=result["config"]["sensing_radius"], mode=result["summary"]["mode"], summary=result["summary"],
                            time=(indices*result["config"]["dt"]).tolist(),
                            position=np.round(result["position"][indices], 4).tolist(),
                            heading=np.round(result["heading"][indices], 4).tolist(),
                            assignment=result["assignment"][indices].tolist(),
                            roles=result["roles"].tolist(), targets=result["targets"].tolist(),
                            adjacency=result["adjacency"][np.minimum(indices, len(result["adjacency"])-1)].astype(int).tolist()))
    data = json.dumps(dict(runs=payload, obstacles=[vars(o) for o in world.obstacles]), separators=(",", ":"))
    html = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>V6 仓储协同控制</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fb;color:#1c2d46;font:15px/1.65 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:36px 24px}h1{font-size:32px;margin:4px 0}small,.muted{color:#627189}
.tag{font:12px monospace;letter-spacing:2px;color:#4364ab}.toolbar,.card{background:white;border:1px solid #dee5ef;border-radius:14px;padding:20px;margin:18px 0}
.toolbar{display:flex;gap:20px;align-items:center;flex-wrap:wrap}button{background:#3159c7;color:white;border:0;border-radius:8px;padding:10px 20px;cursor:pointer}
input[type=range]{flex:1;min-width:180px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}
canvas{width:100%;background:white;border-radius:12px}h2{font-size:19px;margin:0 0 10px}.stat{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stat b{font-size:22px;display:block}
img{max-width:100%;border-radius:12px}a{color:#3159c7}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #eee}
.note{border-left:3px solid #db7a2d;padding-left:14px;color:#58667b}footer{margin:30px 0;color:#65748a;font-size:13px}
@media(max-width:600px){main{padding:16px}.grid{grid-template-columns:1fr}h1{font-size:25px}}
</style><main><div class="tag">TAC CONTROL / VERSION 6.0</div>
<h1>局部协同 · 可执行安全控制</h1><p class="muted">低频目标匹配 → 本地速度场 → 加速度 / 转向安全优化 → 差速运动</p>
<div class="toolbar"><button id="play">播放</button><input id="slider" type="range" min="0" value="0"><b id="time">0.0 s</b>
<label><input id="links" type="checkbox" checked> 协同通信</label><label><input id="sensing" type="checkbox"> 安全感知</label><button id="previous"> 上一步</button><button id="next"> 下一步</button><label><input id="targets" type="checkbox"> 目标匹配</label></div>
<div class="grid" id="runs"></div>
<div class="card"><h2>安全与任务指标</h2><img src="safety_diagnostics.png" alt="安全指标图">
<p class="note">实线为采样间隔距离下界。红色虚线表示物理碰撞阈值，灰色点线表示请求的 CBF 裕量。制动降级事件如实记录，不等价于严格的安全恢复证明。</p></div>
<div class="card"><h2>目标标签变化</h2><img src="assignment_history.png" alt="目标分配历史"></div>
<div class="card"><h2>结果文件</h2><a href="summary.csv">指标 CSV</a> · <a href="REPORT.md">验证报告</a> · <a href="trajectories.png">轨迹图</a> · <a href="warehouse.gif">动画 GIF</a></div>
<footer>本地接口上的单进程仿真；静态地图、理想实时范围传感器。通信延迟和丢包影响协同消息。未宣称完成真实多机网络部署或全局分布一致性估计。</footer>
<script>const DATA=__DATA__;const colors=['#dd8452','#168f82','#526acb'];
const runs=document.querySelector('#runs'),slider=document.querySelector('#slider');slider.max=Math.max(...DATA.runs.map(r=>r.position.length))-1;
DATA.runs.forEach((r,k)=>{const s=r.summary;const d=document.createElement('section');d.className='card';d.innerHTML='<h2>'+r.mode.toUpperCase()+'</h2><div class="stat"><div><small>区域完成率</small><b>'+Math.round(s.region_completion*100)+'%</b></div><div><small>精确槽位完成率</small><b>'+Math.round(s.slot_completion*100)+'%</b></div><div><small>降级次数</small><b>'+s.fallback_count+'</b></div></div><canvas id="c'+k+'" width="720" height="480"></canvas><small>蓝绿 / 橙 / 紫表示固定角色；连线和目标归属可以变化。</small>';runs.appendChild(d)});
let playing=false,timer;document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放';if(playing)timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%(Number(slider.max)+1);draw()},60);else clearInterval(timer)};
document.querySelector('#previous').onclick=()=>{slider.value=Math.max(0,Number(slider.value)-1);draw()};document.querySelector('#next').onclick=()=>{slider.value=Math.min(Number(slider.max),Number(slider.value)+1);draw()};document.querySelector('#sensing').onchange=draw;slider.oninput=draw;document.querySelector('#links').onchange=draw;document.querySelector('#targets').onchange=draw;
function draw(){const index=Number(slider.value);DATA.runs.forEach((r,k)=>{const i=Math.min(index,r.position.length-1),p=r.position[i];const c=document.querySelector('#c'+k),ctx=c.getContext('2d');const X=x=>x*60,Y=y=>480-y*60;
ctx.clearRect(0,0,720,480);ctx.strokeStyle='#edf0f5';ctx.lineWidth=1;for(let x=0;x<=12;x++){ctx.beginPath();ctx.moveTo(X(x),0);ctx.lineTo(X(x),480);ctx.stroke()}for(let y=0;y<=8;y++){ctx.beginPath();ctx.moveTo(0,Y(y));ctx.lineTo(720,Y(y));ctx.stroke()}
DATA.obstacles.forEach(o=>{ctx.fillStyle='#39475d';ctx.fillRect(X(o.xmin),Y(o.ymax),60*(o.xmax-o.xmin),60*(o.ymax-o.ymin))});
r.targets.forEach(t=>{ctx.strokeStyle='#758298';ctx.beginPath();ctx.moveTo(X(t[0])-4,Y(t[1])-4);ctx.lineTo(X(t[0])+4,Y(t[1])+4);ctx.moveTo(X(t[0])+4,Y(t[1])-4);ctx.lineTo(X(t[0])-4,Y(t[1])+4);ctx.stroke()});
if(document.querySelector('#sensing').checked)p.forEach((v,j)=>p.forEach((w,l)=>{if(l>j&&Math.hypot(v[0]-w[0],v[1]-w[1])<=r.sensing_radius){ctx.strokeStyle='#aaa';ctx.globalAlpha=.3;ctx.beginPath();ctx.moveTo(X(v[0]),Y(v[1]));ctx.lineTo(X(w[0]),Y(w[1]));ctx.stroke();ctx.globalAlpha=1}}));p.forEach((v,j)=>{ctx.strokeStyle=colors[r.roles[j]];ctx.globalAlpha=.25;ctx.beginPath();for(let t=Math.max(0,i-30);t<=i;t++){const a=r.position[t][j];t===Math.max(0,i-30)?ctx.moveTo(X(a[0]),Y(a[1])):ctx.lineTo(X(a[0]),Y(a[1]))}ctx.stroke();ctx.globalAlpha=1;
if(document.querySelector('#links').checked)r.adjacency[i][j].forEach((on,l)=>{if(on){const sx=X(p[l][0]),sy=Y(p[l][1]),dx=X(v[0])-sx,dy=Y(v[1])-sy,angle=Math.atan2(dy,dx),ex=X(v[0])-12*Math.cos(angle),ey=Y(v[1])-12*Math.sin(angle);ctx.strokeStyle='#427ec2';ctx.globalAlpha=.55;ctx.beginPath();ctx.moveTo(sx+12*Math.cos(angle),sy+12*Math.sin(angle));ctx.lineTo(ex,ey);ctx.lineTo(ex-7*Math.cos(angle-.5),ey-7*Math.sin(angle-.5));ctx.moveTo(ex,ey);ctx.lineTo(ex-7*Math.cos(angle+.5),ey-7*Math.sin(angle+.5));ctx.stroke();ctx.globalAlpha=1}});
if(document.querySelector('#targets').checked){const t=r.targets[r.assignment[i][j]];ctx.setLineDash([3,5]);ctx.strokeStyle=colors[r.roles[j]];ctx.globalAlpha=.3;ctx.beginPath();ctx.moveTo(X(v[0]),Y(v[1]));ctx.lineTo(X(t[0]),Y(t[1]));ctx.stroke();ctx.globalAlpha=1;ctx.setLineDash([])}
ctx.fillStyle=colors[r.roles[j]];ctx.beginPath();ctx.arc(X(v[0]),Y(v[1]),10.8,0,2*Math.PI);ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(X(v[0]),Y(v[1]));ctx.lineTo(X(v[0]) + 9*Math.cos(r.heading[i][j]),Y(v[1])-9*Math.sin(r.heading[i][j]));ctx.stroke();ctx.lineWidth=1;ctx.fillStyle='#24364f';ctx.font='10px sans-serif';ctx.fillText(j,X(v[0])+10,Y(v[1])-10)});
document.querySelector('#time').textContent=r.time[i].toFixed(2)+' s'})}draw();</script></main></html>"""
    if (output/"mode_comparison.png").exists():
        html = html.replace("<footer>", '<div class="card"><h2>集中式与混合式：同场景对比</h2><img src="mode_comparison.png" alt="通信、到位、路径与耗时对比"><p>消息数量是模拟发送数；运行时间是单进程耗时，不代表多机并行性能。<a href="MODE_COMPARISON.md">查看指标说明</a></p></div><footer>')
    if (output/"network_layers.png").exists():
        html = html.replace("<footer>", '<div class="card"><h2>三层跨角色关系</h2><p>完整动态关系见 <a href="warehouse.gif">3×2 动图</a>。绿色：拥挤；紫色：规划观测与记忆；橙色：让行请求；淡红：纳入的 CBF 约束；粗红：余量接近零的紧约束。下图为 30 秒快照。</p><img src="network_layers.png" alt="六面板网络快照"></div><footer>')
    (output/"index.html").write_text(html.replace("__DATA__", data), encoding="utf-8")


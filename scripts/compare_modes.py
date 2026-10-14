"""Compare both V6 modes under the saved common scenario; no V5 claims."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from amr_control.visualization import load_result

p=Path('output/validation')
rows=[]
for mode in ('centralized','hybrid'):
 r=load_result(p,mode);s=r['summary'];c=r['config']
 error=np.linalg.norm(r['position']-r['targets'][r['assignment']],axis=-1)
 all_done=np.all(error<=c['slot_tolerance'],axis=1)
 sustained=np.flatnonzero(np.logical_and.accumulate(all_done[::-1])[::-1])
 networks={}
 for name in ('cross_congestion','cross_planning','cross_yield','cross_cbf','cross_cbf_binding'):
  matrix=r[name];assert matrix.shape==(c['steps'] if 'steps' in c else round(c['horizon']/c['dt']),c['robots'],c['robots'])
  assert not np.any(matrix[:,r['roles'][:,None]==r['roles'][None]])
  networks[name]={'directed_edge_steps':int(matrix.sum()),'steps_present':int(np.any(matrix,axis=(1,2)).sum())}
 rows.append(dict(mode=mode,messages=s['total_messages'],arrival_seconds=float(sustained[0]*c['dt']) if len(sustained) else None,
  total_path_m=float(np.linalg.norm(np.diff(r['position'],axis=0),axis=-1).sum()),
  pending_stopped_robot_seconds=float(np.sum((r['speed']<.08)&(error>c['slot_tolerance']))*c['dt']),
  runtime_seconds=s['runtime_seconds'],max_constraints=s['max_constraints'],constraint_scope='joint' if mode=='centralized' else 'per robot',
  slot_completion=s['slot_completion'],fallbacks=s['fallback_count'],networks=networks))
(p/'MODE_COMPARISON.json').write_text(json.dumps(rows,indent=2))
fig,axes=plt.subplots(2,2,figsize=(10,7))
for ax,(key,title) in zip(axes.flat,[('messages','Simulated directed sends'),('arrival_seconds','Sustained arrival [s]'),('total_path_m','Fleet path length [m]'),('runtime_seconds','Serial simulation wall time [s]')]):
 values=[r[key] for r in rows];bars=ax.bar(['Centralized','Hybrid'],values,color=['#3159c7','#db7a2d']);ax.bar_label(bars,fmt='%.2f');ax.set_title(title);ax.set_ylim(0,max(values)*1.2)
fig.suptitle('Same V6 scenario: 24 robots, 60 s, unchanged safety margins')
fig.tight_layout();fig.savefig(p/'mode_comparison.png',dpi=150);plt.close(fig)
lines=['# Hybrid comparison','', 'Same V6 scenario; not a controlled V5 comparison. Serial runtime includes telemetry and audit, not hardware parallel deployment.','', '| Metric | Centralized | Hybrid |','|---|---:|---:|']
for key in ('messages','arrival_seconds','total_path_m','pending_stopped_robot_seconds','runtime_seconds','max_constraints'):
 lines.append(f"| {key} | {rows[0][key]} | {rows[1][key]} |")
lines += ['', 'Constraint counts refer to a joint problem versus one robot problem and are not total work counts.', 'Cross-role planning edges mean observed/remembered obstacles supplied to navigation, not proven causal detours. CBF binding means absolute slack <= 1e-5, not a positive KKT multiplier.', 'Third-row lines display unordered pairs when either direction exists; directed matrices are preserved in NPZ. Last frame holds the final control-step networks.']
(p/'MODE_COMPARISON.md').write_text('\n'.join(lines))
print(json.dumps(rows,indent=2))

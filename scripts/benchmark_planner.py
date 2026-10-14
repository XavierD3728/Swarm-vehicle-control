"""Paired planner benchmark on saved states; verifies routes before timing claims."""
import inspect
import json
import time
import textwrap
from pathlib import Path
import numpy as np
from amr_control.local_planner import LocalRoutePlanner
from amr_control import local_planner
from amr_control.config import Config
from amr_control.communication import LocalSensor, LocalView
from amr_control.dynamics import RobotState
from amr_control.geometry import WorldMap, default_map_path
from amr_control.navigation import NavigationField

root=Path('output/validation')
meta=json.loads((root/'hybrid_summary.json').read_text())
config=Config(**meta['config'])
a=np.load(root/'hybrid_trajectory.npz')
world=WorldMap.load(default_map_path())
nav=NavigationField(world,a['targets'],a['target_roles'],config.grid_resolution,config.navigation_inflation,config.obstacle_clearance)
source=textwrap.dedent(inspect.getsource(LocalRoutePlanner.plan))
source=source.replace('    best_candidate = (float("inf"), len(occupied))\n','')
source=source.replace('        candidate = (costs[node]+distance, node)\n','').replace('        # Exact branch-and-bound: a worse cost cannot win, regardless of visibility.\n','')
source=source.replace(' and candidate < best_candidate','').replace('            candidates.append(candidate)\n            best_candidate = candidate','            candidates.append((costs[node]+distance, node))')
namespace=dict(vars(local_planner));exec(source,namespace)
old=namespace['plan']
new=LocalRoutePlanner.plan
cases=[]
for step in (0,200,400,600,800):
 states=[RobotState(p,h,s) for p,h,s in zip(a['position'][step],a['heading'][step],a['speed'][step])]
 observations=LocalSensor(config.sensing_radius).observe(states,step*config.dt)
 for i in range(0,config.robots,3):
  cases.append((LocalView(i,states[i],observations[i],()),a['targets'][a['assignment'][step,i]],step*config.dt))
timings={'original':[],'optimized':[]}
for repeat in range(3):
 paths={}
 for name,method in ([('original',old),('optimized',new)] if repeat%2==0 else [('optimized',new),('original',old)]):
  planner=LocalRoutePlanner(nav,config)
  start=time.perf_counter(); routes=[]
  for view,target,now in cases:
   method(planner,view,target,now);routes.append(planner.path.copy())
  timings[name].append(time.perf_counter()-start);paths[name]=routes
 assert all(np.array_equal(x,y) for x,y in zip(paths['original'],paths['optimized']))
result={'cases':len(cases),'repeats':3,'identical_routes':True,'seconds':timings,'median_speedup':float(np.median(timings['original'])/np.median(timings['optimized']))}
Path('output/optimized').mkdir(parents=True,exist_ok=True)
Path('output/optimized/PLANNER_BENCHMARK.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

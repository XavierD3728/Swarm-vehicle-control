"""Telemetry distinguishes role filtering, range, memory, yielding and binding."""
from types import SimpleNamespace
import numpy as np
from amr_control.communication import LocalView, Observation
from amr_control.config import Config
from amr_control.dynamics import RobotState
from amr_control.interaction import cross_role_networks
from amr_control.safety import reciprocal_allocation


def test_cross_role_telemetry():
    config=Config()
    states=[RobotState(np.array([x, 2.]),0.,0.) for x in (1.,2.,2.4)]
    roles=np.array([0,1,0])
    views=[LocalView(i,state,tuple(Observation(j,other,0.) for j,other in enumerate(states) if i!=j),()) for i,state in enumerate(states)]
    planners=[SimpleNamespace(planning_neighbors=tuple(j for j in range(3) if j!=i),yield_requester=1 if i==0 else None,yielding=lambda now:True) for i in range(3)]
    controls=np.zeros((3,2))
    row,bound=reciprocal_allocation(states[0],states[1],config)
    controls[0]=row*bound/(row@row)
    matrices=cross_role_networks(views,planners,roles,controls,config,'hybrid',0.)
    assert matrices['cross_cbf_binding'][0,1]
    assert matrices['cross_yield'][0,1] and not matrices['cross_yield'][1,0]
    for matrix in matrices.values():
        assert not matrix[0,2] and not matrix[2,0]
        assert not np.any(np.diag(matrix))
    assert matrices['cross_congestion'][0,1]
    # Navigation may retain a neighbor that the safety sensor no longer sees.
    views[0]=LocalView(0,states[0],(),())
    matrices=cross_role_networks(views,planners,roles,controls,config,'hybrid',0.)
    assert matrices['cross_planning'][0,1]
    assert not matrices['cross_cbf'][0,1] and not matrices['cross_congestion'][0,1]

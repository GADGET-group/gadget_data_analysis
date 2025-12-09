import numpy as np
import matplotlib.pyplot as plt

from raw_viewer import process_runs

e23035_run148 = np.array([
    1,0,0,0,0,0,0,0,0,0,
    1,0,0,0,1,0,1,0,0,0,
    1,0,0,0,1,1,0,0,0,0,
    1,0,1,0,0,1,0,1,0,0,
    0,0,1,0,1,0,1,0,0,0,
    0,1,0,0,1,1,0,1,1,0,
    0,1,1,0,0,0,0
    ])

outer_ring_max_counts = process_runs.get_outer_ring_max_counts('e23035', [148])[:len(e23035_run148)]
max_veto_counts = process_runs.get_max_veto_counts('e23035', [148])[:len(e23035_run148)]
track_width = process_runs.get_quantity('charge_width', 'e23035', [148])[:len(e23035_run148)]

plt.figure()
plt.title('should veto')
plt.scatter(max_veto_counts[e23035_run148==1], outer_ring_max_counts[e23035_run148==1], c=track_width[e23035_run148==1], marker='.')
plt.colorbar()
plt.xlabel('single veto pad max')
plt.ylabel('single pad outer ring max')

plt.figure()
plt.title('should not veto')
plt.scatter(max_veto_counts[e23035_run148==0], outer_ring_max_counts[e23035_run148==0], c=track_width[e23035_run148==0], marker='.')
plt.colorbar()
plt.xlabel('single veto pad max')
plt.ylabel('single pad outer ring max')

plt.figure()
plt.scatter(max_veto_counts, outer_ring_max_counts, c=e23035_run148==0, marker='.')
plt.colorbar()
plt.xlabel('single veto pad max')
plt.ylabel('single pad outer ring max')

outer_ring_total = process_runs.get_outer_ring_counts('e23035', [148])[:len(e23035_run148)]
veto_total = process_runs.get_veto_counts('e23035', [148])[:len(e23035_run148)]
plt.figure()
plt.title('should veto')
plt.scatter(veto_total[e23035_run148==1], outer_ring_total[e23035_run148==1], c=track_width[e23035_run148==1], marker='.')
plt.colorbar()
plt.xlabel('total veto pad charge')
plt.ylabel('total outer ring pad charge')

plt.figure()
plt.title('should not veto')
plt.scatter(veto_total[e23035_run148==0], outer_ring_total[e23035_run148==0], c=track_width[e23035_run148==0], marker='.')
plt.colorbar()
plt.xlabel('total veto pad charge')
plt.ylabel('total outer ring pad charge')

plt.figure()
plt.scatter(veto_total, outer_ring_total, c=e23035_run148==0, marker='.')
plt.colorbar()
plt.xlabel('total veto pad charge')
plt.ylabel('total outer ring pad charge')

plt.show(block=False)
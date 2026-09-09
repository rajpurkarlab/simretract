# Simulation results

`simulation_first_run.json` contains the first recorded result for each of the
50 presentations in [`test_core50.json`](../selections/test_core50.json), for
the eight configurations in the paper's main simulation table.
`simulation_summary.csv` contains the corresponding aggregate results.

Grasped means **secured after release**: the gripper completed its close and
the instrument dropped by less than 20 mm when the simulated operator released
it. Timeouts count as failures. Offset and twist are measured only over these
secured trials and reported as mean, population standard deviation (`ddof=0`),
and median, rounded to one decimal. Offset is in millimetres and twist in degrees.

The JSON uses the training preset names as model keys. `verdict.grasp_t_mm` is
the tip offset and `verdict.grasp_twist_deg` is the axial rotation-vector error.
`held` and `verdict.grasp_r_deg` preserve the additional load-retention and
rotation fields used by the scorer; they do not define the paper's grasped rate.
Retry outcomes are excluded.

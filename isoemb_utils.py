"""Local helpers for IsoEmbRate.ipynb.

Nodal interpolation and exact-surface metric history are copied from the
project helpers; no other files from the original project are required.
"""

# ---- surface_lagrange.py ----
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from ngsolve import BND, CF, ET, IntegrationRule, x, y, z


def interpolate_surface_lagrange(fes, evaluate, order):
    """Return (coefficients, info) for a continuous field on a flat surface.

    ``fes`` is a scalar, uniform degree-``order`` H1 space on a triangular
    surface mesh in R^3, without volume elements or deformation. ``evaluate``
    accepts physical points of shape (N, 3) and returns finite scalar (N,)
    or vector (N, d) values. The coefficient array has shape (fes.ndof, d).

    On every reference triangle use nodes (i/k, j/k), i+j <= k, including
    vertices, edge nodes and interior nodes. Solve V c = f(nodes), where V
    contains the *oriented* NGSolve basis values. Shared coefficients agree
    for a continuous field; their averaging only reconciles roundoff. A
    final nodal check rejects incompatible element traces.

    This is point-value interpolation, not an L2 or moment projection.
    No quadrature rule used elsewhere in the solver enters this operation.
    """
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError('order must be a positive integer')
    mesh = fes.mesh
    if (mesh.dim != 3 or mesh.ne != 0 or mesh.GetCurveOrder() != 1
            or mesh.deformation is not None):
        raise ValueError('expected an undeformed, piecewise-flat surface mesh in R^3')
    if fes.type != 'h1ho' or fes.dim != 1:
        raise ValueError('fes must be a scalar H1 space')

    nodes = [(i / order, j / order)
             for i in range(order + 1) for j in range(order + 1 - i)]
    # Weights are unused: the rule is only a container for evaluation points.
    ir = IntegrationRule(nodes, [1.0] * len(nodes))
    coords = CF((x, y, z))
    elements = []
    physical_points = []
    matrices = {}
    for element in mesh.Elements(BND):
        if element.type != ET.TRIG:
            raise ValueError('only triangular surface elements are supported')
        fe = fes.GetFE(element)
        dofs = np.asarray(fes.GetDofNrs(element), dtype=int)
        if fe.order != order or fe.ndof != len(nodes) or np.any(dofs < 0):
            raise ValueError('expected a uniform, unconstrained local P_k space')
        matrix = np.array([fe.CalcShape(u, v).NumPy().copy() for u, v in nodes])
        # The actual matrix captures edge/face orientation; no assumption
        # about global DOF ordering or vertex permutations is needed.
        key = matrix.tobytes()
        if key not in matrices:
            matrices[key] = (matrix, lu_factor(matrix))
        elements.append((dofs, key))
        physical_points.append(np.asarray(coords(mesh.GetTrafo(element)(ir))))
    if not elements:
        raise ValueError('the surface mesh is empty')

    values = np.asarray(evaluate(np.concatenate(physical_points)), dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    npoints = len(elements) * len(nodes)
    if (values.ndim != 2 or values.shape[0] != npoints or values.shape[1] == 0
            or not np.isfinite(values).all()):
        raise ValueError('evaluate must return finite (N,) or (N, d) values')
    values = values.reshape(len(elements), len(nodes), -1)
    coefficients = np.zeros((fes.ndof, values.shape[2]))
    counts = np.zeros(fes.ndof, dtype=int)
    for (dofs, key), local_values in zip(elements, values):
        local_coefficients = lu_solve(matrices[key][1], local_values)
        np.add.at(coefficients, dofs, local_coefficients)
        np.add.at(counts, dofs, 1)
    if np.any(counts == 0):
        raise ValueError('fes contains DOFs outside the triangular surface')
    coefficients /= counts[:, None]

    max_error = max(float(np.max(np.abs(matrices[key][0] @ coefficients[dofs] - local_values)))
                    for (dofs, key), local_values in zip(elements, values))
    scale = max(1.0, float(np.max(np.abs(values))))
    if not np.isfinite(coefficients).all() or max_error > 1e-11 * scale:
        raise RuntimeError('nodal interpolation failed: inconsistent traces or ill-conditioned basis')
    return coefficients, {
        'sample_points': npoints,
        'max_nodal_error': max_error,
    }


# ---- metric_history.py ----
import csv
import math
from pathlib import Path


def summarize_metric_history(rows, tau, T, n_steps, initial_error, final_error):
    """Reject incomplete histories; never substitute endpoint errors for a max."""
    if not isinstance(rows, list) or len(rows) != n_steps + 1 or not rows:
        raise ValueError('metric history must contain all steps 0,...,N_steps')
    for n, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {'step', 'time', 'metricerr'}:
            raise ValueError('invalid metric history row')
        for key in ('step', 'time', 'metricerr'):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('metric history requires finite numeric values')
        expected_time = T if n == n_steps else n * tau
        if (row['step'] != n or row['metricerr'] < 0
                or not math.isclose(row['time'], expected_time, rel_tol=1e-12, abs_tol=1e-14)):
            raise ValueError('metric history has missing steps or invalid times/errors')
    for measured, expected in ((rows[0]['metricerr'], initial_error),
                               (rows[-1]['metricerr'], final_error)):
        if expected is None or not math.isclose(measured, expected, rel_tol=1e-9, abs_tol=1e-13):
            raise ValueError('metric history disagrees with initial/final exact metric error')
    peak = max(rows, key=lambda row: row['metricerr'])  # First time in case of ties.
    return dict(max_metricerr=peak['metricerr'], max_metricerr_time=peak['time'],
                metric_samples=len(rows))


class ExactMetricHistory:
    """Evaluate against analytic J(t)=J0+t*Jv with the fixed exact metric J0^T J0.

    The supplied Jacobians must come from the analytical closest-point lift,
    independently of the finite-element reference used by the solver RHS.
    """
    def __init__(self, mesh, space, J0, Jv, path, order=40):
        import ngsolve as ng
        self.mesh, self.path, self.order = mesh, Path(path), order
        self.rows = []
        self.field = ng.GridFunction(space)
        self.time = ng.Parameter(0.)
        nh = ng.specialcf.normal(3)
        N = ng.OuterProduct(nh, nh)
        P = ng.Id(3) - N
        G0 = J0.trans * J0
        H = P * ng.Inv(G0 + N) * P
        mu = ng.sqrt(ng.Det(G0 + N))
        Jr = J0 + self.time * Jv
        Je = ng.grad(self.field).Trace() - Jr
        defect = Jr.trans*Je + Je.trans*Jr + Je.trans*Je
        self.density = ng.Trace(H*defect*H*defect)*mu

    def record(self, step, time, field):
        import ngsolve as ng
        if step != len(self.rows) or not math.isfinite(time) or time < 0:
            raise ValueError('record metric history in order, starting at step 0')
        if self.rows and time <= self.rows[-1]['time']:
            raise ValueError('metric history times must increase')
        if self.mesh.deformation is not None:
            raise ValueError('measure metric history on the undeformed reference mesh')
        self.field.vec.data = field.vec
        self.time.Set(time)
        squared = float(ng.Integrate(self.density, self.mesh,
                                    definedon=self.mesh.Boundaries('.*'), order=self.order))
        if not math.isfinite(squared) or squared < 0:
            raise ValueError('metric defect is not finite/nonnegative')
        row = dict(step=int(step), time=float(time), metricerr=math.sqrt(squared))
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['step', 'time', 'metricerr'])
            writer.writeheader()
            writer.writerows(self.rows)
        temporary.replace(self.path)
        return row['metricerr']


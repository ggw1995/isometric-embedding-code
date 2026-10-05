"""Nodal Lagrange interpolation in NGSolve's hierarchical surface H1 basis."""

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

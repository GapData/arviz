"""Stats-utility functions for ArviZ"""
import numpy as np

__all__ = ["make_ufunc", "logsumexp"]

def make_ufunc(func, n_dims=2, n_output=1, index=Ellipsis, ravel=True, out=None, args=None, **kwargs):  # noqa: D202
    """Make ufunc from a function.

    Parameters
    ----------
    func : callable
    n_dims : int, optional
        Number of core dimensions not broadcasted. Dimensions are skipped from the end.
        At minimum n_dims > 0.
    n_output : int, optional
        Select number of results returned by `func`.
        If n_output > 1, ufunc returns a tuple of objects else returns an object.
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.
    out : ndarray or tuple of ndarray, optional
        Alternative output location. If defined, ufunc will return None.
    args : tuple, optional
        Arguments for `func`.
    **kwargs
        Other parameters are inserted to `func`.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims < 1:
        raise TypeError("n_dims must be one or higher.")

    def _ufunc(ary):
        """General ufunc for single-output function."""
        target = np.empty(ary.shape[:-n_dims])
        for idx in np.ndindex(target.shape):
            ary_idx = ary[idx].ravel() if ravel else ary[idx]
            target[idx] = np.asarray(func(ary_idx, **kwargs))[index]
        return target

    def _multi_ufunc(ary):
        """General ufunc for multi-output function."""
        targets = tuple(np.empty(ary.shape[:-2]) for _ in range(n_output))
        for idx in np.ndindex(ary.shape[:-2]):
            ary_idx = ary[idx].ravel() if ravel else ary[idx]
            results = func(ary_idx, **kwargs)
            for i, res in enumerate(results):
                targets[i][idx] = np.asarray(res)[index]
        return targets
    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc

    update_docstring(ufunc, func, n_output)
    return ufunc


def update_docstring(ufunc, func, n_output=1):
    """Update ArviZ generated ufunc docstring."""
    module = ""
    name = ""
    docstring = ""
    if hasattr(func, "__module__"):
        module += func.__module__
    if hasattr(func, "__name__"):
        name += func.__name__
    if hasattr(func, "__doc__"):
        docstring += func.__doc__
    ufunc.__doc__ += "\n\n"
    if module or name:
        ufunc.__doc__ += "This function is a thin ufunc wrapper for "
        ufunc.__doc__ += func.__name__ + "."
        ufunc.__doc__ += "\n"
    ufunc.__doc__ += 'Call ufunc from xarray against "chain" and "draw" dimensions:'
    ufunc.__doc__ += "\n\n"
    if n_output > 1:
        output_core_dims = "tuple([] for _ in range({}))".format(n_output)
        ufunc.__doc__ += 'xr.apply_ufunc(ufunc, dataset, input_core_dims=(("chain", "draw"),), output_core_dims={})'.format(output_core_dims)
    else:
        ufunc.__doc__ += 'xr.apply_ufunc(ufunc, dataset, input_core_dims=(("chain", "draw"),))'
    if docstring:
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += "Docstring for "
        ufunc.__doc__ += func.__name__
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += docstring


def logsumexp(ary, *, b=None, b_inv=None, axis=None, keepdims=False, out=None, copy=True):
    """Stable logsumexp when b >= 0 and b is scalar.

    b_inv overwrites b unless b_inv is None.
    """
    # check dimensions for result arrays
    ary = np.asarray(ary)
    if ary.dtype.kind == "i":
        ary = ary.astype(np.float64)
    dtype = ary.dtype.type
    shape = ary.shape
    shape_len = len(shape)
    if isinstance(axis, Sequence):
        axis = tuple(axis_i if axis_i >= 0 else shape_len + axis_i for axis_i in axis)
        agroup = axis
    else:
        axis = axis if (axis is None) or (axis >= 0) else shape_len + axis
        agroup = (axis,)
    shape_max = (
        tuple(1 for _ in shape)
        if axis is None
        else tuple(1 if i in agroup else d for i, d in enumerate(shape))
    )
    # create result arrays
    if out is None:
        if not keepdims:
            out_shape = (
                tuple()
                if axis is None
                else tuple(d for i, d in enumerate(shape) if i not in agroup)
            )
        else:
            out_shape = shape_max
        out = np.empty(out_shape, dtype=dtype)
    if b_inv == 0:
        return np.full_like(out, np.inf, dtype=dtype) if out.shape else np.inf
    if b_inv is None and b == 0:
        return np.full_like(out, -np.inf) if out.shape else -np.inf
    ary_max = np.empty(shape_max, dtype=dtype)
    # calculations
    ary.max(axis=axis, keepdims=True, out=ary_max)
    if copy:
        ary = ary.copy()
    ary -= ary_max
    np.exp(ary, out=ary)
    ary.sum(axis=axis, keepdims=keepdims, out=out)
    np.log(out, out=out)
    if b_inv is not None:
        ary_max -= np.log(b_inv)
    elif b:
        ary_max += np.log(b)
    out += ary_max.squeeze() if not keepdims else ary_max
    # transform to scalar if possible
    return out if out.shape else dtype(out)

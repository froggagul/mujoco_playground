# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training gradient utility functions."""

from typing import Union, Callable, Optional

import jax
import optax

from jax._src.api_util import (
    check_callable,
    _ensure_index,
    _dtype
)
from jax._src.api import (
    _check_input_dtype_jacfwd,
    _check_output_dtype_jacfwd,
    _std_basis,
    _jacfwd_unravel,
    _jvp
)
from jax._src.api import *

def value_and_jacfwd(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """
  [Constructed by analogy to value_and_grad -- see help(value_and_grad) for more.]
  """
  check_callable(fun)
  argnums = _ensure_index(argnums)

  def jacfun(*args, **kwargs):
    max_argnum = argnums if isinstance(argnums, int) else max(argnums)
    if max_argnum >= len(args):
      raise TypeError(f"differentiating with respect to {argnums=} requires at least "
                      f"{max_argnum + 1} positional arguments to be passed by the caller, "
                      f"but got only {len(args)} positional arguments.")
    dbg = debug_info('value_and_jacfwd', fun, args, kwargs, static_argnums=(argnums,) if isinstance(argnums, int) else argnums)

    f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    if not has_aux:
      pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    else:
      pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=True)
      y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)

    if not has_aux:
        return y, jac_tree
    else:
        return (y, aux), jac_tree

  return jacfun

def loss_and_pgrad(
    loss_fn: Callable[..., float],
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  g = value_and_jacfwd(loss_fn, has_aux=has_aux)

  def h(*args, **kwargs):
    value, grad = g(*args, **kwargs)
    return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

  return g if pmap_axis_name is None else h


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
  """Wrapper of the loss function that apply gradient updates.

  Args:
    loss_fn: The loss function.
    optimizer: The optimizer to apply gradients.
    pmap_axis_name: If relevant, the name of the pmap axis to synchronize
      gradients.
    has_aux: Whether the loss_fn has auxiliary data.

  Returns:
    A function that takes the same argument as the loss function plus the
    optimizer state. The output of this function is the loss, the new parameter,
    and the new optimizer state.
  """
  loss_and_pgrad_fn = loss_and_pgrad(
      loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux
  )

  def f(*args, optimizer_state):
    value, grads = loss_and_pgrad_fn(*args)
    params_update, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(args[0], params_update)
    return value, params, optimizer_state

  return f

if __name__ == "__main__":
    def f(x):
        return (x ** 2).sum(), x.sum()
    df = value_and_jacfwd(f, has_aux=True)
    (y, aux), df = df(np.arange(3) * 1.)

    print(f'f(x) = {y}')
    # f(x) = 5.0
    print(f'df(x) = {df}')
    # df(x) = [0. 2. 4.]
    print(f'aux = {aux}')
    # aux = 3.0

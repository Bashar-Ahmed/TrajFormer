from typing import Optional, Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    from torch.types import _dtype as DType
else:
    DType = int

import torch, math
from torch import Tensor, _VF
from torch.nn import Module
from torch.nn.init import constant_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.functional import linear
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.overrides import has_torch_function, has_torch_function_unary

def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")

def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

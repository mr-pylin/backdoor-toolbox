import torch
from torch import optim
from torchvision.transforms import v2

from triggers.target_transform import TargetTriggerTypes
from triggers.transform import TriggerTypes

# uncomment the method you want to execute
# path : from "root"."file" import "class"

# backdoor_toolbox/routines/neutral/
# routine = {
#     "root": "backdoor_toolbox.routines.neutral",
#     "file": "neutral",
#     "class": "NeutralRoutine",
#     "verbose": True,
# }

# backdoor_toolbox/routines/attacks/quantization_attack/
routine = {
    "root": "backdoor_toolbox.routines.attacks.quantization_attack",
    "file": "quantization_attack",
    "class": "QuantizationAttackRoutine",
    "verbose": True,
}

# uncomment the method you want to execute
# path : from "root"."file" import "class"

# [NEUTRAL]
# backdoor_toolbox/routines/neutral/
# routine = {
#     "root": "backdoor_toolbox.routines.neutral",
#     "file": "neutral",
#     "class": "NeutralRoutine",
#     "verbose": True,
# }

# [ATTACK]
# backdoor_toolbox/routines/attacks/multi_attack/
routine = {
    "root": "backdoor_toolbox.routines.attacks.multi_attack",
    "file": "multi_attack",
    "class": "MultiAttackRoutine",
    "verbose": True,
}

# backdoor_toolbox/routines/attacks/quantization_attack/
# routine = {
#     "root": "backdoor_toolbox.routines.attacks.quantization_attack",
#     "file": "quantization_attack",
#     "class": "QuantizationAttackRoutine",
#     "verbose": True,
# }


# [DEFENSE]
# routine = {
#     "root": "backdoor_toolbox.routines.defenses.ensemble_learning",
#     "file": "ensemble_learning",
#     "class": "EnsembleLearningRoutine",
#     "verbose": True,
# }

# routine = {
#     "root": "backdoor_toolbox.routines.defenses.knowledge_distillation",
#     "file": "knowledge_distillation",
#     "class": "KnowledgeDistillationRoutine",
#     "verbose": True,
# }

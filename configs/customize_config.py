CPL_CONFIG ={
    "SafetyBallCircle-multimodal-v0": {'cpl_contrastive_bias': 0.5, },
    "SafetyBallRun-multimodal-v0": {'cpl_contrastive_bias': 1.0, },
    "SafetyBallReach-multimodal-v0": {'cpl_contrastive_bias':1.0, },
    "SafetyHalfCheetahVelocity-multimodal-v0": {'cpl_contrastive_bias': 0.5, },
    "SafetyAntVelocity-multimodal-v0": {'cpl_contrastive_bias': 0.5, },
    "SafetySwimmerVelocity-multimodal-v0": {'cpl_contrastive_bias': 0.5, },
}

IQL_CONFIG ={
    "SafetyBallCircle-multimodal-v0": {'max_steps': 1000000, "iql_expectile": 0.75},
    "SafetyBallRun-multimodal-v0": {'max_steps': 200000, "iql_expectile": 0.75},
    "SafetyBallReach-multimodal-v0": {'max_steps': 1000000, "iql_expectile": 0.9 },
    "SafetyHalfCheetahVelocity-multimodal-v0": {'max_steps': 1000000, "iql_expectile": 0.75 },
    "SafetyAntVelocity-multimodal-v0": {'max_steps': 1000000, "iql_expectile": 0.75 },
    "SafetySwimmerVelocity-multimodal-v0": {'max_steps': 1000000, "iql_expectile": 0.75},
}

DOWNSTREAM_BASELINE_CONFIG = {
    "SafetyBallCircle-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200},
    "SafetyBallRun-downstream-v0": {'max_steps': 500000, "bc_alpha": 20},
    "SafetyBallReach-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200 },
    "SafetyHalfCheetahVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200 },
    "SafetyAntVelocity-downstream-v0": {'max_steps': 3000000, "bc_alpha": 1.0 },
    "SafetySwimmerVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200},
}

DOWNSTREAM_CONFIG = {
    "SafetyBallCircle-downstream-v0": {'max_steps': 2000000, "bc_alpha": 2000},
    "SafetyBallRun-downstream-v0": {'max_steps': 100000, "bc_alpha": 100.0},
    "SafetyBallReach-downstream-v0": {'max_steps': 1000000, "bc_alpha": 100 },
    "SafetyHalfCheetahVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200 },
    "SafetyAntVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 2000 },
    "SafetySwimmerVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200},
}

# DOWNSTREAM_CONFIG = {
#     "SafetyBallCircle-downstream-v0": {'max_steps': 2000000, "bc_alpha": 2000},
#     "SafetyBallRun-downstream-v0": {'max_steps': 100000, "bc_alpha": 100.0},
#     "SafetyBallReach-downstream-v0": {'max_steps': 1000000, "bc_alpha": 100 },
#     "SafetyHalfCheetahVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200 },
#     "SafetyAntVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 2000 },
#     "SafetySwimmerVelocity-downstream-v0": {'max_steps': 1000000, "bc_alpha": 200},
# }
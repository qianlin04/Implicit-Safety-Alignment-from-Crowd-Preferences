import os
import gym
import gymnasium
from gym.envs.registration import register
from gymnasium.envs.registration import register as register_gymnasium

register_gymnasium(
    id="SafetyBallRun-multimodal-with-goal-v0",
    entry_point="jaxrl_m.envs.safety_run:SafetyRun",
    kwargs={"goal_in_state": True,}, # longer horizon covers broader states
)

register_gymnasium(
    id="SafetyBallRun-downstream-v0",
    entry_point="jaxrl_m.envs.safety_run:SafetyRun",
    kwargs={"goal_in_state": True, 'downstream': True,},
)

register_gymnasium(
    id="SafetyBallRun-multimodal-v0",
    entry_point="jaxrl_m.envs.safety_run:SafetyRun",
    kwargs={"goal_in_state": False,},
)   

register_gymnasium(
    id="SafetyBallCircle-multimodal-with-goal-v0",
    entry_point="jaxrl_m.envs.safety_circle:SafetyCircle",
    kwargs={"goal_in_state": True,}, # longer horizon covers broader states
)

register_gymnasium(
    id="SafetyBallCircle-downstream-v0",
    entry_point="jaxrl_m.envs.safety_circle:SafetyCircle",
    kwargs={"goal_in_state": True, 'downstream': True, },
)

register_gymnasium(
    id="SafetyBallCircle-multimodal-v0",
    entry_point="jaxrl_m.envs.safety_circle:SafetyCircle",
    kwargs={"goal_in_state": False, },
)   

register_gymnasium(
    id="SafetyBallReach-multimodal-with-goal-v0",
    entry_point="jaxrl_m.envs.safety_goal_reach:SafetyBallReach",
    kwargs={"goal_in_state": True},
)

register_gymnasium(
    id="SafetyBallReach-downstream-v0",
    entry_point="jaxrl_m.envs.safety_goal_reach:SafetyBallReach",
    kwargs={"goal_in_state": True, 'downstream': True},
)

register_gymnasium(
    id="SafetyBallReach-multimodal-v0",
    entry_point="jaxrl_m.envs.safety_goal_reach:SafetyBallReach",
    kwargs={"goal_in_state": False},
)

for agent_name in ['HalfCheetah', 'Swimmer', 'Ant']:
    register_gymnasium(
        id=f"Safety{agent_name}Velocity-multimodal-with-goal-v0",
        entry_point="jaxrl_m.envs.safety_velocity:SafetyVelocity",
        kwargs={"goal_in_state": True, "agent": agent_name},
    )

    register_gymnasium(
        id=f"Safety{agent_name}Velocity-downstream-v0",
        entry_point="jaxrl_m.envs.safety_velocity:SafetyVelocity",
        kwargs={"goal_in_state": True, 'downstream': True, "agent": agent_name},
    )

    register_gymnasium(
        id=f"Safety{agent_name}Velocity-multimodal-v0",
        entry_point="jaxrl_m.envs.safety_velocity:SafetyVelocity",
        kwargs={"goal_in_state": False, "agent": agent_name},
    )


# compatibible with gym and gymnasium
def make_env(env_id: str, need_truncated=False, **kwargs):
    try:
        env = gym.make(env_id, **kwargs)
        if need_truncated:
            class GymCompatibilityWrapper(gymnasium.Wrapper):
                def step(self, action):
                    obs, reward, done, info = self.env.step(action)
                    truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
                    terminated = done and not truncated
                    return obs, reward, terminated, truncated, info

                def reset(self, **kwargs):
                    obs = self.env.reset(**kwargs)
                    return obs, {}  
            env = GymCompatibilityWrapper(env)
        return env
    except Exception:
        env = gymnasium.make(env_id, **kwargs)
        if not need_truncated:
            class GymCompatibilityWrapper(gymnasium.Wrapper):
                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    if truncated:
                        info['TimeLimit.truncated'] = True
                    return obs, reward, done, info

                def reset(self, **kwargs):
                    obs, info = self.env.reset(**kwargs)
                    return obs  
            env = GymCompatibilityWrapper(env)
        return env
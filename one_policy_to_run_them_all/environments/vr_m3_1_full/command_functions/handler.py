from one_policy_to_run_them_all.environments.vr_m3_1_full.command_functions.random import RandomCommands


def get_command_function(name, env, **kwargs):
    if name == "random":
        return RandomCommands(env, **kwargs)
    else:
        raise NotImplementedError

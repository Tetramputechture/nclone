class SimConfig:
    def __init__(self):
        self.basic_sim = False
        self.full_export = False
        self.tolerance = 1.0

    @classmethod
    def from_args(cls, args):
        config = cls()
        config.basic_sim = args.basic_sim
        config.full_export = args.full_export
        config.tolerance = args.tolerance
        return config


sim_config = None


def init_sim_config(args):
    global sim_config
    sim_config = SimConfig.from_args(args)
    return sim_config

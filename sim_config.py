class SimConfig:
    def __init__(self,
                 basic_sim: bool = False,
                 full_export: bool = False,
                 tolerance: float = 1.0,
                 enable_anim: bool = True,
                 log_data: bool = False):
        self.basic_sim = basic_sim
        self.full_export = full_export
        self.tolerance = tolerance
        self.enable_anim = enable_anim
        self.log_data = log_data

    @classmethod
    def from_args(cls, args=None):
        config = cls()
        if args is None:
            return config
        config.basic_sim = args.basic_sim
        config.full_export = args.full_export
        config.tolerance = args.tolerance
        config.enable_anim = args.enable_anim
        config.log_data = args.log_data
        return config


sim_config = None


def init_sim_config(args=None):
    global sim_config
    sim_config = SimConfig.from_args(args)
    return sim_config

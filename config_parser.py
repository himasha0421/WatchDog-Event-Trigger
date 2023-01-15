import yaml
from easydict import EasyDict

class Config:

    def __init__(self, path):

        with open(path, 'r') as config_file:
            cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        cfg = EasyDict(cfg)
        self.__dict__.update(cfg)

    def initiate_easydict(self, cfg):

        for key in cfg:
            if type(cfg[key]) is dict:
                cfg[key] = self.initiate_easydict(cfg[key])
        return EasyDict(cfg)


conf = Config('config.yml')
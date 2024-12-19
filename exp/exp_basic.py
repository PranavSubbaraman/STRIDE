from models import timer, timer_xl, moirai, patchtst, OFA


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'timer': timer,
            'timer_xl': timer_xl,
            'moirai': moirai,
            'patchtst': patchtst,
            'OFA': OFA
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

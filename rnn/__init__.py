from .rnn_gauss import RNN_GAUSS

def load_model(model_name, params, parser=None):
    model_name = model_name.lower()

    if model_name == 'rnn_gauss':
        return RNN_GAUSS(params, parser)
    else:
        raise NotImplementedError

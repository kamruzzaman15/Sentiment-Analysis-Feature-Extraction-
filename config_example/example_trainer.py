from torch.optim import Adam

class ExampleTrainer:
    def __init__(self,
                 embedding_dim,
                 n_primitive_types,
                 maxdepth,
                 n_layers,
                 n_compressor_layers=1,
                 inverse_distance_weighting=1.):
        self.embedding_dim = embedding_dim
        self.n_primitive_types = n_primitive_types
        self.primitive_types = set(range(n_primitive_types))
        self.maxdepth = maxdepth
        self.n_layers = n_layers
        self.n_compressor_layers = n_compressor_layers
        self.inverse_distance_weighting = inverse_distance_weighting

        
        self.model = # TODO: initialize pytorch model

    def fit(self,
            n_batches,
            lr,
            batch_size,
            verbosity,
            radius):
        optimizer = Adam(list(self.model.parameters()),
                         lr=lr)
        
        # TODO: train using given parameters.

        return self.model.eval()


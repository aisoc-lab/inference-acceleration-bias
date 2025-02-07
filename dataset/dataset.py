class Dataset:
    def __init__(self, args) -> None:
        self.args = args

    def load_dataset(self):
        raise NotImplementedError

    def limit_samples(self):
        """
        Take only num_samples samples from the dataset if num_samples is not None
        """
        # check if num_samples is not None
        num_samples = self.args.get("num_samples", None)
        if num_samples is not None:
            # take only num_samples samples from the dataset
            self.subsets = self.subsets.sample(num_samples)
        else:
            self.subsets = self.subsets

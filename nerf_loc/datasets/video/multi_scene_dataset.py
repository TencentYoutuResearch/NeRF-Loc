from torch.utils.data import ConcatDataset

class MultiSceneDataset(ConcatDataset):
    def __init__(datasets):
        super().__init__(datasets)

    def set_mode(self, mode):
        for ds in self.datasets:
            ds.set_mode(mode)
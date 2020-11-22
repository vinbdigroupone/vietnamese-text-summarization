from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils import tensorsFromPair

class VietDataset(Dataset):
    def __init__(self, pairs, input_lang, output_lang, word2index, mode, cfg):
        self.cfg = cfg
        self.pairs = pairs
        self.training_pairs = [tensorsFromPair(pair, input_lang, output_lang, word2index) 
                                for pair in self.pairs]


    def __len__(self):
        return len(self.training_pairs)
        
    def __getitem__(self, idx):
        in_tensor, tar_tensor = self.training_pairs[idx]
        print(in_tensor.size())
        return in_tensor, tar_tensor

def my_collate(batch):
  text = [item[0] for item in batch]
  target = [item[1] for item in batch]

  text = pad_sequence(text)
  target = pad_sequence(target)

  return text, target


def make_dataloader(cfg, pairs, input_lang, output_lang, word2index):
    _mode = cfg.MODE
    dataset = VietDataset(pairs, input_lang, output_lang, word2index, _mode, cfg)

    #################################################
    #### DEBUG NOT WORKING WITH OVERSAMPLING Bug ####
    #################################################

    # DEBUG_mode = on
    if cfg.DATA.DEBUG:
        dataset = Subset(dataset, 
                        np.random.choice(np.arange(len(dataset)), 72))
                        # np.arrange: return a list from (len(dataset))
                        # np.random.choice: pick 1 random number from the list

    if _mode == 'train':
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset, cfg.TRAIN.BATCH_SIZE, 
                            pin_memory=True, shuffle=shuffle, collate_fn=my_collate,
                            drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

    return dataloader

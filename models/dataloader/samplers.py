import numpy as np
import torch


class CategoriesSampler:
    def __init__(self, label, n_batch, n_cls, n_per):
        # print(label, n_batch, n_cls, n_per)
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls  # 2
        self.n_per = n_per  # 20

        label = np.array(label)  # all data label
        # print(label)
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            # print(ind)
            self.m_ind.append(ind)
        # print(self.m_ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            # print(len(self.m_ind))
            classes = torch.randperm(len(self.m_ind))[
                      : self.n_cls
                      ]  # random sample num_class indices, e.g. 5
            # print(classes)
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class

                pos = torch.randperm(len(l))[
                      : self.n_per
                      ]  # sample n_per data index of this class
                batch.append(l[pos])
                # print(len(l[pos]))
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

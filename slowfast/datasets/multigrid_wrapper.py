from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import math

class LongShortCycleWrapper(object):
    STAGES = [8, 4, 2, 1]

    def __init__(self, loader):
        if not isinstance(loader, DataLoader):
            raise ValueError("loader should be instance of"
                            "torch.utils.data.Dataloader, but got loader={}"
                            .format(loader))
        self.stage = 0
        self.steps = 0
        self.loader = loader

    def __iter__(self):
        input_list = []
        label_list = []
        index_list = []
        for inputs, labels, index in self.loader:
            # Finetune stages
            if self.stage == -1:
                yield 1, inputs, labels, index

            # Long cycle from settings
            scale_b = self.STAGES[self.stage]

            # Short cycle from steps
            if self.steps % 3 == 1:
                scale_b = scale_b * 2
            elif self.steps % 3 == 2:
                scale_b = scale_b * 4

            # Batch accumulate
            input_list.append(inputs[self.stage][self.steps%3])
            label_list.append(labels)
            index_list.append(index)
            if len(input_list) == scale_b:
                if self.steps % 25000 < 30:
                    print(torch.cat(input_list).shape)
                yield self.STAGES[self.stage], [torch.cat(input_list)], torch.cat(label_list), torch.cat(index_list)
                input_list = []
                label_list = []
                index_list = []
        
        # drop remaining

    def set_multgrid(self, gs, rate_cur_lr_step, remaining_lr_step):
        self.steps = gs
        self.stage = math.floor(rate_cur_lr_step * 4) % 4
        if remaining_lr_step <= 1:
            self.stage = -1


# class LongShortCycleWrapper(object):
#     STAGES = [(8, 0.25, math.sqrt(2)/2, math.sqrt(2)/2),
#               (4, 0.5, math.sqrt(2)/2, math.sqrt(2)/2),
#               (2, 0.5, 1.0, 1.0),
#               (1, 1.0, 1.0, 1.0)]

#     def __init__(self, loader):
#         if not isinstance(loader, DataLoader):
#             raise ValueError("loader should be instance of"
#                             "torch.utils.data.Dataloader, but got loader={}"
#                             .format(loader))
#         self.stage = 0
#         self.steps = 0
#         self.loader = loader

#     def __iter__(self):
#         input_list = []
#         label_list = []
#         index_list = []
#         for inputs, labels, index in self.loader:
#             # Finetune stages
#             if self.stage == -1:
#                 yield 1, inputs, labels, index

#             # Long cycle from settings
#             scale_b, scale_t, scale_h, scale_w = self.STAGES[self.stage]

#             # Short cycle from steps
#             if self.steps % 3 == 1:
#                 scale_h = scale_h * math.sqrt(2) / 2
#                 scale_w = scale_w * math.sqrt(2) / 2
#                 scale_b = scale_b * 2
#             elif self.steps % 3 == 2:
#                 scale_h = scale_h / 2
#                 scale_w = scale_w / 2
#                 scale_b = scale_b * 4

#             # Spatial Rescale 
#             # TODO: Support multi pathway
#             inputs = inputs[0].permute([0, 2, 1, 3, 4])
#             B, T, C, H, W = inputs.shape
#             inputs = inputs.reshape([-1, C, H, W])
#             inputs = F.interpolate(inputs, scale_factor=[scale_h, scale_w], align_corners=True, mode="bilinear")
#             _, C, H, W = inputs.shape
#             inputs = inputs.view([B, T, C, H, W])
#             inputs = inputs.permute([0, 2, 1, 3, 4])

#             # Temporal Rescale
#             inputs = torch.index_select(
#                 inputs,
#                 2,
#                 torch.linspace(
#                     0,
#                     inputs.shape[2] - 1,
#                     int(inputs.shape[2] * scale_t),
#                 ).long().cuda(),
#             )

#             # Batch accumulate
#             input_list.append(inputs)
#             label_list.append(labels)
#             index_list.append(index)
#             if len(input_list) == scale_b:
#                 print(torch.cat(input_list).shape)
#                 yield self.STAGES[self.stage][0], [torch.cat(input_list)], torch.cat(label_list), torch.cat(index_list)
#                 input_list = []
#                 label_list = []
#                 index_list = []
        
#         # drop remaining

#     def set_multgrid(self, gs, rate_cur_lr_step, remaining_lr_step):
#         self.steps = gs
#         self.stage = math.floor(rate_cur_lr_step * 4) % 4
#         if remaining_lr_step <= 1:
#             self.stage = -1



# from torch.utils.data.sampler import Sampler
# from torch.utils.data.distributed import DistributedSampler

# class MultiGridBatchSampler(Sampler):
#     r"""
#         Sampler for MultiGrid Strategy
#         Reference: 
#     """

#     def __init__(self, sampler, drop_last):
#         """
#         set batch_size before every iteration.
#         """
#         if not isinstance(sampler, Sampler):
#             raise ValueError("sampler should be an instance of "
#                              "torch.utils.data.Sampler, but got sampler={}"
#                              .format(sampler))
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value, but got "
#                              "drop_last={}".format(drop_last))
        
#         self.sampler = sampler
#         self.drop_last = drop_last

#         self.cur_batch_size = 1

#     def __iter__(self):
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.cur_batch_size:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.cur_batch_size
#         else:
#             return (len(self.sampler) + self.cur_batch_size - 1) // self.cur_batch_size

#     def set_batch_size(self, batch_size):
#         self.cur_batch_size = batch_size


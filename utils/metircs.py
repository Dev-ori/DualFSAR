import torch
import torch.distributed as dist
import numpy as np
from torcheval.metrics import MulticlassAccuracy, MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torcheval.metrics.metric import Metric
from torcheval.metrics.toolkit import get_synced_metric_collection
from collections.abc import MutableMapping

class MetricList:
    def __init__(self, dict_metric:MutableMapping[str, Metric], start_epoch:int = 0):
        """https://pytorch.org/torcheval/stable/torcheval.metrics.html

        Args:
            dict_metric (MutableMapping[str, Metric]): list of torcheval.metrics
            start_epoch (int, optional): Defaults to 0.
        """
        self.dict_metric = dict_metric
        self.next_epoch = start_epoch
        self.tot_result = dict()

        
    def __str__(self):
        pass
        
    def update(self, predicts:torch.Tensor, targets:torch.Tensor):
        for metric in self.dict_metric.values():
            metric.update(predicts, targets)
            
    def compute(self):
        cur_result = dict()
        for key, metric in self.dict_metric.items():
            cur_result[key] = metric.compute().item()
        
        return cur_result
        
    def reset(self):
        for metric in self.dict_metric.values():
            metric.reset()
        
            
    def print(self, epoch:int=None):
        if epoch is None:
            epoch = self.next_epoch - 1
        
        assert epoch < self.next_epoch, "result_dict's epoch must be lower than current epoch"

        max_len = 0
        for key in self.dict_metric.keys():
            max_len = max(max_len, len(key))
            
        for key, value in self.tot_result[epoch].items():
            print(f"{key:<{max_len}} : {value}")
            
    def result_dict(self, epoch:int=None):
        if epoch is None:
            epoch = self.next_epoch - 1
        
        assert epoch < self.next_epoch, "result_dict's epoch must be lower than current epoch"
        result = dict()
        for key, value in self.tot_result[epoch].items():
            result[key] = value
        
        return result
    
    def synced_metric(self):
        if dist.get_world_size() > 1:
            get_synced_metric_collection(self.dict_metric)



        
if __name__ == "__main__":
    macro_acc = MulticlassAccuracy(average='macro', num_classes=4)
    micro_acc = MulticlassAccuracy()
    compusion_matrix = MulticlassConfusionMatrix(num_classes=4)
    f1_score = MulticlassF1Score(num_classes=4)
    precision = MulticlassPrecision(num_classes=4)
    recall = MulticlassRecall(num_classes=4)
    
    
    micro_acc.update(torch.tensor([1,2,3]), torch.tensor([1,2,3]))
    print(micro_acc.compute())
    micro_acc.update(torch.tensor([2]), torch.tensor([3]))
    print(micro_acc.compute())
    # metrics = MetricList({
    #     'macro_acc' : macro_acc,
    #     'micro_acc' : micro_acc,
    #     'precision' : precision,
    #     'recall' : recall,
    #     'f1_score' : f1_score,
    #     'compusion_matrix' : compusion_matrix
    # })

    # input = torch.tensor([0, 2, 1, 3])
    # target = torch.tensor([0, 1, 2, 3])

    # metrics.update(input, target)

    # input = torch.tensor([1,2, 3, 0])
    # target = torch.tensor([0, 1, 2, 3])

    # metrics.update(input, target)

    # metrics.compute()

    # metrics.print()
    # metrics.synced_metric()

    # print(metrics.result_dict())

    # metrics.reset()


    # metrics.compute()

    # metrics.print()
    # metrics.synced_metric()

    # print(metrics.result_dict())
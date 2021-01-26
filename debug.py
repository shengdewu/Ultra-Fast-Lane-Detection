import torch
from utils.loss import SoftmaxFocalLoss
from utils.loss import ParsingRelationLoss
from utils.loss import ParsingRelationDis

if __name__ == '__main__':
    s = SoftmaxFocalLoss(2)
    logit = torch.FloatTensor([[[[1,2],[3,4]],[[0,1],[1,3]],[[2,0],[2,1]]]])
    label = torch.LongTensor([[[1, 0], [0, 1]]])
    v = s.forward(logit, label)
    print(v)

    p = ParsingRelationLoss()
    print(p.forward(logit))

    pr = ParsingRelationDis()
    print(pr.forward(logit))
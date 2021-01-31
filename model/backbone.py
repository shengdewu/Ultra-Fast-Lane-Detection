import torch,pdb
import torchvision
import torch.nn.modules
import model.resnet

class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        model = list(torchvision.models.vgg16_bn(pretrained=pretrained).features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            #resnet_model = torchvision.models.resnet18(pretrained=pretrained)
            resnet_model = model.resnet.resnet18()
        # elif layers == '34':
        #     model = torchvision.models.resnet34(pretrained=pretrained)
        # elif layers == '50':
        #     model = torchvision.models.resnet50(pretrained=pretrained)
        # elif layers == '101':
        #     model = torchvision.models.resnet101(pretrained=pretrained)
        # elif layers == '152':
        #     model = torchvision.models.resnet152(pretrained=pretrained)
        # elif layers == '50next':
        #     model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        # elif layers == '101next':
        #     model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        # elif layers == '50wide':
        #     model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        # elif layers == '101wide':
        #     model = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        else:
            raise NotImplementedError
        
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4

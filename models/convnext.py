import torchvision

def convnext_T( pretrained=True, **kwargs):
    """ Constructs an ConvNeXt-T
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = torchvision.models.convnext_tiny()
    # config = CNBlockConfig(input_channels=3, out_channels=2048, num_layers)
    if pretrained: 
            model = torchvision.models.convnext_tiny(num_classes=2048, weights='IMAGENET1K_V1')
    else: model = torchvision.models.convnext_tiny(num_classes=2048, weights=None)
            
    return model


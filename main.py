import argparse
import time
import os

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.models as models

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--data_path', type=str,
                    default='./data/', help='Path of dataset')
parser.add_argument('--output_path', type=str,
                    default='test', help='Path of output')
parser.add_argument('--imagenet_path', type=str,
                    default='/imagenet/val/', help='Path of the ImageNet dataset')
args = parser.parse_args()
print(args)

# Make directories
os.makedirs(f'logs/', exist_ok=True)
os.makedirs(f'results/{args.output_path}', exist_ok=True)

# Change Torch Hub cache dir
# torch.hub.set_dir('/path/to/cache/models/')


# /////////////// Model Setup ///////////////

def get_net(model_name):
    if model_name == 'alexnet':
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        net = models.alexnet(weights=weights)
        args.test_bs = 256

    elif model_name == 'squeezenet1.0':
        weights = models.SqueezeNet1_0_Weights.IMAGENET1K_V1
        net = models.squeezenet1_0(weights=weights)
        args.test_bs = 256

    elif model_name == 'squeezenet1.1':
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        net = models.squeezenet1_1(weights=weights)
        args.test_bs = 256

    elif model_name == 'vgg11':
        weights = models.VGG11_Weights.IMAGENET1K_V1
        net = models.vgg11(weights=weights)
        args.test_bs = 64

    elif model_name == 'vgg19':
        weights = models.VGG19_Weights.IMAGENET1K_V1
        net = models.vgg19(weights=weights)
        args.test_bs = 64
    
    elif model_name == 'vggbn':
        weights = models.VGG19_BN_Weights.IMAGENET1K_V1
        net = models.vgg19_bn(weights=weights)
        args.test_bs = 64

    elif model_name == 'densenet121':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        net = models.densenet121(weights=weights)
        args.test_bs = 64

    elif model_name == 'densenet169':
        weights = models.DenseNet169_Weights.IMAGENET1K_V1
        net = models.densenet169(weights=weights)
        args.test_bs = 32

    elif model_name == 'densenet201':
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        net = models.densenet201(weights=weights)
        args.test_bs = 32

    elif model_name == 'densenet161':
        weights = models.DenseNet161_Weights.IMAGENET1K_V1
        net = models.densenet161(weights=weights)
        args.test_bs = 32

    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        args.test_bs = 256

    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        net = models.resnet34(weights=weights)
        args.test_bs = 128

    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        net = models.resnet50(weights=weights)
        args.test_bs = 128

    elif model_name == 'resnet50_stylized':
        # model_url = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar'
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        net = models.resnet50()
        checkpoint = torch.load('cache/models/checkpoints/resnet50-stylized.pth.tar')
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint["state_dict"])
        args.test_bs = 128

    elif model_name == 'resnet50_augmix':
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        net = models.resnet50()
        checkpoint = torch.load('cache/models/checkpoints/resnet50-augmix.pth.tar')
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['state_dict'])
        args.test_bs = 128

    elif model_name == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        net = models.resnet101(weights=weights)
        args.test_bs = 32

    elif model_name == 'resnet152':
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        net = models.resnet152(weights=weights)
        args.test_bs = 32

    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        net = models.vit_b_16(weights=weights)
        args.test_bs = 64

    elif model_name == 'vit_b_32':
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        net = models.vit_b_32(weights=weights)
        args.test_bs = 256

    elif model_name == 'vit_l_16':
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        net = models.vit_l_16(weights=weights)
        args.test_bs = 16

    elif model_name == 'vit_l_32':
        weights = models.ViT_L_32_Weights.IMAGENET1K_V1
        net = models.vit_l_32(weights=weights)
        args.test_bs = 16

    elif model_name == 'convnext_base':
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        net = models.convnext_base(weights=weights)
        args.test_bs = 32

    elif model_name == 'swin_b':
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        net = models.swin_b(weights=weights)
        args.test_bs = 32

    elif model_name == 'swin_v2_b':
        weights = models.Swin_V2_B_Weights.IMAGENET1K_V1
        net = models.swin_v2_b(weights=weights)
        args.test_bs = 16

    elif model_name == 'resnext50':
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        net = models.resnext50_32x4d(weights=weights)
        args.test_bs = 64

    elif model_name == 'resnext101':
        weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        net = models.resnext101_32x8d(weights=weights)
        args.test_bs = 32

    elif model_name == 'resnext101_64':
        weights = models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        net = models.resnext101_64x4d(weights=weights)
        args.test_bs = 32

    args.prefetch = 4

    # for p in net.parameters():
    #     p.volatile = True

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()

    torch.manual_seed(1)
    np.random.seed(1)
    if args.ngpu > 0:
        torch.cuda.manual_seed(1)

    net.eval()
    cudnn.benchmark = True  # fire on all cylinders

    preprocess = weights.transforms()

    print(f'Model {model_name} Loaded')

    return net, preprocess

# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


_, class_to_idx = find_classes(args.imagenet_path)


# /////////////// Further Setup ///////////////

def save_csv_log(head, value, is_create=False, file_name='test'):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f'results/{args.output_path}/{file_name}.csv'
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, 'a') as f:
            df.to_csv(f, header=False, index=False)


def evaluate(net, preprocess, domain_name):
    dataset = dset.ImageFolder(
        root=args.data_path + domain_name,
        transform=preprocess)
    data_class_to_idx = dataset.class_to_idx
    idx_to_real_idx = {v: class_to_idx[k]
                       for k, v in data_class_to_idx.items()}
    dataset.target_transform = lambda label: idx_to_real_idx[label]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

    correct = 0
    for _, (data, target) in enumerate(dataloader):
        data = data.cuda()

        output = net(data)

        pred = output.data.max(1)[1]
        correct += pred.eq(target.cuda()).sum()

    accuracy = correct / len(dataset)

    return accuracy.cpu().numpy()

# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////

domains = [
    'Original',
    'Color',
    'Context',
    'Drawing',
    'Weather',
    'Texture'
]

baselines = ['alexnet', 'squeezenet1.0', 'squeezenet1.1',
             'vgg11', 'vgg19', 'vggbn',
             'densenet121', 'densenet169', 'densenet201',
             'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
             'resnext50', 'resnext101', 'resnext101_64',
             'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32',
             'convnext_base',
             'swin_b', 'swin_v2_b',
             'resnet50_stylized', 'resnet50_augmix',
             ]

accuracies = np.zeros([len(baselines), len(domains) + 1])
for i, model in enumerate(baselines):
    net, preprocess = get_net(model)
    model_accuracies = []
    for domain_name in domains:
        accuracy = evaluate(net, preprocess, domain_name) * 100
        model_accuracies.append(accuracy)
        print(f'Domain: {domain_name} | Accuracy (%): {accuracy:.2f}')

    mean_model_accuracy = np.mean(model_accuracies[1:])
    model_accuracies.append(mean_model_accuracy)
    accuracies[i] = np.array(model_accuracies)
    print(f'{model} Average (%): {mean_model_accuracy:.2f}')
    net = net.cpu()

head = np.array(['Model'])
for domain in domains:
    head = np.append(head, [domain])
head = np.append(head, ['Average'])

accuracies = accuracies.round(4)
baselines = np.expand_dims(np.array(baselines), axis=1)
value = np.concatenate([baselines, accuracies.astype(str)], axis=1)

save_csv_log(head, value, is_create=True, file_name='result')

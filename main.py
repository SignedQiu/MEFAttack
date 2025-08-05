import os
import numpy as np
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from utils import save_images, ImageNet, Normalize, TfNormalize
import pretrainedmodels
import timm
from model import tf_adv_inception_v3, tf_ens3_adv_inc_v3, tf_ens4_adv_inc_v3, tf_ens_adv_inc_res_v2
import argparse

normal_list = ['inception_v3', 'inception_v4', 'inc_res_v2', 'dense_121', 'dense_169', 'vgg_19', 'resnet_18', 'resnet_50', 'resnet_101', 'resnet_152', 'xcept', 'pnasnet', 'mobilenet', 'vitb16', 'vitl32', 'pits', 'mlpmixer', 'resmlp']


def get_model(net_name, model_dir):
    """Load converted model"""

    if net_name == 'inception_v3':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inception_v4':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.inceptionv4(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_18':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet18(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_50':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet50(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_101':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet101(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'resnet_152':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.resnet152(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'inc_res_v2':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet').eval().cuda())
    elif net_name == 'mobilenet':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                torchvision.models.mobilenet.mobilenet_v3_large(weights=("pretrained", torchvision.models.mobilenet.MobileNet_V3_Large_Weights.IMAGENET1K_V1)).eval().cuda())
    elif net_name == 'pnasnet':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet').eval())
    elif net_name == 'vitb16':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                torchvision.models.vit_b_16(weights=("pretrained", torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)).eval().cuda())
    elif net_name == 'vitl32':
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                torchvision.models.vit_l_32(weights=("pretrained", torchvision.models.ViT_L_32_Weights.IMAGENET1K_V1)).eval().cuda())
    elif net_name == 'pits':
        pit_model = timm.models.pit.pit_s_224(pretrained=True)
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pit_model.eval().cuda())
    elif net_name == 'mlpmixer':
        mlpmixer_model = timm.models.mlp_mixer.mixer_l16_224(pretrained=True)
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                mlpmixer_model.eval().cuda())
    elif net_name == 'resmlp':
        resmlp_model = timm.models.mlp_mixer.resmlp_12_224(pretrained=True)
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                resmlp_model.eval().cuda())
    elif net_name == "dense_121":
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.densenet121(num_classes=1000, pretrained='imagenet').eval())
    elif net_name == "vgg_19":
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.vgg19_bn(num_classes=1000, pretrained='imagenet').eval())
    elif net_name == "xcept":
        model = torch.nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                pretrainedmodels.xception(num_classes=1000, pretrained='imagenet').eval())
    elif net_name == 'tf_adv_inception_v3':
        model_path = os.path.join(model_dir, net_name + '.npy')
        net = tf_adv_inception_v3
        model = nn.Sequential( 
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval(),)
    elif net_name == 'tf_ens3_adv_inc_v3':
        model_path = os.path.join(model_dir, net_name + '.npy')
        net = tf_ens3_adv_inc_v3
        model = nn.Sequential(
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval(),)
    elif net_name == 'tf_ens4_adv_inc_v3':
        model_path = os.path.join(model_dir, net_name + '.npy')
        net = tf_ens4_adv_inc_v3
        model = nn.Sequential(
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval(),)
    elif net_name == 'tf_ens_adv_inc_res_v2':
        model_path = os.path.join(model_dir, net_name + '.npy')
        net = tf_ens_adv_inc_res_v2
        model = nn.Sequential(
            TfNormalize('tensorflow'),
            net.KitModel(model_path).eval(),)
    else:
        print('Wrong model name!')
    
    model.eval()
    model.requires_grad_(False)
    return model


def mef_attack(save_dir, device, source_model_name, epsilon=16/255, step_size=1.6/255, inner_mu=0.9, outer_mu=0.5, gamma=2, kesai=0.15, sample_num=20, iteration_num=20, batch_size=50):
    
    # prepare dataset
    if source_model_name in ['vitb16', 'vitl32', 'pits', 'mlpmixer', 'resmlp']:
        img_size = 224
    elif source_model_name in ['pnasnet']:
        img_size = 331
    else:
        img_size = 299
    transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    ])
    dataset = ImageNet(dir="./dataset/images", csv_path="./dataset/images.csv", sample_num=1000, transforms=transform)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # prepare model
    source_model = get_model(source_model_name, "/mnt/data/SignedQiu/Security/SecurityForAi/TransferAttack/TransferAttackEval-main/attacks/torch_nets_weight").to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, ((images, labels, _), path) in enumerate(val_loader):

        images = images.to(device)
        labels = labels.to(device)
        img = images.clone()
        grad_pre = torch.zeros_like(images)
        grad_t = torch.zeros_like(images)

        b, c, h, w = images.shape

        grad_list = torch.zeros([sample_num, b, c, h, w]).to(device)
        grad_pgia = torch.zeros([sample_num, b, c, h, w]).to(device)

        for j in range(iteration_num):

            img_x = img.clone().detach()
            for k in range(sample_num):

                img_near = img_x + torch.rand_like(img_x).uniform_(-gamma*epsilon, gamma*epsilon)
                img_min = img_near + kesai*epsilon*(grad_pgia[k])
                img_min.requires_grad_(True)
                logits = source_model(img_min)
                loss = nn.CrossEntropyLoss(reduction='mean')(logits,labels)
                loss.backward()
                grad_list[k] = img_min.grad.detach().clone()
                img_min.grad.zero_()

            grad = (1/sample_num)*grad_list
            grad_pgia = ((grad / torch.mean(torch.abs(grad), (2, 3, 4), keepdim=True)) - inner_mu * grad_pgia)
            grad_t = grad.sum(0)
            grad_t = grad_t / torch.mean(torch.abs(grad_t), (1, 2, 3), keepdim=True)
            input_grad = grad_t + outer_mu * grad_pre
            grad_pre = input_grad
            input_grad = input_grad / torch.mean(torch.abs(input_grad), (1, 2, 3), keepdim=True)
            img = img.data + step_size * torch.sign(input_grad)

            img = torch.where(img > images + epsilon, images + epsilon, img)
            img = torch.where(img < images - epsilon, images - epsilon, img)
            img = torch.clamp(img, min=0, max=1)

            if j == iteration_num - 1: 
                save_images(img.detach().cpu().numpy(), img_list=path, idx=len(path), output_dir=save_dir)


def evaluation(save_dir, device, target_model_name, batch_size=50):

    # prepare dataset
    if target_model_name in ['vitb16', 'vitl32', 'pits', 'mlpmixer', 'resmlp']:
        img_size = 224
    elif target_model_name in ['pnasnet']:
        img_size = 331
    else:
        img_size = 299
    transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    ])
    dataset = ImageNet(dir=save_dir, csv_path="./dataset/images.csv", sample_num=1000, transforms=transform)
    examples_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # prepare model
    target_model = get_model(target_model_name, "/mnt/data/SignedQiu/Security/SecurityForAi/TransferAttack/TransferAttackEval-main/attacks/torch_nets_weight").to(device)

    success_attack_num = 0
    total_sample_num = 0
    for i, ((images, labels, _), _) in enumerate(examples_loader):
        total_sample_num += len(images)
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output_logits = target_model(images)
            if target_model_name in normal_list:
                success_attack_num = success_attack_num + sum(torch.argmax(output_logits, dim=1) != labels).cpu().numpy()
            else:
                success_attack_num = success_attack_num + sum(torch.argmax(output_logits[0], dim=1) != (labels+1)).cpu().numpy()
    
    return success_attack_num / total_sample_num

def main(args):
    device = torch.device("cuda:{}".format(args.gpu_id))

    if args.mode == "attack":
        mef_attack(save_dir=args.save_dir, device=device, source_model_name=args.source_model_name, epsilon=args.epsilon, step_size=args.step_size, inner_mu=args.inner_mu,
                   outer_mu=args.outer_mu, gamma=args.gamma, kesai=args.kesai, sample_num=args.sample_num, iteration_num=args.iteration_num, batch_size=args.batch_size)
    
    elif args.mode == "eval":
        success_rate_rate = evaluation(save_dir=args.save_dir, device=device, target_model_name=args.target_model_name,batch_size=args.batch_size)
        print("attack success rate:{:.2f}".format(success_rate_rate*100))
    else:
        raise ValueError("Mode not support")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["attack", "eval"], default="attack")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--source_model_name", type=str, default="resnet_50")
    parser.add_argument("--target_model_name", type=str, default="inception_v3")
    parser.add_argument("--epsilon", type=float, default=16/255)
    parser.add_argument("--step_size", type=float, default=1.6/255)
    parser.add_argument("--inner_mu", type=float, default=0.9)
    parser.add_argument("--outer_mu", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=2)
    parser.add_argument("--kesai", type=float, default=0.15)
    parser.add_argument("--sample_num", type=int, default=20)
    parser.add_argument("--iteration_num", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    main(args)
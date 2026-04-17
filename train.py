import os
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import argparse
from tqdm import tqdm
from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips

percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

def crop_image_by_part(image, part):
    hw = image.shape[2] // 2
    if part == 0:
        return image[:, :, :hw, :hw]
    if part == 1:
        return image[:, :, :hw, hw:]
    if part == 2:
        return image[:, :, hw:, :hw]
    if part == 3:
        return image[:, :, hw:, hw:]

def train_d(net, data, iteration, total_iterations, label="real"):
    """Train function of discriminator"""
    if label == "real":
        pred, [rec_all, rec_small, rec_part], part, _ = net(data)
        err_real = F.relu(torch.rand_like(pred) * 0.2 + 0.8 - pred).mean()
        loss_rec_real = percept(rec_all, F.interpolate(data, rec_all.shape[2])).sum() + \
                        percept(rec_small, F.interpolate(data, rec_small.shape[2])).sum() + \
                        percept(rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2])).sum()
        err = err_real + loss_rec_real
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part, err_real

    elif label == 'fake':
        pred, [rec_all, rec_small, rec_part], part, feat_out = net(data)
        err_fake = F.relu(torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()

        loss_rec_fake = (iteration / total_iterations) * (percept(rec_all, F.interpolate(data[0], rec_all.shape[2])).sum()) + \
                        (iteration / total_iterations) * (percept(rec_small, F.interpolate(data[0], rec_small.shape[2])).sum()) + \
                        (iteration / total_iterations) * (percept(rec_part, F.interpolate(crop_image_by_part(data[0], part), rec_part.shape[2])).sum())
        err = err_fake + loss_rec_fake
        err.backward(retain_graph=True)
        return pred.mean().item(), rec_all, rec_small, rec_part, err_fake, feat_out

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    eps = 1 * 1e-5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = args.start_iter
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    lamda_MS_D = args.lamda_MS_D
    lamda_MS_G = args.lamda_MS_G

    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,
                                 pin_memory=True))

    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size, batch_size=batch_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['g'].items()})
        netD.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['d'].items()})
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt

    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    for iteration in tqdm(range(current_iteration, total_iterations + 1)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
        noise_2 = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
        noise_conc = torch.cat((noise, noise_2), 0)
        fake_images_conc = netG(noise_conc)

        real_image = DiffAugment(real_image, policy=policy)
        fake_images_conc = [DiffAugment(fake, policy=policy) for fake in fake_images_conc]
        fake_images_noise = [fake_images_conc[0][:real_image.size(0)], fake_images_conc[1][:real_image.size(0)]]
        fake_images_noise_2 = [fake_images_conc[0][real_image.size(0):], fake_images_conc[1][real_image.size(0):]]

        ## 2. train Discriminator
        netD.zero_grad()
        err_dr, rec_img_all, rec_img_small, rec_img_part, d_pred_real = train_d(netD, real_image,
                                                                                               iteration,
                                                                                               total_iterations,
                                                                                               label="real")
        err_df, rec_fake_all, rec_fake_small, rec_fake_part, d_pred_fake, feat_out = train_d(netD,
                                                                                                  [fi.detach() for fi in
                                                                                                   fake_images_noise],
                                                                                                   iteration,
                                                                                                   total_iterations,
                                                                                                   label="fake")
        err_df_2, rec_fake_all_2, rec_fake_small_2, rec_fake_part_2, d_pred_fake_2, feat_out_2 = train_d(netD,
                                                                                                              [fi.detach() for fi in
                                                                                                               fake_images_noise_2],
                                                                                                               iteration,
                                                                                                               total_iterations,
                                                                                                               label="fake")

        lz_D = (torch.mean(torch.abs(feat_out_2 - feat_out))) / (torch.mean(torch.abs(noise_2 - noise)))
        lz_D = (1 / (lz_D + eps)) * lamda_MS_D
        lz_D.backward()
        optimizerD.step()


        ## 3. train Generator
        netG.zero_grad()
        pred_g, [_, _, _], _, feat_last = netD(fake_images_noise)
        pred_g_2, [_, _, _], _, feat_last_2 = netD(fake_images_noise_2)
        err_g = -pred_g.mean()
        err_g_2 = -pred_g_2.mean()

        lz_G = (torch.mean(torch.abs(feat_last_2 - feat_last))) / (torch.mean(torch.abs(noise_2 - noise)))
        lz_G = 1 / (lz_G + eps)
        err_g_all = err_g + err_g_2 + lamda_MS_G * lz_G
        err_g_all.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f" % (err_dr, -err_g.item()))

        if iteration % (save_interval * 1) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration,
                                  nrow=4)
                vutils.save_image(torch.cat([
                    F.interpolate(real_image, 128),
                    rec_img_all, rec_img_small,
                    rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
                vutils.save_image(torch.cat([
                    F.interpolate(fake_images_noise[0], 128),
                    rec_fake_all, rec_fake_small,
                    rec_fake_part]).add(1).mul(0.5), saved_image_folder + '/rec_fake_%d.jpg' % iteration)
            load_params(netG, backup_para)

        if iteration % (save_interval * 50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g': netG.state_dict(), 'd': netD.state_dict()}, saved_model_folder + '/%d.pth' % iteration)
            load_params(netG, backup_para)
            torch.save({'g': netG.state_dict(),
                        'd': netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder + '/all_%d.pth' % iteration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str,
                        default='./100-shot-obama/img',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='obama', help='experiment name')
    parser.add_argument('--iter', type=int, default=100000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--lamda_MS_D', type=float, default=1.0)
    parser.add_argument('--lamda_MS_G', type=float, default=0.1)
    args = parser.parse_args()
    print(args)

    train(args)



import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

class PivotalTuning:
    def __init__(self, decoder, device, lr=1e-4, steps=300, l2_weight=1.0, lpips_weight=0.8):
        self.decoder = decoder
        self.device = device
        self.lr = lr
        self.steps = steps
        self.l2_weight = l2_weight
        self.lpips_weight = lpips_weight

        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)

    def check_requires_grad(self,tensor, name):
        print(f'{name} requires_grad={tensor.requires_grad}, grad_fn={tensor.grad_fn}')

    def resize_to_match(self, source_img, target_img):
        target_h, target_w = target_img.shape[2], target_img.shape[3]
        resized_img = F.interpolate(source_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return resized_img

    def run(self, wcode, origin_img):
        optimizer = torch.optim.Adam([wcode], lr=self.lr)   # 重点：把 wcode 当参数优化

        wcode = wcode.detach().clone().to(self.device).requires_grad_()   # 彻底可导
        origin_img = origin_img.to(self.device)
        origin_img.requires_grad = False

        losses = []

        for step in range(self.steps):
            # generated_img, _ = self.decoder([wcode], input_is_latent=True, randomize_noise=False)
            generated_img, _ = self.decoder(wcode)   # 直接算，别包 with torch.no_grad
            # decoder 内部参数不需要改 (可 frozen)
            for param in self.decoder.parameters():
                param.requires_grad = False

            # 调整图像尺寸
            if generated_img.shape != origin_img.shape:
                origin_img_resized = F.interpolate(origin_img, size=generated_img.shape[2:], mode='bilinear', align_corners=False)
            else:
                origin_img_resized = origin_img
            self.check_requires_grad(wcode, 'wcode')
            self.check_requires_grad(generated_img, 'generated_img')


            # 损失
            l2_loss = nn.MSELoss()(generated_img, origin_img_resized)
            lpips_loss = self.lpips_fn(generated_img, origin_img_resized).mean()
            total_loss = self.l2_weight * l2_loss + self.lpips_weight * lpips_loss
            losses.append(total_loss.item())
            self.check_requires_grad(total_loss, 'total_loss')
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0 or step == self.steps - 1:
                print(f"[PTI Step {step}] Loss: {total_loss.item():.6f}")

        return self.decoder, losses






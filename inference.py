import torch
import yaml
import warnings
import glob

from PIL import Image
from tqdm import tqdm
from models.mapper.passwordmapper import ReverseMapper
from models.mapper.passwordmapper import PasswordMapper
from options.test_options import TrainOptions
from options.test_options import TestOptions
from utils.function import *
from trainer import *

warnings.filterwarnings('ignore')

device = torch.device('cuda')

opts = TrainOptions().parse()
config = yaml.load(open('./configs/' + opts.config + '.yaml', 'r'), Loader=yaml.FullLoader)

trainer = Trainer(config, opts)
trainer.to(device)
trainer.load_checkpoint("./pretrained_models/model.pth")

img_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
input_resize = transforms.Resize([256,256])

# Your images to be anonymized
raw_annotations = glob.glob(r'./sample/origin/00000.jpg')
raw_annotations.sort()

with torch.no_grad():
    trainer.mapper.eval()
    for img_path in tqdm(raw_annotations):
        label = img_path.split('/')[-1]
        image = img_to_tensor(Image.open(img_path)).unsqueeze(0).to(device)
        input_resize = transforms.Resize([256,256])
        input = input_resize(image)
        ori_wcode = trainer.e4e.encoder(input)
        print(f"latent shape: {ori_wcode.shape}")
        if ori_wcode.ndim == 2:
            ori_wcode = ori_wcode.unsqueeze(1).repeat(1, 18, 1)

        if trainer.e4e.latent_avg.shape[1] != ori_wcode.shape[1]:
            latent_avg = trainer.e4e.latent_avg.unsqueeze(1).repeat(1, ori_wcode.shape[1], 1)
        else:
            latent_avg = trainer.e4e.latent_avg

        ori_wcode = ori_wcode + latent_avg
        ori_wcode = ori_wcode + trainer.e4e.latent_avg.repeat(ori_wcode.shape[0], 1, 1)

        recon_img, _ = trainer.StyleGAN([ori_wcode], input_is_latent=True, randomize_noise=False)


        ori_inversion_img, _ = trainer.StyleGAN([ori_wcode], input_is_latent=True, randomize_noise=False)
        p1, r_p1, inv_p, r_inv_p1, r_inv_p2 = generate_code(16, 1, device, inv=True, gen_random_WR=True)
        # p1=[1,1,1,0,0,1,0,0,1,1,0,0,1,0,1,0]
        # p1 = torch.tensor(p1, dtype=torch.float32).unsqueeze(0).to(device)
        # inv_p1=[0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,1]
        # inv_p1 = torch.tensor(inv_p1, dtype=torch.float32).unsqueeze(0).to(device)
        
        # p1=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        # p1 = torch.tensor(p1, dtype=torch.float32).unsqueeze(0).to(device)
        # inv_p1=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # inv_p1 = torch.tensor(inv_p1, dtype=torch.float32).unsqueeze(0).to(device)
        
        # p1=[0,1,1,1,0,1,1,0,0,1,0,0,1,1,0,0]
        # p1 = torch.tensor(p1, dtype=torch.float32).unsqueeze(0).to(device)
        # inv_p1=[1,0,0,0,1,0,0,1,1,0,1,1,0,0,1,1]
        # inv_p1 = torch.tensor(inv_p1, dtype=torch.float32).unsqueeze(0).to(device)
    
        wcode_fake, fake = trainer.mapper(ori_wcode, p1, opts.start_layer, opts.mapping_layers)
        wcode_recon, recon = trainer.mapper(wcode_fake, inv_p1, opts.start_layer, opts.mapping_layers)
        predicted_p = trainer.remapper(wcode_recon)
        predicted_p_embedding = trainer.mapper.passwordmapper(predicted_p)
        original_p_embedding = trainer.mapper.passwordmapper(r_p1)
        mse = torch.nn.functional.mse_loss(predicted_p_embedding, original_p_embedding)
        print(f'预测误差 (MSE): {mse.item()}')
        
        inv_img = tensor2im(ori_inversion_img[0])
        inv_img.save(f'./sample/inversion/{label}')        
        fake_img = tensor2im(fake[0])
        fake_img.save(f'./sample/anonymized/{label}')
        recon_img = tensor2im(recon_img[0])
    
        recon_img.save(f'./sample/recovered/{label}')
        










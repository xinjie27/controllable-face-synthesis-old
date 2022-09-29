# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

from copy import deepcopy
import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from mtcnn import MTCNN
from renderer import renderer128
from Deep3DFaceRecon.util.load_mats import load_lm3d
from Deep3DFaceRecon.util.preprocess import align_img
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
from torch_mtcnn import detect_faces

from functions import supress_stdout

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

RDR = renderer128
# DTR = MTCNN()

#----------------------------------------------------------------------------

class StyleGAN2Loss:
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.mean_gbuffer_loss  = torch.zeros([], device=device) 

    def run_G(self, z, m, c, alpha, update_emas=False):
        x, mean, std, RenderedFace, RenderedMask, RenderedFaceVertex, RenderedFaceTex, RenderedFaceNorm, RenderedLandmarks = self.G.run_enc(m)


        ws = self.G.mapping(z, mean, std, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), m, c, update_emas=False)[:, cutoff:]

        


        img, img_3dmm, face, vertex, tex, norm, landmarks = self.G.synthesis(
            x, 
            ws, 
            RenderedFace, 
            RenderedMask, 
            RenderedFaceVertex, 
            RenderedFaceTex, 
            RenderedFaceNorm, 
            RenderedLandmarks, 
            alpha, 
            update_emas=update_emas)


        return img, img_3dmm, ws, face, vertex, tex, norm, landmarks

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    # @supress_stdout
    def _get_landmarks(self,imgs):
        landmarks_list = []
        def _scale_img(img):
            lo, hi = [-1, 1]
            img = np.asarray(img, dtype=np.float32)
            img = (img - lo) * (255 / (hi - lo))
            img = np.rint(img).clip(0, 255).astype(np.uint8)
            return img
        for img in imgs:
            img = img.detach().cpu().permute(1,2,0).numpy()
            img = _scale_img(img)
            try:
                # landmarks = DTR.detect_faces(img_permuted)[0]["keypoints"]
                _, landmarks = detect_faces(Image.fromarray(img))
            except IndexError:
                print("IndexError: _get_landmarks")
            if landmarks.size == 0:
                landmarks = np.zeros((5,2))
            landmarks = landmarks.reshape((2,5)).transpose()
            landmarks_list.append(landmarks.astype(np.float32))
        return landmarks_list
    
    def _get_coeffs(self, imgs, landmarks, renderer,to_tensor=True):
        # lm3d_std = load_lm3d(RDR.opt.bfm_folder)
        # coeffs_batch = torch.zeros((imgs.shape[0], 257)).to("cuda:0")
        _,W,H = imgs[0].shape
        scaleF = 224 / W
        imgs = F.interpolate((imgs+1)/2, scale_factor=(scaleF,scaleF), mode='bilinear', align_corners=False, antialias=True)
        imgs = torch.clamp(imgs, min=0, max=1)
        
        # i = 0
        # for img, lm in zip(imgs, landmarks):
        #     # lm[:, -1] = H - 1 - lm[:, -1]
        #     # lm = lm.detach().cpu().numpy().astype(np.float32)
        #     # img = T.ToPILImage()((img + 1) / 2)
        #     # img = img.resize((224,224))

        #     # _, img, lm, _ = align_img(img, lm * W / 224, lm3d_std)
        #     # img.save(f'./test/align/{i}_aligned.png')

        #     if to_tensor:
        #         # img = torch.tensor(np.array(img)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        #         img = img.unsqueeze(0)
        #         lm = lm.unsqueeze(0)
        #         # lm = torch.tensor(lm).unsqueeze(0)
        #     data = {
        #     'imgs': img,
        #     'lms': lm
        #     }
        #     renderer.model.set_input(data)  # unpack data from data loader
        #     renderer.model.test()           # run inference
        #     # visuals = renderer.model.get_current_visuals()  # get image results
        #     # renderer.visualizer.display_current_results(visuals, 0, renderer.opt.epoch, dataset=name.split(os.path.sep)[-1], 
        #     # save_results=True, count=i, name=img_name, add_image=False)
        #     coeffs_batch[i] = renderer.model.pred_coeffs
        #     i += 1

        data = {
            'imgs': imgs,
            'lms': landmarks
            }

        renderer.model.set_input(data)  # unpack data from data loader
        renderer.model.forward()           # run inference
        coeffs_batch = renderer.model.pred_coeffs
        
        return coeffs_batch


    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_m, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                loss_3dmm_fade_kimg = 64 * 1000 / 32
                loss_3dmm_weight = 20#min(cur_nimg / (loss_3dmm_fade_kimg * 1e3), 1) * 20

                alpha = min(cur_nimg / (loss_3dmm_fade_kimg * 1e3), 1)
                training_stats.report('Loss/alpha', alpha)

                gen_img, gen_img_3dmm, _gen_ws, GBuffer_face, GBuffer_vertex, GBuffer_tex, GBuffer_norm, landmarks = self.run_G(gen_z, gen_m, gen_c, alpha)
                # lm_list = self._get_landmarks(gen_img)
                coeffs_batch = self._get_coeffs(gen_img_3dmm, landmarks, RDR) # shape (batch_size, 257), same as gen_m

                # 3DMM Loss
                GBuffer_face_recon, Mask, GBuffer_vertex_recon, GBuffer_tex_recon, GBuffer_norm_recon, _ = RDR.render(coeffs_batch)

                Zeros = torch.ones_like(GBuffer_face_recon, device=GBuffer_face_recon.device)
                GBuffer_face_recon = GBuffer_face_recon * Mask + Zeros * (1 - Mask)
                GBuffer_tex_recon = GBuffer_tex_recon * Mask + Zeros * (1 - Mask)
                    
                GBuffer_face_loss = torch.mean(torch.square(GBuffer_face - GBuffer_face_recon), dim=(1,2,3))
                # GBuffer_vertex_loss = torch.mean(torch.square(GBuffer_vertex - GBuffer_vertex_recon), dim=(1,2,3))
                GBuffer_tex_loss = torch.mean(torch.square(GBuffer_tex - GBuffer_tex_recon), dim=(1,2,3))
                GBuffer_norm_loss = torch.mean(torch.square(GBuffer_norm - GBuffer_norm_recon), dim=(1,2,3))

                GBuffer_face_loss_detached = GBuffer_face_loss.clone().detach()
                for i in range(GBuffer_face_recon.shape[0]):
                    if GBuffer_face_loss[i] > 5 * self.mean_gbuffer_loss:
                        if not os.path.exists(f'./test/recon/{cur_nimg}'):
                            os.makedirs(f'./test/recon/{cur_nimg}')
                        RDR.visualize(GBuffer_face[i], f'./test/recon/{cur_nimg}/{i}_face.png')
                        RDR.visualize(GBuffer_face_recon[i], f'./test/recon/{cur_nimg}/{i}_face_recon.png')
                        RDR.visualize(GBuffer_tex[i], f'./test/recon/{cur_nimg}/{i}_tex.png')
                        RDR.visualize(GBuffer_tex_recon[i], f'./test/recon/{cur_nimg}/{i}_tex_recon.png')
                        RDR.visualize(GBuffer_norm[i], f'./test/recon/{cur_nimg}/{i}_norm.png')
                        RDR.visualize(GBuffer_norm_recon[i], f'./test/recon/{cur_nimg}/{i}_norm_recon.png')
                        RDR.visualize(gen_img[i], f'./test/recon/{cur_nimg}/{i}_generated.png')
                        RDR.visualize(gen_img_3dmm[i], f'./test/recon/{cur_nimg}/{i}_generated_blend.png')
                       # GBuffer_face_loss[i] *= 0
                # print(torch.eq(GBuffer_face_loss_detached, GBuffer_face_loss))
                mean_gbuffer_loss = self.mean_gbuffer_loss.lerp(GBuffer_face_loss_detached.mean(), 0.01)
                self.mean_gbuffer_loss.copy_(mean_gbuffer_loss.detach()) 




                loss_3dmm = loss_3dmm_weight * (GBuffer_face_loss + GBuffer_tex_loss + GBuffer_norm_loss)

                #print(f"face loss: {GBuffer_face_loss}, vertex loss: {GBuffer_vertex_loss}, tex loss: {GBuffer_tex_loss}, norm_loss: {GBuffer_norm_loss}")
                training_stats.report('Loss/face', GBuffer_face_loss)
                training_stats.report('Loss/tex', GBuffer_tex_loss)
                training_stats.report('Loss/norm', GBuffer_norm_loss)
                training_stats.report('Loss/3dmm', loss_3dmm)
                training_stats.report('Loss/3dmm_weight', loss_3dmm_weight)
                # print("face loss:", GBuffer_face_loss.mean(), GBuffer_face_loss.shape)
                # training_stats.report('Loss/vertex', GBuffer_vertex_loss)
                # training_stats.report('Loss/tex', GBuffer_tex_loss)
                # training_stats.report('Loss/norm', GBuffer_norm_loss)

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits) + loss_3dmm.view(loss_3dmm.shape[0], 1) # -log(sigmoid(gen_logits))
                # print('loss_Gmain:', loss_Gmain.mean(), loss_Gmain.shape)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        # if phase in ['Greg', 'Gboth']:
        #     with torch.autograd.profiler.record_function('Gpl_forward'):
        #         batch_size = gen_z.shape[0] // self.pl_batch_shrink
        #         gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_m[:batch_size], gen_c[:batch_size])
        #         pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        #         with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
        #             pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        #         pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        #         pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        #         self.pl_mean.copy_(pl_mean.detach())
        #         pl_penalty = (pl_lengths - pl_mean).square()
        #         training_stats.report('Loss/pl_penalty', pl_penalty)
        #         loss_Gpl = pl_penalty * self.pl_weight
        #         training_stats.report('Loss/G/reg', loss_Gpl)
        #     with torch.autograd.profiler.record_function('Gpl_backward'):
        #         loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img = self.run_G(gen_z, gen_m, gen_c, alpha=1, update_emas=True)[0]
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------

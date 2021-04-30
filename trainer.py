import os
import time
import torch
import datetime
import logging
import torch.nn.functional as F
from torchvision.utils import save_image
from modules import Encoder, Transformer, Generator, Reconstructor, Discriminator



class Trainer(object):
    def __init__(self, celeba_loader, config):
        # miscellaneous
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data loader
        self.dataload = celeba_loader

        # model configurations
        self.c64 = config.c64
        self.c256 = config.c256
        self.c2048 = config.c2048
        self.rb6 = config.rb6
        self.attr_dim = config.attr_dim
        self.hair_dim = config.hair_dim

        # training configurations
        self.selected_attrs = config.selected_attrs
        self.train_iters = config.train_iters
        self.num_iters_decay = config.num_iters_decay
        self.n_critic = config.n_critic
        self.d_lr = config.d_lr
        self.r_lr = config.r_lr
        self.t_lr = config.t_lr
        self.e_lr = config.e_lr
        self.decay_rate = config.decay_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.lambda_cls = config.lambda_cls
        self.lambda_cyc = config.lambda_cyc
        self.lambda_gp = config.lambda_gp

        # test configurations
        self.test_iters = config.test_iters

        # directories
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        self.log_dir = config.log_dir

        # step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # initial models
        self.build_models()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_models(self):
        self.E = Encoder(self.c64, self.rb6)
        self.T_Hair = Transformer(self.hair_dim, self.c256, self.rb6)
        self.T_Gender = Transformer(self.attr_dim, self.c256, self.rb6)
        self.T_Smailing = Transformer(self.attr_dim, self.c256, self.rb6)
        self.R = Reconstructor(self.c256)
        self.D_Hair = Discriminator(self.hair_dim, self.c64)
        self.D_Gender = Discriminator(self.attr_dim, self.c64)
        self.D_Smailing = Discriminator(self.attr_dim, self.c64)

        self.e_optim = torch.optim.Adam(self.E.parameters(), self.e_lr, [self.beta1, self.beta2])
        self.th_optim = torch.optim.Adam(self.T_Hair.parameters(), self.t_lr, [self.beta1, self.beta2])
        self.tg_optim = torch.optim.Adam(self.T_Gender.parameters(), self.t_lr, [self.beta1, self.beta2])
        self.ts_optim = torch.optim.Adam(self.T_Smailing.parameters(), self.t_lr, [self.beta1, self.beta2])
        self.r_optim = torch.optim.Adam(self.R.parameters(), self.r_lr, [self.beta1, self.beta2])
        self.dh_optim = torch.optim.Adam(self.D_Hair.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.dg_optim = torch.optim.Adam(self.D_Gender.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.ds_optim = torch.optim.Adam(self.D_Smailing.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.E, 'Encoder')
        self.print_network(self.T_Hair, 'Transformer for Hair Color')
        self.print_network(self.T_Gender, 'Transformer for Gender')
        self.print_network(self.T_Smailing, 'Transformer for Smailing')
        self.print_network(self.R, 'Reconstructor')
        self.print_network(self.D_Hair, 'D for Hair Color')
        self.print_network(self.D_Gender, 'D for Gender')
        self.print_network(self.D_Smailing, 'D for Smailing')

        self.E.to(self.device)
        self.T_Hair.to(self.device)
        self.T_Gender.to(self.device)
        self.T_Smailing.to(self.device)
        self.R.to(self.device)
        self.D_Gender.to(self.device)
        self.D_Smailing.to(self.device)
        self.D_Hair.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        print(name)
        print("The number of parameters: {}".format(num_params))
        print(model)
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def reset_grad(self):
        self.e_optim.zero_grad()
        self.th_optim.zero_grad()
        self.tg_optim.zero_grad()
        self.ts_optim.zero_grad()
        self.r_optim.zero_grad()
        self.dh_optim.zero_grad()
        self.dg_optim.zero_grad()
        self.ds_optim.zero_grad()

    def update_lr(self, e_lr, d_lr, r_lr, t_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.e_optim.param_groups:
            param_group['lr'] = e_lr
        for param_group in self.dh_optim.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.dg_optim.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.ds_optim.param_groups:
            param_group['lr'] = d_lr
        for param_group in self.r_optim.param_groups:
            param_group['lr'] = r_lr
        for param_group in self.th_optim.param_groups:
            param_group['lr'] = t_lr
        for param_group in self.tg_optim.param_groups:
            param_group['lr'] = t_lr
        for param_group in self.ts_optim.param_groups:
            param_group['lr'] = t_lr

    def create_labels(self, c_org, c_dim=5, selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair']:
                hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def train(self):
        data_loader = self.dataload

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, 5, self.selected_attrs)

        d_lr = self.d_lr
        r_lr = self.r_lr
        t_lr = self.t_lr
        e_lr = self.e_lr

        # Start training
        print('Starting point==============================')
        start_time = time.time()

        for i in range(0, self.train_iters):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels
            try:
                x_real, label_real = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_real = next(data_iter)

            rand_idx = torch.randperm(label_real.size(0))
            label_feak = label_real[rand_idx]

            x_real = x_real.to(self.device)
            # labels for hair color
            label_h_real = label_real[:, 0:3]
            label_h_feak = label_feak[:, 0:3]
            # labels for gender
            label_g_real = label_real[:, 3:4]
            label_g_feak = label_feak[:, 3:4]
            # labels for smailing
            label_s_real = label_real[:, 4:]
            label_s_feak = label_feak[:, 4:]

            label_h_real = label_h_real.to(self.device)
            label_h_feak = label_h_feak.to(self.device)
            label_g_real = label_g_real.to(self.device)
            label_g_feak = label_g_feak.to(self.device)
            label_s_real = label_s_real.to(self.device)
            label_s_feak = label_s_feak.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Computer loss with real images
            h_src, h_cls = self.D_Hair(x_real)
            d_h_loss_real = -torch.mean(h_src)
            d_h_loss_cls = F.binary_cross_entropy_with_logits(h_cls, label_h_real, reduction='sum') / h_cls.size(0)

            g_src, g_cls = self.D_Gender(x_real)
            d_g_loss_real = -torch.mean(g_src)
            d_g_loss_cls = F.binary_cross_entropy_with_logits(g_cls, label_g_real, reduction='sum') / g_cls.size(0)

            s_src, s_cls = self.D_Smailing(x_real)
            d_s_loss_real = -torch.mean(s_src)
            d_s_loss_cls = F.binary_cross_entropy_with_logits(s_cls, label_s_real, reduction='sum') / s_cls.size(0)

            # Generate fake images and computer loss
            # Retrieve features of real image
            features = self.E(x_real)
            # Transform attributes from one value to an other
            t_h_features = self.T_Hair(features.detach(), label_h_feak)
            t_g_features = self.T_Gender(features.detach(), label_g_feak)
            t_s_features = self.T_Smailing(features.detach(), label_s_feak)
            # Reconstruct images from transformed attributes
            x_h_feak = self.R(t_h_features.detach())
            x_g_feak = self.R(t_g_features.detach())
            x_s_feak = self.R(t_s_features.detach())

            # Computer loss with fake images
            h_src, h_cls = self.D_Hair(x_h_feak.detach())
            d_h_loss_fake = torch.mean(h_src)

            g_src, g_cls = self.D_Gender(x_g_feak.detach())
            d_g_loss_fake = torch.mean(g_src)

            s_src, s_cls = self.D_Smailing(x_s_feak.detach())
            d_s_loss_fake = torch.mean(s_src)

            # Compute loss for gradient penalty
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_h_hat = (alpha * x_real.data + (1 - alpha) * x_h_feak.data).requires_grad_(True)
            #x_h_hat = (alpha * x_real.data + (1-alpha) * x_h_feak.data).requires_grad_(True).to(torch.float16)
            x_g_hat = (alpha * x_real.data + (1 - alpha) * x_g_feak.data).requires_grad_(True)
            #x_g_hat = (alpha * x_real.data + (1-alpha) * x_g_feak.data).requires_grad_(True).to(torch.float16)
            x_s_hat = (alpha * x_real.data + (1 - alpha) * x_s_feak.data).requires_grad_(True)
            #x_s_hat = (alpha * x_real.data + (1-alpha) * x_s_feak.data).requires_grad_(True).to(torch.float16)

            out_src, _ = self.D_Hair(x_h_hat)
            d_h_loss_gp = self.gradient_penalty(out_src, x_h_hat)
            out_src, _ = self.D_Gender(x_g_hat)
            d_g_loss_gp = self.gradient_penalty(out_src, x_g_hat)
            out_src, _ = self.D_Smailing(x_s_hat)
            d_s_loss_gp = self.gradient_penalty(out_src, x_s_hat)

            # Backward and optimize
            d_loss = d_h_loss_real + d_g_loss_real + d_s_loss_real + \
                     d_h_loss_fake + d_g_loss_fake + d_s_loss_fake + \
                     self.lambda_gp * (d_h_loss_gp + d_g_loss_gp + d_s_loss_gp) + \
                     self.lambda_cls * (d_h_loss_cls + d_g_loss_cls + d_s_loss_cls)
            #d_loss = d_h_loss_real + d_h_loss_fake + self.lambda_gp * d_h_loss_gp + self.lambda_cls * d_h_loss_cls


            self.reset_grad()
            d_loss.backward()
            self.dh_optim.step()
            self.dg_optim.step()
            self.ds_optim.step()

            # Logging
            loss = {}
            loss['D/h_loss_real'] = d_h_loss_real.item()
            loss['D/g_loss_real'] = d_g_loss_real.item()
            loss['D/s_loss_real'] = d_s_loss_real.item()
            loss['D/h_loss_fake'] = d_h_loss_fake.item()
            loss['D/g_loss_fake'] = d_g_loss_fake.item()
            loss['D/s_loss_fake'] = d_s_loss_fake.item()
            loss['D/h_loss_cls'] = d_h_loss_cls.item()
            loss['D/g_loss_cls'] = d_g_loss_cls.item()
            loss['D/s_loss_cls'] = d_s_loss_cls.item()
            loss['D/h_loss_gp'] = d_h_loss_gp.item()
            loss['D/g_loss_gp'] = d_g_loss_gp.item()
            loss['D/s_loss_gp'] = d_s_loss_gp.item()

            # =================================================================================== #
            #                  3. Train the encoder, transformer and reconstructor                #
            # =================================================================================== #

            if(i+1) % self.n_critic == 0:
                # Generate fake images and compute loss
                # Retrieve features of real image
                features = self.E(x_real)
                # Transform attributes from one value to an other
                t_h_features = self.T_Hair(features, label_h_feak)
                t_g_features = self.T_Gender(features, label_g_feak)
                t_s_features = self.T_Smailing(features, label_s_feak)
                # Reconstruct images from transformed attributes
                x_h_feak = self.R(t_h_features)
                x_g_feak = self.R(t_g_features)
                x_s_feak = self.R(t_s_features)

                # Computer loss with fake images
                h_src, h_cls = self.D_Hair(x_h_feak)
                etr_h_loss_fake = -torch.mean(h_src)
                etr_h_loss_cls = F.binary_cross_entropy_with_logits(h_cls, label_h_feak, reduction='sum') / h_cls.size(0)

                g_src, g_cls = self.D_Gender(x_g_feak)
                etr_g_loss_fake = -torch.mean(g_src)
                etr_g_loss_cls = F.binary_cross_entropy_with_logits(g_cls, label_g_feak, reduction='sum') / g_cls.size(0)

                s_src, s_cls = self.D_Smailing(x_s_feak)
                etr_s_loss_fake = -torch.mean(s_src)
                etr_s_loss_cls = F.binary_cross_entropy_with_logits(s_cls, label_s_feak, reduction='sum') / s_cls.size(0)

                # Real - Encoder - Reconstructor - Real loss
                x_re = self.R(features)
                er_loss_cyc = torch.mean(torch.abs(x_re - x_real))

                # Real - Encoder - Transform, Real - Encoder - Transform - Reconstructor - Encoder loss
                h_fake_features = self.E(x_h_feak)
                g_fake_features = self.E(x_g_feak)
                s_fake_features = self.E(x_s_feak)

                etr_h_loss_cyc = torch.mean(torch.abs(t_h_features - h_fake_features))
                etr_g_loss_cyc = torch.mean(torch.abs(t_g_features - g_fake_features))
                etr_s_loss_cyc = torch.mean(torch.abs(t_s_features - s_fake_features))

                # Backward and optimize
                etr_loss = etr_h_loss_fake + etr_g_loss_fake + etr_s_loss_fake + \
                           self.lambda_cls * (etr_h_loss_cls + etr_g_loss_cls + etr_s_loss_cls) + \
                           self.lambda_cyc * (er_loss_cyc + etr_h_loss_cyc + etr_g_loss_cyc + etr_s_loss_cyc)
                #etr_loss = etr_h_loss_fake + self.lambda_cls * etr_h_loss_cls + self.lambda_cyc * (er_loss_cyc + etr_h_loss_cyc)



                self.reset_grad()
                etr_loss.backward()
                self.e_optim.step()
                self.th_optim.step()
                self.tg_optim.step()
                self.ts_optim.step()
                self.r_optim.step()

                # Logging.
                loss['ETR/h_loss_fake'] = etr_h_loss_fake.item()
                loss['ETR/g_loss_fake'] = etr_g_loss_fake.item()
                loss['ETR/s_loss_fake'] = etr_s_loss_fake.item()
                loss['ETR/h_loss_cls'] = etr_h_loss_cls.item()
                loss['ETR/g_loss_cls'] = etr_g_loss_cls.item()
                loss['ETR/s_loss_cls'] = etr_s_loss_cls.item()
                loss['ER/er_loss_cyc'] = er_loss_cyc.item()
                loss['ETR/h_loss_cyc'] = etr_h_loss_cyc.item()
                loss['ETR/g_loss_cyc'] = etr_g_loss_cyc.item()
                loss['ETR/s_loss_cyc'] = etr_s_loss_cyc.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Translate fixed images for debugging.
            if (i + 1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        xf = self.E(x_fixed)
                        xth = self.T_Hair(xf, c_fixed[:, 0:3])
                        xtg = self.T_Gender(xth, c_fixed[:, 3:4])
                        xts = self.T_Smailing(xtg, c_fixed[:, 4:5])
                        x_fake_list.append(self.R(xts))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i + 1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Print out training information.
            if (i + 1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.train_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                
                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # save model checkpoints
            if (i+1) % self.model_save_step == 0:
                E_path = os.path.join(self.model_save_dir, '{}-E.ckpt'.format(i+1))
                D_h_path = os.path.join(self.model_save_dir, '{}-D_h.ckpt'.format(i+1))
                D_g_path = os.path.join(self.model_save_dir, '{}-D_g.ckpt'.format(i+1))
                D_s_path = os.path.join(self.model_save_dir, '{}-D_s.ckpt'.format(i+1))
                R_path = os.path.join(self.model_save_dir, '{}-R.ckpt'.format(i+1))
                T_h_path = os.path.join(self.model_save_dir, '{}-T_h.ckpt'.format(i+1))
                T_g_path = os.path.join(self.model_save_dir, '{}-T_g.ckpt'.format(i+1))
                T_s_path = os.path.join(self.model_save_dir, '{}-T_s.ckpt'.format(i+1))
                torch.save(self.E.state_dict(), E_path)
                torch.save(self.D_Hair.state_dict(), D_h_path)
                torch.save(self.D_Gender.state_dict(), D_g_path)
                torch.save(self.D_Smailing.state_dict(), D_s_path)
                torch.save(self.R.state_dict(), R_path)
                torch.save(self.T_Hair.state_dict(), T_h_path)
                torch.save(self.T_Gender.state_dict(), T_g_path)
                torch.save(self.T_Smailing.state_dict(), T_s_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # decay learning rates
            if (i+1) % self.lr_update_step == 0 and (i+1) > self.num_iters_decay:
                e_lr -= (self.e_lr / float(self.decay_rate))
                d_lr -= (self.d_lr / float(self.decay_rate))
                r_lr -= (self.r_lr / float(self.decay_rate))
                t_lr -= (self.t_lr / float(self.decay_rate))
                self.update_lr(e_lr, d_lr, r_lr, t_lr)
                print ('Decayed learning rates, e_lr: {}, d_lr: {}, r_lr: {}, t_lr: {}.'.format(e_lr, d_lr, r_lr, t_lr))
















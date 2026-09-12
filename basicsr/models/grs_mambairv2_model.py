import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class GRSMambaIRv2Model(SRModel):
    """L1-only RGB/depth training, with baseline-identical evaluation partitions."""

    def feed_data(self, data):
        super().feed_data(data)
        self.depth = data['depth'].to(self.device)

    def optimize_parameters(self, current_iter):
        # GRS first experiment: preserve baseline optimizer/scheduler; L1 only.
        if self.cri_pix is None or self.cri_perceptual is not None:
            raise ValueError('GRS first experiment requires pixel loss only.')
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.depth)
        loss = self.cri_pix(self.output, self.gt)
        loss.backward()
        self.optimizer_g.step()
        stats = self.get_bare_model(self.net_g).geometry_stats
        self.log_dict = self.reduce_loss_dict(dict(l_pix=loss, **stats))
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test_selfensemble(self):
        raise NotImplementedError('Use baseline-matched partition inference for GRS evaluation.')

    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # GRS: apply exactly the RGB partition padding to depth.
        depth_img = F.pad(self.depth, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        depth_chops = []
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
            depth_chops.append(depth_img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop, depth_chop in zip(img_chops, depth_chops):
                    out = self.net_g_ema(chop, depth_chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop, depth_chop in zip(img_chops, depth_chops):
                    out = self.net_g(chop, depth_chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

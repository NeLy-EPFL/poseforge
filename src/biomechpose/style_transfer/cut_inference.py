import torch
import numpy as np
import torchvision
from pathlib import Path
from PIL import Image
from cut.models.cut_model import CUTModel
from cut.options.option_stats import OptionsWrapper


class Options:
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        ngf: int,
        netG: str,
        image_side_length: int,
        nce_layers: list[int],
    ):
        # Variables listed as "don't-care" are not actually used in the inference
        # pipeline, but they are needed for initializing the model instances
        self.isTrain = False
        self.gpu_ids = [0]
        self.checkpoints_dir = ""  # don't-care
        self.name = ""  # don't-care
        self.preprocess = None  # don't-care
        self.nce_layers = ",".join([str(x) for x in nce_layers])  # don't-care
        self.nce_idt = True  # don't-care
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.netG = netG
        self.normG = "instance"  # don't-care
        self.no_dropout = True  # don't-care
        self.init_type = None  # don't-care
        self.init_gain = None  # don't-care
        self.no_antialias = False  # don't-care
        self.no_antialias_up = False  # don't-care
        self.load_size = image_side_length
        self.crop_size = image_side_length
        self.stylegan2_G_num_downsampling = 1
        self.netF = "mlp_sample"  # don't-care
        self.netF_nc = None


class InferencePipeline:
    def __init__(
        self,
        netG_ckpt_path: Path,
        *,
        input_nc: int,
        output_nc: int,
        ngf: int,
        netG: str,
        image_side_length: int,
        nce_layers: list[int],
        device: str | torch.device = "cuda",
    ):
        if device == "cpu":
            raise NotImplementedError(
                "CPU inference is still buggy (TODO: figure this out). Use GPU instead."
            )

        _opt = Options(input_nc, output_nc, ngf, netG, image_side_length, nce_layers)
        self.opt = OptionsWrapper(_opt)
        self.model = CUTModel(self.opt)
        self.input_nc = input_nc
        self.image_side_length = image_side_length
        self._is_model_initialized = False
        self.netG_ckpt_path = netG_ckpt_path
        self.device = device
        normalize_mean = (0.5,) * input_nc
        normalize_std = (0.5,) * input_nc
        self._input_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((image_side_length, image_side_length)),
                torchvision.transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        self._denormalize_transform = torchvision.transforms.Normalize(
            tuple(-m / s for m, s in zip(normalize_mean, normalize_std)),
            tuple(1 / s for s in normalize_std),
        )

    def infer(self, input_images: list[Image.Image]) -> np.ndarray:
        input_images_transformed = torch.stack(
            [self._input_transforms(img) for img in input_images]
        ).to(self.device)

        # Data-dependent model initialization
        if not self._is_model_initialized:
            self._initialize_model(input_images_transformed)

        with torch.no_grad():
            output_images = self.model.netG(input_images_transformed)
            output_images = self._denormalize_transform(output_images)

        # Permute to (batch_size, height, width, channels) as expected by
        # non-torch libraries, and convert to uint8 numpy arrays
        output_images = output_images.permute(0, 2, 3, 1)
        output_images = output_images.detach().cpu().numpy()
        output_images = (output_images * 255).clip(0, 255).astype(np.uint8)
        return output_images

    def _initialize_model(self, input_images):
        # Move model to device first - this ensures ALL components are on the correct device
        self.model.netG = self.model.netG.to(self.device)
        self.model.netF = self.model.netF.to(self.device)
        # no netD during inference - used only during training

        # Data-dependent initialization
        # When forward is called, the forward method of the PatchSampleF
        # layers will call create_mlp based on feature shape
        self.model.netG(input_images)

        # Load model weights
        net = getattr(self.model, "netG")
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(self.netG_ckpt_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict)

        # Ensure model is on device after loading weights
        self.model.netG = self.model.netG.to(self.device)
        self.model.netF = self.model.netF.to(self.device)

        # Print network architecture
        self.model.print_networks(verbose=True)

        # Parallelize (only useful for multi-GPU case)
        if self.device != "cpu":
            parallel_net = torch.nn.DataParallel(
                self.model.netG, device_ids=self.opt.gpu_ids
            )
            setattr(self.model, "netG", parallel_net)

        self._is_model_initialized = True

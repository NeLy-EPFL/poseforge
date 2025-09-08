import torch
import numpy as np
import torchvision
import logging
import gc
import imageio.v2 as imageio
from tqdm import trange
from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from cut.models.cut_model import CUTModel
from cut.options.option_stats import OptionsWrapper


class _CUTOptions:
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
        print_architecture: bool = False,
    ):
        _opt = _CUTOptions(input_nc, output_nc, ngf, netG, image_side_length, nce_layers)
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
        self._print_architecture = print_architecture

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
        if self._print_architecture:
            self.model.print_networks(verbose=True)

        # Parallelize (only useful for multi-GPU case)
        if self.device != "cpu":
            parallel_net = torch.nn.DataParallel(
                self.model.netG, device_ids=self.opt.gpu_ids
            )
            setattr(self.model, "netG", parallel_net)

        self._is_model_initialized = True

    def detect_max_batch_size(
        self,
        input_image_shape: tuple,
        exponential: bool = True,
        start: int = 1,
        end: int | None = None,
    ) -> int:
        """Detect the maximum batch size that can be used for inference
        without running out of memory. The batch size tested starts from
        `start` and increases either exponentially (if `exponential` is True)
        or linearly (if `exponential` is False) until out-of-memory is
        detected. If `end` is specified, the search will stop once the batch
        size reaches `end`."""
        batch_size_hist = [start]
        logging.info("Detecting maximum batch size for inference...")
        while True:
            batch_size = batch_size_hist[-1]
            if end is not None and batch_size > end:
                logging.info(f"Exceeded maximum batch size limit: {end}")
                break
            logging.info(f"Testing batch size: {batch_size}")
            try:
                dummy_input_arr = np.zeros(
                    (batch_size, *input_image_shape), dtype=np.uint8
                )
                dummy_input_pil = [to_pil_image(frame) for frame in dummy_input_arr]
                self.infer(dummy_input_pil)
                if exponential:
                    batch_size_hist.append(batch_size * 2)
                else:
                    batch_size_hist.append(batch_size + 1)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    logging.info(
                        f"Out-of-memory error detected at batch size: {batch_size}."
                    )
                    break
                else:
                    raise e

        del dummy_input_arr, dummy_input_pil
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        max_batch_size = batch_size_hist[-2]
        logging.info(f"Maximum batch size is: {max_batch_size}")
        return max_batch_size


def process_simulation(
    inference_pipeline: InferencePipeline,
    input_video_path: Path,
    output_video_path: Path,
    batch_size: int | None = None,
    progress_bar: bool = True,
) -> None:
    """Run style transfer on a NeuroMechFly-rendered behavior clip using a
    trained style transfer model.

    Args:
        inference_pipeline (InferencePipeline): The
            `biomechpose.style_transfer.InferencePipeline` object for
            a trained style transfer model.
        input_video_path (Path): Path to the input video (i.e.
            NeuroMechFly-rendered behavior clip).
        output_video_path (Path): Path to the output video (i.e. the video
            styled to look like experimental data will be saved here).
        batch_size (int | None, optional): Number of frames to process in
            each batch during inference. If None, the maximum possible
            batch size will be automatically detected. Defaults to None.
        progress_bar (bool, optional): Whether to show a progress bar during
            inference. Defaults to True.
    """
    # Load input video
    with imageio.get_reader(str(input_video_path), "ffmpeg") as reader:
        fps = reader.get_meta_data()["fps"]
        video_frames = [frame for frame in reader]

    # Auto-detect batch size if not specified
    if batch_size is None:
        batch_size = inference_pipeline.detect_max_batch_size(
            video_frames[0].shape, start=1, end=len(video_frames)
        )

    # Create output directory if it doesn't exist
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video writer
    with imageio.get_writer(
        str(output_video_path),
        "ffmpeg",
        fps=fps,
        codec="libx264",
        quality=10,  # 10 is highest for imageio, lower is lower quality
        ffmpeg_params=["-crf", "18", "-preset", "slow"],  # lower crf = higher quality
    ) as video_writer:
        # Process frames in batches
        for i in trange(0, len(video_frames), batch_size, disable=not progress_bar):
            # Get batch of frames (each frame is HWC format, uint8, 0-255)
            input_batch_frames = np.array(video_frames[i : i + batch_size])
            input_batch_pil = [to_pil_image(frame) for frame in input_batch_frames]
            output_batch = inference_pipeline.infer(input_batch_pil)
            for j in range(output_batch.shape[0]):
                frame = output_batch[j]
                video_writer.append_data(frame)


def get_inference_pipeline(
    checkpoint_path: Path, params: dict, device: str | torch.device = "cuda"
) -> InferencePipeline:
    """Create a `biomechpose.style_transfer.InferencePipeline` object that
    can be used to inference on a trained style transfer model.

    Args:
        checkpoint_path (Path): Path to the model checkpoint.
        params (dict): Hyperparameters of the model architecture required
            to create the torch Module.
        device (str | torch.device, optional): Torch device to run
            inference on. Defaults to "cuda" (strongly recommended).

    Returns:
        InferencePipeline: The inference pipeline for style transfer.
    """
    # Some parameters can be assumed
    input_nc = params.get("input_nc", 3)
    output_nc = params.get("output_nc", 3)
    image_side_length = params.get("image_side_length", 256)
    nce_layers = params.get("nce_layers", [0, 4, 8, 12, 16])

    # Other parameters must be stated explicitly
    ngf = params["ngf"]
    netG = params["net"]

    # Determine device ("cpu" for CPU vs. "cuda" for GPU)
    if device == "cuda" and not torch.cuda.is_available():
        logging.error("CUDA device requested but not available. Using CPU instead.")
        device = "cpu"

    inference_pipeline = InferencePipeline(
        checkpoint_path,
        input_nc=input_nc,
        output_nc=output_nc,
        ngf=ngf,
        netG=netG,
        image_side_length=image_side_length,
        nce_layers=nce_layers,
        device=device,
    )
    return inference_pipeline

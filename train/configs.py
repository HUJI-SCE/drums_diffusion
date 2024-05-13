from dataclasses import dataclass
import os
@dataclass
class TrainingConfig:
    #LOGGING CONFIGS
    wandb = "055dd77a9574488aae64d39a2a77a6376d7b1fa5"

    # PATHS CONFIGS
    # root_dir = ""
    root_dir =os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir,"out_dir")
    path_to_components = os.path.join(root_dir,'models')
    path_to_class_file = os.path.join(root_dir,"inference/pipline.py")
    """ the path to the real annotation file """
    annotation_file_path = os.path.join(root_dir,"training/train/annotation.csv")
    """ the path to the simulation annotation file """
    # annotation_file_path = os.path.join(root_dir,"training/train/annotation_simulation.csv")

    # TRAINING CONFIGS
    loss_noise_factor = 0.3
    loss_diff_drums_factor = 1 - loss_noise_factor

    train_batch_size = 32
    num_epochs = 50

    #TODO: change the eval_batch_size to a wanted number
    eval_batch_size = num_epochs + 1  # how many images to sample during evaluation

    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    eval_epoch = 10
    save_model_epochs = 5
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision

    # HUGGINGFACE CONFIGS
    #TODO: change push_to_hub to True if you want to push the model to the hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_model_id = "michaelpiro1/drums_diff"  # the name of the repository to create on the HF Hub
    hub_private_repo = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

    #AUDIO CONFIGS
    SAMPLE_RATE = 44100
    N_FFT = 1024
    HOP_LENGTH = 160
    WIN_LENGTH = 1024
    N_MELS = 64
    FMAX = int(SAMPLE_RATE / 2)
    FMIN = 0
    AUDIO_LEN_SEC = 5
    TARGET_MEL_LENGTH = 256
    NUM_SAMPLES = int((TARGET_MEL_LENGTH - 1) * HOP_LENGTH)
    TARGET_LENGTH_SEC = NUM_SAMPLES / SAMPLE_RATE

    #DEMUCS CONFIGS
    DEMUCS_FILE_TYPE = "--mp3"
    DEMUCS_DTYPE = '--float32'
    DEMUCS_TWO_STEMS = "--two-stems"
    DEMUCS_ROLE = "drums"
    DEMUCS_FLAG = "-o"
    DEMUCS_MODEL_FLAG = "-n"
    DEMUCS_MODEL = "mdx_extra"
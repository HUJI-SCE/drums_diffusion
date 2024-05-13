import sys
# sys.path.append("C:\\Users\\michaelpiro1\\AppData\\Local\\miniconda3\\envs\\demucs\\demucs")
import shutil
import multiprocessing
import os
import torchaudio
import csv
import demucs.separate
import zipfile
from configs import TrainingConfig
# import numpy
# import librosa
# FILE_TYPE = "--mp3"
# DTYPE = '--float32'
# TWO_STEMS = "--two-stems"
# ROLE = "drums"
# FLAG = "-o"
# MODEL_FLAG = "-n"
# MODEL = "mdx_extra"
FILE_TYPE = TrainingConfig.DEMUCS_FILE_TYPE
DTYPE = TrainingConfig.DEMUCS_DTYPE
TWO_STEMS = TrainingConfig.DEMUCS_TWO_STEMS
ROLE = TrainingConfig.DEMUCS_ROLE
FLAG = TrainingConfig.DEMUCS_FLAG
MODEL_FLAG = TrainingConfig.DEMUCS_MODEL_FLAG
MODEL = TrainingConfig.DEMUCS_MODEL

# SAVE_PATH = "/Users/mac/pythonProject1/pythonProject/utils"
# GRAVEYARD = "/Users/mac/pythonProject1/pythonProject/utils/graveyard"
# PAIRS = "pairs"
# MODEL_FLAG = "-n"
# MODEL = "mdx_extra"
# NEW_DIR_NAME = "mdx_extra"
# DEMUCS_OUT_DIR = os.path.join(SAVE_PATH,NEW_DIR_NAME)
# PAIRS_DIR = os.path.join(SAVE_PATH,PAIRS)
# LEN_IN_SEC = 5
# OVERLAP_IN_SEC = 0.25
# DUMP_SHORTER = True
# EXT = '.mp3'
# NO_DRUMS_EXT = 'no_drums' + EXT
# DRUMS_EXT = 'drums' + EXT
# CSV_FILE_PATH = "/Users/mac/pythonProject1/pythonProject/utils"

def foo(arg):
    save_path = "D:\\yuval.shaffir\\3"
    SAVE_PATH = save_path
    args = [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL] + [arg]
    demucs.separate.main(args)


def create_dataset_csv(path_to_orig,path_to_sep, csv_file_name='dataset.csv'):
    # Define a list of audio file extensions
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']

    # # Ensure the directory exists
    # if not os.path.exists(dir_path):
    #     raise ValueError("The provided directory does not exist")
    file_names = []
    dirs = os.listdir(path_to_orig)
    for dir in dirs:
        path = os.path.join(path_to_orig,dir)
        if os.path.isfile(path):
            continue
        files_in_dir = os.listdir(path)
        path_to_files = [os.path.join(path,file_name) for file_name in files_in_dir]
        file_names += path_to_files

    with open(csv_file_name, 'w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['file_name','drums_path','no_drums_path'])
        print("collecting files...")

        for i in range(len(file_names)):
            ext = os.path.splitext(file_names[i])[1].lower()
            name = os.path.basename(os.path.splitext(file_names[i])[0].lower())
            separated_dir = os.path.join(path_to_sep,name)
            if not os.path.exists(separated_dir):
                print(f"The directory {separated_dir} does not exist, file didn't separated")
            else:
                d_path = os.path.join(separated_dir,f"drums{ext}")
                no_d_path = os.path.join(separated_dir,f"no_drums{ext}")
                is_drums = os.path.exists(d_path)
                is_no_drums = os.path.exists(no_d_path)
                if is_drums and is_no_drums:
                    writer.writerow([file_names[i], d_path, no_d_path])
                else:
                    print(f"The directory {separated_dir} exist!, but file didn't separated")


def apply_demucs_create_anno_file(audio_data_dir):
    """
    Extracts all the audio files from in_path and its subdirectories, then cuts each audio file into
    segments of specified length with overlap, and saves them to out_path using librosa. If dump_shorter
    is True, segments shorter than length_in_sec are not saved.
    """
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    graveyard = None
    dir_name = os.path.basename(audio_data_dir)
    csv_file_name = f"C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train\\annotation{dir_name}.csv"
    save_path = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train"

    rel_paths, files = make_all_files_list(audio_data_dir, graveyard)
    print(len(rel_paths))
    SAVE_PATH = save_path
    p = multiprocessing.Pool()
    res = p.map(foo,rel_paths)
    p.close()
    # for i in rel_paths:
    #     args =  [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL] + [i]
    #     demucs.separate.main(args)
    demucs_out_dir = os.path.join(SAVE_PATH,MODEL)
    # Prepare to write to the CSV file
    # with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
    with open(csv_file_name, 'w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['file_name', 'path_from_dir','drums_path','no_drums_path'])
        print("collecting files...")
        for i in range(len(files)):
            ext = os.path.splitext(files[i])[1].lower()
            name = os.path.splitext(files[i])[0].lower()
            separated_dir = os.path.join(demucs_out_dir,name)
            if not os.path.exists(separated_dir):
                raise ValueError(f"The directory {separated_dir} does not exist")
            drums_file = f"drums{ext}"
            no_drums_file = f"no_drums{ext}"
            audio_drums = os.path.join(separated_dir,drums_file)
            audio_no_drums = os.path.join(separated_dir,no_drums_file)
            if not os.path.exists(audio_drums):
                raise ValueError(f"The file {audio_drums} does not exist")
            if not os.path.exists(audio_no_drums):
                raise ValueError(f"The file {audio_no_drums} does not exist")
            writer.writerow([files[i], rel_paths[i],audio_drums,audio_no_drums])


def make_all_files_list(dir_path,graveyard):
    rel_paths = []
    files_names = []
    audio_extensions = ['.mp3', '.wav']
    # Ensure the directory exists
    if not os.path.exists(dir_path):
        raise ValueError(f"The provided directory: {dir_path} does not exist")
    if graveyard is not None:
        if not os.path.exists(graveyard):
            os.makedirs(graveyard)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Check if the file has an audio extension
            if os.path.splitext(file)[1].lower() in audio_extensions:
                # Construct the path relative to the provided directory
                abs_path = os.path.join(root, file)
                if not check_file(abs_path,graveyard):
                    continue
                # Write the file name and its relative path to the CSV
                rel_paths.append(abs_path)
                files_names.append(file)
    return rel_paths,files_names


def check_file(file_path,files_graveyard,move_corrupted = False):
    try:
        audio, sr = torchaudio.load(file_path)  # Load audio with its native sampling rate
        # if sr != SAMPLE_RATE:
        #     return False
        return True
    except:
        if move_corrupted:
            name = os.path.basename(file_path)
            grave = os.path.join(files_graveyard,name)
            shutil.move(file_path, grave)
        return False
def extract_zip_file(zip_path,destination_path):
    # global to

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(destination_path)
# if __name__ == '__main__':
#     start = sys.argv[1]
#     end = sys.argv[2]
#     data_prep_path = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train\\utils\\data_prep_utils.py"
#     audio_data_dir = "D:\\yuval.shaffir\\fma_small"
#     csv_file_name = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train\\annotation.csv"
#     save_path = "C:\\Users\\michaelpiro1\\PycharmProjects\\training\\training\\train"
#     dirs = os.listdir(audio_data_dir)
#     for i in range(start,end,1):
#         path = os.path.join(audio_data_dir,dirs[i])
#         dirs[i] = path
#         apply_demucs_create_anno_file(path)
#     # p = multiprocessing.Pool()
#     # p.map(apply_demucs_create_anno_file,dirs)
#     # p.close()
#
#
#         # graveyard = ""
#         # audio_data_dir, csv_file_name, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
#         # if len(sys.argv) == 5:
#         #     graveyard = sys.argv[4]
#         # elif len(sys.argv) == 4:
#         #     graveyard = None
#         # else:
#         #     raise ValueError("wrong arguments")
#
if __name__ == '__main__':
    start = int(sys.argv[1])
    end = int(sys.argv[2])
    save_path = "D:\\yuval.shaffir\\separated"
    SAVE_PATH = save_path
    audiopath = "D:\\yuval.shaffir\\fma_small"
    alldirs = os.listdir(audiopath)[start:end]
    # "D:\\yuval.shaffir\\fma_small\\008"
    for dir in alldirs:
        path_to_dir = os.path.join(audiopath,dir)
        files = os.listdir(path_to_dir)
        l = []
        for i in files:
            p = os.path.join(path_to_dir,i)
            print(p)
            l.append(p)
        args = [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL] + l
        demucs.separate.main(args)
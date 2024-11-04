# the train script for the model

import os

# due to some weird reasons, pytorch lightning will take all cpu cores busy
# so we need to limit the number of threads used by openmp
# 4 per gpu is typically enough and will not slow down the training
os.environ['OMP_NUM_THREADS'] = '8'


import shutil
import time
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import importlib
import time
import pickle

from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor, ModelSummary, RichProgressBar, EarlyStopping
from config.global_config import global_config, console


import my_dataset

import warnings

# Overall settings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')  # the API of torch_geometric is a little old, so we surpress the warnings
warnings.filterwarnings("ignore", ".*does not have many workers which may be a bottleneck.*")
warnings.filterwarnings("ignore", "PossibleUserWarning: The number of training batches.*")
warnings.filterwarnings("ignore", ".*PossibleUserWarning: Your `val_dataloader`'s sampler has shuffling enabled*")
warnings.filterwarnings("ignore", ".*PossibleUserWarning: Your `test_dataloader`'s sampler has shuffling enabled*")
warnings.filterwarnings("ignore", ".*UserWarning: Checkpoint directory*")
 


# set the precision of the matrix multiplication. May be useful for some GPUs
torch.set_float32_matmul_precision('high') # 'medium', 'high'

# set the environment variable to make sure that the GPU operations are deterministic (synced)
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if global_config.cleaner_console:
    import logging
    pl._logger.setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")




def check_gpu():
    """
    find all available GPUs and print their usage
    """
    if torch.cuda.is_available():
        gpu_usages = []
        for i in range(torch.cuda.device_count()):
            # get their power usage
            power_usage = os.popen(f'nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --id={i}').read()
            power_usage = power_usage.split()[-1]
            gpu_usages.append(float(power_usage))
        # we think those with < 50W power usage are not used
        gpu_usages = np.array(gpu_usages)
        gpu_usages = gpu_usages < 150
        gpu_usages = np.where(gpu_usages)[0]
        return gpu_usages
    else:
        raise RuntimeError("No GPU is available!")
    




def experiment():
    
    
    import src.pipeline as pipeline
    
    EXPERIMENT_NAME: str=global_config.exp_name
    EPOCHS: int=global_config.epochs
    
    # automatically allocate GPUs
    auto_alloc_gpu = ""
    if global_config.auto_alloc_gpu and global_config.need_gpu:
        auto_alloc_gpu = f"(auto alloc {global_config.auto_alloc_gpu_num} GPUs)"
        gpu_usages = check_gpu()
        if len(gpu_usages) < global_config.auto_alloc_gpu_num:
            raise RuntimeError("Not enough GPUs are available!")
        else:
            global_config.gpu_id = gpu_usages[:global_config.auto_alloc_gpu_num]
            global_config.gpu_id = [int(i) for i in global_config.gpu_id]
            
    console.rule("Experiment Information")
    console.log("Experiment Name:", EXPERIMENT_NAME, "-- mode:" , global_config.mode, style="green blink bold")
    console.log("GPUs used:", str(global_config.gpu_id), auto_alloc_gpu, style="green")
    console.log("Data folder:", global_config.data_folder, ", batch size:", global_config.batch_size, ", load ratio:", str(min(global_config.load_ratio, 1)*100) + "%", style="green")
    
    # set random seed
    if global_config.mannual_seed != -1:
        np.random.seed(global_config.mannual_seed)
        torch.manual_seed(global_config.mannual_seed)    
    
    # define the model
    my_model = pipeline.MyPipeline()
    
    ##########################################
    # Load the model from the checkpoint
    ##########################################
    
    
    if global_config.mode == "train_fine_tune":
        
        # for fine-tuning, load the "base" model. Otherwise, load the latest checkpoint
        checkpoint_path = None
        for file in sorted(os.listdir('out/checkpoints')):
            # since the listdir is sorted, the last one is the latest one
            if global_config.fine_tune_base_model + "ckpt" in file:
                checkpoint_path = 'out/checkpoints/' + file
        if checkpoint_path is None:
            raise FileNotFoundError(f"The base model ({global_config.fine_tune_base_model}) for fine-tuning does not exist!")
        my_model = pipeline.MyPipeline.load_from_checkpoint(checkpoint_path)
        console.log("You are Fine-tuning a model!", style="yellow")
        console.log(f"The base model is {global_config.fine_tune_base_model}, loadding from:", checkpoint_path, style="yellow")
        
    else:
        # check if there already been a checkpoint with the same experiment name
        checkpoint_path = None
        for file in sorted(os.listdir('out/checkpoints')):
            # since the listdir is sorted, the last one is the latest one
            if global_config.exp_name + "ckpt" in file:
                checkpoint_path = 'out/checkpoints/' + file
        if checkpoint_path is not None:
            console.log("There already exists a checkpoint with the same exp_name. Will load from that.", style="yellow")
            console.log("Loading from:", checkpoint_path, style="yellow")
            #my_model = pipeline.MyPipeline.load_from_checkpoint('out/checkpoints/' + file)
        else:
            console.log("There is no checkpoint with the same exp_name. Will start from scratch.", style="yellow")

    # get the dataset
    data_module = my_dataset.MyLapDataset()
    # set the logger
    logger = TensorBoardLogger('out/tb_logs', name=EXPERIMENT_NAME, default_hp_metric=False)
    logger.log_hyperparams(global_config.__dict__)
    
    # set the callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    device_stats = DeviceStatsMonitor(cpu_stats=True)
    model_summary = ModelSummary(max_depth=1)
    checkpoint_callback = ModelCheckpoint(
        monitor='epoch',    
        dirpath='out/checkpoints',
        #filename=EXPERIMENT_NAME + 'ckpt-{epoch:02d}-{val_loss:.2f}_' + global_config.checksum,
        filename=EXPERIMENT_NAME + 'ckpt-{epoch:02d}_' + global_config.checksum,
        save_top_k=2,
        mode='max',
        every_n_epochs=global_config.check_val_every_n_epoch,
    )
    

    if global_config.mode == "profile":
        profiler = "simple"
    else:
        profiler = None
        
    # define the trainer
    trainer = pl.Trainer(
                         max_epochs=EPOCHS,             # number of epochs  
                         log_every_n_steps=5,        # log every n steps
                         check_val_every_n_epoch=global_config.check_val_every_n_epoch, # check validation every n epochs
                         precision="32",             # use 32-bit floating point
                         accelerator="gpu",         # use GPU
                         devices=global_config.gpu_id,             # use all GPUs if available
                         logger=logger,              # tensorboard logger
                         profiler=profiler,
                         callbacks=[
                                    checkpoint_callback,
                                    lr_monitor,
                                 #   device_stats,
                                    RichProgressBar(),
                         ],
                         gradient_clip_val= 0.5,     # gradient clipping
                         gradient_clip_algorithm="norm",  # gradient clipping algorithm
                         #detect_anomaly=True,
                        )
        
        
    if global_config.mode == "train":
        # train the model with the dataset
        if global_config.auto_retrain_when_exploded:
            raise NotImplementedError("auto_retrain_when_exploded is not implemented yet.")
            
        else:
            # otherwise, train the model, no matter what it experiences during training
            trainer.fit(model=my_model, datamodule=data_module, ckpt_path=checkpoint_path)

    elif global_config.mode == "train_fine_tune":
        # fine-tune the model
        trainer.fit(model=my_model, datamodule=data_module, ckpt_path=None)
        
    elif global_config.mode == "profile":
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            trainer.fit(model=my_model, datamodule=data_module, ckpt_path=checkpoint_path)
        #prof.export_chrome_trace("out/profile/" + EXPERIMENT_NAME + ".json")
        #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        
    elif global_config.mode == "test_visualize" or global_config.mode == "test_loss":
        # evaluate the model with the dataset
        trainer.test(model=my_model, datamodule=data_module, ckpt_path=checkpoint_path)

    elif global_config.mode == "cache_pkl":
        # cache the h_graph into pkl files
        data_module.cache_pkl()
       # data_module.check_quality()
        
    elif global_config.mode == "visualize_graph_tree":
        # visualize the graph tree
        raise NotImplementedError
       # import src.graphtree.h_graph_visualization as h_graph_visualization
       # h_graph_visualization.HGraphVisualization.main()
        
    else:
        raise NotImplementedError



if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='EigenPrediction Parameters')
    parser.add_argument('--config_file', default='config/global_config.py',
                        help='config file', type=str)

    args = parser.parse_args()
    
    # assert that the config file exists
    assert os.path.exists(args.config_file), "The config file does not exist!"
    
    config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    
    global_config.update_config(**config.global_config.__dict__)
    
    
    # start training
    time_start = time.time()        
    experiment()
    time_end = time.time()
    # done
    console.rule("Finished!")
    console.log(f"Time elapsed: {time_end - time_start} seconds.", style="yellow")

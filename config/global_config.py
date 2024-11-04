

class GlobalConfiguration():
    def __init__(self):
        
        # Opt: "train" - training
        #      "train_fine_tune" - fine-tuning
        #      "test_loss" - check the performance on test dataset
        #      "test_visualize" - visualize the result/application on test set
        
        #      "cache_pkl" - prepare data & check data quality
        #      "profile" - do a quick profiling, to find the time bottleneck
        #      "visualize_graph_tree" - draw the h_graph at different layers
        self.mode                            = "train"     
        self.exp_name                        = "may_8_version_temp"
        self.mannual_seed                    = 1            # -1 for random seed
        self.num_workers                     = 0             # how many workers to use in dataloader. 0 for no worker.
        self.comments                        = "my experiment"
        self.epochs                          = 500
        self.check_val_every_n_epoch         = 10
        self.visualize_every_n_epoch         = 50
        
        # GPU configuration
        self.auto_alloc_gpu                  = True          # whether to allocate gpu automatically. if False, you need to specify the gpu_id.
        self.auto_alloc_gpu_num              = 4             # if auto_alloc_gpu is True, how many gpu to allocate.
        self.gpu_id                          = [0]           # if auto_alloc_gpu is False, which gpu to use.

        
        # 数据集配置
        self.data_folder                     = "data/big_2_k_8"
        self.load_data_workers               = 1       # deprecated
        self.load_data_at_start              = True     # deprecated
        self.batch_size                      = 8
        self.load_ratio                      = 1.0             # the ratio of data to load. 1.0 for all data, 0.1 for 10% of the data. this is useful for debugging.
        self.shuffle_val_dataset             = False             # whether to shuffle the validation dataset. this is useful for debugging.
        self.preload_all_data_into_memory    = False            # whether to preload all data into memory. not recommended if the dataset is too large.
                
        # 模型通用配置
        self.model                           = "unet"
        self.input_feature                   = "PN"             # TODO: not implemented yet
        self.embedding_dim                   = 256               # the dimension of the embedding
        self.normalization_type              = "gn"              # Gnn normalization. options: "bn", "gn", "ln", "in", "none"
        # UNet 专用配置
        self.unet_encoder_blocks             = [3, 2, 3, 3, 4]
        self.unet_decoder_blocks             = [3, 2, 3, 3, 4]
        self.unet_encoder_feature_dim        = [128, 128, 128, 256, 256, 512]
        self.unet_decoder_feature_dim        = [512, 256, 256, 128, 128, 128]
        # GNN 专用配置
        self.gnn_num_layers                  = 4                # deprecated
        self.gnn_hidden_channels             = 256              # deprecated
        self.gnn_conv_type                   = "my"            # Gnn convolution type. options: "my", "sage", "gat"
        self.gnn_input_signal                = "all_one"       # input signal. "all_one", "xyz". note that "all_one" should be used together with "my", while "sage" 和 "gat" can be used with "all_one" or "xyz"。

        
        # 
        self.use_ref_mesh                    = True            # whether to use a GT mesh as reference. if False, store_eigens_of_lap_mat and eigens_of_lap_mat_num will be automatically set to False and 64.
        self.store_eigens_of_lap_mat         = True             # whether to store the eigens of the laplacian matrix.
        self.eigens_of_lap_mat_num           = 64              # how many eigens to store when storing the eigens of the laplacian matrix.
        
        self.normal_aware_pooling            = False            # deprecated
        self.graph_tree_layers               = 4                # deprecated
        self.normal_weight                   = 10.0             # deprecated
        
        #self.graph_tree_from_pc_allow        = False            
        #self.graph_tree_from_pc_verbose      = False            #

        # optimizer & scheduler
        self.optimizer                       = "adamw"
        self.learning_rate                   = None             # None for default learning rate
        self.weight_decay                    = None             # None for default weight decay
        self.scheduler                       = "linear"         
        self.auto_retrain_when_exploded      = False             # deprecated
        self.auto_retrain_threshold          = 1.2              # deprecated
        self.auto_retrain_patience           = 5                # deprecated
        
        ############################################
        # Fine-tuning
        ############################################
        self.fine_tune_base_model            = "new_1_feature"  # fine-tuning on which existing model. 
        
        ############################################
        # train/val/test data
        ############################################
        
        # 使用点云的配置
        self.train_graph_data_construt       = "pc"             # "pc", "mesh"
        self.val_graph_data_construt         = "pc"             # "pc", "mesh"
        self.point_cloud_knn_k               = 8               # the K for knn when constructing the graph from point cloud.
        self.cache_optimal_mesh_graph        = False            # deprecated
        
        ############################################
        
        self.random_move_vertex_strength_train = 0.000      # Deprecated: data augmentation
        self.random_move_vertex_strength_val   = 0.000      # Deprecated: data augmentation
        self.using_wang_2016_kinect_dataset    = False            # whether to use Wang 2016 Kinect dataset's noise mesh. if True, the noise mesh from Wang 2016 Kinect dataset will be read. TODO: not implemented in this release.

        ############################################
        # Compare
        ############################################
        
        self.compare_with_robust_method        = True           # [Sharp and Crane 2020]
        
        self.compare_with_belkin_method        = False            # I only reproduce the result of Belkin 2008 in a docker, and upload a docker image to github repo is difficult. so if you want to compare with Belkin 2008, please contact me.
        self.compare_with_belkin_folder        = "data/belkin/output_lap_shapenet_sparse/"  # 
        
        ############################################
        # NeuralLap
        ############################################

        self.use_vertex_mass                 = True            # whether to use mass matrix
        
        # probe functions
        self.probe_function_type             = ['eigen', 'tri_random',]        # probes when training
        self.probe_function_channels         = 14               # for triangular functions, how many channels are used.
        self.probe_eigen_start               = 0                # when using eigen as probe, from which eigen to start.
        self.probe_eigen_end                 = None               # when using eigen as probe, to which eigen to end. None for the last cahced eigen.
        self.val_probe_function_type         = [ 'eigen', 'tri_xyz_test', 'xyz_test' ]            # probes when validating and testing
        self.visualize_pc_channel            = "my_default"              # deprecated
        
        # loss function
        self.vertex_wise_loss_metric         = "mse"            # "mse", "mae", "relative" 
        self.vertex_wise_loss_weight         = 1.0              # 
        self.process_before_calculate_loss   = "factor"             # additional balancing of loss. "none": nothing; "normalize": normalize the vertex result to a normal distribution of mean 0 and std 1; "factor": multiply the loss by (1 / (mean + 0.1))
        
        # 
        self.smoothing_regulization_factor   = 0.0              # deprecated
        
        ############################################
        # Tests
        ############################################
    
        self.do_visualization_on_tensorboard = False            # deprecated
        self.do_log_little_files             = False            # deprecated
        self.do_visualizaion_on_plotly       = False            # use plotly to visualize?
        
        ############################################
        # Application 
        # not provided in this release
        ############################################
        '''
        self.do_application_test             = False            # 
        self.do_what_kind_of_application     = [
                                                "heat_diffusion_heat", "heat_diffusion_smoothing", "spectral_filter", "eigens_of_laplacian", "arap", "geodesic_distance"
                                                ] 
        
        # filtering
        self.application_test_filter         = False            # 
        self.application_filter_list         = ["lowpass", "highpass"]
        
        self.application_test_heat_diffusion = False            # 
        self.application_diffusion_initial   = "grid"           # 
        self.application_diffusion_time_step = 0.001            # 
        self.application_diffusion_num_steps = 1000             # 
        
        self.application_test_smoothing      = False            # 
        self.application_smoothing_time_step = 0.001            # 
        self.application_smoothing_num_steps = 1000             # 
        
        # ARAP 
        self.application_arap_preset_force   = "guitar"         # 
        
        # geodesic distance 
        self.application_geodesic_source     = -1               # 
        '''
        ############################################
        # Misc
        ############################################
        self.cleaner_console                 = False             # whether to clean the console output. if False, some additional information will be printed to the console for debugging. Note that setting this to True will eliminate all warnings.
        
        self.automatic_config()
    
    
    def automatic_config(self):
        """
        Non-public method. Automatically configure the parameters.
        """
        
        # check if the mode is valid
        if self.gnn_conv_type == "my":
            assert self.gnn_input_signal == "all_one"
        else:
            pass
        
        if self.cache_optimal_mesh_graph == True:
            assert False
       
        
        # noise test
        if self.using_wang_2016_kinect_dataset:
            assert self.random_move_vertex_strength_train == 0.0
            assert self.random_move_vertex_strength_val == 0.0
            
        # GT reference mesh exists?
        if self.use_ref_mesh == False:
            self.store_eigens_of_lap_mat = False     
        
        # do we need to preload all data into memory?
        if self.preload_all_data_into_memory == True:
            self.num_workers = 0
        
        # generate a experiment name if not explicitly specified
        if self.exp_name == "" or self.exp_name == None:
            self.exp_name = f"{self.data_folder.split('/')[-1]}_{self.epochs}epochs_{self.batch_size}batchsize"
            
        # do we need to use GPU?
        if self.mode == "cache_pkl" or self.mode == "visualize_graph_tree":
            self.need_gpu = True
        else:
            self.need_gpu = True
            
        # change batch size
        if self.mode == "test_loss" or self.mode == "test_visualize":
            self.batch_size = 1
            if self.auto_alloc_gpu: 
                self.auto_alloc_gpu_num = 1
        
        # number of input signal channel
        self.input_channels          = 4

        # default learning rate of adamw and adam
        if self.learning_rate == None:
            self.learning_rate = 1e-3
        if self.weight_decay == None:
            self.weight_decay = 1e-2
        
            
        # profile mode?
        if self.mode == "profile":
            self.epochs = 3
            self.check_val_every_n_epoch = 111111
            self.visualize_every_n_epoch = 111111
        elif self.mode == "training_data_check":
            self.epochs = 1
            self.batch_size = 1
        
        # generate the checksum of the config
        import hashlib
        # check if a checksum already exists
        if hasattr(self, 'checksum') and False:
            pass
        else:
            temp_dict = self.__dict__.copy()
            temp_dict['mode'] = ''                  # we dont want to include mode in the checksum
            temp_dict = str(temp_dict).encode()
            self.checksum = hashlib.md5(temp_dict).hexdigest()
        
        if self.mode == "test":
            self.batch_size = 1
            print("TIP: You are using test mode. The batch size will be set to 1.")
        
        # sanity check
        pass
    
        
    def update_config(self, **kwargs):
        """
        Upload the configuration. This method should be called only before training.
        """
        self.__dict__.update(kwargs)
        self.automatic_config()

        
    def get_config(self):
        """
        get the configuration
        """
        return self.__dict__
        
        
#
global_config = GlobalConfiguration()

from rich.console import Console
console = Console()
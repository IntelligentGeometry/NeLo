

class GlobalConfiguration():
    def __init__(self):
        

        self.mode                            = "cache_pkl"     
        self.exp_name                        = "may_2_version"
        
        self.data_folder                     = "data/chair"
        

        self.mannual_seed                    = 1
        self.batch_size                      = 1
        self.auto_alloc_gpu_num              = 1
   

        

global_config = GlobalConfiguration()

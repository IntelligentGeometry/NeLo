

class GlobalConfiguration():
    def __init__(self):

        self.mode                            = "test_visualize"     
        self.exp_name                        = "may_2_version"
        
        self.data_folder                     = "data/desk"
        
        self.do_visualizaion_on_plotly       = True 

        self.mannual_seed                    = 1
        self.batch_size                      = 1
   
        self.compare_with_robust_method      = True

        
        

global_config = GlobalConfiguration()

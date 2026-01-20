##################################################################
## Install SLIDE package
#library(devtools)
#devtools::install_github("jishnu-lab/SLIDE") Requires R ≥ 4.2
##################################################################
# load the yaml file and check the input files
yaml_path = ".../SLIDE/SLIDE.yaml"
input_params <- yaml::yaml.load_file(yaml_path)
SLIDE::checkDataParams(input_params)
##################################################################
##                            STEP 1                            ##
##################################################################
## Approximate the SLIDE model's performance for different delta and lambda. 
SLIDE::optimizeSLIDE(input_params, sink_file = FALSE)
##################################################################
##                            STEP 2                            ##
##################################################################
#SLIDE::SLIDEcv(yaml_path, nrep = 2000, k = 20)

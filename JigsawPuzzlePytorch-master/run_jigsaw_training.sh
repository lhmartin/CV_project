IMAGENET_FOLD=../sslime/extra_scripts/data

GPU=${1} # gpu used
CHECKPOINTS_FOLD=${2} #path_to_output_folder

#python JigsawTrain.py ${IMAGENET_FOLD} --checkpoint=${CHECKPOINTS_FOLD} \
#                      --classes=1000 --batch 128 --lr=0.001 --gpu=${GPU} --cores=10
python3 JigsawTrain.py ${IMAGENET_FOLD} --classes=1000 --batch 128 --lr=0.001 --cores=10
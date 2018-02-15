cd ./mxnet

#build training datasets
python build_dataset_baseline.py $1 299
python build_dataset_simplecut.py $1 299
python build_dataset_simplecut.py $1 280
cd ..

cd ./mxnet/data

python ../im2rec.py --list True --recursive True  --shuffle True --num-thread 20 train_baseline baseline_299
cp train_baseline.lst train_simplecut.lst
cp train_baseline.lst train_280.lst
#python ../im2rec.py --list True --recursive True  --shuffle True --num-thread 20 train_280 simplecut_280

#use extact same file list to convert images to .rec   
python ../im2rec.py --quality 100 --num-thread 20 --shuffle False train_280 simplecut_280
python ../im2rec.py --quality 100 --num-thread 20 --shuffle False train_baseline baseline_299
python ../im2rec.py --quality 100 --num-thread 20 --shuffle False train_simplecut simplecut_299
cd ../..

#build meta feature file
python build_feature_csv.py

cd ./mxnet
python fmow_cnn_only.py  --gpus 0,1,2,3 --pretrained-model imagenet1k-resnext-101  --data-train data/train_280.rec  --model-prefix ./model/imagenet1k-resnext-101-cnn-only-all  --batch-size 36 --num-classes 63 --num-examples 426960  --disp-batches 1000  --lr 0.01 --max-random-rotate-angle 360 --num-epochs 8 --lr-step-epochs 6

python fmow_cnn_only.py  --gpus 0,1,2,3 --pretrained-model imagenet11k-place365ch-resnet-152  --data-train data/train_280.rec --model-prefix ./model/imagenet11k-place365ch-resnet-152-cnn-only-all  --batch-size 36 --num-classes 63 --num-examples 426960  --disp-batches 1000  --lr 0.01 --max-random-rotate-angle 360 --num-epochs 6 --lr-step-epochs 5

python fmow_cnn_meta.py  --gpus 0,1,2,3 --pretrained-model imagenet11k-resnet-152     --data-train data/train_simplecut.rec  --model-prefix ./model/imagenet11k-resnet-152-cnn-meta-simplecut-all  --batch-size 36 --num-classes 63 --num-examples 426960  --disp-batches 1000  --lr 0.01 --max-random-rotate-angle 360

python fmow_cnn_meta.py  --gpus 0,1,2,3 --pretrained-model imagenet11k-resnet-152     --data-train data/train_baseline.rec --model-prefix ./model/imagenet11k-resnet-152-cnn-meta-baseline-all  --batch-size 36 --num-classes 63 --num-examples 426960  --disp-batches 1000  --lr 0.01 --max-random-rotate-angle 360

cd ..


rm -rf ./baseline/data/input
rm -rf ./baseline/data/output

cd ./baseline/code

python3 runBaseline.py -prepare test
python3 runBaseline.py -cnn

cd ../..





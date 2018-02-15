if [ "$#" -ne 4 ]; then
    echo "Illegal number of parameters"
    echo "------------examples-----------------"
    echo "bash test.sh /data/train /data/test_provisional /work/out.txt pretrained"
    echo "bash test.sh /data/train /data/test /work/out.txt pretrained"
    echo "bash test.sh /data/train /data/test /work/out.txt model"
    exit
fi

rm -rf ./baseline/data/input
rm -rf ./baseline/data/output

cd ./mxnet
python build_test_dataset_simplecut.py $2 280
python build_test_dataset_simplecut.py $2 299
python build_test_dataset_baseline.py  $2 299
python test_meta.py ./$4/imagenet11k-resnet-152-cnn-meta-simplecut-all 7 299 simplecut_test  
python test_meta.py ./$4/imagenet11k-resnet-152-cnn-meta-baseline-all 7 299  baseline_test
python test_meta.py ./$4/imagenet11k-resnet-152-cnn-meta-simplecut-all 7 299 baseline_test  
python test_meta.py ./$4/imagenet11k-resnet-152-cnn-meta-baseline-all 7 299  simplecut_test
python test_meta.py ./$4/imagenet11k-place365ch-resnet-152-cnn-only-all 6 280  simplecut_test
python test_meta.py ./$4/imagenet1k-resnext-101-cnn-only-all 8 280  simplecut_test
cd ..


cd ./baseline/code

python3 runBaseline.py -prepare $2
if [ $4 == "pretrained" ]; then 
cp ../data/working/cnn_models_pretrained/cnn_image_and_metadata.model ../data/working/cnn_models
fi
python3 runBaseline.py -test_cnn 

if [ $4 == "pretrained" ]; then 
rm ../data/working/cnn_models/cnn_image_and_metadata.model
fi 
cd ../..

cd ./mxnet
python create_sub.py $3
cd ..

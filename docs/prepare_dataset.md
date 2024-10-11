

## NuScenes
Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running

If you use AutoDL, you needn't download.


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**AutoDL version**
Note: Please ensure that your device has at least 500GB of free space.
```
unzip -d /root/autodl-tmp/BEVFormer/data /root/autodl-pub/nuScenes/CANbusexpansion/can_bus.zip
tar -zxvf /root/autodl-pub/nuScenes/Fulldatasetv1.0/Test/v1.0-test_blobs.tgz -C /root/autodl-tmp/BEVFormer/data/nuscenes
tar -zxvf /root/autodl-pub/nuScenes/Fulldatasetv1.0/Test/v1.0-test_meta.tar -C /root/autodl-tmp/BEVFormer/data/nuscenes
cd /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
for tar in *.tgz;  do tar xzvf $tar -C /root/autodl-tmp/BEVFormer/data/nuscenes; done
export PYTHONPATH="/root/autodl-tmp/BEVFormer":$PYTHONPATH
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**
```
bevformer
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train.pkl
|   |   ├── nuscenes_infos_temporal_val.pkl
```

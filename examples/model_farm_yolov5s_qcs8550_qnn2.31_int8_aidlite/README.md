## Model Information
### Source model

- Input shape: 640x640
- Number of parameters: 7.2M
- Model size: 29.0 MB
- Output shape: 1x25200x85

Source model repository: [yolov5](https://github.com/ultralytics/yolov5)

### Converted model

- Precision: INT8
- Backend: QNN2.31
- Target Device: SNM972 QCS8550

## Model Conversion Reference
User can find model conversion reference at [aimo.aidlux.com](https://aimo.aidlux.com/#/public/48e5f27a-7e4a-4a70-8f8c-4a501b123ae5)

## Inference with AidLite SDK

### SDK installation
Model Farm uses AidLite SDK as the model inference SDK. For details, please refer to the [AidLite Developer Documentation](https://docs.aidlux.com/guide/software/sdk/aidlite/aidlite-sdk)

- install AidLite SDK

```bash
# Install the appropriate version of the aidlite sdk
sudo aid-pkg update
sudo aid-pkg install aidlite-sdk
# Download the qnn version that matches the above backend. Eg Install QNN2.23 Aidlite: sudo aid-pkg install aidlite-qnn223
sudo aid-pkg install aidlite-{QNN VERSION}
# eg: Install QNN 2.23 Aidlite: sudo aid-pkg install aidlite-qnn223
```

- Verify AidLite SDK

```bash
# aidlite sdk c++ check
python3 -c "import aidlite; print(aidlite.get_library_version())"

# aidlite sdk python check
python3 -c "import aidlite; print(aidlite.get_py_library_version())"
```

### Run python demo

```bash
cd model_farm_yolov5s_qcs8550_qnn2.31_int8_aidlite
python3  python/run_test.py --target_model ./models/cutoff_yolov5s_qcs8550_w8a8.qnn231.ctx.bin --imgs ./python/bus.jpg --invoke_nums 10

```

### Run c++ demo

```bash
cd model_farm_yolov5s_qcs8550_qnn2.31_int8_aidlite/cpp
mkdir build 
cd build 
cmake ..
make
./run_yolov5
```
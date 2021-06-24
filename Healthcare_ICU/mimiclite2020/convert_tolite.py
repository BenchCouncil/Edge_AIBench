from tensorflow.contrib import lite
converter = lite.TFLiteConverter.from_keras_model_file('mimic3models/in_hospital_mortality/keras_states/k_lstm.n16.d0.3.dep2.bs8.ts1.0.epoch23.test0.28048032904211445.state')
tfmodel = converter.convert()
open("model.tflite","wb").write(tfmodel)

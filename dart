import 'package:tflite/tflite.dart';

class BrainTumorModel {
  Future<String> loadModel() async {
    String res = await Tflite.loadModel(
        model: "assets/brain_tumor_model.tflite",
        labels: "assets/labels.txt",
    );
    return res;
  }

  Future<List?> predictImage(String imagePath) async {
    var output = await Tflite.runModelOnImage(
      path: imagePath,
      numResults: 2,
      threshold: 0.5,
      asynch: true,
    );
    return output;
  }
}

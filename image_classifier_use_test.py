from image_classifier import image_classifier

classifier = image_classifier(top_k=20)
#classifier.get_cnn_feature_of_image("./img/1.jpg")
classifier.get_bing_cnn_feature_of_image("./img/1.jpg")

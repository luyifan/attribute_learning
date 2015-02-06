from bing_cluster import bing_cluster
b = bing_cluster()
b.load_bing_model()
boxes=b.get_bing_of_image("./img/1.jpg")
#distance_matrix=b.get_iou_distance_matrix(boxes)
boxes=b.cluster_boxes(boxes)
print boxes



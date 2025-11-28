import numpy as np


def calculate_iou(box, boxes):
    # 计算输入框的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    ious = []
    for other_box in boxes:
        # 计算其他框的面积
        other_area = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
        
        # 计算交集的坐标
        intersection_x1 = max(box[0], other_box[0])
        intersection_y1 = max(box[1], other_box[1])
        intersection_x2 = min(box[2], other_box[2])
        intersection_y2 = min(box[3], other_box[3])
        
        # 计算交集的面积
        intersection_area = max(intersection_x2 - intersection_x1, 0) * max(intersection_y2 - intersection_y1, 0)
        
        # 计算并添加IOU值
        iou = intersection_area / (box_area + other_area - intersection_area)
        ious.append(iou)
    
    ious = max(ious)
    
    return ious

def resize_box(box, reduction, img_size):    
    """
    从中心点缩放
    """
    
    new_x1 = box[0] + reduction
    new_y1 = box[1] + reduction
    new_x2 = box[2] - reduction
    new_y2 = box[3] - reduction
    
    # 限制缩放，避免缩太多导致box过小
    if new_x2>new_x1 and new_y2>new_y1 and \
        new_x1>0 and new_y1>0 and new_x2<img_size and new_y2<img_size :
        return np.array([new_x1, new_y1, new_x2, new_y2])
    else:
        return box

def boxes_process(boxes, img_size=768, adjust_steps=250, 
                  distance=1, reduction=1, iou_threshold=0.1,
                  min_size=0.2):
    # 计算中心点与焦点
    box_centers = []
    for box in boxes:
        box_centers.append([int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)])

    box_centers = np.array(box_centers)
    intersection_point = np.mean(box_centers, axis=0)
    
    is_move = True
    scale_stop = [False for i in range(len(boxes))]
    for step in range(adjust_steps):
        length = boxes[:, 2] - boxes[:, 0]
        sorted_indices = np.argsort(length)[::-1]   # 从大到小排序
        for idx, i in enumerate(sorted_indices):
            ious_pre = calculate_iou(boxes[i], np.delete(boxes, i, axis=0))
            box_new = resize_box(boxes[i], reduction, img_size)
            ious_post = calculate_iou(box_new, np.delete(boxes, i, axis=0))

            if ious_post>iou_threshold:    # 如果iou还是比较大，则继续缩小
                if box_new[2]-box_new[0]>=min_size*img_size and box_new[3]-box_new[1]>=min_size*img_size:   # 不能缩过头
                    boxes[i] = box_new

            """
            缩放完，进行移动
            """
            for s in range(10):
                # 重新计算中心点与焦点
                box_centers = []
                for box in boxes:
                    box_centers.append([int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)])

                box_centers = np.array(box_centers)
                intersection_point = np.mean(box_centers, axis=0)
                
                move_stop = [False for i in range(len(boxes))]
                for j, center in enumerate(box_centers):
                    ious_pre = calculate_iou(boxes[j], np.delete(boxes, j, axis=0))
                    
                    if ious_pre>iou_threshold:     # 重叠较多，需要移动
                        # 计算中心点与焦点之间的向量
                        direction_vector = intersection_point - center
                        # 归一化向量，并乘以移动的距离
                        move_vector = -direction_vector / np.linalg.norm(direction_vector) * distance
                        # 更新框坐标
                        box_new = boxes[j] + np.round(np.concatenate((move_vector, move_vector), axis=None)).astype(int)
                        
                        ious_post = calculate_iou(box_new, np.delete(boxes, j, axis=0))

                        # 限制box在图像内，并且只有iou变小才能移动box
                        if box_new[0]>0 and box_new[1]>0 and box_new[2]<img_size and box_new[3]<img_size and \
                            box_new[2]>box_new[0] and box_new[3]>box_new[1] and ious_post<ious_pre:
                                    boxes[j] = box_new

                        ious_new = calculate_iou(boxes[i], np.delete(boxes, i, axis=0))
                        # 表示IOU前后没发生变化，移动结束
                        # 或者IOU小于一定阈值，移动结束
                        if ious_new == ious_pre or ious_new<=iou_threshold:    
                            move_stop[i] = True
                        
                
                if False not in move_stop:  # 判断是否全部移动完成
                    break
    
    return boxes



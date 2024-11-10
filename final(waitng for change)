import cv2
import numpy as np
import os

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.0001,  # 取最红的前0.005%的点
            'min_distance': 10        # 两个红点之间的最小距离
        }
        # 加载面部检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def detect_face_features(self, image):
        """检测脸部和嘴唇区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        face_mouth_mask = np.ones(image.shape[:2], dtype=np.uint8)
        
        for (x, y, w, h) in faces:
            # 定义脸部感兴趣区域
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # 检测嘴唇
            mouths = self.mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
            
            # 在嘴唇区域创建mask
            for (mx, my, mw, mh) in mouths:
                # 扩大一点嘴唇区域的mask
                mouth_y1 = y + my - int(mh * 0.2)
                mouth_y2 = y + my + mh + int(mh * 0.2)
                mouth_x1 = x + mx - int(mw * 0.2)
                mouth_x2 = x + mx + mw + int(mw * 0.2)
                
                # 确保坐标在图像范围内
                mouth_y1 = max(0, mouth_y1)
                mouth_y2 = min(image.shape[0], mouth_y2)
                mouth_x1 = max(0, mouth_x1)
                mouth_x2 = min(image.shape[1], mouth_x2)
                
                face_mouth_mask[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = 0
                
        return face_mouth_mask

    def find_red_peaks(self, a_channel, mask):
        """找到A通道中最红的区域，排除mask中的区域"""
        # 应用mask到A通道
        masked_a = a_channel.copy()
        masked_a[mask == 0] = 0
        
        # 获取图像尺寸
        height, width = a_channel.shape
        total_pixels = np.sum(mask > 0)  # 只计算未被mask的像素
        
        # 计算要选取的像素数量
        num_peaks = int(total_pixels * self.red_params['top_percentage'])
        
        # 找到前N个最红的像素位置
        flat_indices = np.argpartition(masked_a.ravel(), -num_peaks)[-num_peaks:]
        row_indices, col_indices = np.unravel_index(flat_indices, masked_a.shape)
        
        # 将坐标和对应的值组合在一起
        coords_and_values = [
            (y, x, masked_a[y, x]) 
            for y, x in zip(row_indices, col_indices) 
            if masked_a[y, x] > 0  # 只保留非0值
        ]
        
        # 按值降序排序
        coords_and_values.sort(key=lambda x: x[2], reverse=True)
        
        # 使用非最大值抑制来筛选峰值
        selected_peaks = []
        for y, x, val in coords_and_values:
            # 检查是否距离已选择的点太近
            too_close = False
            for selected_y, selected_x, _ in selected_peaks:
                distance = np.sqrt((y - selected_y)**2 + (x - selected_x)**2)
                if distance < self.red_params['min_distance']:
                    too_close = True
                    break
            
            if not too_close:
                selected_peaks.append((y, x, val))
        
        return selected_peaks

    def visualize_red_detection(self, image):
        """使用LAB颜色空间检测红色区域"""
        # 获取嘴唇mask
        face_mouth_mask = self.detect_face_features(image)
        
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 归一化A通道以便显示
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        
        # 创建伪彩色图像
        a_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)
        
        # 找到红色峰值，使用mask排除嘴唇区域
        red_peaks = self.find_red_peaks(a, face_mouth_mask)
        
        # 在原图和colormap上标记检测到的区域
        result_image = image.copy()
        colormap_with_marks = a_colormap.copy()
        
        # 创建热力图可视化
        heatmap = np.zeros_like(a_normalized)
        
        # 在图像上标记找到的红色峰值
        for y, x, val in red_peaks:
            # 在结果图像上画圆
            cv2.circle(result_image, (x, y), 5, (0, 0, 255), 2)  # 红色圆圈
            cv2.circle(result_image, (x, y), 1, (0, 255, 0), -1) # 绿色中心点
            
            # 在colormap上也画圆
            cv2.circle(colormap_with_marks, (x, y), 5, (255, 255, 255), 2)  # 白色圆圈
            cv2.circle(colormap_with_marks, (x, y), 1, (0, 255, 0), -1)     # 绿色中心点
            
            # 在热力图上添加高斯分布
            y_coords, x_coords = np.ogrid[-y:a_normalized.shape[0]-y, -x:a_normalized.shape[1]-x]
            mask = x_coords*x_coords + y_coords*y_coords <= 100
            heatmap[mask] = np.maximum(heatmap[mask], val)
        
        # 归一化热力图并创建彩色版本
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # 创建半透明叠加效果
        alpha = 0.3
        overlay = cv2.addWeighted(image, 1-alpha, heatmap_color, alpha, 0)
        
        # 可视化mask（用于调试）
        mask_visualization = cv2.cvtColor((face_mouth_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        results = {
            'original': image,
            'a_channel': a_normalized,
            'a_colormap': a_colormap,
            'mask': mask_visualization,
            'colormap_with_marks': colormap_with_marks,
            'heatmap': heatmap_color,
            'overlay': overlay,
            'result': result_image
        }
        
        return results

    def analyze_image(self, image_path):
        """分析图像并显示处理步骤"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("无法读取图像")

        # 调整图像大小
        height, width = image.shape[:2]
        max_dimension = 800
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, (int(width * scale), int(height * scale)))

        # 获取处理结果
        results = self.visualize_red_detection(image)
        
        # 显示所有处理步骤
        for title, img in results.items():
            self.show_debug_image(title, img)
            # 保存结果
            cv2.imwrite(f"{title}.jpg", img)
            print(f"已保存：{title}.jpg")

if __name__ == "__main__":
    try:
        detector = AcneDetector(debug=True)
        image_path = input("请输入图片路径：")
        detector.analyze_image(image_path)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"发生错误：{str(e)}")

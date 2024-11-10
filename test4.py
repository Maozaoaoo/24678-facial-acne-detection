import cv2
import numpy as np
import os

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.skin_params = {
            'h_min1': 0,
            'h_max1': 19,
            'h_min2': 176,
            'h_max2': 180,
            's_min': 32,
            'v_min': 135
        }

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def visualize_red_detection(self, image):
        """可视化LAB空间中A通道的红色检测过程"""
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 归一化A通道以便显示
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        
        # 创建伪彩色图像以更好地显示红色区域
        a_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)
        
        # 使用自适应阈值处理
        a_thresh = cv2.adaptiveThreshold(
            a,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            a_thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 在colormap和原图上标记轮廓
        colormap_with_contours = a_colormap.copy()
        result_image = image.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 <= area <= 200:  # 面积限制
                # 在colormap上画白色轮廓
                cv2.drawContours(colormap_with_contours, [cnt], -1, (255, 255, 255), 1)
                # 在原图上画红色轮廓
                cv2.drawContours(result_image, [cnt], -1, (0, 0, 255), 1)
        
        # 保存所有处理步骤的结果
        results = {
            'original': image,
            'a_channel': a_normalized,
            'a_colormap': a_colormap,
            'colormap_with_contours': colormap_with_contours,
            'threshold': a_thresh,
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
import cv2
import numpy as np
import os


class FaceSkinAnalyzer:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Cascade classifier file not found at {cascade_path}")

        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def create_skin_mask(self, image):
        """创建肤色掩码"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义肤色范围
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 150, 255])

        # 创建肤色掩码
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # 形态学操作改善掩码
        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        return skin_mask

    def detect_skin_issues(self, face_region):
        """改进的皮肤问题检测"""
        # 创建肤色掩码
        skin_mask = self.create_skin_mask(face_region)

        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # 痘痘检测（更严格的红色范围）
        lower_acne = np.array([0, 50, 50])
        upper_acne = np.array([10, 255, 255])
        acne_mask = cv2.inRange(hsv, lower_acne, upper_acne)

        # 将肤色掩码应用到痘痘检测结果
        acne_mask = cv2.bitwise_and(acne_mask, skin_mask)

        # 使用高斯模糊减少噪声
        acne_mask = cv2.GaussianBlur(acne_mask, (5, 5), 0)

        # 寻找轮廓
        contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 分析轮廓
        skin_issues = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 20 < area < 200:  # 调整面积阈值
                x, y, w, h = cv2.boundingRect(cnt)

                # 计算轮廓的圆形度
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # 只选择较圆的区域作为痘痘
                if circularity > 0.4:
                    skin_issues.append((x, y, w, h, 'acne'))

        return skin_issues

    def analyze_image(self, image_path):
        """主要分析函数"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found at {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image at {image_path}")

        # 调整图片大小以提高处理速度
        height, width = image.shape[:2]
        max_dimension = 800
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        result_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用不同的参数进行人脸检测
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return [], result_image

        results = []
        for (x, y, w, h) in faces:
            face_region = image[y:y + h, x:x + w]

            # 检测皮肤问题
            skin_issues = self.detect_skin_issues(face_region)

            # 在结果图片上标注
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            for (ix, iy, iw, ih, issue_type) in skin_issues:
                # 使用半透明覆盖
                overlay = result_image.copy()
                cv2.rectangle(overlay, (x + ix, y + iy), (x + ix + iw, y + iy + ih), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)

                # 添加标签
                cv2.putText(result_image, issue_type, (x + ix, y + iy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            results.append({
                'face_location': (x, y, w, h),
                'skin_issues_count': len(skin_issues)
            })

        return results, result_image


if __name__ == "__main__":
    try:
        analyzer = FaceSkinAnalyzer()

        # 获取当前文件的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, 'face.jpg')

        if not os.path.exists(image_path):
            image_path = input("请输入图片的完整路径：").strip()

        results, marked_image = analyzer.analyze_image(image_path)

        if results:
            print("检测结果:", results)

            # 显示结果
            cv2.imshow("Face Analysis Results", marked_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 保存结果
            output_path = os.path.join(current_dir, "result.jpg")
            cv2.imwrite(output_path, marked_image)
            print(f"结果已保存至：{output_path}")
        else:
            print("未检测到人脸，请检查图片")

    except Exception as e:
        print(f"错误: {str(e)}")
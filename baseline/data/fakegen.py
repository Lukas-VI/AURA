"""合成伪造图（deepfake）生成。

问题背景：本仓库的 faceswap/simswap 预训练权重缺失（checkpoint 为空），
无法直接跑通原版换脸生成器；而 AURA 需要一个“伪造图 -> 对应真实源图”的检索任务。
因此这里实现一个自包含的“合成换脸”攻击：

    fake = 源图(source)的整幅画面（背景/姿态/光照保持不变）
           + 供体图(donor)的人脸区域（按【前景掩膜】贴入源图的前景区域）

掩膜由 FaceBackgroundSplit / FaceParseMask 提供（几何椭圆或 BiSeNet 人脸解析），
与模型内部的前后景分离使用【同一套前景定义】——这是四元组损失成立的前提。

关键点：伪造图保留的是【源图】的背景，但人脸来自另一个身份。
这正好符合 AURA 的核心假设——背景是伪造图唯一的“身份指纹”，
检索时应能把伪造图匹配回它真正的源图。
"""
import cv2
import numpy as np


class SyntheticFaceSwap:
    """基于前景掩膜 + OpenCV 的轻量合成换脸器（无需预训练模型）。"""

    def __init__(self, splitter):
        self.splitter = splitter

    def swap(self, source_bgr: np.ndarray, donor_bgr: np.ndarray,
             source_mask: np.ndarray = None) -> np.ndarray:
        """把 donor 的前景（人脸）贴到 source 的前景区域，返回伪造图（uint8 BGR，尺寸同 source）。

        source_mask 可由调用方传入（训练时已从缓存取好），避免重复解析；
        为 None 时用 self.splitter 现算。
        """
        # 统一尺寸：donor 与 source 尺寸不同时先缩放到 source
        if donor_bgr.shape[:2] != source_bgr.shape[:2]:
            donor_bgr = cv2.resize(donor_bgr,
                                   (source_bgr.shape[1], source_bgr.shape[0]),
                                   interpolation=cv2.INTER_AREA)

        mask = source_mask if source_mask is not None else self.splitter.mask(source_bgr)
        mask = np.asarray(mask, np.float32)
        if mask.shape[:2] != source_bgr.shape[:2]:    # 掩膜尺寸与图不一致时对齐到图
            mask = cv2.resize(mask, (source_bgr.shape[1], source_bgr.shape[0]),
                              interpolation=cv2.INTER_AREA)
        if mask.ndim == 2:
            mask = mask[..., None]

        # 前景取 donor 内容、背景保持 source 原样（掩膜边缘羽化 -> 自然过渡）
        out = source_bgr.astype(np.float32) * (1.0 - mask) + donor_bgr.astype(np.float32) * mask
        return out.astype(np.uint8)

    def background_attack(self, img_bgr: np.ndarray, mode: str = "blur",
                          rng: np.random.Generator = None) -> np.ndarray:
        """只改动“背景区域”，前景（人脸）保持原样。

        用于构造四元组里的 q2_pos（背景被修改的正样本）以及诊断实验。
        mode: blur(模糊) / color(色偏) / jpeg(压缩) / mix(随机混合)
        """
        rng = rng or np.random.default_rng(0)
        mask = self.splitter.mask(img_bgr)[..., None]  # 前景掩膜

        if mode == "mix":
            mode = str(rng.choice(["blur", "color", "jpeg"]))

        if mode == "blur":
            k = 15  # 固定较大模糊核，效果明显
            attacked = cv2.GaussianBlur(img_bgr, (k, k), 0)
        elif mode == "color":
            # 随机整体色偏 + 亮度调整
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + rng.uniform(-20, 20)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.6, 1.4), 0, 255)
            hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.6, 1.4), 0, 255)
            attacked = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        elif mode == "jpeg":
            quality = int(rng.integers(20, 45))
            ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            attacked = cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img_bgr.copy()
        else:
            attacked = img_bgr.copy()

        # 合成：前景用原图，背景用被攻击图
        out = img_bgr.astype(np.float32) * mask + attacked.astype(np.float32) * (1.0 - mask)
        return out.astype(np.uint8)

    def foreground_attack(self, img_bgr: np.ndarray, donor_bgr: np.ndarray) -> np.ndarray:
        """只改动“前景人脸”（等价于合成换脸），背景保持原样 -> 四元组里的 q1_pos。"""
        return self.swap(img_bgr, donor_bgr)

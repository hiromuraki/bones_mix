from pathlib import Path
import numpy as np
import json
import pickle


class Serializer:
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # 将 numpy 数组转为普通列表
            if isinstance(obj, np.floating):
                return float(obj)
            return super(Serializer.NumpyEncoder, self).default(obj)

    @classmethod
    def to_json(cls, keypoints_data) -> str:
        return json.dumps(keypoints_data, cls=cls.NumpyEncoder, indent=2)

    @classmethod
    def save_keypoints_to_json(cls, keypoints_data, output_path: Path = Path("keypoints.json")):
        # 2. 使用转换器写入文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(keypoints_data, f, cls=cls.NumpyEncoder, indent=2)
        print(f"✅ 关键点数据已成功保存至: {output_path}")

    @classmethod
    def save_keypoints_to_pickle(cls, keypoints_data, output_path: Path = Path("keypoints.pkl")):
        with open(output_path, "wb") as f:
            pickle.dump(keypoints_data, f)
        print(f"✅ 关键点数据已成功保存至: {output_path}")

import torch
import torch.onnx
import numpy as np
import onnx
import onnxruntime as ort
import os
import sys

def convert_pth_to_onnx():
    """
    将 Yolo-FastestV2 的 .pth 权重导出为 .onnx。
    依赖于项目中的 Yolo-FastestV2 源码来构建模型结构。
    """
    # 文件路径配置
    pth_path = r"D:\embodiedcar\embodiedcar\CppVISUAL\modelzoo\coco2017-0.241078ap-model.pth"
    onnx_path = r"D:\embodiedcar\embodiedcar\CppVISUAL\modelzoo\coco2017-0.241078ap-model.onnx"
    yfv2_root = r"D:\embodiedcar\embodiedcar\Yolo-FastestV2"

    # 检查文件是否存在
    if not os.path.exists(pth_path):
        print(f"错误：找不到 .pth 文件: {pth_path}")
        return False
    if not os.path.isdir(yfv2_root):
        print(f"错误：找不到 Yolo-FastestV2 源码目录: {yfv2_root}")
        return False

    # 确保能 import 到 Yolo-FastestV2 的模块
    if yfv2_root not in sys.path:
        sys.path.append(yfv2_root)

    try:
        # 导入模型定义
        from model.detector import Detector

        # 配置（与 data/coco.data 一致）
        classes = 80
        anchor_num = 3
        input_height = 352
        input_width = 352

        print("构建 Yolo-FastestV2 模型...")
        # 设 load_param=True 以跳过内部从 ./model/backbone/backbone.pth 的加载
        # export_onnx=True 会在 forward 里做激活与输出格式整理
        model = Detector(classes=classes, anchor_num=anchor_num, load_param=True, export_onnx=True)

        print("加载权重...")
        checkpoint = torch.load(pth_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            # 兼容不同保存方式
            state_dict = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
        else:
            # 直接保存的 nn.Module（极少见）
            print("检测到完整模型对象，尝试直接使用其 state_dict")
            state_dict = getattr(checkpoint, "state_dict", lambda: None)()
            if state_dict is None:
                print("错误：无法从检查点读取 state_dict")
                return False

        # 有些 state_dict 可能带有前缀，尝试去除常见前缀
        new_state = {}
        for k, v in state_dict.items():
            nk = k
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            new_state[nk] = v

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        if missing:
            print(f"警告：缺失权重: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"警告：多余权重: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")

        model.eval()
        print("模型与权重加载完成。")

        # 创建虚拟输入（BCHW）
        dummy_input = torch.randn(1, 3, input_height, input_width)
        print(f"创建虚拟输入: {tuple(dummy_input.shape)}")

        # 导出为 ONNX
        print("开始导出 ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["P3", "P4"],  # 与 detector.py export_onnx 分支返回的两个输出对应
            dynamic_axes={
                "images": {0: "batch_size"},
                "P3": {0: "batch_size"},
                "P4": {0: "batch_size"},
            },
            verbose=False,
        )
        print(f"ONNX 导出完成: {onnx_path}")

        # 验证 ONNX 模型
        print("验证 ONNX 模型...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 模型验证通过。")

        # 简单推理测试（形状检查）
        print("ONNX 运行时推理测试...")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        out_names = [o.name for o in sess.get_outputs()]
        test_inp = np.random.randn(1, 3, input_height, input_width).astype(np.float32)
        outs = sess.run(out_names, {inp_name: test_inp})
        print("推理成功：输出形状：", [tuple(o.shape) for o in outs])

        return True

    except Exception as e:
        print(f"转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_model_info(pth_path):
    """获取模型信息"""
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')
        print("\n=== 模型文件信息 ===")
        
        if isinstance(checkpoint, dict):
            print("检查点类型: dict")
            print("包含的键:", list(checkpoint.keys()))
            
            for key, value in checkpoint.items():
                if hasattr(value, 'shape') or hasattr(value, '__len__'):
                    if hasattr(value, 'shape'):
                        print(f"  {key}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
                    else:
                        print(f"  {key}: {type(value)}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                else:
                    print(f"  {key}: {type(value)}")
        else:
            print("检查点类型:", type(checkpoint))
            print("模型结构:", checkpoint)
            
    except Exception as e:
        print(f"获取模型信息失败: {e}")

if __name__ == "__main__":
    # 首先显示模型信息，帮助你了解模型结构
    pth_path = r"D:\embodiedcar\embodiedcar\CppVISUAL\modelzoo\coco2017-0.241078ap-model.pth"
    get_model_info(pth_path)
    
    print("\n开始转换...")
    success = convert_pth_to_onnx()
    
    if success:
        print("\n✅ 转换成功！")
        print("现在你可以在C++中使用ONNX Runtime加载这个.onnx文件进行推理")
    else:
        print("\n❌ 转换失败")
        print("请根据上面的模型信息，修改模型创建部分的代码")
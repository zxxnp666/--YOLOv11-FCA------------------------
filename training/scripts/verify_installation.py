"""
环境验证脚本
检查所有依赖是否正确安装
"""

import sys
import os

def check_python_version():
    """检查Python版本"""
    print("=" * 60)
    print("检查Python版本...")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python版本满足要求 (>= 3.8)")
        return True
    else:
        print("✗ Python版本过低，需要 >= 3.8")
        return False

def check_pytorch():
    """检查PyTorch安装"""
    print("\n" + "=" * 60)
    print("检查PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠ CUDA不可用，将使用CPU（速度较慢）")
        
        return True
    except ImportError as e:
        print(f"✗ PyTorch未安装: {e}")
        return False

def check_opencv():
    """检查OpenCV安装"""
    print("\n" + "=" * 60)
    print("检查OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
        return True
    except ImportError as e:
        print(f"✗ OpenCV未安装: {e}")
        return False

def check_other_packages():
    """检查其他依赖包"""
    print("\n" + "=" * 60)
    print("检查其他依赖包...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'yaml': 'PyYAML',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn',
        'tensorboard': 'TensorBoard',
    }
    
    all_ok = True
    for package, name in packages.items():
        try:
            if package == 'PIL':
                import PIL
                version = PIL.__version__
            elif package == 'yaml':
                import yaml
                version = yaml.__version__ if hasattr(yaml, '__version__') else 'N/A'
            else:
                mod = __import__(package)
                version = mod.__version__
            
            print(f"✓ {name}: {version}")
        except ImportError:
            print(f"✗ {name}: 未安装")
            all_ok = False
    
    return all_ok

def check_ultralytics():
    """检查Ultralytics YOLO"""
    print("\n" + "=" * 60)
    print("检查Ultralytics YOLO...")
    try:
        import ultralytics
        print(f"✓ Ultralytics版本: {ultralytics.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Ultralytics未安装: {e}")
        return False

def check_directory_structure():
    """检查项目目录结构"""
    print("\n" + "=" * 60)
    print("检查项目目录结构...")
    
    required_dirs = [
        'data',
        'models',
        'modules',
        'train',
        'utils',
        'evaluation',
        'checkpoints',
        'results',
        'demo',
        'scripts',
        'third_party'
    ]
    
    all_ok = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (缺失)")
            all_ok = False
    
    return all_ok

def check_deanet():
    """检查DEA-Net代码"""
    print("\n" + "=" * 60)
    print("检查DEA-Net代码...")
    
    deanet_path = os.path.join('third_party', 'DEA-Net')
    if os.path.exists(deanet_path):
        print(f"✓ DEA-Net代码已克隆: {deanet_path}")
        
        # 检查关键文件
        key_files = ['README.md', 'requirements.txt']
        for file in key_files:
            file_path = os.path.join(deanet_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (缺失)")
        
        return True
    else:
        print(f"✗ DEA-Net代码未找到: {deanet_path}")
        print("  请运行: cd third_party && git clone https://github.com/cecret3350/DEA-Net.git")
        return False

def main():
    """主函数"""
    print("\n")
    print("=" * 60)
    print("   雾天交通目标检测系统 - 环境验证")
    print("=" * 60)
    
    results = []
    
    # 执行各项检查
    results.append(("Python版本", check_python_version()))
    results.append(("PyTorch", check_pytorch()))
    results.append(("OpenCV", check_opencv()))
    results.append(("其他依赖包", check_other_packages()))
    results.append(("Ultralytics YOLO", check_ultralytics()))
    results.append(("目录结构", check_directory_structure()))
    results.append(("DEA-Net代码", check_deanet()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("验证结果汇总:")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ 所有检查通过！环境配置成功！")
        print("\n下一步:")
        print("1. 下载DEA-Net预训练权重")
        print("2. 准备数据集")
        print("3. 开始开发")
    else:
        print("⚠ 部分检查未通过，请根据提示修复")
    print("=" * 60)
    print()

if __name__ == '__main__':
    main()

import os, random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize


def find_clear_image(clear_path, clear_id):
    """自动检测clear图片的扩展名"""
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        candidate = clear_id + ext
        if os.path.exists(os.path.join(clear_path, candidate)):
            return candidate
    return None


def extract_clear_id(hazy_image_name):
    """
    从hazy图片名提取clear_id
    支持多种命名格式：
    1. 标准格式：0007_01_0.9085.png -> clear_id = 0007
    2. 简单格式：1.jpg -> clear_id = 1
    3. 带扩展名格式：5082.jpg_1_0.9.png -> clear_id = 5082.jpg
    """
    # 先去除扩展名
    base_name = os.path.splitext(hazy_image_name)[0]
    
    # 检查是否包含下划线
    if '_' in base_name:
        # 标准格式：取第一个下划线之前的部分
        clear_id = base_name.split('_')[0]
    else:
        # 简单格式：整个base_name就是clear_id
        clear_id = base_name
    
    return clear_id


def check_image_size(img_path, min_size=256):
    """检查图片尺寸是否满足要求"""
    try:
        img = Image.open(img_path)
        w, h = img.size
        img.close()
        return w >= min_size and h >= min_size
    except:
        return False


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, min_size=256, use_night_augmentation=False):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.min_size = min_size
        self.use_night_augmentation = use_night_augmentation
        
        # 导入夜晚数据增强
        if use_night_augmentation:
            try:
                from .night_augmentation import NightAugmentation
                self.night_aug = NightAugmentation()
                print("✓ Night augmentation enabled")
            except:
                print("⚠️  Night augmentation not available, using standard augmentation")
                self.use_night_augmentation = False
        
        # 获取所有图片文件
        all_hazy_images = [f for f in os.listdir(hazy_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 过滤掉尺寸过小的图片
        self.hazy_image_list = []
        skipped_count = 0
        
        print(f"\n正在过滤训练数据集...")
        print(f"最小尺寸要求: {min_size}×{min_size}")
        
        for hazy_name in all_hazy_images:
            hazy_path_full = os.path.join(hazy_path, hazy_name)
            clear_id = extract_clear_id(hazy_name)
            clear_name = find_clear_image(clear_path, clear_id)
            
            if clear_name is None:
                skipped_count += 1
                continue
            
            clear_path_full = os.path.join(clear_path, clear_name)
            
            # 检查两张图片的尺寸
            if check_image_size(hazy_path_full, min_size) and check_image_size(clear_path_full, min_size):
                self.hazy_image_list.append(hazy_name)
            else:
                skipped_count += 1
        
        self.clear_image_list = os.listdir(clear_path)
        
        print(f"✅ 有效图片: {len(self.hazy_image_list)} 张")
        print(f"⚠️  已跳过: {skipped_count} 张（尺寸过小或找不到对应图片）")
        print()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_id = extract_clear_id(hazy_image_name)
        
        # 自动检测clear图片的扩展名
        clear_image_name = find_clear_image(self.clear_path, clear_id)
        
        if clear_image_name is None:
            # 这种情况理论上不会发生，因为已经在__init__中过滤了
            raise FileNotFoundError(
                f"Cannot find clear image for hazy image: {hazy_image_name}\n"
                f"Extracted clear_id: {clear_id}"
            )

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        try:
            hazy = Image.open(hazy_image_path).convert('RGB')
            clear = Image.open(clear_image_path).convert('RGB')
            
            # 双重保险：再次检查尺寸
            w, h = hazy.size
            if w < self.min_size or h < self.min_size:
                # 如果还是太小，使用resize放大（不推荐，但至少不会崩溃）
                hazy = hazy.resize((max(w, self.min_size), max(h, self.min_size)), Image.BILINEAR)
                clear = clear.resize((max(w, self.min_size), max(h, self.min_size)), Image.BILINEAR)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load images:\n"
                f"  Hazy: {hazy_image_path}\n"
                f"  Clear: {clear_image_path}\n"
                f"  Error: {e}"
            )

        crop_params = RandomCrop.get_params(hazy, [256, 256])
        rotate_params = random.randint(0, 3) * 90

        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)

        hazy = rotate(hazy, rotate_params)
        clear = rotate(clear, rotate_params)

        to_tensor = ToTensor()  # PyTorch的ToTensor会自动归一化

        hazy = to_tensor(hazy)  # 将PIL图像转为tensor并归一化到[0,1]
        clear = to_tensor(clear) # 将PIL图像转为tensor并归一化到[0,1]
        
        # 应用夜晚数据增强（如果启用）
        if self.use_night_augmentation:
            hazy, clear = self.night_aug(hazy, clear)

        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, min_size=256):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.min_size = min_size
        
        # 获取所有图片文件
        all_hazy_images = [f for f in os.listdir(hazy_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 过滤掉尺寸过小的图片
        self.hazy_image_list = []
        skipped_count = 0
        
        for hazy_name in all_hazy_images:
            hazy_path_full = os.path.join(hazy_path, hazy_name)
            clear_id = extract_clear_id(hazy_name)
            clear_name = find_clear_image(clear_path, clear_id)
            
            if clear_name is None:
                skipped_count += 1
                continue
            
            clear_path_full = os.path.join(clear_path, clear_name)
            
            # 检查两张图片的尺寸
            if check_image_size(hazy_path_full, min_size) and check_image_size(clear_path_full, min_size):
                self.hazy_image_list.append(hazy_name)
            else:
                skipped_count += 1
        
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        
        if skipped_count > 0:
            print(f"测试集: 有效 {len(self.hazy_image_list)} 张, 跳过 {skipped_count} 张")

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_id = extract_clear_id(hazy_image_name)
        
        # 自动检测clear图片的扩展名
        clear_image_name = find_clear_image(self.clear_path, clear_id)
        
        if clear_image_name is None:
            raise FileNotFoundError(
                f"Cannot find clear image for hazy image: {hazy_image_name}\n"
                f"Extracted clear_id: {clear_id}"
            )

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, min_size=256):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.min_size = min_size
        
        # 获取所有图片文件
        all_hazy_images = [f for f in os.listdir(hazy_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 过滤掉尺寸过小的图片
        self.hazy_image_list = []
        skipped_count = 0
        
        for hazy_name in all_hazy_images:
            hazy_path_full = os.path.join(hazy_path, hazy_name)
            clear_id = extract_clear_id(hazy_name)
            clear_name = find_clear_image(clear_path, clear_id)
            
            if clear_name is None:
                skipped_count += 1
                continue
            
            clear_path_full = os.path.join(clear_path, clear_name)
            
            # 检查两张图片的尺寸
            if check_image_size(hazy_path_full, min_size) and check_image_size(clear_path_full, min_size):
                self.hazy_image_list.append(hazy_name)
            else:
                skipped_count += 1
        
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_id = extract_clear_id(hazy_image_name)
        
        # 自动检测clear图片的扩展名
        clear_image_name = find_clear_image(self.clear_path, clear_id)
        
        if clear_image_name is None:
            raise FileNotFoundError(
                f"Cannot find clear image for hazy image: {hazy_image_name}\n"
                f"Extracted clear_id: {clear_id}"
            )

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}

    def __len__(self):
        return len(self.hazy_image_list)

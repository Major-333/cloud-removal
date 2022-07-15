import os
if __name__ == '__main__':
    dir = '/home/major333@corp.sse.tongji.edu.cn/workspace/remote-sensing/data/checkpoints/TSOCR_V0'
    names = os.listdir(dir)
    for name in names:
        if name.endswith('_1.json') or name.endswith('_1'):
            path = os.path.join(dir, name)
            os.remove(path)
    names = os.listdir(dir)
    for name in names:
        chips = name.split('_')
        for chip in chips:
            if 'epoch' in chip:
                path = os.path.join(dir, name)
                new_path = os.path.join(dir, chip)
                if name.endswith('.json'):
                    new_path = f'{new_path}.json'
                os.rename(path, new_path)
                break


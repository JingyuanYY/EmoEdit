import os

def get_all_paths(dir_root, suffix: list):
    txt_file_list = []
    for root, _, file_path in os.walk(dir_root):
        for file in file_path:
            for suffix_name in suffix:
                if file.endswith(suffix_name):
                    tmp = os.path.join(root, file)
                    txt_file_list.append(tmp)
    txt_file_list.sort()
    return txt_file_list
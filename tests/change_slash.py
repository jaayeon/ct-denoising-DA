import os


def change_os_slash(dir_name):
    dir_name = dir_name.split('\\')
    out_dir_name = '/'.join(dir_name)

    return out_dir_name


if __name__ == "__main__":
    dir_name = os.getcwd()
    print(dir_name)
    out_dir_name = change_os_slash(dir_name)
    print(out_dir_name)
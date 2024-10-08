import os
import configparser
import re


def read_config(filename="config.ini"):
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(filename)
    return config


def update_file(filename, replacements):
    with open(filename, "r") as file:
        content = file.read()

    for old, new in replacements.items():
        content = re.sub(re.escape(old), new, content)

    with open(filename, "w") as file:
        file.write(content)


def main():
    config = read_config()

    paths = config["Paths"]
    project_root = paths["project_root"]
    project_root = project_root[:-1] if project_root.endswith("/") else project_root
    if not os.path.exists(project_root):
        raise ValueError(f"project root dir {project_root} does not exist")

    replacements = {
        "/path/to/heedless-backbones": project_root,
        "heedlessbackbones.com": config["Server"]["domain"],
        "linux_user": config["User"]["linux_user"],
        "linux_group": config["User"]["linux_group"],
        "conda activate model_stats": f"conda activate {config['Conda']['environment']}",
    }

    files_to_update = [
        "heedless-backbones.nginxconf",
        "heedless-backbones.service",
        "gunicorn.sh",
        "init_systemd.sh",
    ]

    for filename in files_to_update:
        update_file(filename, replacements)
        print(f"Updated {filename}")


if __name__ == "__main__":
    main()

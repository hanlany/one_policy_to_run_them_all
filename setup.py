import os
import pathlib

from setuptools import find_packages, setup


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
package_root = here / "one_policy_to_run_them_all"


def read_requirements_file(filename):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def find_environment_assets():
    return [
        str(path.relative_to(package_root))
        for data_dir in (package_root / "environments").glob("*/data")
        for path in data_dir.rglob("*")
        if path.is_file()
    ]


def setup_package():
    setup(
        name="one_policy_to_run_them_all",
        description="one_policy_to_run_them_all",
        long_description=long_description,
        url="https://github.com/nico-bohlinger/one_policy_to_run_them_all",
        author="Nico Bohlinger",
        author_email="nico.bohlinger@gmail.com",
        version="0.0.1",
        packages=find_packages(
            include=["experiments", "one_policy_to_run_them_all", "one_policy_to_run_them_all.*"]
        ),
        package_data={
            "experiments": ["configs/*.yaml"],
            "one_policy_to_run_them_all": find_environment_assets(),
        },
        install_requires=read_requirements_file("requirements.txt"),
        license="MIT",
    )


if __name__ == "__main__":
    setup_package()

import setuptools

# バージョンは一箇所で管理するのが一般的
VERSION = "0.1.0"

# READMEをlong_descriptionとして読み込む（エンコーディング指定）
with open("README.md", "r", encoding="utf-8") as file_object:
    LONG_DESCRIPTION = file_object.read()

# requirements.txtから依存関係を読み込む
with open("requirements.txt", encoding="utf-8") as file_object:
    INSTALL_REQUIRES = file_object.read().splitlines()

setuptools.setup(
    name="vap-o",
    version=VERSION,
    author="Kazuyo Onishi",
    author_email="kazuyo.onishi@riken.jp",
    description="Onset Proximity for turn-taking timing prediction",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/OU-sigma-dis/VAP-O",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.11",  # 対応するPythonのバージョンを指定
)

from setuptools import find_packages, setup

setup(
    name="nlp_radiology",
    version="1.0.0",
    description="This project houses my honours project.",
    author="Hareeshan Elankeeran",
    author_email="h.elankeeran.21@abdn.ac.uk",
    packages=find_packages(),
    url="https://github.com/harrykeeran12/cs4525-honours-project",
    install_requires=[
        "flask",
        "ollama-python",
        "scikit-learn",
        "pytorch>2.3.0",
        "transformers>=4.50.0",
        "mauve-text",
        "bert_score",
        "hugging_face_hub",
        "sentence-transformers==3.4.1",
        "ipykernel",
        "datasets",
        "jupyter",
        "ipywidgets",
        "ipykernel",
        "python-dotenv",
        "pydantic",
        "evaluate",
        "rouge-score",
    ],
    data_files=[
        "scripts",
        "src"
        "tests",
        "datasets",
        "logs",
        "manuals",
        "modelfiles",
    ],
    scripts=["webview/app.py"],
    setup_requires=["setuptools_scm"],
    include_package_data=True,
)

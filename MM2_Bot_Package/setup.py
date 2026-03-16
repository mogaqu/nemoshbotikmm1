from setuptools import setup, find_packages

setup(
    name="MM2_Bot",
    version="3.0.0",
    description="Enhanced Candy Hunter Bot for Murder Mystery 2",
    author="Bot Developer",
    author_email="developer@example.com",
    packages=find_packages(),
    install_requires=[
        "python>=3.14.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "pywin32>=306",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "colorama>=0.4.6",
        "keyboard>=0.13.5",
        "mss>=9.0.1"
    ],
    python_requires=">=3.14.0",
    entry_points={
        "console_scripts": [
            "mm2-bot=MM2_Bot_Package.run_enhanced_bot:main",
            "mm2-bot-enhanced=MM2_Bot_Package.run_enhanced_bot:main",
            "mm2-bot-predator=MM2_Bot_Package.run_predator_bot:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.14",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Games/Entertainment",
    ],
    license="MIT",
    keywords="roblox murder mystery bot automation",
    zip_safe=False
)
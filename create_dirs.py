import os

dirs = [
    "llm-framework/config",
    "llm-framework/data",
    "llm-framework/models/layers",
    "llm-framework/models/heads",
    "llm-framework/training",
    "llm-framework/inference",
    "llm-framework/evaluation",
    "llm-framework/utils",
    "llm-framework/scripts",
    "llm-framework/tests"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    # create __init__.py in all directories
    parts = d.split('/')
    for i in range(1, len(parts) + 1):
        sub_dir = "/".join(parts[:i])
        init_file = os.path.join(sub_dir, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, 'w').close()

print("Directories created successfully.")

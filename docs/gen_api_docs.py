import os
from pathlib import Path

NAV_PATH = Path("django_chain")  # This will be the sub-folder inside docs/

MODULES_TO_DOCUMENT = [
    "models",
    "exceptions",
]

for module in MODULES_TO_DOCUMENT:
    filepath = NAV_PATH / Path(module.replace(".", os.sep) + ".md")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as fd:
        print(f"::: {module}", file=fd)

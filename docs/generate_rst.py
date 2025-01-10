import os
import glob

template = """
{header_name}
{header_separator}

.. doxygennamespace:: {filepath}
   :project: newton-isaac-sim
   :members:
   :protected-members:
   :private-members:
   :undoc-members:
"""


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def generate_pages():
    headers = glob.glob("../core/**/*.py", recursive=True)

    for header in headers:
        name = os.path.basename(header)
        print(f"Generating {name}")

        name_length = len(name)

        with open(f"api/{name}.rst", "w") as f:
            f.write(
                template.format(
                    header_name=name,
                    header_separator="=" * name_length,
                    filepath=header.removeprefix("../")
                    .replace(".py", "")
                    .replace("/", "::"),
                )
            )


if __name__ == "__main__":
    ensure_dir("api/")
    generate_pages()

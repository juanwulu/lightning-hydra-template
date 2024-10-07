# Code Released under MIT License
#
# Copyright (c) 2024 David Juanwu Lu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
import os
import sys
from typing import Iterable

from omegaconf import OmegaConf

# export project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["PROJECT_ROOT"] = project_root
sys.path.append(project_root)

# Disable Tensorflow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Register custom resolvers for `OmegaConf`
def join_string_underscore(texts: Iterable[str]) -> str:
    """Join strings with underscore."""
    return "_".join(str(item) for item in texts)


def parse_git_sha(clean: bool = False) -> str:
    """Parse git information and export to environment variables.

    Args:
        clean (bool, optional): If to check whether the git repository is
            clean. Defaults to False.

    Returns:
        str: The short SHA of the current git repository.

    Raises:
        RuntimeError: If the repository is not clean and `clean
    """
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = str(repo.head.object.hexsha)
    short_sha = str(repo.git.rev_parse(sha, short=7))
    is_clean = not repo.is_dirty()

    if clean and not is_clean:
        raise RuntimeError(
            "Clean your git repository before running experiments! "
            f"GIT SHA: {sha}, Status: {'Clean' if is_clean else 'Dirty'}"
        )
        os._exit(1)

    os.environ["PROJECT_GIT_SHA"] = sha
    os.environ["PROJECT_GIT_SHORT_SHA"] = short_sha

    return short_sha


OmegaConf.register_new_resolver(
    "join_string_underscore",
    join_string_underscore,
    replace=False,
    use_cache=False,
)
OmegaConf.register_new_resolver(
    "parse_git_sha",
    lambda clean: parse_git_sha(clean=clean),
    replace=False,
    use_cache=False,
)

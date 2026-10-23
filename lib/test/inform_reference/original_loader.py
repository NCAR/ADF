"""Import the original cookbook modules, at the commit pinned in cookbook_ref.json,
without touching the ADF ones.

The reference is the cookbook as committed on GitHub, never a working copy:
original_dir() extracts the pinned commit's files (git archive, so uncommitted
edits in a local clone cannot leak in) into a cache directory and returns it.
The commit comes from, in order:

  1. INFORM_COOKBOOK_DIR - a local clone of the cookbook that has the commit;
  2. otherwise a shallow fetch of just that commit from GitHub, kept as a bare
     repository in the cache.

The cache is INFORM_COOKBOOK_CACHE, or ~/.cache/adf/inform-cookbook, with one
subdirectory per commit.  If neither source works, CookbookUnavailable is
raised (the tests skip on it).

The cookbook and the ADF use the same module names (cam_diagnostic,
cam_era5_functions, inform_utils), and the originals import each other by those
names.  load_originals() imports them from original_dir() with the ADF copies
moved out of sys.modules for the duration, then puts everything back, so each
side keeps pointing at its own copies:

    orig = load_originals()
    orig["cam_diagnostic"].plot_obs_cam_nd_lwc_pdfs(...)
"""
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_HERE = Path(__file__).parent
REF = json.loads((_HERE / "cookbook_ref.json").read_text())

#Files the cookbook never committed (see README.md); not part of the reference:
LOCAL_ONLY_DIR = _HERE / "original"

#The modules the ADF was built from, in dependency order:
ORIGINAL_MODULES = ("inform_utils", "process_data_products_utils", "cam_era5_functions",
                    "cam_diagnostic")

_CACHE = {}


class CookbookUnavailable(RuntimeError):
    """The pinned cookbook commit could not be obtained."""


def _cache_root():
    root = os.environ.get("INFORM_COOKBOOK_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "adf" / "inform-cookbook"


def _git(*args, git_dir):
    return subprocess.run(["git", f"--git-dir={git_dir}", *args], check=True,
                          capture_output=True, timeout=300).stdout


def _has_commit(git_dir, sha):
    try:
        return _git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}",
                    git_dir=git_dir).decode().strip() == sha
    except (subprocess.SubprocessError, OSError):
        return False


def _git_dir_with_commit(sha):
    """A git directory holding `sha`: the INFORM_COOKBOOK_DIR clone, or a bare
    repository in the cache, fetched from GitHub if needed."""
    errors = []
    local = os.environ.get("INFORM_COOKBOOK_DIR")
    if local:
        try:
            git_dir = subprocess.run(["git", "-C", local, "rev-parse", "--absolute-git-dir"],
                                     check=True, capture_output=True,
                                     timeout=60).stdout.decode().strip()
        except (subprocess.SubprocessError, OSError) as err:
            git_dir = None
            errors.append(f"INFORM_COOKBOOK_DIR={local} is not a git clone ({err})")
        if git_dir and _has_commit(git_dir, sha):
            return git_dir
        if git_dir:
            errors.append(f"INFORM_COOKBOOK_DIR={local} does not have commit {sha} "
                          "(git fetch there, or unset it to fetch from GitHub)")

    bare = _cache_root() / "repo.git"
    try:
        if not bare.is_dir():
            bare.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "--quiet", "--bare", str(bare)], check=True,
                           capture_output=True, timeout=60)
        if not _has_commit(bare, sha):
            _git("fetch", "--quiet", "--depth", "1", REF["repo"], sha, git_dir=bare)
    except (subprocess.SubprocessError, OSError) as err:
        stderr = getattr(err, "stderr", b"") or b""
        errors.append(f"fetching {sha} from {REF['repo']} failed: {err} {stderr.decode().strip()}")
    if _has_commit(bare, sha):
        return bare
    raise CookbookUnavailable("; ".join(errors))


def original_dir():
    """The pinned commit's reference files, extracted into the cache."""
    sha = REF["commit"]
    out = _cache_root() / sha
    if (out / ".complete").is_file():
        return out
    git_dir = _git_dir_with_commit(sha)
    try:
        archive = _git("archive", "--format=tar", sha, "--", *REF["files"], git_dir=git_dir)
    except subprocess.SubprocessError as err:
        raise CookbookUnavailable(f"git archive of {sha} failed: {err}") from err
    #Extract next to the final location, then rename: concurrent runs never see
    #a half-written directory.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{sha[:12]}-"))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)
        (tmp / ".complete").write_text(f"{REF['repo']} {sha}\n")
        try:
            tmp.rename(out)
        except OSError:
            if not (out / ".complete").is_file():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def branch_head():
    """The pinned branch's current commit on GitHub, or None if unreachable."""
    try:
        out = subprocess.run(["git", "ls-remote", REF["repo"], f"refs/heads/{REF['branch']}"],
                             check=True, capture_output=True, timeout=60).stdout.decode()
    except (subprocess.SubprocessError, OSError):
        return None
    return out.split()[0] if out.strip() else None


def load_originals():
    """Return {name: module} for the original cookbook modules."""
    if _CACHE:
        return _CACHE

    src = str(original_dir())
    saved = {n: sys.modules.pop(n) for n in ORIGINAL_MODULES if n in sys.modules}
    sys.path.insert(0, src)
    try:
        for name in ORIGINAL_MODULES:
            _CACHE[name] = importlib.import_module(name)
    finally:
        sys.path.remove(src)
        for name in ORIGINAL_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return _CACHE

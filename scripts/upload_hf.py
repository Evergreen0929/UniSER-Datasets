"""Upload packed WebDataset shards to the Hugging Face Hub.

Wraps `HfApi.upload_large_folder` so a long upload survives interruptions and
resumes from where it stopped. Designed to be re-run safely — already-uploaded
files are skipped via content-hash dedup.

Prerequisites:
    pip install -U "huggingface_hub[hf_xet]"
    hf auth login            # paste a write token
    export HF_XET_HIGH_PERFORMANCE=1     # enable Xet fast transfer

Typical usage:
    # upload <folder>/* to repo root (no subdir)
    python scripts/upload_hf.py --folder /mnt/localssd/staging/shards

    # upload to repo subdir 'shards-dryrun/' via transparent hardlink staging
    python scripts/upload_hf.py --folder /mnt/localssd/staging/dryrun \\
        --path-in-repo shards-dryrun

    # dry-run (just list files, no upload)
    python scripts/upload_hf.py --folder /mnt/localssd/staging/dryrun --dry-run
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, login


REPO_DEFAULT = "jdzhang0929/uniser-haze-dataset"


def stage_with_subpath(source: Path, subpath: str) -> Path:
    """Build a temporary directory where `source/` appears at `subpath/` inside.

    `upload_large_folder` does not support `path_in_repo` directly — it uploads
    the folder layout verbatim. To put files under a subdir on the Hub, we
    create a temp dir whose internal structure is `<temp>/<subpath>/<files>`
    and hard-link every file from `source` into it (zero extra disk).

    The temp dir is placed *next to* the source folder so the hardlinks live
    on the same filesystem. If we used the OS default tempdir (often /tmp),
    cross-device hardlinks fail and the fallback copy can blow out /tmp.

    Returns the temp dir, which the caller must clean up.
    """
    # Place the wrapper on the same filesystem as the source so hardlinks work.
    wrapper = Path(tempfile.mkdtemp(prefix="hfupload_", dir=str(source.parent)))
    target = wrapper / subpath
    target.mkdir(parents=True, exist_ok=True)
    for f in source.iterdir():
        if not f.is_file():
            continue
        dst = target / f.name
        try:
            os.link(f, dst)            # hardlink (cheap, same FS)
        except OSError as e:
            # Cross-device hardlink (errno 18) should not happen now that the
            # wrapper sits next to the source, but fall back to copy just in
            # case (e.g. if the source is a bind mount).
            shutil.copy2(f, dst)
    return wrapper


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True,
                    help="Local folder whose contents will be uploaded.")
    ap.add_argument("--repo-id", default=REPO_DEFAULT,
                    help=f"HF dataset repo id (default: {REPO_DEFAULT}).")
    ap.add_argument("--path-in-repo", default=None,
                    help="Optional subdir inside the dataset repo. When set, the "
                         "script transparently stages files via hardlinks. Omit "
                         "to upload at repo root.")
    ap.add_argument("--allow", nargs="*",
                    default=["*.tar", "*.json", "*.md", "*.yaml"],
                    help="Glob patterns to include (relative to --folder).")
    ap.add_argument("--ignore", nargs="*",
                    default=[".tmp/*", "*.pyc"],
                    help="Glob patterns to skip.")
    ap.add_argument("--num-workers", type=int, default=4,
                    help="Concurrent upload workers (default: 4).")
    ap.add_argument("--dry-run", action="store_true",
                    help="List files that would be uploaded and exit.")
    ap.add_argument("--token", default=None,
                    help="Override HF token (otherwise uses `hf auth login` cache).")
    args = ap.parse_args()

    if not os.environ.get("HF_XET_HIGH_PERFORMANCE"):
        print("WARN: HF_XET_HIGH_PERFORMANCE is unset — uploads will be slow.",
              file=sys.stderr)

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"ERROR: folder does not exist: {folder}")

    if args.dry_run:
        print(f"[dry-run] would upload these from {folder}:")
        import fnmatch
        for f in sorted(folder.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(folder).as_posix()
            if any(fnmatch.fnmatch(rel, p) for p in args.ignore):
                continue
            if not any(fnmatch.fnmatch(rel, p) for p in args.allow):
                continue
            on_hub = f"{args.path_in_repo}/{rel}" if args.path_in_repo else rel
            print(f"  {rel}  -> {on_hub}  ({f.stat().st_size / 1e6:.1f} MB)")
        return

    if args.token:
        login(token=args.token)

    api = HfApi()
    # Ensure the repo exists; visibility (private) defaults to True only when creating.
    api.create_repo(args.repo_id, repo_type="dataset", private=True, exist_ok=True)

    # Build the upload root (with or without subpath staging).
    cleanup_dir = None
    if args.path_in_repo:
        upload_root = stage_with_subpath(folder, args.path_in_repo)
        cleanup_dir = upload_root
        print(f"staged via hardlinks: {folder} -> {upload_root}/{args.path_in_repo}/")
    else:
        upload_root = folder

    where = f"{args.repo_id}/{args.path_in_repo}" if args.path_in_repo else args.repo_id
    print(f"uploading {folder}  ->  {where}")
    print(f"  allow:  {args.allow}")
    print(f"  ignore: {args.ignore}")
    print(f"  workers: {args.num_workers}")

    try:
        api.upload_large_folder(
            folder_path=str(upload_root),
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=args.allow,
            ignore_patterns=args.ignore,
            num_workers=args.num_workers,
        )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    print("\nupload complete.")


if __name__ == "__main__":
    main()

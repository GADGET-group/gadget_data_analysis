import os
import sys
import shutil
import subprocess
import traceback

import uproot


DEFAULT_ROOT = (
    "/egr/research-tpc/shared/proc_runs/e25058/new_h5/"
    "e25058_run128.root"
)


def git_output(args):
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception as exc:
        return f"Could not collect git information: {exc}\n"


def main():
    original = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)

    if not os.path.exists(original):
        raise FileNotFoundError(f"ROOT file not found:\n{original}")

    base, ext = os.path.splitext(original)
    test_copy = base + "_metadata_test" + ext

    print("=" * 72)
    print("Uproot version:", uproot.__version__)
    print("Original file :", original)
    print("Original size :", os.path.getsize(original) / 1e9, "GB")
    print("=" * 72)

    # Read-only validation of the original.
    with uproot.open(original) as f:
        print("\nOriginal ROOT objects:")
        print(f.classnames())

        if "events" in f:
            events = f["events"]
            print("events type   :", type(events))
            print("event entries :", events.num_entries)
            print("event fields  :", events.keys())

        print("metadata exists:", "metadata" in f)

    # Never overwrite an old test file silently.
    if os.path.exists(test_copy):
        print(f"\nRemoving previous test copy:\n{test_copy}")
        os.remove(test_copy)

    print("\nCreating test copy...")
    print("The ORIGINAL file will not be modified.")

    # Prefer a copy-on-write reflink; if unavailable, fall back to shutil.copy2.
    try:
        subprocess.run(
            ["cp", "--reflink=auto", "--sparse=always", original, test_copy],
            check=True,
        )
    except Exception as exc:
        print("cp --reflink=auto failed:", exc)
        print("Falling back to shutil.copy2 (this may copy the full file).")
        shutil.copy2(original, test_copy)

    print("Test copy     :", test_copy)
    print("Test copy size:", os.path.getsize(test_copy) / 1e9, "GB")

    # Match the metadata structure used in process_runs.py.
    git_version = git_output(["rev-parse", "--verify", "HEAD"])
    git_status = git_output(["status"])
    git_diff = git_output(["diff"])

    metadata = {
        "git_version": [git_version],
        "git_status": [git_status],
        "git_diff": [git_diff],
    }

    print("\nAttempting to add metadata to TEST COPY only...")
    print("-" * 72)

    try:
        with uproot.update(test_copy) as f:
            f["metadata"] = metadata

        print("\nSUCCESS: metadata write completed without an exception.")

    except Exception:
        print("\nFAILED while adding metadata.")
        print("\nFull traceback:")
        traceback.print_exc()

    print("\n" + "-" * 72)
    print("Reopening TEST COPY to inspect final state...")

    try:
        with uproot.open(test_copy) as f:
            print("Objects:")
            print(f.classnames())

            if "events" in f:
                print("Event entries:", f["events"].num_entries)

            print("Metadata present:", "metadata" in f)

            if "metadata" in f:
                print("Metadata fields:", f["metadata"].keys())

    except Exception:
        print("Could not reopen test copy.")
        traceback.print_exc()

    print("\nOriginal remains untouched:")
    print(original)
    print("\nTest copy may be deleted after inspection:")
    print(test_copy)


if __name__ == "__main__":
    main()
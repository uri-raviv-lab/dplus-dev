# wrapper.py — debug-friendly wrapper that finds an embedded MSI, copies to temp and runs it
import sys
import os
import tempfile
import shutil
import subprocess
import traceback

def find_msi_in_bundle():
    """Search for .msi inside the bundle (sys._MEIPASS) or next to exe/script.
       Return full path to the largest .msi found (or None)."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".msi"):
                p = os.path.join(root, f)
                try:
                    size = os.path.getsize(p)
                except OSError:
                    size = 0
                candidates.append((size, p))
    if not candidates:
        return None
    candidates.sort(reverse=True)  # pick largest (helps avoid tiny placeholders)
    return candidates[0][1]

def main():
    cwd = os.getcwd()
    run_log = os.path.join(cwd, "wrapper_run.log")
    with open(run_log, "w", encoding="utf-8") as log:
        def lprint(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=log)

        lprint("=== wrapper start ===")
        lprint("cwd:", cwd)
        lprint("sys._MEIPASS:", getattr(sys, "_MEIPASS", None))
        try:
            msi_path = find_msi_in_bundle()
            lprint("found msi:", msi_path)
            if not msi_path:
                lprint("ERROR: no MSI found inside bundle or next to exe.")
                lprint("Check that you used --add-data \"Dplus.installer.msi;.\" and that the filename matches.")
                input("Press Enter to exit...")
                return

            size = os.path.getsize(msi_path)
            lprint("msi size (bytes):", size)
            if size < 1024:
                lprint("WARNING: MSI looks very small (<1KB). This suggests it might be an empty/placeholder file.")

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".msi")
            tmp.close()
            shutil.copyfile(msi_path, tmp.name)
            lprint("copied embedded MSI to temp:", tmp.name)

            # Create install log next to current working dir so you can open it later
            install_log = os.path.join(cwd, "msi_install.log")
            cmd = ["msiexec", "/i", tmp.name, "/l*v", install_log]
            lprint("running:", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                lprint("msiexec finished successfully.")
            except subprocess.CalledProcessError as e:
                lprint("msiexec failed, returncode:", getattr(e, "returncode", None))
                lprint("See install log:", install_log)
                lprint("Traceback:")
                traceback.print_exc(file=log)
                traceback.print_exc()
            except Exception:
                lprint("Unexpected exception while running msiexec:")
                traceback.print_exc(file=log)
                traceback.print_exc()
        except Exception:
            lprint("Unexpected overall exception:")
            traceback.print_exc(file=log)
            traceback.print_exc()
        finally:
            try:
                os.remove(tmp.name)
                lprint("removed temp msi:", tmp.name)
            except Exception as e:
                lprint("could not remove temp msi (maybe still in use):", e)

        lprint("=== wrapper end ===")


if __name__ == "__main__":
    main()

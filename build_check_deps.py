"""Used by build_gui_exe_conda.bat — avoids fragile quoting in .bat and shows clear errors."""
import sys

def main():
    mods = [
        'tkinterdnd2',
        'torch',
        'onnxruntime',
        'faster_whisper',
        'ctranslate2',
    ]
    for name in mods:
        try:
            __import__(name)
        except ImportError as e:
            print(f'ERROR: missing module "{name}": {e}', file=sys.stderr)
            print(f'  conda activate <env> && pip install -r requirements.txt', file=sys.stderr)
            return 1
    try:
        __import__('PyInstaller')
    except ImportError as e:
        print(f'ERROR: missing PyInstaller: {e}', file=sys.stderr)
        print('  pip install -r requirements-build.txt', file=sys.stderr)
        return 1
    print('Dependency check OK.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

# env.sh
# Usage:
#   source ./env.sh export   # export variables into current shell
#   source ./env.sh echo     # print the export commands

# --- arg check (assumes you source this file) ---
if [ -z "$1" ]; then
  echo "Usage: source ./env.sh {export|echo}"
  return 1 2>/dev/null
fi

if [ "$1" != "export" ] && [ "$1" != "echo" ]; then
  echo "Error: use 'export' or 'echo'."
  return 1 2>/dev/null
fi

MODE="$1"

# --- compute values based on *current working directory* (like your original) ---
CWD="${PWD:-$(pwd -P)}"

# NOTE: original had PYTHON_PATH; correct var name is PYTHONPATH
PYTHONPATH_NEW="${CWD}/neon/py:${CWD}/XLB${PYTHONPATH:+:${PYTHONPATH}}"
LD_LIBRARY_PATH_NEW="${CWD}/neon/build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# --- apply or print ---
if [ "$MODE" = "export" ]; then
  export PYTHONPATH="$PYTHONPATH_NEW"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_NEW"
  echo "PYTHONPATH and LD_LIBRARY_PATH exported."
else
  printf 'export PYTHONPATH=%q\n' "$PYTHONPATH_NEW"
  printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH_NEW"
fi

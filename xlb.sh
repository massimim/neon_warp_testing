set -e
set -x

if [ -z "$1" ]; then
  echo "Usage: $0 <branch_name> [ssh|http]"
  echo "  <branch_name>: The branch name to use."
  echo "  [ssh|http]: Optional. The protocol to use for cloning. Default is 'http'."
  exit 1
fi

BRANCH_NAME=$1
PROTOCOL=${2:-http}

if [ "$PROTOCOL" != "ssh" ] && [ "$PROTOCOL" != "http" ]; then
  echo "Error: Invalid protocol. Use 'ssh' or 'http'."
  exit 1
fi

if [ "$PROTOCOL" == "ssh" ]; then
  REPO_URL="git@github.com:hsalehipour/XLB.git"
else
  REPO_URL="https://github.com/hsalehipour/XLB.git"
fi

echo "Cloning branch '$BRANCH_NAME' using protocol '$PROTOCOL'..."
git clone -b $BRANCH_NAME $REPO_URL
